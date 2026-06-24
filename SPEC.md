# SPEC — Modelo Preditivo de Resultados do Mata-Mata da Copa do Mundo 2026

> Status: rascunho v1 (fonte de dados verificada) · Última atualização: 2026-06-24 · Stack: Python · Workflow: spec-first

---

## 1. Problema e escopo

### 1.1 Objetivo
Prever o resultado das partidas do **mata-mata** da Copa do Mundo FIFA 2026 a partir de dados
de desempenho coletados durante a **fase de grupos** (em andamento) e de fatores contextuais do
confronto.

### 1.2 Pergunta central
Dado um confronto de mata-mata entre dois times A e B, qual a distribuição de probabilidade do
**resultado ao final dos 90 minutos regulamentares**?

```
P(resultado_90min | features) → {vitória_A, empate, vitória_B}
```

Trata-se de um problema de **classificação multiclasse (3 classes)** com saída probabilística.

### 1.3 Por que "resultado em 90 minutos" e não "vencedor do confronto"
No mata-mata não há empate na classificação final (prorrogação + pênaltis decidem). Optamos por
modelar o resultado em 90 minutos por três motivos:

1. **Máximo de dados de treino** — todas as partidas de futebol (grupos, amistosos, eliminatórias,
   Copas anteriores) têm um resultado em 90 minutos, então o modelo aproveita um histórico amplo.
2. **Interpretabilidade** — a classe "empate" carrega sinal útil: indica jogo equilibrado /
   candidato a prorrogação.
3. **Previsão de pênaltis é mal-condicionada** — séries de pênaltis têm amostra pequena e alta
   aleatoriedade; forçá-las no target degradaria o modelo. A classe "empate" funciona como proxy
   de "jogo aberto".

### 1.4 Convenção de rotulagem (evita ambiguidade home/away)
A Copa tem sede única (sem mando de campo real), então não usamos home/away. Definimos:

- **Time A** = sempre o time com **maior ranking FIFA** no momento da partida.
- **Time B** = o adversário.

Isso fixa a orientação de todas as features de diferença (`delta = A − B`) e remove ambiguidade
no target.

### 1.5 Fora de escopo (v1)
- Previsão de placar exato (gols por time).
- Previsão de quem avança em séries de pênaltis.
- Modelagem ao vivo intra-partida (planejada como evolução futura — ver §8).
- Escalações, lesões e suspensões (requer dados em tempo real + NLP).

---

## 2. Dados e fonte

> **Estratégia: 100% gratuita.** A verificação real da API (2026-06-24, §2.7) descartou usar a
> API-Football como fonte da WC2026. Adotamos fontes livres distintas para treino e para inferência.

### 2.1 Fonte de inferência (WC2026) — football-data.org (free) · VERIFICADO (2026-06-24)
- **Acesso confirmado:** competição `WC` (id 2000), temporada **2026-06-11 → 2026-07-19**,
  `currentMatchday=3` (fase de grupos em andamento). Plano gratuito, **10 req/min**.
- **Estrutura verificada (104 partidas, formato de 48 seleções):**
  `GROUP_STAGE` 72 · `LAST_32` 16 · `LAST_16` 8 · `QUARTER_FINALS` 4 · `SEMI_FINALS` 2 ·
  `THIRD_PLACE` 1 · `FINAL` 1. **As partidas do mata-mata já aparecem agendadas** (alvo da previsão).
- **Disponível por partida:** `score` (final/intervalo/vencedor/duração regular‑prorrogação),
  `stage`, `matchday`, `group`, datas, `venue`. **Standings** com 12 grupos (pontos, gols pró/contra,
  saldo, jogos).
- **NÃO disponível (free) — verificado:** **nenhum bloco de estatísticas por partida** (sem chutes,
  sem posse), sem escalações/cartões, **sem xG**. Temporadas históricas (2022, 2018…) **restritas**
  no free (só a temporada corrente). → motiva a arquitetura de features em duas camadas (§3.0).
- **Uso:** fonte da camada base (Tier 1) — resultados (target) + agregados de gols da fase de grupos.

### 2.2 Fonte de treino histórico — StatsBomb Open Data (free)
- **Cobertura confirmada (GitHub, gratuito):** competition_id 43 — **WC2022 e WC2018**; além de
  **Euro 2024, Euro 2020 e Copa América 2024** (torneios de seleções recentes e relevantes para
  mata-mata). Inclui dados de eventos e **xG do StatsBomb**.
- **Uso:** conjunto de treino/validação. Cobre a intenção de "últimas 3–4 competições de seleções"
  (inviável via API-Football free, que só libera 2022–2024).

### 2.3 Fonte secundária / cross-check — API-Football (free)
- **Plano:** gratuito, 100 req/dia. **Não acessa 2026** (só temporadas 2022–2024 — verificado).
- **Uso restrito:** validação cruzada da WC2022 e coleta de partidas internacionais de 2022–2024
  (eliminatórias, Nations League, Copa América/Euro) para **enriquecer o treino**, se necessário.
- Endpoints: `/fixtures`, `/fixtures/statistics`, `/teams/statistics`, `/standings`.

### 2.4 Princípio de paridade treino↔inferência (DECISÃO MAIS IMPORTANTE DO PROJETO)
As features de inferência da WC2026 (football-data.org free) são **só gols + contexto** — sem
chutes, posse ou xG. Para evitar distribution shift, **o treino deve usar exatamente a mesma camada
de features ativa na inferência** (ver arquitetura de duas camadas em §3.0): o modelo de produção
(Tier 1) treina só com gols+contexto, mesmo tendo o StatsBomb com dados ricos. Os dados ricos só
entram num modelo Tier 2 separado, em que treino e inferência compartilham a mesma fonte de
chutes/xG.

> ⚠️ **É a violação mais fácil de cometer "no calor do código"** — o StatsBomb oferece xG/posse de
> bandeja no treino, e usá-los infla a métrica de validação enquanto a previsão real (sem esses
> campos) degrada silenciosamente.

**Checklist de paridade — rodar ANTES de qualquer `fit()`:**
- [ ] A lista de colunas do dataset de treino é **idêntica** à do dataset de inferência (mesmo nome,
      mesma ordem). Assertar em código: `assert train_cols == infer_cols`.
- [ ] Nenhuma feature do treino vem de campo **ausente na inferência** (chutes, posse, xG real do
      StatsBomb) — a menos que `tier2_enabled=True` em ambos.
- [ ] A mesma `tier` está ativa nos dois lados (um modelo Tier 1 nunca prevê com features Tier 2 e
      vice-versa).
- [ ] A mesma fórmula/normalização é aplicada a treino e inferência (fit do scaler só no treino).
- [ ] Teste automatizado em `tests/` que falha se as colunas divergirem.

### 2.5 Estratégia de coleta e cache
- Persistir todas as respostas em cache local (JSON/Parquet) para respeitar limites de requisição e
  garantir reprodutibilidade.
- Camada de coleta separada da camada de features (raw → processed).
- StatsBomb Open Data: clonar/baixar o repositório uma vez (não tem rate limit de API).

### 2.6 Granularidade e janela temporal
- **Treino:** torneios de seleções recentes via StatsBomb (WC2022, WC2018, Euro 2024/2020, Copa
  América 2024) — equilíbrio entre volume e recência de estilo. Copas muito antigas (estilo/regras
  divergentes) ficam de fora.
- **Inferência:** agregados da fase de grupos da WC2026 (3 jogos/time) + ranking/contexto vigentes.

### 2.7 Disponibilidade de xG — VERIFICADO (2026-06-24)
Verificação real executada na conta do usuário (`src/collect/check_xg_coverage.py` + chamadas
diretas a `v3.football.api-sports.io`):

- ❌ **WC2026 indisponível no plano Free** da API-Football — erro explícito da API:
  *"Free plans do not have access to this season, try from 2022 to 2024."*
- ❌ **xG ausente mesmo na WC2022 free** — as estatísticas disponíveis são posse, chutes (no gol/
  fora/dentro-fora da área/bloqueados), passes, faltas, defesas, cartões — **sem `expected_goals`**.
- ❌ **football-data.org free** não fornece xG.
- ⚠️ **FBref/Opta** perdeu a licença em jan/2026 — sem xG atualizado da temporada corrente.

**Conclusão:** xG não está disponível para a WC2026 por via gratuita. O modelo de produção é
**goals-based (Tier 1, §3.1)**; chutes/xG só entram como **camada opcional via scraping (Tier 2,
§3.3)**, com degradação graciosa.

### 2.8 Ranking FIFA — fonte externa estática
Nenhuma das fontes acima fornece o ranking FIFA, que é insumo central (convenção A/B em §1.4 e
features de §3.1–3.2). Mantê-lo como **snapshot estático** (CSV versionado) extraído da publicação
oficial da FIFA, atualizado manualmente a cada janela de ranking. Baixo esforço, alto valor.

---

## 3. Features planejadas

Todas as features de confronto são orientadas pela convenção A = maior ranking FIFA (§1.4).
Onde aplicável, computar como **delta** (`valor_A − valor_B`) além dos valores absolutos.

### 3.0 Arquitetura de features em duas camadas (decisão de projeto)
A verificação (§2.7) mostrou que o football-data.org free **não fornece chutes/posse por partida**
para a WC2026 — só placar, classificação e contexto. Por isso o modelo é estruturado em camadas:

- **Camada base (Tier 1) — OBRIGATÓRIA, 100% gratuita.** Apenas features de **gols + contexto**,
  disponíveis garantidamente pelo football-data.org free. **O modelo deve treinar, validar e prever
  usando somente esta camada.** É o caminho de produção.
- **Camada de enriquecimento (Tier 2) — OPCIONAL, com degradação graciosa.** Features de
  **chutes/xG** obtidas por scraping (FotMob / FootyStats / RealGM). Se a fonte quebrar, ficar
  indisponível ou violar ToS, **o modelo roda normalmente sem ela** (cai para Tier 1). Nunca é
  pré-requisito.

**Paridade treino↔inferência por camada:** o treino deve usar exatamente a camada ativa na
inferência. Modelo Tier 1 treina só com gols+contexto (mesmo tendo StatsBomb com dados ricos,
estes **não** entram como features no Tier 1). Quando o Tier 2 estiver ativo, chutes/xG entram em
treino e inferência com a **mesma fórmula**. São, na prática, dois modelos versionados.

### 3.1 Camada base (Tier 1) — desempenho na fase de grupos · OBRIGATÓRIA

| Categoria | Features | Fonte |
|-----------|----------|-------|
| Resultado bruto | Pontos, saldo de gols, gols marcados, gols sofridos | football-data.org (standings + matches) |
| Eficiência por gols | Gols marcados/jogo, gols sofridos/jogo, clean sheets | idem |
| Qualidade do adversário | Média do ranking FIFA dos adversários nos grupos | ranking externo (§2.8) |
| Forma recente | Resultado dos últimos 2–3 jogos (ordinal V/E/D) | football-data.org |

### 3.2 Camada base (Tier 1) — contexto do confronto · OBRIGATÓRIA

| Categoria | Features | Fonte |
|-----------|----------|-------|
| Diferença de ranking FIFA | `delta_ranking = ranking_A − ranking_B` | ranking externo (§2.8) |
| Histórico H2H | Win rate histórico A vs B (com prior/shrinkage — amostra pequena) | StatsBomb / histórico |
| Confederação | CONMEBOL / UEFA / AFC / CAF / CONCACAF / OFC (efeito de estilo) | estático |
| Fase da competição | LAST_32 / oitavas / quartas / semi / final (pressão crescente) | football-data.org (`stage`) |
| Descanso | Dias desde o último jogo de cada time (`delta_descanso`) | datas das partidas |

### 3.3 Camada de enriquecimento (Tier 2) — chutes/xG · OPCIONAL, degradação graciosa
Só ativa se houver fonte de scraping funcional. Implementar atrás de um flag `tier2_enabled`; em
qualquer falha (timeout, parse, bloqueio), **logar e seguir com Tier 1**.

| Categoria | Features (se disponível) |
|-----------|--------------------------|
| Eficiência ofensiva | xG médio/jogo, chutes a gol/jogo, chutes totais/jogo, conversão (gols/xG) |
| Solidez defensiva | xG concedido médio, chutes a gol concedidos/jogo |
| Controle | % posse média (quando a fonte fornecer) |

Cada feature do Tier 2 carrega um flag `*_disponivel`; o pipeline nunca substitui ausência por
valor arbitrário — a feature simplesmente não entra no vetor do modelo Tier 1.

### 3.4 Features deliberadamente excluídas (v1)
- **Escalação / lesões:** requer dados em tempo real e NLP.
- **Odds de casas de apostas:** sinal forte, mas cria dependência externa e potencial *leakage*
  na avaliação independente do modelo. Pode entrar como benchmark externo, não como feature.

### 3.5 Pré-processamento
- **Normalização:** padronização (z-score) das features numéricas contínuas.
- **Encoding:** one-hot para confederação; ordinal para fase da competição e forma recente.
- **Valores ausentes:** imputação explícita + flag de ausência. Evitar imputar silenciosamente.
- **Suavização de H2H:** aplicar prior (ex.: shrinkage para a taxa base) dado o tamanho de amostra.

---

## 4. Arquitetura do modelo

Abordagem incremental — começar simples, subir complexidade apenas se justificado pela métrica.

### 4.1 Baseline
- **Regressão logística multinomial** (3 classes) sobre o conjunto de features padronizadas.
- Serve como referência interpretável e de baixa variância.

### 4.2 Modelo principal
- **Gradient boosting (XGBoost / LightGBM)** com objetivo `multi:softprob`.
- Captura interações não-lineares entre features (ex.: ranking × forma × descanso).

### 4.3 Calibração de probabilidades
- A saída precisa ser **probabilisticamente calibrada** (log-loss é a métrica principal).
- Aplicar calibração pós-treino se necessário (Platt scaling / isotônica) e avaliar com
  **reliability diagram**.

### 4.4 Validação
- **Validação temporal** (não usar k-fold aleatório que vaza futuro→passado): treinar em Copas
  anteriores, validar em Copas mais recentes; para 2026, prever rodada a rodada do mata-mata.
- Cuidado com **data leakage**: nenhuma feature pode usar informação posterior à partida prevista.

### 4.5 Saída do sistema
Para cada confronto: vetor `[P(vitória_A), P(empate), P(vitória_B)]` somando 1, mais a classe
de maior probabilidade e uma medida de confiança.

---

## 5. Métrica de avaliação

### 5.1 Métrica principal — Log-loss (cross-entropy multiclasse)
Penaliza previsões confiantes e erradas; recompensa probabilidades bem calibradas. Mais adequada
que acurácia porque o valor do modelo está na **qualidade das probabilidades**, não só no acerto
da classe.

### 5.2 Métricas secundárias
- **Acurácia** da classe de maior probabilidade (interpretável, mas insuficiente isolada).
- **Brier score** multiclasse (complementa o log-loss na avaliação de calibração).
- **Reliability diagram** (inspeção visual de calibração).

---

## 6. Critério de sucesso

O modelo é considerado bem-sucedido se, na validação temporal:

1. **Log-loss < baseline ingênuo**, onde o baseline é o melhor entre:
   - distribuição uniforme (0,33 / 0,33 / 0,33), e
   - frequência histórica de resultados de mata-mata de Copas (prior fixo).
2. **Calibração razoável** — reliability diagram próximo da diagonal (sem excesso de confiança).
3. **Acurácia ≥ baseline de maioria** (sanidade — não regredir abaixo de "sempre o favorito FIFA").

Meta-alvo (a refinar após o baseline): superar o baseline ingênuo de log-loss com margem
consistente em múltiplas rodadas de validação, não apenas em uma.

---

## 7. Estrutura de projeto (proposta)

```
polvo-paul-26/
├── SPEC.md
├── data/
│   ├── raw/              # cache: football-data.org + StatsBomb open-data + API-Football
│   ├── reference/        # snapshot de ranking FIFA (CSV versionado, §2.8)
│   └── processed/        # features prontas para treino
├── src/
│   ├── collect/          # clientes de fonte + cache
│   │   ├── footballdata.py     # WC2026 (Tier 1)
│   │   ├── statsbomb.py        # treino histórico
│   │   ├── scrape_shots.py     # Tier 2 opcional (degradação graciosa)
│   │   └── check_xg_coverage.py
│   ├── features/         # Tier 1 (obrigatório) + Tier 2 (opcional)
│   ├── models/           # baseline + gradient boosting + calibração
│   └── evaluate/         # métricas, validação temporal, plots
├── notebooks/            # exploração
└── tests/
```

---

## 8. Evolução futura (pós-v1)

- **Previsão ao vivo intra-partida** (atualizar probabilidades conforme o jogo evolui: placar,
  cartões, tempo restante). Requer fonte paga com endpoint live (ex.: API-Football plano Pro,
  `/fixtures/live`) — o tier gratuito só oferece dados atrasados.
- **Variante xG-enabled** usando o xG real do StatsBomb para backtesting histórico (não aplicável à
  WC2026, que não tem xG gratuito — ver §2.4).
- **Modelagem de placar** (distribuição de gols por time, ex.: Poisson/Dixon-Coles).
- **Etapa de desempate** (modelo/heurística separada para prorrogação e pênaltis).
- **Incorporação de escalações/lesões** quando houver pipeline de dados em tempo real.

---

## 9. Decisões resolvidas e em aberto

### Resolvidas nesta sessão (verificado em 2026-06-24)
- [x] **Fonte de dados** — estratégia 100% gratuita: football-data.org (free) para WC2026 +
      StatsBomb Open Data para treino histórico + API-Football (free) como cross-check (§2.1–2.3).
- [x] **WC2026 na API-Football Free** — confirmado **indisponível** (só 2022–2024). Descartada como
      fonte da inferência.
- [x] **football-data.org free verificado** — WC2026 acessível (104 jogos, 12 grupos, mata-mata
      agendado), mas **sem stats por partida** (sem chutes/posse) e histórico restrito (§2.1).
- [x] **Disponibilidade de xG** — confirmado **indisponível por via gratuita** para a WC2026 (§2.7).
- [x] **Arquitetura de features** — duas camadas: Tier 1 goals-based **obrigatório**, Tier 2
      chutes/xG via scraping **opcional com degradação graciosa** (§3.0).
- [x] **Janela de treino** — StatsBomb: WC2022, WC2018, Euro 2024/2020, Copa América 2024 (§2.6).
- [x] **Previsão ao vivo** — fora da v1; movida para evolução futura (§8). v1 é **pré-partida**.

### Em aberto (a resolver na implementação)
- [ ] 🔴 **BLOQUEADOR #1 — snapshot de ranking FIFA (CSV, §2.8).** Sem ele a convenção A/B (§1.4)
      não funciona e **nenhuma feature de contexto fecha** (delta_ranking, qualidade do adversário).
      Resolver **antes de qualquer modelo** — ver ordem em §10.
- [ ] Definir o tratamento exato de H2H (prior/shrinkage) e seu peso.
- [ ] Escolher entre XGBoost e LightGBM após primeiro benchmark.
- [ ] Escolher a fonte de scraping do Tier 2 (FotMob / FootyStats / RealGM) e checar ToS.
- [ ] Validar a profundidade histórica de fixtures que o football-data.org free expõe para a WC2026
      (e se há atraso relevante para previsão pré-jogo do mata-mata).
- [ ] Definir a fórmula exata do proxy de xG (pesos por tipo de chute / localização) — só Tier 2.

---

## 10. Ordem de implementação (sequência obrigatória)

A ordem abaixo existe porque dois passos são **bloqueadores** de tudo que vem depois. Não pular.

1. 🔴 **Ranking FIFA (CSV de referência) — BLOQUEADOR #1.** Sem o ranking não há convenção A/B
   (§1.4) nem features de contexto (§3.2). É o **primeiro entregável**, antes de qualquer coleta de
   features ou modelo. Fonte: snapshot oficial da FIFA → `data/reference/fifa_ranking.csv`.
2. **Coletores Tier 1.** `footballdata.py` (WC2026: fixtures, standings, resultados) e
   `statsbomb.py` (treino histórico). Persistir em cache (§2.5).
3. **Build de features Tier 1** (gols + contexto), unindo resultados + ranking FIFA.
4. 🟡 **GATE DE PARIDADE (§2.4) — antes de qualquer `fit()`.** Rodar o checklist de paridade e o
   teste automatizado de igualdade de colunas treino↔inferência. **Nenhum modelo é treinado antes
   deste gate passar.**
5. **Baseline** (regressão logística multinomial) → métrica vs. baseline ingênuo (§5–6).
6. **Modelo principal** (gradient boosting) + calibração.
7. *(Opcional, a qualquer momento depois)* **Tier 2** (scraping de chutes/xG) com degradação
   graciosa — nunca bloqueia os passos acima.
