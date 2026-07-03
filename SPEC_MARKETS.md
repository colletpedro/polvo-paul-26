# SPEC_MARKETS — Modo 2: Mercados de Contagem (Gols, Escanteios, Cartões)

> Status: rascunho v1 · Última atualização: 2026-07-01 · Stack: Python · Workflow: spec-first
> Documento irmão do `SPEC.md` (Modo 1 — resultado/vencedor, CONGELADO). Este spec cobre
> exclusivamente o Modo 2. Nada aqui edita ou depende de alterar o Modo 1.

---

## 0. Relação com o Modo 1 (fronteira dura)

O projeto passa a ter **dois modos separados e versionados independentemente**:

- **Modo 1 — Resultado/vencedor.** XGBoost + calibração sigmoid, treinado, testado, **congelado**,
  rodando ao vivo na WC2026. Documentado em `SPEC.md` e `docs/RESULTS.md`. **Este spec não o altera.**
- **Modo 2 — Mercados de contagem.** Gols, escanteios e cartões por time, modelados como distribuições
  de contagem, das quais os mercados (over/under, placar, ambas marcam, etc.) são **derivados**.

**Cerca elétrica (§9 item 0):** o desenvolvimento do Modo 2 não pode tocar em nenhum arquivo de que o
Modo 1 depende sem que `tests/test_parity.py` E um novo `tests/test_mode1_frozen.py` passem. O Modo 1
está em produção ao vivo; qualquer regressão silenciosa nele é inaceitável.

---

## 1. Problema e escopo

### 1.1 Objetivo
Para cada partida de seleções, prever a **distribuição de contagem** de três quantidades, por time:

```
gols_A, gols_B      → Poisson (ou Poisson bivariado / Dixon-Coles p/ correlação)
escanteios_A, _B    → Poisson
cartões_A, _B       → Negative Binomial (sobredispersão — evento raro, alta variância)
```

A partir das distribuições ajustadas, **derivar os mercados** (o usuário deriva, o modelo entrega a
distribuição): over/under de gols/escanteios/cartões, placar exato, ambas marcam, total por time, etc.

### 1.2 Por que contagem por time e não classificação direta de mercado
Um modelo de contagem por time é **uma fonte, muitos mercados**. Ajustada a distribuição de gols de A e
de B, qualquer mercado de gols sai por integração/soma da distribuição — over 2.5, placar exato, BTTS,
handicap. Treinar um classificador por mercado (over 9.5 escanteios sim/não) multiplicaria modelos e
desperdiçaria a estrutura de contagem. **Decisão: modelar a contagem, derivar o mercado.**

### 1.3 Domínio de aplicação — SOMENTE torneios de seleções
A sondagem (§2.6) confirmou empiricamente que **não há fonte gratuita com contagem estruturada para
jogos de seleção fora de torneios principais** (eliminatórias, Nations League, amistosos de baixo
nível). Portanto o universo de treino e inferência do Modo 2 são **torneios de seleções**: WC, Euro,
Copa América, AFCON. Ligas de clube ficam **fora** (ver §2.5 — fallback condicional, não plano).

### 1.4 Fora de escopo (v1)
- Previsão ao vivo intra-partida de contagem (mesma restrição de fonte live do Modo 1, §8 do SPEC.md).
- Mercados de jogador individual (chutes de X, cartão de Y) — exige dado a nível de jogador na inferência.
- Modelagem de correlação entre os três alvos (ex.: jogo faltoso → mais cartões E mais escanteios).
  Fica como evolução futura; v1 modela os três de forma independente.

---

## 2. Dados e fonte

> **Estratégia: gratuita.** A verificação empírica (§2.6) confirmou cobertura de torneios recentes
> **e** profundidade histórica do SofaScore (passo 0, 2026-07-01). Com isso, a arquitetura padrão passa
> a ser **SofaScore-only** para treino e inferência (paridade trivial, §2.4); StatsBomb recua para
> **validador / ground truth** da Camada 3 (§2.3). O antigo BLOQUEADOR #1 está **resolvido** (§10 passo 0).

### 2.1 Camada 1 — Ground truth / validador · StatsBomb Open Data · VERIFICADO (2026-07-01)
- **Cobertura confirmada:** 314 partidas em 6 torneios de seleção — WC2022 (64), WC2018 (64),
  Euro 2024 (51), Euro 2020 (51), Copa América 2024 (32), AFCON 2023 (52). Já coletado e cacheado.
- **Derivação de contagem confirmada empiricamente (`check_counts_coverage.py`):**
  - **Escanteio** = evento `Pass` com `pass.type.name == "Corner"` (contagem por time).
  - **Cartão** = evento `Bad Behaviour` com `bad_behaviour.card.name` ∈ {Yellow Card, Red Card,
    Second Yellow}.
  - **Gols** = já disponíveis via placar.
- **Papel:** fonte confiável de referência. Valida o pipeline, valida a engenharia de atributos,
  serve de ground truth para a validação cruzada (§2.3) e para os testes automatizados. Sem ToS,
  sem scraping, estável.
- **Limite:** não cobre WC2026 em andamento, nem eliminatórias/amistosos.

### 2.2 Camada 2 — Fonte operacional · SofaScore (API interna não-oficial) · VERIFICADO
- **Acesso:** API interna `api.sofascore.com/api/v1`. Bloqueia `urllib`/`curl` padrão por **TLS
  fingerprinting (Varnish)**; contornado com `curl_cffi` (impersonação de TLS do Chrome). VERIFICADO.
- **Cobertura de `/event/{id}/statistics` — recente, VERIFICADO (2026-06-30):**
  - Euro 2024, Copa América 2024, WC2026, Brasileirão Série A → **100%**.
  - Amistosos internacionais (ID 851) → **~20%** (só jogos de alto nível).
- **Profundidade histórica — VERIFICADO (2026-07-01, `check_sofascore_depth.py`):** `/statistics` com
  **escanteios + cartões amarelos ~100%** em TODOS os torneios principais testados, em temporadas
  antigas. Piso histórico: **~2008–2010 para WC e Euro** (WC2010/2014/2018/2022, Euro2008/2012/2016/2020
  todos 5/5); **~2015 para confederações menores** (Copa América, AFCON, Asian Cup, Gold Cup a partir de
  ~2015 todos 5/5; Asian Cup **2011 → 404**, primeiro piso observado). **Isto encerra o BLOQUEADOR #1:**
  o volume de treino histórico do Modo 2 vem do próprio SofaScore.
- **IDs de `unique-tournament` confirmados (passo 0):** WC=**16**, Euro=**1**, Copa América=**133**,
  AFCON=**270**, AFC Asian Cup=**246**, CONCACAF Gold Cup=**140**, UEFA Nations League=**10783**
  (finalmente localizado — os antigos 113/1061 eram chutes errados). Eliminatórias (fora do domínio,
  §1.3, mas mapeadas): WCQ UEFA=**11**, CONMEBOL=**295**, CAF=**13**, CONCACAF=**14**, AFC=**308**.
- **Campos confirmados (consistentes entre competições testadas):** `cornerKicks`, `shotsOnGoal`,
  `ballPossession` sempre presentes; `yellowCards`/`redCards` de **presença condicional** — ausentes
  quando o valor é 0 (tratar ausência = 0).
- **Rate limit — VERIFICADO:** 20/20 requisições com `sleep(3s)` retornaram HTTP 200, sem bloqueio no
  nosso ambiente. **Não** assumir o limite "de fórum" de 25–30s; medir sob volume real de coleta.

#### Requisitos VERIFICADOS revelados pelo passo 0 (obrigatórios na implementação)
1. **xG (`expectedGoals`) só existe a partir de ~2020** (Euro2020, WC2022, Asian Cup 2023, 2024+;
   ausente antes). ⇒ tratar `expectedGoals` como **Tier 2 opcional** do Modo 2, **nunca requisito**.
   O modelo de contagem base (gols/escanteios/cartões) **não depende de xG**. Se xG entrar como feature,
   o corpus com xG começa em ~2020, não antes — restrição de paridade a respeitar.
2. **Resolução de ID por nome é frágil** — o passo 0 casou "WCQ" com **basquete FIBA** por homonímia
   (`/search/all` retorna FIBA/futsal/rugby primeiro). ⇒ o coletor `sofascore.py` **DEVE validar
   `category.sport.name == "Football"` e a categoria/confederação por ID** antes de aceitar qualquer
   torneio. Nunca confiar só no nome retornado pela busca.
3. **Amistosos (851) degradam para ~20% de cobertura antes de ~2021.** ⇒ entram como **bônus
   condicional, nunca base**. Avaliar **excluí-los do treino de cartões** (viés de intensidade: jogo
   amistoso tem arbitragem e disciplina diferentes de torneio oficial). Questão em aberto — ver §9.
- **Papel:** fonte primária de treino histórico **e** inferência de contagem ao vivo na WC2026
  (única fonte que cobre a Copa em andamento).
- **Risco registrado:** fonte não-oficial; o uso pode não estar coberto pelos Termos de Serviço do
  SofaScore. Dataset coletado é para uso pessoal de ML, **sem redistribuição**. Estrutura de endpoint
  pode mudar sem aviso — mitigado pela Camada 3.

### 2.3 Camada 3 — Validação cruzada como GATE de unificação (não detector passivo)
Nos jogos presentes nas **duas** fontes (Euro 2024, Copa América 2024 — StatsBomb tem eventos,
SofaScore tem `/statistics`), comparar automaticamente escanteios, cartões, posse e finalizações.

Isso cumpre **três** funções, em ordem de importância:

1. **Gate de paridade de definição.** Se, no mesmo jogo, StatsBomb deriva 5 escanteios e SofaScore
   reporta 6, existe divergência de definição a tratar antes de qualquer unificação. Este é o teste
   §2.4 do SPEC.md aplicado **entre fontes**.
2. **Piso de erro de medição.** A discrepância média entre as fontes nos mesmos jogos é o **piso de
   erro** do Modo 2 — nenhum modelo prevê contagem com precisão maior que a das próprias fontes.
   Quantificar isso calibra a expectativa de sucesso (§6).
3. **Detecção de drift.** Se o SofaScore mudar endpoint/estrutura, a comparação contínua acusa.

**Papel deste gate após o passo 0 (§10 passo 1):** com a arquitetura já definida como **SofaScore-only**
(§2.2, §2.4), StatsBomb **não** é candidato a unificação de corpus — é o **ground truth** que valida se
as contagens do SofaScore batem com uma fonte de evento confiável. O gate agora confirma que a **fonte
operacional é fidedigna** antes de confiarmos no histórico dela para treino. Aprovar paridade de
definição **e** quantificar o piso de erro de medição (§6.2 item 2) continua obrigatório.

### 2.4 Princípio de paridade treino↔inferência (herdado do §2.4 do SPEC.md)
Vale integralmente para o Modo 2, em duas dimensões:
- **Temporal:** as features de contagem usadas no treino devem ser idênticas (nome, ordem, definição)
  às da inferência. Gate automatizado em `tests/`, análogo ao `test_parity.py` do Modo 1.
- **Entre fontes:** se treino e inferência usarem fontes diferentes (ex.: treino StatsBomb, inferência
  SofaScore WC2026), a definição de cada contagem deve ser validada equivalente pela Camada 3 **antes**
  do `fit()`. Divergência não-tratada = distribution shift silencioso.

> **ADOTADO (passo 0 concluído, 2026-07-01):** a arquitetura principal é **mesma fonte nos dois lados**
> — **SofaScore no treino E na inferência** —, pois a profundidade histórica foi provada suficiente
> (§2.2). Isso torna a paridade entre fontes **trivial** (só resta a paridade temporal). O plano B
> (treino StatsBomb + inferência SofaScore) fica como **fallback**, não caminho padrão, e só seria
> ativado se o gate da Camada 3 reprovasse a fidedignidade do SofaScore.

### 2.5 Ligas de clube — fallback condicional, NÃO plano
football-data.co.uk tem `HC/AC` (escanteios) e `HY/AY/HR/AR` (cartões) para ~20 ligas, CSV pronto,
sem ToS. **Mas cobre só clubes.** Entra **apenas** se o volume de treino de seleção (§2.1 + §2.2 pós-
verificação) for insuficiente para os modelos rodarem — e, nesse caso, reabre o problema de
transferência clube→seleção (distribution shift de domínio), que passa a exigir tratamento explícito
e medição de degradação contra os jogos de seleção. Registrado como risco, não como caminho padrão.

### 2.6 Sondagens de cobertura já executadas (2026-07-01)
- `src/collect/check_counts_coverage.py` — confirmou derivação de escanteio/cartão no StatsBomb em
  todas as 6 competições, estrutura 100% consistente. 314 partidas, 86 de mata-mata.
  **Derivação canônica (corrigida 2026-07-02 pelo gate da Camada 3):** cartão vem de DOIS tipos de
  evento — `Bad Behaviour.card` **E** `Foul Committed.card` (a maioria dos amarelos); a derivação antiga
  só olhava `Bad Behaviour` e **subcontava ~70% dos amarelos**. Regra única em
  `statsbomb.derive_counts`: amarelo = Yellow + Second Yellow · vermelho = Red + Second Yellow.
  **Médias por jogo (314 partidas, contagem total dos dois times):** ~9.1 escanteios · ~3.5 amarelos ·
  ~0.13 vermelhos. Por competição (amarelos/jogo): WC2022 3.5 · WC2018 3.2 · Euro2024 4.5 · Euro2020 3.0 ·
  CopaAm2024 3.9 · AFCON2023 3.4. (Antes da correção os amarelos apareciam ~1/jogo — subestimados.)
- `src/collect/check_sofascore_coverage.py` — confirmou acesso via `curl_cffi`, 100% de cobertura de
  `/statistics` em torneios recentes + WC2026, campos-chave e rate limit.
- `src/collect/check_sofascore_depth.py` — passo 0 (2026-07-01): descoberta de IDs (Fase A) + cobertura
  histórica de `/statistics` em temporadas pré-2024 (Fase B). Confirmou profundidade até ~2008–2010
  (WC/Euro) e ~2015 (confederações menores). **Encerrou o BLOQUEADOR #1.**

### 2.7 Estratégia de coleta e cache (dois pipelines DISTINTOS)
"Coletar histórico" e "atualizar ao vivo" são pipelines separados, com cadência e risco diferentes:
- **Corpus histórico de treino:** coleta única, cache permanente (JSON/Parquet). Jogos antigos não
  mudam. Zero atualização recorrente — evita requisição desperdiçada e exposição de ToS à toa.
- **Feed ao vivo WC2026:** atualiza conforme os jogos do mata-mata terminam. Só este é "diário".

---

## 3. Volume de dados — leitura honesta

Diferente do Modo 1 (limitado a mata-mata → 47 linhas de treino), o Modo 2 usa **toda partida** como
observação de contagem, independentemente da fase — cada jogo entrega contagem por time em grupos E
mata-mata. Portanto:

- **Base StatsBomb:** 314 partidas × 2 times = **~628 observações time-partida** de contagem.
- **Com profundidade SofaScore (se confirmada):** potencialmente muito maior.

Isso é materialmente melhor que o Modo 1. **Ressalva por alvo:**
- **Gols e escanteios:** volume confortável para Poisson.
- **Cartões:** evento raro e sobredisperso; ~628 observações é aceitável mas com intervalos largos.
  É o alvo de maior incerteza. O modelo roda; a utilidade real se mede, não se assume (mesma disciplina
  do `docs/RESULTS.md` do Modo 1 — documentar underperformance honestamente se for o caso).

---

## 4. Features planejadas

Orientadas pela mesma convenção A/B do Modo 1 (A = maior ranking FIFA, §1.4 do SPEC.md), reaproveitando
`fifa_ranking_full.csv` e o mapa de aliases já existentes.

### 4.1 Features de contagem (médias históricas por time, pré-jogo)
| Categoria | Features |
|-----------|----------|
| Ataque (gols) | gols marcados/jogo, gols sofridos/jogo do adversário |
| Ataque (escanteios) | escanteios a favor/jogo, escanteios cedidos/jogo do adversário |
| Disciplina | cartões/jogo do time, cartões provocados/jogo do adversário |
| Contexto | ranking FIFA, delta_ranking, confederação, fase, descanso (herdados do Tier 1) |

### 4.2 Pré-processamento
- Contagens médias como taxa (rate) — insumo natural do Poisson.
- Presença condicional do SofaScore: campo ausente de cartão = 0 (nunca imputar valor arbitrário).
- Encoding de confederação/fase idêntico ao Modo 1, para consistência.

### 4.3 Definição temporal dos targets de contagem
Régua temporal por mercado (decisão fechada, não reabrir sem motivo forte):

- **GOLS:** target e feature = placar de 90min (`*_goals_reg` / normaltime do SofaScore),
  EXCLUINDO prorrogação e pênaltis. Escolha consciente, alinhada ao scoring de 90min do Modo 1
  (`live_score.py`) e à convenção de mercado de gols. Feature e target lêem o MESMO campo
  (`_team_side_counts`, `counts.py:141-142`) — sem mismatch histórico-sujo/alvo-limpo.

- **ESCANTEIOS e CARTÕES:** target e feature = jogo inteiro, INCLUINDO prorrogação. Não por
  escolha de conveniência, mas porque o SofaScore não expõe variante 90min dessas contagens em
  NENHUM ponto (treino ou inferência ao vivo). A observabilidade no scoring força jogo-inteiro:
  só se pode treinar pra prever o que se consegue observar ao pontuar, e a única contagem
  observável de escanteio/cartão na WC2026 (SofaScore) é a de jogo inteiro. É simétrica
  feature/target (nenhuma métrica mistura réguas dentro de si).

- **ASSIMETRIA CONHECIDA:** gols=90min, escanteios/cartões=jogo-inteiro. Afeta apenas jogos de
  mata-mata que foram à prorrogação (nunca jogos de grupo). LIMITAÇÃO: a régua jogo-inteiro de
  escanteio/cartão pode NÃO casar com a liquidação padrão de linhas de casa de aposta (usualmente
  90min). Qualquer benchmark do Modo 2 contra linha real de mercado exige confirmar antes a regra
  de liquidação da linha, sob pena de comparação inválida.

- **ESCOPO DE VERIFICAÇÃO do fix reg90 de gols:** confirmado por medição em 9/9 jogos
  ET-sem-pênalti, via oráculo StatsBomb (período+minuto), cobrindo 5 edições e 3 confederações
  (WC2018, Euro2020×4, Euro2024×2, CopaAm2024, AFCON2023). Não medido: 10 jogos pós-90min em
  CopaAm2021 (3) e AFCON2021 (7), ausentes do StatsBomb — aceites por inferência estrutural (campo
  consistente nas 5 edições testadas), aceite consciente. A superfície de risco real é ≤10 (só o
  subconjunto ET-depois-pênalti; jogos direto-nos-pênaltis têm normaltime=90min trivialmente).

- **Resolução de benchmark por campo.** O Modo 2 será avaliado em dois níveis, com escopo distinto
  por mercado: (a) *benchmark contra realizado* — válido para os três mercados (gols, escanteios,
  cartões), sem restrição, usando a régua temporal definida acima para cada campo; (b) *benchmark
  contra linha de mercado* — habilitado apenas para gols, cuja régua (90min via `*_goals_reg`)
  coincide com a liquidação padrão de linhas de gols. Para escanteios e cartões, o benchmark contra
  linha fica **bloqueado** até que a regra de liquidação da casa de referência seja confirmada: a
  régua jogo-inteiro destes campos inclui prorrogação, divergindo sistematicamente da liquidação
  90min exatamente nos jogos de mata-mata — o subconjunto onde o modelo opera ao vivo. A comparação
  contra realizado não é afetada por esta limitação.

---

## 5. Arquitetura do modelo

Um modelo de contagem por alvo, por time. Incremental — começar simples.

### 5.1 Gols
- **Poisson** por time (baseline). Evoluir para **Poisson bivariado / Dixon-Coles** se a validação
  mostrar correlação relevante entre gols de A e B. Deriva placar exato, over/under, BTTS.
- Target: placar de **90min** (`*_goals_reg`), excluindo prorrogação e pênaltis (§4.3).

### 5.2 Escanteios
- **Poisson** por time. Verificar dispersão; se sobredisperso, migrar para Negative Binomial.
- Target: contagem de **jogo inteiro** (inclui prorrogação — única régua exposta pela fonte, §4.3).

### 5.3 Cartões
- **Negative Binomial** por time (assume sobredispersão — cartão tem variância > média, o que o
  Poisson não captura). Deriva over/under de cartões, total por time.
- Target: contagem de **jogo inteiro** (inclui prorrogação, §4.3).

### 5.4 Calibração
- Distribuições de contagem precisam ser **calibradas** (lição do Modo 1: calibração foi o maior ganho).
- Avaliar com PIT histogram / reliability para contagem, não só ponto médio.

### 5.5 Baseline de comparação obrigatório
Cada alvo compara contra um baseline ingênuo: **taxa média do torneio como parâmetro Poisson fixo**
(ex.: "todo time faz a média histórica de escanteios"). O modelo só se justifica se superar isso —
mesmo critério de honestidade do §6 do SPEC.md. O baseline é Poisson-de-taxa-média-do-torneio para
**TODOS** os alvos, **inclusive cartões**: comparar o modelo NegBin de cartões contra um baseline
Poisson de taxa média é **intencional** — o baseline ingênuo é único (taxa média), independente da
família do modelo candidato.

### 5.6 Forma funcional do modelo v1 (PINADA antes de qualquer `fit()`)
**Caminho A (v1, padrão)** — modelo multiplicativo estilo Dixon-Coles simplificado, **sem parâmetro
ajustado por otimização**:

    λ_alvo(time) = ataque_shrunk(time) × defesa_shrunk(oponente) / média_do_torneio

- **ataque** = taxa A-FAVOR shrunk do time (janela N=10, shrink k=5, §4); **defesa** = taxa CONTRA
  shrunk do oponente. Reusa exatamente as features de `counts.py` — **zero parâmetro otimizado** no v1.
- **`média_do_torneio` = média global EXPANSIVA por-jogo** (só jogos com data < a do jogo corrente) —
  exatamente a que alimenta o prior do shrinkage em `counts.py` (`_global_prior_means`). É pura de
  (data, corpus), **leak-free** e **idêntica treino/inferência**. Caso-borda de prior vazio (primeiros
  jogos de WC2018): segue a MESMA convenção de `counts.py` — prior vazio ⇒ média `= 0.0`; como esses
  jogos nunca são pontuados na validação (2024), o divisor nunca é 0 no conjunto avaliado (guardar
  defensivamente mesmo assim: divisor 0 ⇒ sem previsão, nunca fabricar valor).
- **Sem fator casa** (sede neutra em torneios de seleção).
- **NegBin (cartões):** parâmetro de dispersão `r` **GLOBAL** (um por torneio inteiro), não por time.
- **Caminho B (escalada condicional):** GLM sobre as 25 features **só se** o v1 não superar o baseline
  §5.5. Registrado como escalada, **não** como plano.
- **Validação:** split **temporal explícito** (§7 — corte 2023-01-01, AFCON2023→validação),
  **codificado** (nunca k-fold aleatório).
- **Métrica de critério:** log-loss de mercado (§6, com o escopo por campo da §4.3 — realizado p/ os 3
  alvos, linha só p/ gols). **Diagnóstico:** RPS truncado em 20 + PIT.

---

## 6. Métrica e critério de sucesso

### 6.1 Métricas
- **Contagem (média):** MAE e RMSE do número previsto vs real, por alvo.
- **Distribuição:** log-loss dos mercados binários derivados (over/under). **Escopo por campo (§4.3):**
  contra **realizado** para os 3 alvos; contra **linha de mercado** apenas para **gols** (90min) —
  escanteios/cartões vs linha ficam bloqueados até confirmar a regra de liquidação da casa.
- **Calibração:** PIT / reliability da distribuição de contagem.

### 6.2 Critério de sucesso (por alvo, na validação temporal)
1. **Superar o baseline de taxa média** (Poisson fixo) em log-loss dos mercados derivados, medido
   contra **realizado** (válido p/ gols, escanteios e cartões) e, adicionalmente, contra **linha de
   mercado apenas para gols** (§4.3 — escanteios/cartões vs linha bloqueados até confirmar liquidação).
2. **Erro de contagem não pior que o piso de medição** quantificado pela Camada 3 (§2.3 função 2) —
   não se cobra do modelo precisão maior que a das fontes.
3. **Calibração razoável** da distribuição.

Meta honesta: como no Modo 1, um resultado negativo bem-diagnosticado (ex.: "cartões são imprevisíveis
com features pré-jogo") é resultado válido e documentável, não fracasso.

---

## 7. Validação
- **Validação temporal** com **atribuição explícita por edição** (corte limpo em **2023-01-01**),
  usando as datas REAIS de cada edição no corpus:

  | Conjunto | Edições | Datas reais (corpus) |
  |----------|---------|----------------------|
  | **TREINO** | WC2018 · Euro2020 · CopaAm2021 · AFCON2021 · WC2022 | 2018-06 · 2021-06/07 · 2021-06/07 · 2022-01/02 · 2022-11/12 |
  | **VALIDAÇÃO** | AFCON2023 · Euro2024 · CopaAm2024 | **2024-01/02** · 2024-06/07 · 2024-06/07 |

  Todo o treino é **≤ 2022-12-18**; toda a validação é **≥ 2024-01-13**. Corte único por data, sem
  sobreposição.
- **Racional (leak-free):** AFCON2023 foi **jogada em jan–fev de 2024**, portanto qualquer split que a
  coloque no TREINO com WC2022 na VALIDAÇÃO **vaza futuro→passado** via agregados de treino (λ do
  baseline §5.5, `r` do NegBin §5.6, `média_do_torneio` §5.6): estimativas informadas por 2024 seriam
  usadas para pontuar um jogo de 2022. Por isso AFCON2023 vai para a **validação**.
- **NOTA — não espelha o Modo 1 (intencional):** o split do Modo 1 contém exatamente esse leak
  (edição jogada em 2024 no treino, edição de 2022 na validação). Registrado como **observação**; o
  Modo 1 está congelado e **não** é alterado por esta decisão.
- **Nunca k-fold aleatório** (vaza futuro→passado).
- **WC2026 estritamente reservada para inferência** — nunca entra em treino/validação.
- **Degradação graciosa (§3.3 do SPEC.md):** se `/statistics` do SofaScore falhar para um jogo da
  WC2026, esse jogo fica **sem previsão de contagem** e o Modo 1 segue normal. Nunca fabricar valor.

---

## 8. Estrutura de projeto (adições ao repo)

```
polvo-paul-26/
├── SPEC.md                          # Modo 1 — CONGELADO
├── SPEC_MARKETS.md                  # Modo 2 — este documento
├── docs/
│   ├── RESULTS.md                   # Modo 1 (pronto)
│   └── RESULTS_MARKETS.md           # Modo 2 (futuro)
├── src/
│   ├── collect/
│   │   ├── footballdata.py          # Modo 1 (existente)
│   │   ├── statsbomb.py             # compartilhado (existente)
│   │   ├── check_counts_coverage.py     # sondagem (feito)
│   │   ├── check_sofascore_coverage.py  # sondagem (feito)
│   │   ├── check_sofascore_depth.py     # sondagem profundidade (feito — passo 0)
│   │   └── sofascore.py             # NOVO — coletor operacional (curl_cffi)
│   ├── features/
│   │   ├── tier1.py                 # Modo 1 (existente)
│   │   └── counts.py                # NOVO — features de contagem
│   ├── models/
│   │   ├── (modelos Modo 1)         # existentes, CONGELADOS
│   │   └── counts/                  # NOVO — poisson/negbin por alvo
│   └── evaluate/
│       ├── error_analysis.py        # Modo 1 (existente)
│       └── counts_metrics.py        # NOVO — MAE/RMSE/PIT por alvo
└── tests/
    ├── test_parity.py               # existente
    ├── test_mode1_frozen.py         # NOVO — cerca elétrica do Modo 1
    └── test_counts_parity.py        # NOVO — paridade do Modo 2 (temporal + entre fontes)
```

---

## 9. Decisões resolvidas e em aberto

### Resolvidas
- [x] **Escopo:** três alvos (gols, escanteios, cartões), contagem por time, mercados derivados.
- [x] **Domínio:** só torneios de seleção; ligas fora (fallback condicional §2.5).
- [x] **Modelagem:** Poisson (gols, escanteios) / Negative Binomial (cartões).
- [x] **Arquitetura de fonte:** 3 camadas — StatsBomb ground truth, SofaScore operacional,
      validação cruzada como gate de unificação.
- [x] **Derivação de contagem no StatsBomb** confirmada empiricamente.
- [x] **Acesso, cobertura recente, campos e rate limit do SofaScore** confirmados.
- [x] **🔴→✅ BLOQUEADOR #1 — profundidade histórica do SofaScore RESOLVIDA** (passo 0, 2026-07-01):
      `/statistics` com escanteios+cartões ~100% até ~2008–2010 (WC/Euro) e ~2015 (confederações
      menores). Ver §2.2.
- [x] **Arquitetura de fonte definida: SofaScore-only** (treino + inferência); StatsBomb = validador
      da Camada 3. Plano B (StatsBomb treino) vira fallback. Ver §2.4.
- [x] **ID da Nations League** localizado: **10783** (§2.2).
- [x] **Modo 1 congelado** — fronteira dura, cerca elétrica via teste.

### Em aberto (a resolver na implementação)
- [ ] **Gate da Camada 3** — definições de contagem batem entre StatsBomb (ground truth) e SofaScore
      (operacional)? Confirma fidedignidade do histórico + quantifica piso de erro (§10 passo 1).
- [ ] **Amistosos (851) no treino de cartões?** Cobertura ~20% pré-2021 + viés de intensidade
      (disciplina/arbitragem de amistoso ≠ torneio oficial). Avaliar excluí-los do treino de cartões
      (entram só como bônus condicional, nunca base — §2.2 requisito 3).
- [ ] Poisson simples vs bivariado/Dixon-Coles para gols (após primeiro benchmark).
- [ ] Confirmar sobredispersão de escanteios (Poisson vs NegBin).

---

## 10. Ordem de implementação (sequência obrigatória)

Dois passos são bloqueadores. Não pular — mesma disciplina do §10 do SPEC.md.

0. ✅ **CONCLUÍDO (2026-07-01) — PROFUNDIDADE HISTÓRICA DO SOFASCORE (ex-BLOQUEADOR #1).**
   `check_sofascore_depth.py` mediu `/statistics` em torneios pré-2024: **escanteios + cartões ~100%**
   até ~2008–2010 (WC/Euro) e ~2015 (confederações menores); IDs de torneio descobertos e Nations
   League localizada (§2.2). **Decisão destravada: SofaScore-only** para treino + inferência (§2.4).
   Requisitos revelados (xG ≥2020 opcional, validar esporte por ID, amistosos como bônus) em §2.2.

1. 🟡 **GATE DA CAMADA 3 — validação cruzada de definição.** Nos jogos em ambas as fontes (Euro 2024,
   Copa América 2024), comparar escanteios/cartões/posse StatsBomb (ground truth) vs SofaScore
   (operacional). Papel ajustado pós-passo 0: **não decide unificação** (arquitetura já é SofaScore-only)
   — **confirma que a fonte operacional é fidedigna** antes de confiar no histórico dela para treino,
   E quantifica o piso de erro (§6.2 item 2). **Nenhum `fit()` sobre histórico SofaScore antes deste gate.**

2. **Coletor operacional** `sofascore.py` (curl_cffi, cache, dois pipelines §2.7) + finalizar StatsBomb
   como derivador de contagem.

3. **Build de features de contagem** `counts.py` (§4), unindo contagens + contexto + ranking.

4. ✅ **CONCLUÍDO — GATE DE PARIDADE DO MODO 2.** `tests/test_counts_parity.py` (11 testes): paridade
   **temporal** (corte estrito `<`), de **colunas** e de **valores** treino↔inferência. A paridade de
   FEATURES **entre fontes é trivial por construção** (SofaScore-only nos dois lados, §2.4) — não há
   segunda fonte no vetor de features. O StatsBomb permanece como **validador ground-truth** do gate de
   3 pernas (`cross_validation --check-cache`): cache==API **394/394** exato, API~StatsBomb 97%/99%,
   cache~StatsBomb nos 83. **Nenhum `fit()` antes deste gate — agora liberado.**

5. ✅ **CONCLUÍDO — MODELOS DE CONTAGEM v1.** `src/models/counts_model.py`: baseline Poisson de
   taxa-média-treino + v1 multiplicativo DC-simplificado (§5.6), split temporal leak-free (§7),
   convolução NegBin truncada, avaliação log-loss/RPS/PIT (§6). **Veredito medido (validação 135
   jogos):** escanteios — v1 supera o baseline por margem **marginal** (−0.0017), único alvo com sinal
   direcional real (corr 0.227); gols e cartões — v1 **não supera** (ruído, corr 0.054 e −0.027).
   v1 calibrado (PIT plano); o gargalo é discriminação, não calibração — teto de sinal baixo (§6.2,
   resultado negativo válido). **Caminho B (GLM) NÃO escalado por decisão diagnóstica:** sobre alvos
   com corr≈0 amplificaria ruído; mover gols/cartões exige features novas, não novo modelo. Detalhes
   em `docs/RESULTS.md` (Modo 2 v1).

6. **Derivação de mercados** a partir das distribuições ajustadas + métricas (§6).

7. **Inferência WC2026 ao vivo** via SofaScore, com degradação graciosa (§7).

### Pré-requisito paralelo (Modo 1) — ✅ CONCLUÍDO
- `src/evaluate/live_score.py` do Modo 1 — instrumento de scoring ao vivo da Copa. **Entregue e
  rodado** (implementação em `5037cf7`; previsões×mercado do LAST_32 documentadas em
  `docs/RESULTS.md` §6, commit `af0da6e`): 7 jogos do LAST_32 pontuados com `actual_result`
  preenchido em `data/processed/predictions_log.csv`. Scoring recorrente conforme os jogos terminam.