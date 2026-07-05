# Resultados da Fase de Modelagem

> Referência: SPEC.md §4–6 · Stack: Python · Atualizado: 2026-06-29

---

## 1. Setup experimental

### Dados e split temporal (§4.4)

| | Competições | Partidas |
|---|---|---|
| **Treino** | WC2018 · Euro 2020 · AFCON 2023 | **47** mata-mata |
| **Validação** | WC2022 · Euro 2024 · Copa América 2024 | **39** mata-mata |

Apenas partidas de mata-mata (`stage_ord > 0`). Fase de grupos entra exclusivamente como fonte das features de desempenho pré-confronto, nunca como linha de treino.

Split estritamente temporal: nenhum torneio do futuro vaza para o treino.

### Features (Tier 1 — goals-based obrigatório, §3.1–3.2)

Todas as features derivam de gols + contexto — sem xG, posse ou chutes. A restrição é imposta pela inferência real (WC2026 via football-data.org free não fornece stats por partida), seguindo o princípio de paridade treino↔inferência do §2.4.

| Categoria | Features |
|-----------|----------|
| Desempenho na fase de grupos | pontos, gols marcados/sofridos, saldo, gols por jogo, clean sheets, forma recente |
| Contexto do confronto | ranking FIFA (A e B), delta de ranking, confederação, fase do torneio, dias de descanso |

27 features no total · target: `result_90min ∈ {A_win, draw, B_win}`.

### Distribuição do target (treino)

| Classe | n | % |
|--------|---|---|
| A_win (favorito vence) | 23 | 48.9% |
| draw | 13 | 27.7% |
| B_win (zebra) | 11 | 23.4% |

Desequilíbrio moderado — A_win é a classe majoritária, draw e B_win somam ~51%.

---

## 2. Resultados — todos os modelos

Avaliação no conjunto de validação (39 partidas). Métricas principais: §5.1–5.2.

| Modelo | log-loss | Acurácia | Brier | Notas |
|--------|----------|----------|-------|-------|
| **Baseline uniforme** | 1.0986 | 0.590 | 0.667 | [1/3, 1/3, 1/3] fixo |
| **Baseline prior histórico** | **0.9659** | 0.590 | 0.575 | frequência do treino fixo |
| Logístico (§4.1) | 1.3167 | 0.590 | 0.668 | |
| XGBoost cru (§4.2) | 1.1670 | 0.462 | 0.676 | n_est=50, depth=2, lr=0.05 |
| XGBoost + isotonic (§4.3) | 1.2971 | 0.410 | 0.778 | descartado — overfit |
| **XGBoost + sigmoid (§4.3)** | **1.0425** | **0.513** | **0.625** | **modelo de produção** |

> Métrica principal: **log-loss** (cross-entropy multiclasse). Penaliza previsões confiantes e erradas; recompensa calibração probabilística. Melhor = menor.

### Observações por modelo

**Logístico** — pior log-loss que o XGBoost cru, mas maior acurácia. O logístico aposta quase sempre em A_win (favorito), o que maximiza acertos de classe mas produz probabilidades mal distribuídas.

**XGBoost cru** — melhora log-loss em −0.15 vs logístico, mas acurácia cai (modelo espalha mais probabilidade entre as classes, o que é correto probabilisticamente mas reduz o acerto da classe predita).

**XGBoost + isotonic** — piora em todas as métricas. Com ~9 amostras de calibração por fold (cv=5, treino=47) e classes minoritárias com 1–2 positivos por fold, o isotonic não tem dados suficientes e overfita: torna as previsões mais extremas em vez de suavizá-las.

**XGBoost + sigmoid** — maior ganho obtido. Platt scaling ajusta apenas 2 parâmetros por classe (slope + intercept), o que é robusto com amostras pequenas. Log-loss cai −0.12 vs cru, brier melhora de 0.676 → 0.625. **Este é o modelo escolhido para produção.**

---

## 3. Critério de sucesso (§6.1) — avaliação honesta

O critério de sucesso estabelecido no SPEC é:

> **Log-loss do modelo < melhor baseline ingênuo** (uniforme ou prior histórico).

| | log-loss | Resultado |
|---|---|---|
| XGBoost + sigmoid (produção) | 1.0425 | |
| Melhor baseline ingênuo (prior) | 0.9659 | |
| **Diferença** | **+0.0766** | **✗ critério NÃO atingido** |

O melhor modelo ficou **0.077 acima do prior histórico**. O critério §6.1 não foi atingido nesta fase.

O prior histórico é um adversário difícil neste contexto: com ~49% de A_win no treino, prever "o favorito vence com 49% de chance" por padrão já captura boa parte do sinal disponível nas features de gols+contexto.

---

## 4. Diagnóstico da análise de erro

### Os 10 piores erros (maior log-loss individual)

Os 4 piores são todos empates de mata-mata previstos como vitória do favorito com 70%+ de confiança:

| # | Jogo | Resultado real | P(B_win) | log-loss |
|---|------|----------------|----------|----------|
| 1 | Brasil vs Croácia (QF WC2022) | empate | 0.723 | 2.974 |
| 2 | Argentina vs França (Final WC2022) | empate | 0.700 | 2.952 |
| 3 | França vs Portugal (QF Euro 2024) | empate | 0.399 | 2.692 |
| 4 | Espanha vs Marrocos (R16 WC2022) | empate | 0.768 | 2.391 |

### Interpretação

O padrão é claro: **o modelo não diferencia "favorito controlou o jogo" de "favorito travou o jogo"**. Brasil dominou Croácia nos 90 minutos com posse e volume de finalização — mas empatou. Argentina dominou França na maior parte da final — mas empatou após levar 2 gols rapidamente no segundo tempo.

Com features exclusivamente de gols + contexto (sem xG, sem chutes, sem posse), o modelo não tem como capturar essa distinção. Ranking FIFA alto + bom desempenho na fase de grupos → modelo prevê vitória com alta confiança → quando o jogo termina empatado, a penalidade de log-loss é máxima.

Esta é a limitação estrutural do Tier 1 goals-based, não um problema de tuning ou calibração.

---

## 5. Decisões registradas e próximos passos

### O que foi feito e funcionou

| Técnica | Ganho em log-loss | Decisão |
|---------|-------------------|---------|
| Expansão do treino (+ AFCON 2023) | −0.030 vs sem AFCON | ✓ mantido |
| XGBoost vs logístico | −0.150 | ✓ XGBoost é o modelo base |
| Calibração sigmoid | −0.125 vs XGBoost cru | ✓ produção |
| Calibração isotonic | +0.130 (piorou) | ✗ descartado |

A calibração sigmoid foi o maior ganho "barato" — sem novos dados, sem novas features.

### Teto do Tier 1 e o caminho para mover o número

Features goals-based têm teto natural em empates de mata-mata de alto nível: dois times equilibrados na fase de grupos e com ranking próximo produzem as mesmas features que dois times equilibrados onde um domina completamente — mas apenas um empata.

**Tier 2 (xG + chutes)** é o único caminho com real chance de mover o log-loss abaixo do prior. xG captura volume de finalização qualificado, que discrimina exatamente os jogos que hoje produzem os maiores erros. Porém, como documentado em §2.7, xG não está disponível para a WC2026 por via gratuita — portanto Tier 2 criaria distribution shift na inferência real e fica fora da v1.

### Foco da v1

O objetivo real do projeto é prever a WC2026 em andamento, não otimizar métricas em dados históricos. O modelo Tier 1 + sigmoid calibrado:

- É deployável imediatamente para a fase de mata-mata da WC2026 (início junho/2026)
- Usa apenas features disponíveis gratuitamente (standings + ranking FIFA)
- Produz probabilidades calibradas (brier 0.625) razoavelmente úteis para inferência pré-partida
- Não viola o princípio de paridade treino↔inferência (§2.4)

Otimizações adicionais (Tier 2, mais dados históricos de outras confederações) são evolução futura (§8).

---

## 6. Validação ao vivo — LAST_32 WC2026 vs mercado de apostas

> Previsões geradas em 2026-06-29T20:54Z via `python src/predict.py`.
> Cache de inferência atualizado com `--force` (football-data.org, dados reais).
> Probabilidades de mercado: faixas indicativas convertidas de odds médias
> de casas europeias (Bet365, William Hill, Pinnacle) no dia anterior ao jogo.
> Confrontos concluídos antes desta escrita: Canadá×África do Sul (A_win ✓),
> Brasil×Japão (A_win ✓) — não incluídos aqui pois não passaram pelo filtro TIMED.

### 6.1 Tabela de previsões × mercado

Colunas do modelo: P(A) = P(time melhor rankeado vence em 90 min), P(E) = empate, P(B) = P(time pior rankeado vence).
Mercado: P(favorito) conforme implícita nas odds 1X2, não separada por time A/B.

| Jogo (time A vs time B) | Modelo P(A) | P(E) | P(B) | Predito | Mercado P(fav.) | Alinhamento |
|-------------------------|-------------|------|------|---------|-----------------|-------------|
| Morocco vs **Netherlands** | 30.3% | 23.0% | 46.7% | NED | 36–44% (NED) | ≈ alinhado |
| Norway vs **Ivory Coast** | 32.2% | 23.7% | 44.1% | CIV | 38–46% (CIV) | ≈ alinhado |
| **France** vs Sweden | 60.1% | 20.7% | 19.2% | FRA | 70–76% (FRA) | ↓ subestima |
| **Mexico** vs Ecuador | 48.7% | 24.1% | 27.1% | MEX | 52–58% (MEX) | ≈ alinhado |
| **England** vs Congo DR | 53.6% | 23.4% | 23.0% | ENG | 70–77% (ENG) | ↓ subestima |
| **Belgium** vs Senegal | **61.4%** | 20.6% | 18.0% | BEL | 55–60% (BEL) | ↑ **supera** ← único |
| **United States** vs Bosnia-Herz. | 40.3% | 31.9% | 27.8% | USA | 55–62% (USA) | ↓ subestima |
| **Spain** vs Austria | 54.2% | 24.4% | 21.5% | ESP | 65–72% (ESP) | ↓ subestima |
| **Portugal** vs Croatia | 56.2% | 22.0% | 21.8% | POR | 54–62% (POR) | ≈ alinhado |
| **Switzerland** vs Algeria | 54.6% | 20.6% | 24.8% | SUI | 56–63% (SUI) | ≈ alinhado |
| Australia vs **Egypt** | 30.9% | 24.1% | 45.0% | EGY | 48–55% (EGY) | ↓ subestima |
| **Argentina** vs Cape Verde Is. | 50.7% | 26.5% | 22.8% | ARG | 82–87% (ARG) | ↓↓ subestima muito |
| **Colombia** vs Ghana | 50.0% | 27.2% | 22.8% | COL | 58–65% (COL) | ↓ subestima |

### 6.2 Padrão observado: alinhamento em jogos equilibrados, cegueira em desníveis

O modelo comporta-se de forma diferente conforme o grau de desequilíbrio do confronto:

**Jogos equilibrados (entropia > 1.0)** — Morocco×Neth, Norway×CIV, Mexico×ECU, Portugal×CRO, Switzerland×ALG: modelo dentro ou próximo da faixa do mercado. Ambos concordam na direção do favorito, com diferenças ≤ 8 p.p. no P(favorito). As features goals-based capturam bem o sinal quando os times têm histórico recente comparável e ranking próximo.

**Desníveis claros** — Argentina×Cabo Verde (Δrankings = 60 posições), Spain×Austria (Δ = 21), England×Congo DR (Δ = 42), USA×Bosnia (Δ = 42): o modelo subestima dramaticamente o favorito. O caso mais severo é Argentina: o modelo prevê 50.7%, o mercado atribui 82–87%. A diferença de 30+ p.p. não é ruído — é uma falha estrutural.

A causa é direta: o conjunto de treino tem 47 partidas de mata-mata entre seleções de nível similar (WC, Euro, AFCON). **O modelo nunca viu uma seleção de elite enfrentar uma de médio porte no mata-mata**, então não aprendeu a amplificar a diferença de ranking quando ela é grande. Na fase de grupos do treino — que serve apenas como fonte de features — esse padrão existe, mas os 47 confrontos de mata-mata propriamente ditos são entre times que chegaram até lá, criando um viés de seleção: o dataset de treino é sobrerrepresentado de matchups de qualidade.

### 6.3 Conexão com o diagnóstico histórico: efeito de dois lados da calibração sigmoid

Na validação histórica (§4), o XGBoost cru **era superconfiante**: previa vitórias com 70%+ em jogos que terminavam em empate (§4, top-4 erros). A calibração sigmoid corrigiu esse viés — reduzindo probabilidades extremas — e melhorou log-loss em 0.12.

O mesmo mecanismo que resolveu um problema criou outro:

| Situação | XGBoost cru | XGBoost + sigmoid | Mercado |
|----------|-------------|-------------------|---------|
| Favorito amplo (Argentina vs Cabo Verde) | ~70% → superconfiante em histórico | **~51% → subconfiante** | ~85% |
| Favorito moderado (Portugal vs Croácia) | ~65–70% → ligeiramente alto | ~56% → próximo | ~58% |
| Jogo equilibrado (Mexico vs Ecuador) | ~55% → razoável | ~49% → próximo | ~55% |

O sigmoid "achatou" a curva de probabilidades de forma uniforme. Para jogos equilibrados isso é adequado; para desníveis extremos, o achatamento vai longe demais. Um modelo com mais dados de mismatches — ou uma feature explícita de "magnitude do desequilíbrio" — calibraria de forma assimétrica.

Este é o diagnóstico para a v2: o problema não é a calibração em si, é a falta de exemplos de mismatch no treino que o calibrador não consegue compensar post-hoc.

### 6.4 Caso de teste: Bélgica × Senegal

Bélgica vs Senegal é o único dos 13 jogos onde o modelo (61.4%) **supera a confiança do mercado** (~55–60%). É também o jogo com menor entropia (H=0.933) — o mais "confiante" entre as previsões.

O motivo provável: Bélgica teve desempenho de grupo superior a Senegal nesta edição da WC2026, e as features goals-based capturam essa diferença mais fortemente do que o mercado, que pondera o histórico de ambas as seleções, probabilidades de prorrogação e eventual desempenho da AFCON 2023 (onde Senegal não chegou muito longe).

Este jogo é o **principal caso de teste ao vivo** da v1:

- Se **Bélgica vencer**: evidência de que o modelo captura algo que o mercado subestima neste confronto específico. Resultado favorável à v1.
- Se **Senegal vencer ou empatar**: resultado consistente com o mercado estar certo e o modelo excessivamente confiante — possível artefato da proeminência de times UEFA no treino (Euro 2020 e Euro 2024 são 30/47 das linhas).

O resultado de Bélgica×Senegal e dos outros 12 jogos do LAST_32 será registrado em `data/processed/predictions_log.csv` (coluna `actual_result`) para avaliação post-hoc assim que os jogos terminem.

#### Post-hoc (atualizado 2026-07-03)

O jogo terminou **2-2 nos 90 minutos** — Bélgica só decidiu na prorrogação (3-2). Pela régua de 90min do Modo 1 (`live_score.py`: `EXTRA_TIME`/`PENALTY_SHOOTOUT` → empate), o resultado é **draw**.

Este é exatamente o segundo cenário que o §6.4 havia antecipado: o mercado estava mais próximo da realidade do que o modelo. O modelo deu 61.4% para Bélgica vencer nos 90min (a previsão mais confiante das 13) num jogo que na verdade terminou empatado — log-loss individual **1.5799**, um dos piores do lote. Não há indício de que o modelo tenha "captado algo que o mercado subestimava"; o caso se resolveu **contra** a v1.

**Nota de reconciliação:** a primeira pontuação (rodada em 2026-07-01, quando o jogo havia acabado de terminar) registrou `A_win` no log — o feed da football-data.org, no momento da consulta, aparentemente ainda não expunha o campo `duration=EXTRA_TIME` de forma completa. A consulta seguinte à API (2026-07-03) já retornava a informação de prorrogação corretamente; `live_score.py` foi ajustado para reconciliar (corrigir e logar) células já preenchidas que divergem de uma nova leitura, em vez de nunca revisitar dados antigos.

**Acumulado atualizado (11 jogos decididos até 2026-07-03):**

| | log-loss |
|---|---|
| Modelo | **0.9237** |
| Mercado | **0.7641** |
| Gap | **+0.1596** (modelo pior que o mercado) |

Acurácia ao vivo: 7/11 (63.6%). Uma execução anterior deste acumulado (antes da correção de Bélgica×Senegal) havia reportado 0.8358 vs 0.6772 — esses números **não são reproduzíveis** com o dado atual da API e ficam superados por este post-hoc; o gap direcional (modelo pior que o mercado) se mantém nos dois cálculos.

---

# Modo 2 — Mercados de contagem (v1)

> Referência: SPEC_MARKETS.md §5–7 · Stack: Python (SofaScore-only) · Atualizado: 2026-07-05

Alvos: gols, escanteios, cartões — contagem por time, mercados over/under derivados. Este é o resultado do **passo 5** (§10): baseline + modelo v1 + avaliação. A leitura honesta abaixo vale a mesma disciplina do Modo 1: um resultado negativo bem-diagnosticado é resultado, não fracasso (§6.2).

## M2.1 Setup — split temporal leak-free (§7)

| | Edições | Jogos |
|---|---|---|
| **Treino** | WC2018 · Euro2020 · CopaAm2021 · AFCON2021 · WC2022 | **259** |
| **Validação** | AFCON2023 · Euro2024 · CopaAm2024 | **135** |

Corte único por data em **2023-01-01** (todo treino ≤ 2022-12-18; toda validação ≥ 2024-01-13). Diferente do Modo 1, este split **não** espelha o do Modo 1 — e intencionalmente. AFCON2023 foi **jogada em jan–fev de 2024**; colocá-la no treino com WC2022 (2022) na validação, como faz o split do Modo 1, vaza futuro→passado pelos agregados de treino (λ do baseline, `r` do NegBin, `média_do_torneio`). Por isso AFCON2023 vai para a validação (§7). O leak do split do Modo 1 fica **registrado como observação**; o Modo 1 está congelado e não é alterado.

Todas as features vêm de `counts.py` (janela N=10, shrink k=5) e passaram o gate de paridade treino↔inferência (`test_counts_parity.py`, §10 passo 4).

## M2.2 Baseline vs v1 — resultados (§5.5 / §5.6 / §6)

- **Baseline (§5.5):** Poisson de taxa média do **treino**, por alvo (um λ por-time fixo, sem features). Poisson para os 3 alvos, inclusive cartões (baseline ingênuo único). λ/time: gols=1.22 · escanteios=4.59 · cartões=1.85.
- **v1 (§5.6):** multiplicativo estilo Dixon-Coles simplificado, **zero parâmetro otimizado**: `λ_alvo(time) = ataque_shrunk(time) × defesa_shrunk(oponente) / média_expansiva`. Gols/escanteios via Poisson; cartões via Negative Binomial (r global = 10.40, momentos no treino), com o total do jogo por **convolução truncada** (validada vs Monte Carlo, tolerância 1e-3).

**Log-loss over/under contra realizado** (validação; linhas gols 2.5, escanteios 8.5/9.5/10.5, cartões 3.5/4.5; menor é melhor):

| alvo | modelo | AFCON2023 | CopaAm2024 | Euro2024 | **Agregado** |
|---|---|---|---|---|---|
| gols | baseline | 0.6685 | 0.6780 | 0.6793 | **0.6748** |
| gols | v1 | 0.6719 | 0.6395 | 0.7450 | **0.6918** |
| escanteios | baseline | 0.6690 | 0.6698 | 0.6915 | **0.6777** |
| escanteios | v1 | 0.6873 | 0.6783 | 0.6630 | **0.6760** |
| cartões | baseline | 0.6475 | 0.6681 | 0.6637 | **0.6585** |
| cartões | v1 | 0.6490 | 0.7173 | 0.7230 | **0.6931** |

Todos os valores rondam **0.69** (o log-loss de uma moeda). Tanto o baseline de taxa fixa quanto o v1 sem parâmetro ajustado são **quase não-informativos** nesses over/under.

## M2.3 Veredito por alvo (sem suavizar)

O teste que separa "ruído puro" de "sinal com escala errada" é o acerto de **direção** do desvio (v1 previu acima do baseline quando o jogo teve contagem acima da média?) e a correlação de sinal:

| alvo | v1 − baseline (agg) | dir_hit | corr | veredito |
|---|---|---|---|---|
| escanteios | **−0.0017** | 61.5% | **+0.227** | v1 melhor variante, ganho **marginal**; tem sinal real, teto baixo |
| gols | +0.0170 | 54.1% | +0.054 | v1 **não supera**; sinal ≈ ruído |
| cartões | +0.0347 | 51.9% | −0.027 | v1 **não supera**; **ruído puro** |

- **Escanteios** — único alvo com sinal direcional real (corr 0.227, 61.5% de acerto de direção). Ainda assim, o ganho sobre o baseline é **marginal** (−0.0017 em log-loss), dentro do ruído. É o teto do que a forma multiplicativa entrega aqui.
- **Gols** — v1 fica **+0.0170 pior que o baseline**. A correlação de sinal (0.054) é indistinguível de ruído; o produto ataque×defesa não carrega direção jogo-a-jogo para gols.
- **Cartões** — v1 fica **+0.0347 pior que o baseline**. Correlação **−0.027**: ruído puro, o pior caso. Há também um viés de nível em Euro2024 (v1 prevê 3.41 cartões/jogo vs 4.59 reais, −1.17), mas mesmo corrigindo o nível não haveria discriminação a extrair.

RPS truncado@20 (agregado) confirma: escanteios v1 (1.993) < baseline (2.029); gols (0.828 vs 0.800) e cartões (1.448 vs 1.372) pioram.

## M2.4 A tentativa de divisor local (caminho A) e sua falha

O diagnóstico sugeriu que o problema de escanteios poderia ser de **escala**: o divisor `média_do_torneio` é uma média expansiva **global** que mistura confederações de base diferente (cartões por-time: Euro 2.29 · CopaAm 2.08 · AFCON 1.83). Testou-se trocar o divisor de escanteios por uma **média local por-torneio** (expansiva dentro do torneio, leak-free; fallback à global no 1º jogo de cada torneio — apenas 1 jogo/torneio caiu no fallback).

**Falhou.** Log-loss agregado de escanteios **piorou de 0.6760 → 0.7361** (Euro2024 foi de 0.6630 → 0.7914; só CopaAm2024 melhorou). A hipótese "problema é escala corrigível pelo divisor" **não se sustentou**: o divisor médio mal se move (Euro local 4.56 vs global 4.58) — não é efeito de nível, é **variância**. A média expansiva dentro do torneio tem amostra minúscula no começo de cada edição, oscila muito e **injeta ruído em λ** (corr caiu de 0.227 → 0.154; PIT menos uniforme). Uma "taxa local" limpa exigiria a média do torneio inteiro, que **vazaria**. **v1-global permanece a melhor variante de escanteios.**

Este experimento negativo não é um beco sem saída: ele **descarta empiricamente** a hipótese concorrente "o problema é escala corrigível pelo divisor", estreitando o diagnóstico para "**teto de sinal baixo**" em vez de deixá-lo como suposição. Eliminar uma hipótese rival é evidência a favor da conclusão principal.

## M2.5 Calibração vs discriminação

O PIT do v1 é razoavelmente plano nos 3 alvos (médias 0.482 / 0.492 / 0.526, histogramas 5-bins sem picos fortes) ⇒ **o v1 é calibrado**. O gargalo **não é calibração, é discriminação**: o modelo acerta o nível médio mas não separa jogo-a-jogo. Isso indica que o **teto de previsibilidade** de over/under de contagem a partir de features de forma pré-jogo é **intrinsecamente baixo** — resultado válido por §6.2, análogo (em espírito) ao teto do Tier 1 do Modo 1: features de forma não distinguem os casos que mais importam. Não é defeito de implementação.

## M2.6 Decisão registrada: caminho B (GLM) NÃO justificado

O §5.6 prevê escalar para um GLM sobre as 25 features **só se** o v1 não superar o baseline. Os números dizem para **não** escalar:

- **Escanteios** já toca o teto de sinal (corr 0.227) — um GLM não encontraria muito mais direção.
- **Gols e cartões** têm corr ≈ 0. Ajustar um GLM sobre 259 jogos para prever alvos sem sinal direcional **amplificaria ruído** — exatamente o erro de overfit diagnosticado no Modo 1 (isotonic com dados escassos piorou tudo). Mais capacidade de modelo sobre sinal ausente não cria sinal.

Mover gols/cartões exigiria **features novas** (árbitro do jogo, disciplina do confronto, importância/estágio, xG quando houver), **não um novo modelo** — e isso está fora do escopo do v1.

**Benchmark contra linha de mercado:** pendente — não há coleta de odds. Pela §4.3, só gols estaria habilitado a comparar contra linha (90min), e gols é justamente um resultado negativo — prioridade baixa.

### Conclusão do v1 (Modo 2)
Uma variante (escanteios) supera o baseline por margem marginal e com sinal real mas de teto baixo; gols e cartões não superam, sendo cartões ruído puro. O v1 está correto e calibrado — o limite é de sinal, não de implementação. Escalar para GLM não se justifica; o próximo lever honesto é **dados/features novas**, não mais modelo — e a candidata mais provável de carregar sinal direcional para cartões é o **árbitro nominal** (efeito de árbitro sobre cartões é documentado e é pré-jogo observável), o ponto de partida natural.

## M2.7 — Árbitro nominal → cartões (teste isolado, negativo)

Hipótese: o rigor histórico do árbitro (média de cartões nos jogos que apitou antes) carrega sinal sobre cartões do jogo, além do que as features atuais (corr −0.027, M2.3) já não capturam.

**Fonte e blindagem:** `referee.id`/`referee.name` não vêm no cache canônico (`_STAT_KEYS` de `sofascore.py` só extrai corner/cartão/posse/xG) — coletados à parte via `GET /event/{id}` (394/394, 0 nulo, preservado em `data/probes/referees.jsonl`). **Proibido usar os agregados de carreira da API** (`referee.yellowCards`/`redCards`/`games`) como feature — são totais até hoje e vazam jogos futuros; a feature testada foi calculada jogo-a-jogo a partir do próprio corpus, corte temporal `<` estrito, chaveada por `referee.id` (nunca por nome).

**Poder do teste:** 116 árbitros distintos no corpus; mediana 3 jogos/árbitro, máximo 10. 278/394 jogos com ≥1 apitada prévia do mesmo árbitro; 191/394 com ≥2.

**Correlação (feature = média de cartões totais nos jogos anteriores do mesmo árbitro; target = cartões totais do jogo):**

| Amostra | n | Pearson | Spearman |
|---|---|---|---|
| ≥1 jogo prévio | 278 | −0.0601 | −0.0744 |
| ≥2 jogos prévios | 191 | −0.0190 | −0.0408 |

**Veredito: negativo.** A correlação **enfraquece** (não fortalece) quando se exige mais histórico prévio por árbitro — o oposto do padrão esperado se o sinal estivesse sendo apenas diluído por ruído de amostra pequena. Isso descarta "falta de dado" como explicação e aponta para ausência real de sinal nessa formulação. Não virou feature.

## M2.8 — Posse → escanteios (negativo em duas etapas: promissor in-sample, morre out-of-sample)

Hipótese: posse média histórica do time adiciona sinal **incremental** sobre escanteios, além da média histórica de escanteios que o v1 já usa (não interessa a correlação bruta, que é redundante por construção).

**Etapa 1 — incremental in-sample (n=705/788 obs time-jogo elegíveis, ≥1 jogo prévio):**

| Métrica | Pearson | Spearman |
|---|---|---|
| Bruta: posse × escanteios reais | 0.2549 | 0.2308 |
| **Incremental** (parcial, controlando pela média histórica de escanteios do v1) | **0.1252** | 0.0993 |

Sinal aparentemente real e não-redundante — não colapsa para perto de zero ao controlar pelo baseline.

**Etapa 2 — screen out-of-sample (split §7: 259 treino / 135 validação; GLM Poisson por-time, `baseline_feat` e `possession_feat` ambas shrunk k=5/N=10 para paridade de tratamento; score reusado de `counts_model.py` — mesmas `_logloss`/`_rps_trunc`/`_poisson_total_pmf`/linhas [8.5, 9.5, 10.5]):**

| Modelo | logloss | RPS@20 |
|---|---|---|
| A: baseline (só escanteios históricos) | 0.6795 | 2.0260 |
| B: baseline + posse | 0.6877 | 2.0761 |
| **delta (B − A)** | **+0.0082** | **+0.0501** |

**B piora A nas duas métricas.** Diagnóstico: no treino a posse rouba peso do coeficiente do baseline (padronizado: 0.117 → 0.043), e essa realocação **degrada** a validação — feature correlacionada no treino que não generaliza. Robustez confirmada com features padronizadas (mesmo delta).

**Lição registrada (benchmark de corte para testes futuros):** um incremental in-sample de Pearson ≈ **0.125** já demonstrou não sobreviver a este split. Testes futuros de features incrementais só valem escalar para o screen OOS se materialmente **acima** desse patamar.

## M2.9 — Chutes no alvo → gols (negativo in-sample, não avançou ao OOS)

Hipótese: chutes no alvo médios do time (proxy mecânico de criação de chance) adicionam sinal incremental sobre gols, além da média histórica de gols do v1.

**Fonte:** `shotsOnGoal` ("Shots on target") do `/statistics`, não presente no cache canônico — coletado à parte (394/394, 0 nulo, cobertura 100% uniforme nas 8 edições do corpus, confirmada em amostra de 40 e depois nos 394; preservado em `data/probes/shot_stats.jsonl`).

**Armadilha de nomenclatura documentada:** a chave `totalShotsOnGoal` tem `name` **"Total shots"** (total, não no alvo); a chave `shotsOnGoal` tem `name` **"Shots on target"** (a correta). Extração deve casar chave↔name, nunca assumir pela chave isolada.

**Teste incremental (n=705, mesma metodologia shrunk k=5/N=10 do M2.8):**

| Métrica | Pearson | Spearman |
|---|---|---|
| Bruta: chutes no alvo × gols reais | 0.1150 | 0.0754 |
| **Incremental** (parcial, controlando pela média histórica de gols do v1) | **0.0603** | 0.0210 |

**Veredito: negativo, parado antes do screen OOS.** O incremental (0.0603) é cerca de metade do benchmark de corte estabelecido em M2.8 (0.1252, que já morreu OOS) — não valeria gastar o screen num sinal in-sample mais fraco que um que já sabemos não sobreviver. Nota interpretativa: até a correlação **bruta** de chutes-no-alvo (0.115) é fraca — o proxy mais mecânico de gol que existe quase não correlaciona neste corpus, evidência de **teto de regime de dado** (N=10, elenco rotativo de seleção), não de feature mal escolhida.

## M2.10 — Odds de mercado: sondagem e coleta (infraestrutura + limitações da fonte)

Pré-requisito para comparar o Modo 2 contra a linha de mercado (pendente desde M2.6): sondagem de disponibilidade de odds O/U de gols no SofaScore.

**Endpoint confirmado:** `GET /api/v1/event/{id}/odds/1/all` → `{"markets": [...], "eventId": ...}`. **Provider único** (`sourceId=69800370` constante em todos os jogos testados) — **não é consenso de mercado**, é uma casa de apostas isolada. Odds em formato **fracionário** (`"X/Y"`, decimal = X/Y + 1).

**Mercados relevantes:** `marketId=9` "Match goals" — O/U com múltiplas linhas simultâneas por jogo (`choiceGroup` de "0.5" a "7.5" conforme o jogo); BTTS nativo (`marketName="Both teams to score"`, choices Yes/No).

**Buraco de cobertura confirmado:** **AFCON2023 — 0/11 jogos amostrados têm qualquer mercado além de 1X2** (amostra ampliada de 3→11 especificamente para não concluir de N pequeno). É um buraco de **edição específica**, não uma tendência de época suave (AFCON2021, mais antiga, tem cobertura perfeita). AFCON2023 é uma das 3 edições do split de validação §7 — ficou fora de qualquer teste que dependa de odds. Também 5 jogos individuais de WC2018 sem linha O/U (causa não investigada).

**Coleta realizada:** O/U 2.5 gols + BTTS, decimal bruto e de-vig (normalização over/under para somar 1), nas 7 edições com cobertura confirmada (AFCON2023 excluída) — **337/342 jogos (98,5%)**. Preservado em `data/probes/odds_ou_btts.jsonl`. **BTTS coletado mas nunca testado** — decisão registrada: sem valor de informação após o resultado negativo do M2.11 abaixo.

## M2.11 — Dixon-Coles de força vs mercado O/U 2.5 (quinto negativo — fecha o grupo "rotear em volta do teto")

Direção separada do v1 multiplicativo: um estimador de força **Dixon-Coles clássico** (ataque/defesa por seleção + fator ρ de correção de placares baixos, sem fator casa — sedes neutras —, sem decaimento temporal), calibrado por MLE **só nos gols reais do treino**, testado contra a probabilidade de-vig do mercado (M2.10) na linha O/U 2.5.

**Blindagens de vazamento ativas:** odds nunca entraram na estimação (só benchmark); parâmetros calibrados exclusivamente nos 259 jogos de treino do split §7 e congelados antes de pontuar a validação; time sem histórico no treino usa fallback definido no treino (atk = média do treino, def = 0).

**Calibração:** MLE convergiu (`success=True`, nll=640.64), ρ=0.0738 (dentro dos bounds). Sanity de força sem inversão de sinal: France (atk +0.96, 2.00 gols/jogo no treino), England (atk +0.83), Italy (def +1.18, 0.43 sofridos/jogo), Brazil (def +1.07) no topo — consistente com o esperado. Distribuição de P(over 2.5) não colapsou (min 0.053, mediana 0.466, max 0.931).

**Resultado (validação: 83 jogos com odds válidas — Euro2024 51 + CopaAm2024 32; AFCON2023 sem odds; fallback usado em 18/83, times Albania/Georgia/Jamaica/Romania/Slovenia):**

| Fonte | log-loss O/U 2.5 |
|---|---|
| DC (força, só gols) | 0.7394 |
| Mercado de-vig (provider único) | 0.6828 |
| **delta (DC − mercado)** | **+0.0566** (DC pior) |

**Robustez — DC perde em todo corte**, não é artefato de fallback nem de uma edição só:

| Corte | n | DC | mercado | delta |
|---|---|---|---|---|
| Sem fallback (ambos times conhecidos) | 65 | 0.7272 | 0.6731 | +0.0542 |
| Com fallback (≥1 time novo) | 18 | 0.7832 | 0.7180 | +0.0652 |
| CopaAm2024 | 32 | 0.6742 | 0.6578 | +0.0164 |
| Euro2024 | 51 | 0.7803 | 0.6985 | +0.0818 |

**Diagnóstico central: o DC é sobreconfiante.** Desvio-padrão de P(over) do DC = 0.197 vs 0.079 do mercado, com a **mesma média** (0.472, ambos próximos da taxa real 0.410) — o DC assume posições mais extremas que o mercado sem ganhar acurácia, e o log-loss pune exatamente esse excesso de confiança. É o mesmo padrão de sobreconfiança já visto na comparação do Modo 1 contra o mercado e no PIT do v1 do Modo 2 (M2.5). Etiqueta honesta: o adversário aqui era **uma casa isolada** (provider único, não consenso) — o alvo mais fácil disponível — e ainda assim o DC perde.
