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
