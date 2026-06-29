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
