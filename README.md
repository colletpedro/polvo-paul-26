# polvo-paul-26

Modelo preditivo para torneios de seleção (Copa do Mundo 2026): probabilidade de resultado da partida (Mode 1) e mercados de contagem — gols, escanteios, cartões (Mode 2).

> ⚠️ **README provisório.** Será atualizado ao final da Copa do Mundo 2026 com os números definitivos de Mode 1 e do shadow mode de escanteios (Mode 2). Os resultados abaixo refletem o estado parcial em **2026-07-07**.

## Status atual

**Mode 1 — resultado da partida.** Congelado, em produção, com pré-registro contínuo de previsões antes de cada apito.
- Log-loss calibrado (XGBoost + sigmoid): **1.0425**.
- **Não atingiu a meta própria** — o baseline de prior histórico ingênuo faz **0.9659**, melhor que o modelo de produção.
- Ao vivo, o modelo fica **atrás do mercado de apostas** de forma consistente (ex.: 0.9437 modelo vs 0.8556 mercado em 19 jogos decididos até agora). Isso é o resultado esperado dado o diagnóstico do projeto, não uma alegação de vantagem — e a amostra ainda é pequena para conclusões fortes.

**Mode 2 — mercados de contagem.** Desenvolvimento **encerrado** após seis testes de hipótese negativos (árbitro nominal, posse de bola, chutes no alvo, um estimador de força Dixon-Coles vs mercado, e densificação do corpus de treino). Só **escanteios** segue em acompanhamento ao vivo (shadow mode) — o único mercado com algum sinal de discriminação encontrado (correlação +0.227 em teste offline, ganho marginal sobre o baseline). Gols e cartões não têm sinal recuperável no regime de dado disponível.

## O que o projeto faz

- **Mode 1:** XGBoost + calibração sigmoid para probabilidade de vitória do time A / empate / vitória do time B, a partir de features de forma pré-jogo (goals-based).
- **Mode 2:** modelagem de contagem (Poisson / Binomial Negativa) para gols, escanteios e cartões por time, com mercados over/under derivados diretamente da distribuição prevista.

## Metodologia

- **Spec-first:** decisões de arquitetura fechadas em documento antes de qualquer código (`SPEC.md`, `SPEC_MARKETS.md`).
- **Paridade rigorosa treino↔inferência:** as mesmas funções de feature rodam em treino e em produção, usando só dado que também está disponível no momento da inferência.
- **Corte temporal estrito:** toda feature histórica usa apenas jogos anteriores ao jogo sendo previsto — sem vazamento de futuro.
- **Pré-registro versionado:** previsões são gravadas e commitadas no git *antes* do resultado do jogo, como carimbo de tempo à prova de ajuste retroativo. Nenhuma previsão é alterada depois que o resultado é conhecido.
- **Resultado negativo como resultado válido:** seis hipóteses de feature/arquitetura foram testadas e descartadas no Mode 2, cada uma com critério de saída definido *antes* do teste rodar e diagnóstico causal registrado — não só "não funcionou", mas por quê.

## Estrutura do repositório

```
src/
  collect/     # coleta de dados (football-data.org, SofaScore, StatsBomb) + manifest de integridade
  features/    # construção de features (Tier 1 goals-based; contagens Modo 2)
  models/      # baseline, modelo calibrado (Mode 1), modelo de contagem (Mode 2)
  evaluate/    # validação cruzada, análise de erro, scoring ao vivo (live_score*.py)
  predict.py         # inferência pré-jogo Mode 1
  predict_counts.py  # inferência pré-jogo Mode 2 (escanteios, shadow mode)
data/
  processed/   # predictions_log.csv, counts_predictions_log.csv (pré-registro versionado)
  reference/   # rankings FIFA, manifest do cache SofaScore
  probes/      # dados de sondagens de hipótese (árbitro, chutes, odds) — re-deriváveis
  raw/         # cache de coleta (não versionado — reproduzível via src/collect)
tests/         # testes de paridade treino↔inferência
docs/
  RESULTS.md   # números completos, histórico de decisões e diagnósticos
```

## Documentação adicional

Os números completos, tabelas de resultados e o histórico detalhado de decisões (incluindo todos os testes de hipótese negativos do Mode 2) estão em:

- [`docs/RESULTS.md`](docs/RESULTS.md) — resultados e diagnósticos, Mode 1 e Mode 2.
- [`SPEC.md`](SPEC.md) — especificação do Mode 1.
- [`SPEC_MARKETS.md`](SPEC_MARKETS.md) — especificação do Mode 2.

---

Última atualização: 2026-07-07. Próxima atualização planejada: ao final da Copa do Mundo 2026, com os números definitivos de Mode 1 e do shadow mode de escanteios.
