# Dados de referência

## fifa_ranking.csv

Snapshot do ranking FIFA/Coca-Cola masculino para os **48 participantes da Copa 2026**.
Insumo do **BLOQUEADOR #1** (SPEC.md §2.8, §10): sustenta a convenção A/B (§1.4) e as features de
contexto (§3.1–3.2: `delta_ranking`, qualidade do adversário).

| Campo | Descrição |
|-------|-----------|
| `team` | Nome canônico no padrão **football-data.org** (chave de join com a coleta da WC2026) |
| `tla` | Código de 3 letras |
| `fifa_rank` | Posição no ranking FIFA (1 = melhor) |
| `fifa_points` | Pontos FIFA |
| `confederation` | UEFA / CONMEBOL / CAF / AFC / CONCACAF / OFC |
| `ranking_date` | Data do snapshot |

**Snapshot:** `2026-06-11` — última publicação FIFA antes do mata-mata da Copa 2026
(próxima atualização oficial: 2026-07-20). Como é estático, **atualizar manualmente** a cada janela
de ranking (§2.8).

**Fonte e verificação:** tabela completa do ranking FIFA (211 seleções) extraída e os 48 times da
Copa casados por nome/alias (Czechia↔Czech Republic, Ivory Coast↔Côte d'Ivoire, Türkiye, etc.).
Cross-check contra valores conhecidos da publicação de jun/2026: Argentina #1, Brasil #6,
México #14, EUA #17, Canadá #30. Faixa dos participantes: rank 1 (Argentina) a 85 (Nova Zelândia).
