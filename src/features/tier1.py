"""Build de features Tier 1 (obrigatório) — gols + contexto.

SPEC.md §3.1, §3.2, §3.0, §10 passo 3.

Produz um DataFrame com uma linha por partida de mata-mata, contendo:
  - Agregados da fase de grupos para cada time (pontos, gols, saldo, clean sheets, forma)
  - Contexto do confronto (ranking FIFA, confederação, fase, descanso)
  - Target: result_90min ∈ {'A_win', 'draw', 'B_win'} (A = maior ranking FIFA, ou seja, rank menor)

⚠️  PARIDADE TREINO↔INFERÊNCIA (SPEC.md §2.4):
    Este módulo usa SOMENTE features derivadas de gols + contexto.
    NÃO usa xG, posse, passes ou qualquer campo rico do StatsBomb.
    O scaler (normalização) deve ser fit APENAS no conjunto de treino e aplicado a ambos.

Uso:
  from src.features.tier1 import build_training_dataset, build_inference_dataset
  train_df = build_training_dataset()
  infer_df = build_inference_dataset()  # linhas com status TIMED têm result_90min = NaN
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import pandas as pd

from src.collect.statsbomb import fetch_all_training_matches
from src.collect.footballdata import fetch_matches, fetch_standings

# ---------------------------------------------------------------------------
# Constantes e mapeamentos
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parents[2]
_RANKING_WC2026 = _ROOT / "data" / "reference" / "fifa_ranking.csv"
_RANKING_FULL   = _ROOT / "data" / "reference" / "fifa_ranking_full.csv"

# Encoding ordinal de fase (compartilhado treino e inferência — SPEC.md §3.5)
STAGE_ORD: dict[str, int] = {
    # StatsBomb
    "Group Stage":      0,
    "Round of 16":      1,
    "3rd Place Final":  1,
    "Quarter-finals":   2,
    "Semi-finals":      3,
    "Final":            4,
    # football-data.org
    "GROUP_STAGE":      0,
    "LAST_32":          1,
    "LAST_16":          1,
    "QUARTER_FINALS":   2,
    "SEMI_FINALS":      3,
    "THIRD_PLACE":      3,
    "FINAL":            4,
}

# Aliases: nome StatsBomb/football-data.org → nome canônico no ranking FIFA
_TEAM_ALIASES: dict[str, str] = {
    # StatsBomb → ranking full
    "Czech Republic":       "Czechia",
    "IR Iran":              "Iran",
    "Korea Republic":       "South Korea",
    "Türkiye":              "Turkey",
    "USA":                  "United States",
    "Northern Ireland":     "Northern Ireland",
    "Ivory Coast":          "Côte d'Ivoire",
    "Cape Verde":           "Cape Verde Islands",
    "Cabo Verde":           "Cape Verde Islands",
    "DR Congo":             "Congo DR",
    # football-data.org → ranking full
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Cape Verde Islands":   "Cape Verde Islands",
    "Congo DR":             "Congo DR",
}

# Forma: mapeamento de string W/D/L para peso ordinal
_FORM_WEIGHTS = {"W": 1.0, "D": 0.0, "L": -1.0}


# ---------------------------------------------------------------------------
# Helpers: ranking FIFA
# ---------------------------------------------------------------------------

def _load_ranking(wc2026_only: bool = False) -> pd.DataFrame:
    """Carrega ranking FIFA. wc2026_only=True usa apenas os 48 participantes."""
    path = _RANKING_WC2026 if wc2026_only else _RANKING_FULL
    df = pd.read_csv(path, dtype={"fifa_rank": int, "fifa_points": float})
    return df


def _build_rank_lookup(wc2026_only: bool = False) -> dict[str, dict]:
    """Retorna dict {team_name: {fifa_rank, fifa_points, confederation}}."""
    df = _load_ranking(wc2026_only)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        lookup[row["team"]] = {
            "fifa_rank":      int(row["fifa_rank"]),
            "fifa_points":    float(row["fifa_points"]),
            "confederation":  row["confederation"],
        }
    return lookup


def _resolve_team(name: str, lookup: dict[str, dict]) -> Optional[dict]:
    """Resolve nome de time para entrada do ranking, aplicando aliases."""
    canonical = _TEAM_ALIASES.get(name, name)
    return lookup.get(canonical) or lookup.get(name)


# ---------------------------------------------------------------------------
# Helpers: encoding de features
# ---------------------------------------------------------------------------

def _encode_form(form_str: Optional[str], n: int = 3) -> float:
    """Converte string de forma ('W,D,L') em score médio [-1, 1].

    Usa os últimos n resultados. Retorna 0.0 se form_str for None/vazio.
    """
    if not form_str:
        return 0.0
    parts = [r.strip() for r in form_str.split(",") if r.strip() in _FORM_WEIGHTS]
    recent = parts[-n:] if len(parts) >= n else parts
    if not recent:
        return 0.0
    return sum(_FORM_WEIGHTS[r] for r in recent) / len(recent)


def _days_rest(date_str: str, prev_date_str: Optional[str]) -> Optional[float]:
    """Dias entre duas partidas. Retorna None se prev_date_str ausente."""
    if not prev_date_str:
        return None
    from datetime import date
    fmt = "%Y-%m-%d"
    d1 = date.fromisoformat(date_str[:10])
    d0 = date.fromisoformat(prev_date_str[:10])
    diff = (d1 - d0).days
    return float(diff) if diff >= 0 else None


# ---------------------------------------------------------------------------
# Build de features — StatsBomb (treino)
# ---------------------------------------------------------------------------

def _statsbomb_group_stats(matches: list[dict], team_name: str) -> dict:
    """Agrega estatísticas da fase de grupos para um time a partir de partidas StatsBomb.

    Considera apenas matches com stage='Group Stage'.
    """
    gp = gs = gf = ga = 0
    for m in matches:
        stage = m.get("competition_stage", {}).get("name", "")
        if stage != "Group Stage":
            continue
        ht = m["home_team"]["home_team_name"]
        at = m["away_team"]["away_team_name"]
        if ht != team_name and at != team_name:
            continue
        hs, as_ = m["home_score"], m["away_score"]
        gp += 1
        if ht == team_name:
            gf += hs; ga += as_
            if hs > as_: gs += 3
            elif hs == as_: gs += 1
        else:
            gf += as_; ga += hs
            if as_ > hs: gs += 3
            elif as_ == hs: gs += 1
    if gp == 0:
        return {}
    clean_sheets = sum(
        1 for m in matches
        if m.get("competition_stage", {}).get("name") == "Group Stage"
        and (m["home_team"]["home_team_name"] == team_name and m["away_score"] == 0
             or m["away_team"]["away_team_name"] == team_name and m["home_score"] == 0)
    )
    # Forma: últimas 3 partidas de grupo (V/E/D)
    group_matches = [
        m for m in matches
        if m.get("competition_stage", {}).get("name") == "Group Stage"
        and (m["home_team"]["home_team_name"] == team_name
             or m["away_team"]["away_team_name"] == team_name)
    ]
    group_matches.sort(key=lambda x: x["match_date"])
    form_parts = []
    for m in group_matches[-3:]:
        ht = m["home_team"]["home_team_name"]
        hs, as_ = m["home_score"], m["away_score"]
        if ht == team_name:
            form_parts.append("W" if hs > as_ else ("D" if hs == as_ else "L"))
        else:
            form_parts.append("W" if as_ > hs else ("D" if as_ == hs else "L"))
    return {
        "points": gs, "goals_for": gf, "goals_against": ga,
        "goal_diff": gf - ga, "played": gp,
        "goals_for_pg": round(gf / gp, 3), "goals_against_pg": round(ga / gp, 3),
        "clean_sheets": clean_sheets,
        "form_score": _encode_form(",".join(form_parts)),
    }


def build_training_dataset() -> pd.DataFrame:
    """Constrói DataFrame de treino a partir do StatsBomb Open Data.

    Uma linha por partida de mata-mata (qualquer stage != 'Group Stage').
    A = time com menor fifa_rank (melhor posição). Target incluído.

    ⚠️  Nenhuma feature de xG, posse ou passes é incluída (SPEC.md §2.4).
    """
    rank_lookup = _build_rank_lookup(wc2026_only=False)
    all_matches = fetch_all_training_matches()

    # Agrupa por (competition, season) para calcular stats de grupo por torneio
    from itertools import groupby
    from operator import itemgetter

    def comp_key(m: dict) -> tuple:
        return (m["competition"]["competition_id"], m["season"]["season_id"])

    all_matches_sorted = sorted(all_matches, key=comp_key)
    rows = []

    for (cid, sid), comp_matches in groupby(all_matches_sorted, key=comp_key):
        comp_matches = list(comp_matches)
        label = comp_matches[0].get("_training_label", f"comp_{cid}_s{sid}")
        knockout = [
            m for m in comp_matches
            if m.get("competition_stage", {}).get("name") != "Group Stage"
            and m["match_status"] == "available"
        ]

        for m in knockout:
            home = m["home_team"]["home_team_name"]
            away = m["away_team"]["away_team_name"]
            home_r = _resolve_team(home, rank_lookup)
            away_r = _resolve_team(away, rank_lookup)

            # Determina A (melhor ranking = rank menor numericamente)
            if home_r and away_r:
                if home_r["fifa_rank"] <= away_r["fifa_rank"]:
                    team_A, team_B, rank_A, rank_B = home, away, home_r, away_r
                    hs, as_ = m["home_score"], m["away_score"]
                else:
                    team_A, team_B, rank_A, rank_B = away, home, away_r, home_r
                    hs, as_ = m["away_score"], m["home_score"]
            else:
                # Ranking ausente: usa home como A por default, flags NaN
                team_A, team_B = home, away
                rank_A = home_r or {}
                rank_B = away_r or {}
                hs, as_ = m["home_score"], m["away_score"]

            # Target
            if hs > as_:   result = "A_win"
            elif hs < as_: result = "B_win"
            else:          result = "draw"

            # Agregados de grupo
            stats_A = _statsbomb_group_stats(comp_matches, team_A)
            stats_B = _statsbomb_group_stats(comp_matches, team_B)

            # Data da última partida de grupo (para descanso)
            def last_group_date(team: str) -> Optional[str]:
                gms = sorted(
                    [mx for mx in comp_matches
                     if mx.get("competition_stage", {}).get("name") == "Group Stage"
                     and (mx["home_team"]["home_team_name"] == team
                          or mx["away_team"]["away_team_name"] == team)],
                    key=lambda x: x["match_date"]
                )
                return gms[-1]["match_date"] if gms else None

            row = {
                "source":        "statsbomb",
                "competition":   label,
                "match_id":      m["match_id"],
                "match_date":    m["match_date"],
                "stage":         m["competition_stage"]["name"],
                "stage_ord":     STAGE_ORD.get(m["competition_stage"]["name"], 1),
                "team_A":        team_A,
                "team_B":        team_B,
                # Grupo A
                "A_points":         stats_A.get("points"),
                "A_goals_for":      stats_A.get("goals_for"),
                "A_goals_against":  stats_A.get("goals_against"),
                "A_goal_diff":      stats_A.get("goal_diff"),
                "A_goals_for_pg":   stats_A.get("goals_for_pg"),
                "A_goals_against_pg": stats_A.get("goals_against_pg"),
                "A_clean_sheets":   stats_A.get("clean_sheets"),
                "A_form_score":     stats_A.get("form_score"),
                # Grupo B
                "B_points":         stats_B.get("points"),
                "B_goals_for":      stats_B.get("goals_for"),
                "B_goals_against":  stats_B.get("goals_against"),
                "B_goal_diff":      stats_B.get("goal_diff"),
                "B_goals_for_pg":   stats_B.get("goals_for_pg"),
                "B_goals_against_pg": stats_B.get("goals_against_pg"),
                "B_clean_sheets":   stats_B.get("clean_sheets"),
                "B_form_score":     stats_B.get("form_score"),
                # Ranking e contexto
                "A_fifa_rank":    rank_A.get("fifa_rank"),
                "A_fifa_points":  rank_A.get("fifa_points"),
                "A_confederation": rank_A.get("confederation"),
                "B_fifa_rank":    rank_B.get("fifa_rank"),
                "B_fifa_points":  rank_B.get("fifa_points"),
                "B_confederation": rank_B.get("confederation"),
                "delta_rank":     (rank_A.get("fifa_rank", float("nan"))
                                   - rank_B.get("fifa_rank", float("nan"))) if rank_A and rank_B else None,
                # Descanso
                "A_days_rest": _days_rest(m["match_date"], last_group_date(team_A)),
                "B_days_rest": _days_rest(m["match_date"], last_group_date(team_B)),
                # Target
                "result_90min": result,
            }
            row["delta_rest"] = (
                (row["A_days_rest"] - row["B_days_rest"])
                if row["A_days_rest"] is not None and row["B_days_rest"] is not None
                else None
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build de features — football-data.org (inferência WC2026)
# ---------------------------------------------------------------------------

def build_inference_dataset() -> pd.DataFrame:
    """Constrói DataFrame de inferência para os jogos do mata-mata da WC2026.

    Usa standings (fase de grupos) + matches para datas/descanso.
    result_90min = NaN para partidas ainda TIMED (não disputadas).

    ⚠️  Mesmas colunas de Tier 1 que build_training_dataset() — SPEC.md §2.4.
    """
    rank_lookup = _build_rank_lookup(wc2026_only=True)

    standings_raw = fetch_standings()
    matches_raw   = fetch_matches()

    # -- Standings → dict {tla: stats de grupo}
    group_stats: dict[str, dict] = {}
    for group in standings_raw.get("standings", []):
        for row in group.get("table", []):
            team = row["team"]
            tla  = team["tla"]
            gp   = row["playedGames"]
            gf   = row["goalsFor"]
            ga   = row["goalsAgainst"]
            pts  = row["points"]
            form_raw = row.get("form") or ""
            group_stats[tla] = {
                "name":          team["name"],
                "points":        pts,
                "goals_for":     gf,
                "goals_against": ga,
                "goal_diff":     row["goalDifference"],
                "played":        gp,
                "goals_for_pg":  round(gf / gp, 3) if gp else 0.0,
                "goals_against_pg": round(ga / gp, 3) if gp else 0.0,
                "clean_sheets":  sum(
                    1 for m in matches_raw.get("matches", [])
                    if m["stage"] == "GROUP_STAGE"
                    and m["status"] == "FINISHED"
                    and (
                        (m["homeTeam"]["tla"] == tla and (m["score"]["fullTime"]["away"] or 0) == 0)
                        or (m["awayTeam"]["tla"] == tla and (m["score"]["fullTime"]["home"] or 0) == 0)
                    )
                ),
                "form_score": _encode_form(form_raw),
            }

    # -- Última data de jogo na fase de grupos por TLA
    last_group: dict[str, str] = {}
    for m in matches_raw.get("matches", []):
        if m["stage"] != "GROUP_STAGE" or m["status"] != "FINISHED":
            continue
        date_str = m["utcDate"][:10]
        for tla in (m["homeTeam"]["tla"], m["awayTeam"]["tla"]):
            if tla not in last_group or date_str > last_group[tla]:
                last_group[tla] = date_str

    # -- Partidas do mata-mata
    knockout_stages = {"LAST_32", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "THIRD_PLACE", "FINAL"}
    rows = []

    for m in matches_raw.get("matches", []):
        if m["stage"] not in knockout_stages:
            continue

        home_tla = m["homeTeam"]["tla"]
        away_tla = m["awayTeam"]["tla"]
        home_name = m["homeTeam"].get("name", home_tla)
        away_name = m["awayTeam"].get("name", away_tla)

        home_r = _resolve_team(home_name, rank_lookup) or _resolve_team(home_tla, rank_lookup)
        away_r = _resolve_team(away_name, rank_lookup) or _resolve_team(away_tla, rank_lookup)

        # Convenção A/B
        if home_r and away_r and home_r["fifa_rank"] <= away_r["fifa_rank"]:
            team_A, tla_A, rank_A = home_name, home_tla, home_r
            team_B, tla_B, rank_B = away_name, away_tla, away_r
            score_A = m["score"]["fullTime"].get("home")
            score_B = m["score"]["fullTime"].get("away")
        else:
            team_A, tla_A, rank_A = away_name, away_tla, away_r or {}
            team_B, tla_B, rank_B = home_name, home_tla, home_r or {}
            score_A = m["score"]["fullTime"].get("away")
            score_B = m["score"]["fullTime"].get("home")

        stats_A = group_stats.get(tla_A, {})
        stats_B = group_stats.get(tla_B, {})

        match_date = m["utcDate"][:10]
        days_A = _days_rest(match_date, last_group.get(tla_A))
        days_B = _days_rest(match_date, last_group.get(tla_B))

        # Target (NaN se TIMED)
        result = None
        if m["status"] == "FINISHED" and score_A is not None and score_B is not None:
            if score_A > score_B:   result = "A_win"
            elif score_A < score_B: result = "B_win"
            else:                   result = "draw"

        row = {
            "source":        "footballdata",
            "competition":   "FIFA World Cup 2026",
            "match_id":      m["id"],
            "match_date":    match_date,
            "status":        m["status"],
            "stage":         m["stage"],
            "stage_ord":     STAGE_ORD.get(m["stage"], 1),
            "team_A":        team_A,
            "team_B":        team_B,
            "tla_A":         tla_A,
            "tla_B":         tla_B,
            "A_points":           stats_A.get("points"),
            "A_goals_for":        stats_A.get("goals_for"),
            "A_goals_against":    stats_A.get("goals_against"),
            "A_goal_diff":        stats_A.get("goal_diff"),
            "A_goals_for_pg":     stats_A.get("goals_for_pg"),
            "A_goals_against_pg": stats_A.get("goals_against_pg"),
            "A_clean_sheets":     stats_A.get("clean_sheets"),
            "A_form_score":       stats_A.get("form_score"),
            "B_points":           stats_B.get("points"),
            "B_goals_for":        stats_B.get("goals_for"),
            "B_goals_against":    stats_B.get("goals_against"),
            "B_goal_diff":        stats_B.get("goal_diff"),
            "B_goals_for_pg":     stats_B.get("goals_for_pg"),
            "B_goals_against_pg": stats_B.get("goals_against_pg"),
            "B_clean_sheets":     stats_B.get("clean_sheets"),
            "B_form_score":       stats_B.get("form_score"),
            "A_fifa_rank":    rank_A.get("fifa_rank"),
            "A_fifa_points":  rank_A.get("fifa_points"),
            "A_confederation": rank_A.get("confederation"),
            "B_fifa_rank":    rank_B.get("fifa_rank"),
            "B_fifa_points":  rank_B.get("fifa_points"),
            "B_confederation": rank_B.get("confederation"),
            "delta_rank":     (rank_A.get("fifa_rank", float("nan"))
                               - rank_B.get("fifa_rank", float("nan"))) if rank_A and rank_B else None,
            "A_days_rest":   days_A,
            "B_days_rest":   days_B,
            "delta_rest":    (days_A - days_B) if days_A is not None and days_B is not None else None,
            "result_90min":  result,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)
