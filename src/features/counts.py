"""Build de features de CONTAGEM (Modo 2) — passo 3 do SPEC_MARKETS.md §10.

SPEC_MARKETS.md §4 (features) e §2.4 (paridade treino↔inferência).

PRINCÍPIO CENTRAL (§2.4): cada feature é FUNÇÃO PURA de (time, data, corpus). A MESMA
função canônica `build_count_features` roda no treino e na inferência — a paridade sai
por CONSTRUÇÃO, não por dois caminhos que "deveriam" ser iguais. Os builders de treino e
inferência são chamadores finos dessa função única.

Fonte: corpus de contagem do SofaScore — `sofascore.fetch_historical_counts()` (394 jogos).

Regras (sem vazamento de futuro):
  - Corpus ordenado por `start_timestamp` (hora de início); `event_id` só desempata. NUNCA
    ordenar por event_id primário (ID de cadastro ≠ ordem cronológica).
  - Janela móvel N=10: os últimos 10 jogos de CADA time com `start_timestamp` ESTRITAMENTE
    MENOR ('<') que o do jogo atual. <10 anteriores → usa os que houver (n games).
  - Rastreio SEPARADO a favor/contra, por time: escanteios feitos/cedidos, cartões do
    time/provocados, gols marcados/sofridos (dobra as features).
  - Shrinkage k=5: media_ajustada = (n·media_time + k·media_global)/(n+k). A media_global é
    EXPANSIVA — média de todos os jogos ANTERIORES ao atual, nunca o corpus inteiro. n=0 →
    recebe só o prior global (sem caso especial: primeiro jogo é apenas n=0).
  - n_games_A / n_games_B entram como FEATURES (confiança da média).
  - Orientação A/B (A = melhor ranking FIFA = rank menor) em todas as features de delta.

Cuidados de dados:
  - GOLS: usar `*_goals_reg` (normaltime, 90min). O placar `*_score` do SofaScore INCLUI
    pênaltis em mata-mata (ex.: 0-0 nos pênaltis vira 4-2) — jamais usar como gols.
  - CARTÕES: total = amarelos + vermelhos por time (ausência já normalizada a 0 na coleta).
  - xG é Tier 2 OPCIONAL (ausente em WC2018/AFCON) → NÃO entra em FEATURE_COLS_COUNTS base;
    é bloco separado, ativável por `include_xg=True` (FEATURE_COLS_COUNTS_XG).

NÃO treina modelo. Só features + targets.

Uso:
  from src.features.counts import build_count_training_dataset, build_count_inference_dataset
  from src.features.counts import build_count_features, FEATURE_COLS_COUNTS
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.collect.sofascore import fetch_historical_counts
from src.features import tier1

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
N_WINDOW = 10       # janela móvel por time
SHRINK_K = 5        # força do prior no shrinkage

# Aliases: nomes do SofaScore → nome canônico no ranking FIFA. Estende o mapa do Tier 1
# com os dois casos que faltavam para os nomes do SofaScore (medido no corpus).
_COUNTS_ALIASES: dict[str, str] = {
    **tier1._TEAM_ALIASES,
    "Iran":        "IR Iran",
    "South Korea": "Korea Republic",
}

# Fase (roundInfo.name do SofaScore) → ordinal, alinhado ao STAGE_ORD do Tier 1.
# Jogos de fase de grupos não têm roundInfo.name (None) → 0.
_STAGE_ORD_SOFA: dict[Optional[str], int] = {
    None:                  0,   # fase de grupos
    "Round of 16":         1,
    "Match for 3rd place": 1,
    "Quarterfinals":       2,
    "Semifinals":          3,
    "Final":               4,
}

# --- Colunas canônicas (ordem FIXA — o gate de paridade §4/§2.4 verifica esta lista) ---
META_COLS: list[str] = [
    "source", "tournament", "season", "event_id", "date", "start_timestamp",
    "team_A", "team_B", "round_name", "home_is_A",
]

FEATURE_COLS_COUNTS: list[str] = [
    "stage_ord",
    # Time A — médias móveis (shrunk) a favor/contra + confiança
    "A_corners_for", "A_corners_against",
    "A_cards_for", "A_cards_against",
    "A_goals_for", "A_goals_against",
    "A_n_games",
    # Time B
    "B_corners_for", "B_corners_against",
    "B_cards_for", "B_cards_against",
    "B_goals_for", "B_goals_against",
    "B_n_games",
    # Contexto (herdado do Tier 1)
    "A_fifa_rank", "A_fifa_points", "A_confederation",
    "B_fifa_rank", "B_fifa_points", "B_confederation",
    "delta_rank",
    "A_days_rest", "B_days_rest", "delta_rest",
]

# Bloco xG opcional (Tier 2, flag-gated — NÃO faz parte da base).
FEATURE_COLS_COUNTS_XG: list[str] = [
    "A_xg_for", "A_xg_against", "B_xg_for", "B_xg_against",
]

TARGET_COLS: list[str] = [
    "target_corners_A", "target_corners_B",
    "target_cards_A", "target_cards_B",
    "target_goals_A", "target_goals_B",
]


# ---------------------------------------------------------------------------
# Helpers de resolução de ranking
# ---------------------------------------------------------------------------

def _resolve(name: str, lookup: dict[str, dict]) -> Optional[dict]:
    canonical = _COUNTS_ALIASES.get(name, name)
    return lookup.get(canonical) or lookup.get(name)


def _num(x: Any) -> float:
    """None/NaN → 0.0; caso contrário float(x)."""
    if x is None:
        return 0.0
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f  # NaN → 0


def _team_side_counts(g: dict, side: str) -> dict[str, float]:
    """Contagens de um lado do jogo (corners, cards=amar+verm, goals 90min)."""
    other = "away" if side == "home" else "home"
    return {
        "corners_for": _num(g.get(f"{side}_corners")),
        "corners_against": _num(g.get(f"{other}_corners")),
        "cards_for": _num(g.get(f"{side}_yellow")) + _num(g.get(f"{side}_red")),
        "cards_against": _num(g.get(f"{other}_yellow")) + _num(g.get(f"{other}_red")),
        "goals_for": _num(g.get(f"{side}_goals_reg")),
        "goals_against": _num(g.get(f"{other}_goals_reg")),
        "xg_for": g.get(f"{side}_xg"),
        "xg_against": g.get(f"{other}_xg"),
    }


# ---------------------------------------------------------------------------
# Núcleo: features de um time a partir da janela móvel + shrinkage
# ---------------------------------------------------------------------------

_METRICS = ("corners", "cards", "goals")
_DIRECTIONS = ("for", "against")


def _global_prior_means(prior: list[dict]) -> dict[str, float]:
    """Média EXPANSIVA global por métrica sobre TODAS as obs time-jogo anteriores.

    Cada jogo anterior contribui 2 observações (home e away). Para corners/cards/goals a
    média 'for' e 'against' são simétricas no agregado, então uma média por métrica basta.
    """
    sums = {m: 0.0 for m in _METRICS}
    count = 0
    for g in prior:
        h = _team_side_counts(g, "home")
        a = _team_side_counts(g, "away")
        for m in _METRICS:
            sums[m] += h[f"{m}_for"] + a[f"{m}_for"]
        count += 2
    return {m: (sums[m] / count if count else 0.0) for m in _METRICS}


def _team_features(prefix: str, team: str, rank: Optional[dict], prior: list[dict],
                   gmean: dict[str, float], cur_date: str, include_xg: bool) -> dict:
    """Features de um time: janela dos últimos N jogos anteriores + shrinkage + contexto."""
    games = [g for g in prior if g["home_team"] == team or g["away_team"] == team]
    games.sort(key=lambda g: (g["start_timestamp"], g["event_id"]))
    window = games[-N_WINDOW:]
    n = len(window)

    sums = {f"{m}_{d}": 0.0 for m in _METRICS for d in _DIRECTIONS}
    for g in window:
        side = "home" if g["home_team"] == team else "away"
        c = _team_side_counts(g, side)
        for m in _METRICS:
            for d in _DIRECTIONS:
                sums[f"{m}_{d}"] += c[f"{m}_{d}"]

    out: dict[str, Any] = {}
    for m in _METRICS:
        for d in _DIRECTIONS:
            team_mean = (sums[f"{m}_{d}"] / n) if n else 0.0
            # Shrinkage: n=0 → só o prior global (media_global de m).
            out[f"{prefix}_{m}_{d}"] = (
                (n * team_mean + SHRINK_K * gmean[m]) / (n + SHRINK_K)
            )
    out[f"{prefix}_n_games"] = n

    # Contexto de ranking
    out[f"{prefix}_fifa_rank"] = rank.get("fifa_rank") if rank else None
    out[f"{prefix}_fifa_points"] = rank.get("fifa_points") if rank else None
    out[f"{prefix}_confederation"] = rank.get("confederation") if rank else None

    # Descanso: dias desde o último jogo anterior do time (qualquer, não só da janela).
    last_date = games[-1]["date"] if games else None
    out[f"{prefix}_days_rest"] = tier1._days_rest(cur_date, last_date)

    if include_xg:
        # xG só onde disponível nos jogos da janela (prior). n_xg pode ser < n.
        xg_for = [c for g in window
                  for c in [_team_side_counts(g, "home" if g["home_team"] == team else "away")]
                  if c["xg_for"] is not None]
        n_xg = len(xg_for)
        out[f"{prefix}_xg_for"] = (
            sum(_num(c["xg_for"]) for c in xg_for) / n_xg if n_xg else None)
        out[f"{prefix}_xg_against"] = (
            sum(_num(c["xg_against"]) for c in xg_for) / n_xg if n_xg else None)
    return out


# ---------------------------------------------------------------------------
# FUNÇÃO CANÔNICA — a mesma no treino e na inferência (§2.4)
# ---------------------------------------------------------------------------

def build_count_features(match: dict, corpus: list[dict], rank_lookup: dict[str, dict],
                         include_xg: bool = False) -> dict:
    """Features pré-jogo de UM confronto, função pura de (times, data, corpus).

    match: dict com home_team, away_team, start_timestamp, date, round_name.
    corpus: lista de jogos (dicts). Só jogos com start_timestamp < o do match e has_stats
            entram na história — nenhum vazamento de futuro.
    Retorna dict com FEATURE_COLS_COUNTS + chaves de meta (team_A/team_B/home_is_A).
    """
    home, away = match["home_team"], match["away_team"]
    hr = _resolve(home, rank_lookup)
    ar = _resolve(away, rank_lookup)

    # Orientação A/B: A = melhor ranking (rank menor). Sem ranking → home vira A (fallback).
    home_is_A = True if not (hr and ar) else (hr["fifa_rank"] <= ar["fifa_rank"])
    if home_is_A:
        team_A, rank_A, team_B, rank_B = home, hr, away, ar
    else:
        team_A, rank_A, team_B, rank_B = away, ar, home, hr

    ts = match["start_timestamp"]
    cur_date = match["date"]
    # História: estritamente anterior ('<') e com contagens válidas.
    prior = [g for g in corpus if g["start_timestamp"] < ts and g.get("has_stats")]
    gmean = _global_prior_means(prior)

    feat_A = _team_features("A", team_A, rank_A, prior, gmean, cur_date, include_xg)
    feat_B = _team_features("B", team_B, rank_B, prior, gmean, cur_date, include_xg)

    out: dict[str, Any] = {**feat_A, **feat_B}
    out["stage_ord"] = _STAGE_ORD_SOFA.get(match.get("round_name"), 0)
    out["delta_rank"] = (
        (rank_A["fifa_rank"] - rank_B["fifa_rank"]) if rank_A and rank_B else None)
    ra, rb = out["A_days_rest"], out["B_days_rest"]
    out["delta_rest"] = (ra - rb) if ra is not None and rb is not None else None

    # Meta de orientação (para o caller alinhar os targets)
    out["team_A"] = team_A
    out["team_B"] = team_B
    out["home_is_A"] = home_is_A
    return out


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _load_corpus() -> list[dict]:
    """Corpus histórico como lista de dicts, ordenado por (start_timestamp, event_id)."""
    df = fetch_historical_counts()
    if df.empty:
        return []
    df = df.sort_values(["start_timestamp", "event_id"]).reset_index(drop=True)
    return df.to_dict("records")


def _match_targets(match: dict, home_is_A: bool) -> dict:
    """Contagens REAIS do jogo, orientadas A/B (gols = 90min regulamentar)."""
    side_A = "home" if home_is_A else "away"
    cA = _team_side_counts(match, side_A)
    side_B = "away" if home_is_A else "home"
    cB = _team_side_counts(match, side_B)
    return {
        "target_corners_A": cA["corners_for"], "target_corners_B": cB["corners_for"],
        "target_cards_A": cA["cards_for"],     "target_cards_B": cB["cards_for"],
        "target_goals_A": cA["goals_for"],     "target_goals_B": cB["goals_for"],
    }


def _assemble_row(match: dict, feat: dict, include_xg: bool,
                  with_targets: bool) -> dict:
    """Monta uma linha na ordem canônica: META + FEATURES (+xG) (+TARGETS)."""
    cols = FEATURE_COLS_COUNTS + (FEATURE_COLS_COUNTS_XG if include_xg else [])
    row = {
        "source": "sofascore",
        "tournament": match.get("tournament"),
        "season": match.get("season"),
        "event_id": match.get("event_id"),
        "date": match.get("date"),
        "start_timestamp": match.get("start_timestamp"),
        "team_A": feat["team_A"],
        "team_B": feat["team_B"],
        "round_name": match.get("round_name"),
        "home_is_A": feat["home_is_A"],
    }
    for c in cols:
        row[c] = feat.get(c)
    if with_targets:
        row.update(_match_targets(match, feat["home_is_A"]))
    else:
        for t in TARGET_COLS:
            row[t] = None
    return row


# ---------------------------------------------------------------------------
# Builders finos — treino e inferência (mesma função canônica)
# ---------------------------------------------------------------------------

def build_count_training_dataset(include_xg: bool = False) -> pd.DataFrame:
    """Dataset de treino: cada jogo com /statistics, em ordem temporal.

    FEATURES = médias móveis ANTERIORES (pré-jogo). TARGETS = contagens REAIS do jogo.
    Nunca featuriza um jogo com o próprio resultado (janela usa só '<').
    """
    corpus = _load_corpus()
    rank_lookup = tier1._build_rank_lookup(wc2026_only=False)
    rows = []
    for m in corpus:  # já em ordem temporal
        if not m.get("has_stats"):
            continue
        feat = build_count_features(m, corpus, rank_lookup, include_xg=include_xg)
        rows.append(_assemble_row(m, feat, include_xg, with_targets=True))
    df = pd.DataFrame(rows)
    cols = (META_COLS + FEATURE_COLS_COUNTS
            + (FEATURE_COLS_COUNTS_XG if include_xg else []) + TARGET_COLS)
    return df[cols]


def build_count_inference_dataset(corpus_atualizado: list[dict],
                                  include_xg: bool = False) -> pd.DataFrame:
    """Dataset de inferência: jogos da WC2026, MESMA função canônica.

    corpus_atualizado = histórico + jogos da Copa já disputados (+ eventuais stubs de
    confrontos futuros com home_team/away_team/start_timestamp/date/round_name). Featuriza
    os jogos WC2026 (tournament_id=16, season_year=2026). Targets vazios (jogo por disputar).
    """
    rank_lookup = tier1._build_rank_lookup(wc2026_only=False)
    corpus = sorted(corpus_atualizado, key=lambda g: (g["start_timestamp"], g["event_id"]))
    wc = [m for m in corpus
          if m.get("tournament_id") == 16 and m.get("season_year") == 2026]
    rows = []
    for m in wc:
        feat = build_count_features(m, corpus, rank_lookup, include_xg=include_xg)
        rows.append(_assemble_row(m, feat, include_xg, with_targets=False))
    df = pd.DataFrame(rows)
    cols = (META_COLS + FEATURE_COLS_COUNTS
            + (FEATURE_COLS_COUNTS_XG if include_xg else []) + TARGET_COLS)
    if df.empty:  # nenhum jogo WC2026 no corpus ainda
        return pd.DataFrame(columns=cols)
    return df[cols]


# ---------------------------------------------------------------------------
# CLI / demonstração (sem treinar nada)
# ---------------------------------------------------------------------------

def _print_diagnostics(df: pd.DataFrame) -> None:
    print(f"\nshape: {df.shape}  ({len(FEATURE_COLS_COUNTS)} features base + "
          f"{len(TARGET_COLS)} targets + {len(META_COLS)} meta)")
    print("\nprimeiras linhas (subconjunto de colunas):")
    show = ["date", "team_A", "team_B", "A_n_games", "B_n_games",
            "A_corners_for", "B_corners_for", "A_cards_for",
            "target_corners_A", "target_corners_B", "target_goals_A", "target_goals_B"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[show].head(8).to_string(index=False))

    # Distribuição de n_games (A e B juntos = 2 obs/jogo)
    ng = pd.concat([df["A_n_games"], df["B_n_games"]], ignore_index=True)
    print(f"\nDISTRIBUIÇÃO de n_games (janela por time, {len(ng)} obs = 2/jogo):")
    vc = ng.value_counts().sort_index()
    for k, v in vc.items():
        bar = "█" * min(40, v // 3)
        print(f"  n={int(k):>2}: {v:>4}  {bar}")
    n_lt10 = int((ng < N_WINDOW).sum())
    print(f"  → {n_lt10}/{len(ng)} ({100*n_lt10/len(ng):.0f}%) com n<{N_WINDOW} (janela incompleta)")
    print(f"  → média n_games = {ng.mean():.1f}")


def main() -> None:
    print("=== build_count_training_dataset() ===")
    df = build_count_training_dataset()
    _print_diagnostics(df)

    # Sanity de não-vazamento: o 1º jogo cronológico deve ter A_n_games=B_n_games=0.
    first = df.iloc[0]
    assert first["A_n_games"] == 0 and first["B_n_games"] == 0, \
        "1º jogo deveria ter janela vazia (n=0) — possível vazamento."
    print("\n✓ sanity: 1º jogo cronológico tem n_games=0 (sem vazamento de futuro).")

    # Demonstração da inferência com a MESMA função canônica (stub de confronto WC2026).
    print("\n=== build_count_inference_dataset() — demo com stub WC2026 ===")
    corpus = _load_corpus()
    last_ts = max(g["start_timestamp"] for g in corpus)
    stub = {
        "tournament_id": 16, "season_year": 2026, "tournament": "FIFA World Cup",
        "season": "World Cup 2026", "event_id": 999_000_001,
        "start_timestamp": last_ts + 86_400 * 365, "date": "2026-06-15",
        "home_team": "Argentina", "away_team": "Spain", "round_name": None,
        "has_stats": False,
    }
    inf = build_count_inference_dataset(corpus + [stub])
    print(f"shape: {inf.shape}  (targets vazios: "
          f"{inf[TARGET_COLS].isna().all().all() if not inf.empty else 'n/a'})")
    if not inf.empty:
        show = ["date", "team_A", "team_B", "A_n_games", "B_n_games",
                "A_corners_for", "B_corners_for", "delta_rank"]
        print(inf[show].to_string(index=False))
        assert list(inf.columns) == list(df.columns), "PARIDADE quebrada: colunas divergem!"
        print("✓ paridade: inferência produz EXATAMENTE as mesmas colunas do treino.")


if __name__ == "__main__":
    main()
