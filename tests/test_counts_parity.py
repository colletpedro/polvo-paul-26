"""Gate de paridade treino↔inferência das features de CONTAGEM — SPEC_MARKETS.md §2.4/§10 passo 4.

Análogo do `test_parity.py` do Modo 1, para o Modo 2. A promessa central da spec: a
inferência produz EXATAMENTE as mesmas colunas de feature que o treino, na mesma ordem,
com a MESMA semântica — porque `build_count_features` é função pura única e os builders de
treino/inferência são chamadores finos dela. Este teste transforma essa garantia em regressão.

Duas camadas (o cache `data/raw/sofascore/` NÃO é git-tracked — um clone limpo não o tem):

  CAMADA A — corpus SINTÉTICO, sempre roda (CI-safe, sem rede, sem cache). Mini-corpus
             construído no teste, no schema real dos `hist_*.jsonl`. Cobre paridade de
             colunas e de valores, corte temporal estrito, orientação A/B e n_games.
  CAMADA B — smoke test sobre o cache REAL (skip se ausente). Reproduz a verificação do
             Passo 3: treino 394×41, inferência com as mesmas colunas, sem NaN em colunas
             não-nullable. Só leitura local, sem rede.

Se um teste falhar, ele é um ACHADO sobre counts.py — não se ajusta o teste pra passar.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features import counts, tier1
from src.features.counts import (
    FEATURE_COLS_COUNTS,
    FEATURE_COLS_COUNTS_XG,
    META_COLS,
    TARGET_COLS,
    build_count_features,
    build_count_inference_dataset,
    build_count_training_dataset,
)

_DAY = 86_400
_BASE_TS = 1_600_000_000  # timestamp fixo (sem Date.now — determinístico)


# ===========================================================================
# Helpers de fixture — registros no MESMO schema dos hist_*.jsonl
# ===========================================================================

def _rec(event_id, day_offset, home, away, *, hc, ac, hy=0, ay=0, hr=0, ar=0,
         hg=0, ag=0, hxg=None, axg=None, round_name=None,
         tournament_id=1, season_year=2020, has_stats=True):
    """Um jogo no formato de registro do cache (espelha _build_row do coletor)."""
    ts = _BASE_TS + day_offset * _DAY
    date = f"2020-{1 + (day_offset // 28):02d}-{1 + (day_offset % 28):02d}"
    return {
        "tournament_id": tournament_id, "tournament": "Synthetic Cup",
        "season_id": 999, "season": "Synthetic 2020", "season_year": season_year,
        "event_id": event_id, "date": date, "start_timestamp": ts,
        "home_team": home, "away_team": away,
        "home_score": hg, "away_score": ag,
        "home_goals_reg": hg, "away_goals_reg": ag,
        "round_name": round_name, "cup_round_type": None, "round_num": 1,
        "home_corners": hc, "away_corners": ac,
        "home_yellow": hy, "away_yellow": ay, "home_red": hr, "away_red": ar,
        "home_possession": 50.0, "away_possession": 50.0,
        "home_xg": hxg, "away_xg": axg,
        "has_stats": has_stats,
    }


def _ranks(order):
    """rank_lookup {time: {fifa_rank, fifa_points, confederation}} a partir de uma ordem."""
    return {
        team: {"fifa_rank": i + 1, "fifa_points": 2000.0 - i, "confederation": "UEFA"}
        for i, team in enumerate(order)
    }


def _feat_cols(df):
    return [c for c in df.columns if c in FEATURE_COLS_COUNTS]


def _eq(a, b):
    if pd.isna(a) and pd.isna(b):
        return True
    return a == b


# ===========================================================================
# CAMADA A — corpus sintético (CI-safe)
# ===========================================================================

# Corpus com história + um jogo "WC2026" (tid=16, 2026) jogado (has_stats) que é ao
# mesmo tempo alvo de inferência E linha de treino — permite comparar os dois modos.
def _corpus_ab():
    corpus = [
        _rec(1, 1,  "Alpha", "Charlie", hc=8, ac=3, hy=2, ay=1, hg=2, ag=0),
        _rec(2, 2,  "Bravo", "Delta",   hc=4, ac=5, hy=1, ay=3, hg=1, ag=1),
        _rec(3, 3,  "Alpha", "Delta",   hc=7, ac=2, hy=0, ay=2, hg=3, ag=1),
        _rec(4, 4,  "Bravo", "Charlie", hc=3, ac=6, hy=2, ay=0, hg=0, ag=2),
        _rec(5, 5,  "Charlie", "Alpha", hc=4, ac=9, hy=1, ay=1, hg=1, ag=2),
        _rec(6, 6,  "Delta", "Bravo",   hc=2, ac=5, hy=3, ay=1, hg=1, ag=1),
        _rec(7, 7,  "Alpha", "Bravo",   hc=6, ac=4, hy=1, ay=2, hg=2, ag=2),
        _rec(8, 8,  "Charlie", "Delta",  hc=5, ac=5, hy=0, ay=0, hg=0, ag=0),
        # jogo "WC2026" jogado (has_stats=True): alvo de inferência e linha de treino
        _rec(9, 30, "Alpha", "Bravo",   hc=5, ac=7, hy=2, ay=3, hg=1, ag=2,
             hxg=1.5, axg=2.1, round_name="Final", tournament_id=16, season_year=2026),
    ]
    return sorted(corpus, key=lambda g: (g["start_timestamp"], g["event_id"]))


@pytest.fixture
def patched_ab(monkeypatch):
    """Injeta o corpus sintético + rank_lookup controlado nos builders."""
    corpus = _corpus_ab()
    lookup = _ranks(["Alpha", "Bravo", "Charlie", "Delta"])
    monkeypatch.setattr(counts, "_load_corpus", lambda: list(corpus))
    monkeypatch.setattr(tier1, "_build_rank_lookup", lambda wc2026_only=False: dict(lookup))
    return corpus


# ---- 1. PARIDADE DE COLUNAS -------------------------------------------------

def test_column_parity_base(patched_ab):
    """Treino e inferência produzem EXATAMENTE as mesmas colunas, na mesma ordem."""
    train = build_count_training_dataset()
    infer = build_count_inference_dataset(patched_ab)

    expected = META_COLS + FEATURE_COLS_COUNTS + TARGET_COLS
    assert list(train.columns) == expected, "colunas do TREINO divergem do canônico"
    assert list(infer.columns) == expected, "colunas da INFERÊNCIA divergem do canônico"
    assert list(train.columns) == list(infer.columns), "treino ≠ inferência (colunas/ordem)"
    assert _feat_cols(train) == FEATURE_COLS_COUNTS
    assert _feat_cols(infer) == FEATURE_COLS_COUNTS


def test_column_parity_with_xg(patched_ab):
    """Com xG ligado, o bloco FEATURE_COLS_COUNTS_XG entra igual nos dois modos."""
    train = build_count_training_dataset(include_xg=True)
    infer = build_count_inference_dataset(patched_ab, include_xg=True)

    expected = META_COLS + FEATURE_COLS_COUNTS + FEATURE_COLS_COUNTS_XG + TARGET_COLS
    assert list(train.columns) == expected, "colunas do TREINO (xG) divergem"
    assert list(infer.columns) == expected, "colunas da INFERÊNCIA (xG) divergem"
    for c in FEATURE_COLS_COUNTS_XG:
        assert c in train.columns and c in infer.columns


# ---- 2. PARIDADE DE VALORES -------------------------------------------------

def test_value_parity_same_match(patched_ab):
    """O MESMO jogo featurizado no treino e na inferência é numericamente idêntico."""
    train = build_count_training_dataset(include_xg=True)
    infer = build_count_inference_dataset(patched_ab, include_xg=True)

    wc_id = 9  # o jogo tid=16/2026, presente nos dois datasets
    trow = train[train["event_id"] == wc_id]
    irow = infer[infer["event_id"] == wc_id]
    assert len(trow) == 1 and len(irow) == 1, "jogo-alvo ausente em algum dos modos"
    trow, irow = trow.iloc[0], irow.iloc[0]

    cols = FEATURE_COLS_COUNTS + FEATURE_COLS_COUNTS_XG
    diffs = [(c, trow[c], irow[c]) for c in cols if not _eq(trow[c], irow[c])]
    assert not diffs, f"features divergem treino×inferência (função deveria ser pura): {diffs}"


# ---- 3. CORTE TEMPORAL ESTRITO ('<', nunca '<=') ---------------------------

def test_strict_temporal_cutoff():
    """Jogo com timestamp IGUAL ao do alvo NÃO entra na janela; posterior tampouco."""
    lookup = _ranks(["Alpha", "Bravo", "Charlie"])
    # 3 jogos de Alpha ESTRITAMENTE antes; 1 no MESMO ts do alvo; 1 depois — os dois
    # últimos "envenenados" (corners=99) para que qualquer vazamento seja detectável.
    corpus = [
        _rec(1, 10, "Alpha", "Charlie", hc=4, ac=4),
        _rec(2, 11, "Alpha", "Charlie", hc=4, ac=4),
        _rec(3, 12, "Alpha", "Charlie", hc=4, ac=4),
        _rec(4, 20, "Alpha", "Charlie", hc=99, ac=99),  # MESMO ts do alvo
        _rec(5, 21, "Alpha", "Charlie", hc=99, ac=99),  # depois do alvo
    ]
    target = _rec(99, 20, "Alpha", "Bravo", hc=0, ac=0)  # ts == jogo #4
    feat = build_count_features(target, corpus + [target], lookup)

    # A = Alpha (melhor ranking). Só os 3 jogos '<' entram → n_games == 3 (não 4/5).
    assert feat["team_A"] == "Alpha"
    assert feat["A_n_games"] == 3, (
        f"n_games={feat['A_n_games']} — corte deveria ser '<' (3), não '<=' (4) nem incluir posterior"
    )
    # Média não pode ter sido inflada pelos jogos envenenados (corners=99).
    assert feat["A_corners_for"] < 10, (
        f"A_corners_for={feat['A_corners_for']} — jogo de ts igual/posterior vazou para a janela"
    )


# ---- 4. ORIENTAÇÃO A/B (A = melhor ranking FIFA) ---------------------------

def test_ab_orientation_follows_ranking():
    """A = melhor ranking; inverter o ranking inverte os lados coerentemente."""
    corpus = [
        _rec(1, 1, "Alpha", "Charlie", hc=8, ac=1),   # Alpha: escanteios altos
        _rec(2, 2, "Alpha", "Charlie", hc=8, ac=1),
        _rec(3, 3, "Bravo", "Charlie", hc=2, ac=7),   # Bravo: escanteios baixos
        _rec(4, 4, "Bravo", "Charlie", hc=2, ac=7),
    ]
    target = _rec(99, 10, "Alpha", "Bravo", hc=0, ac=0)

    f1 = build_count_features(target, corpus + [target], _ranks(["Alpha", "Bravo", "Charlie"]))
    f2 = build_count_features(target, corpus + [target], _ranks(["Bravo", "Alpha", "Charlie"]))

    assert f1["team_A"] == "Alpha" and f1["team_B"] == "Bravo"
    assert f2["team_A"] == "Bravo" and f2["team_B"] == "Alpha"
    # Inverter ranking troca os lados: o A do caso 2 == o B do caso 1 (e vice-versa).
    assert _eq(f2["A_corners_for"], f1["B_corners_for"])
    assert _eq(f2["B_corners_for"], f1["A_corners_for"])
    assert f1["A_fifa_rank"] == 1 and f2["A_fifa_rank"] == 1  # A é sempre o rank 1
    # delta_rank = rank_A - rank_B, e A é SEMPRE o de melhor ranking → delta_rank <= 0.
    # É INVARIANTE sob relabeling (permanece -1, não inverte de sinal), pois os rótulos
    # A/B são definidos pelo ranking, não pelo time. Mesma convenção do Tier 1: a feature
    # codifica a MAGNITUDE do gap, não a direção (essa vem do próprio A/B).
    assert f1["delta_rank"] == f2["delta_rank"] == -1
    assert f1["delta_rank"] <= 0


# ---- 5. n_games: estreante=0, cap em N=10 ----------------------------------

def test_n_games_reflects_window():
    """n_games = nº real de jogos anteriores; 0 para estreante, cap em N_WINDOW."""
    # Alpha tem 12 jogos anteriores; Bravo é estreante (nenhum jogo prévio).
    corpus = [_rec(i, i, "Alpha", "Charlie", hc=5, ac=5) for i in range(1, 13)]
    target = _rec(99, 20, "Alpha", "Bravo", hc=0, ac=0)
    feat = build_count_features(target, corpus + [target], _ranks(["Alpha", "Bravo", "Charlie"]))

    assert feat["team_A"] == "Alpha"
    assert feat["A_n_games"] == counts.N_WINDOW == 10, (
        f"A_n_games={feat['A_n_games']} — deveria capar em {counts.N_WINDOW}"
    )
    assert feat["B_n_games"] == 0, f"Bravo é estreante — B_n_games deveria ser 0"


def test_debut_both_teams_zero():
    """Primeiro jogo do corpus: ambos os times estreantes → n_games 0/0, sem vazamento."""
    target = _rec(1, 5, "Alpha", "Bravo", hc=3, ac=4)
    feat = build_count_features(target, [target], _ranks(["Alpha", "Bravo"]))
    assert feat["A_n_games"] == 0 and feat["B_n_games"] == 0


# ===========================================================================
# CAMADA B — smoke test sobre o cache REAL (skip se ausente)
# ===========================================================================

_CACHE_DIR = Path(__file__).parents[1] / "data" / "raw" / "sofascore"
_HAS_CACHE = _CACHE_DIR.exists() and any(_CACHE_DIR.glob("hist_*.jsonl"))
_SKIP_REASON = (
    f"cache SofaScore ausente em {_CACHE_DIR} (data/raw/ não é git-tracked) — "
    "smoke test da Camada B pulado"
)

# Colunas de feature legitimamente NULLABLE:
#  - days_rest/delta_rest: NaN quando o time é estreante no corpus (sem jogo anterior);
#  - xG: Tier 2, ausente em WC2018/AFCON.
_NULLABLE_FEATURES = {"A_days_rest", "B_days_rest", "delta_rest"}
_NONNULLABLE_FEATURES = [c for c in FEATURE_COLS_COUNTS if c not in _NULLABLE_FEATURES]


@pytest.fixture(scope="module")
def real_train():
    return build_count_training_dataset()


@pytest.mark.skipif(not _HAS_CACHE, reason=_SKIP_REASON)
def test_real_training_shape(real_train):
    """Reproduz o Passo 3: 394 jogos × 41 colunas (10 meta + 25 feat + 6 target)."""
    assert list(real_train.columns) == META_COLS + FEATURE_COLS_COUNTS + TARGET_COLS
    assert real_train.shape == (394, 41), f"shape inesperado: {real_train.shape} (esperado 394×41)"


@pytest.mark.skipif(not _HAS_CACHE, reason=_SKIP_REASON)
def test_real_inference_columns_match_training(real_train):
    """Inferência (com stub WC2026) produz EXATAMENTE as colunas do treino."""
    corpus = counts._load_corpus()
    last_ts = max(g["start_timestamp"] for g in corpus)
    stub = {
        "tournament_id": 16, "season_year": 2026, "tournament": "FIFA World Cup",
        "season": "World Cup 2026", "event_id": 999_000_001,
        "start_timestamp": last_ts + _DAY * 365, "date": "2026-06-15",
        "home_team": "Argentina", "away_team": "Spain", "round_name": None,
        "has_stats": False,
    }
    infer = build_count_inference_dataset(corpus + [stub])
    assert len(infer) >= 1, "inferência não featurizou o stub WC2026"
    assert list(infer.columns) == list(real_train.columns)
    assert infer[TARGET_COLS].isna().all().all(), "targets de inferência deveriam ser vazios"


@pytest.mark.skipif(not _HAS_CACHE, reason=_SKIP_REASON)
def test_real_no_nan_in_nonnullable_features(real_train):
    """Zero NaN nas features não-nullable; days_rest/delta_rest e xG são Tier2/nullable."""
    nan_counts = {c: int(real_train[c].isna().sum()) for c in _NONNULLABLE_FEATURES}
    offenders = {c: n for c, n in nan_counts.items() if n > 0}
    assert not offenders, f"NaN em colunas não-nullable: {offenders}"


@pytest.mark.skipif(not _HAS_CACHE, reason=_SKIP_REASON)
def test_real_rest_nan_iff_debut(real_train):
    """Invariante: days_rest é NaN EXATAMENTE quando o time é estreante (n_games==0)."""
    assert (real_train["A_days_rest"].isna() == (real_train["A_n_games"] == 0)).all(), (
        "A_days_rest NaN não coincide com A_n_games==0"
    )
    assert (real_train["B_days_rest"].isna() == (real_train["B_n_games"] == 0)).all(), (
        "B_days_rest NaN não coincide com B_n_games==0"
    )
