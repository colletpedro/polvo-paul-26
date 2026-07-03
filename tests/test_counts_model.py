"""Higiene do passo 5 — split leak-free, distribuições válidas, convolução NegBin vs Monte Carlo.

Não valida performance (isso é a avaliação); valida CORREÇÃO das peças (SPEC_MARKETS §5/§7).
Sem rede. Camada sobre cache real é guardada por skipif.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import counts_model as cm

_CACHE = Path(__file__).parents[1] / "data" / "raw" / "sofascore"
_HAS_CACHE = _CACHE.exists() and any(_CACHE.glob("hist_*.jsonl"))
_SKIP = pytest.mark.skipif(not _HAS_CACHE, reason="cache SofaScore ausente (data/raw/ não git-tracked)")


# ---------------------------------------------------------------------------
# Distribuições: PMF válida (soma ≈ 1 no suporte truncado)
# ---------------------------------------------------------------------------

def test_poisson_total_pmf_sums_to_one():
    pmf = cm._poisson_total_pmf(9.5, 60)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert (pmf >= 0).all()


def test_negbin_total_pmf_sums_to_one():
    pmf = cm._negbin_total_pmf(2.1, 3.4, r=6.0)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert (pmf >= 0).all()


def test_negbin_pmf_reduces_to_poisson_when_r_infinite():
    """r=inf ⇒ NegBin degenera para Poisson (sem sobredispersão)."""
    nb = cm._nb_pmf(3.0, float("inf"), 40)
    from scipy.stats import poisson
    assert np.allclose(nb, poisson.pmf(np.arange(41), 3.0), atol=1e-12)


# ---------------------------------------------------------------------------
# Convolução NegBin vs Monte Carlo (caso fixo, tolerância 1e-3)
# ---------------------------------------------------------------------------

def test_negbin_convolution_matches_monte_carlo():
    """PMF do total por convolução ≈ simulação MC de NB(mu_a,r)+NB(mu_b,r)."""
    mu_a, mu_b, r = 2.3, 3.1, 5.0
    pmf = cm._negbin_total_pmf(mu_a, mu_b, r)
    cdf = np.cumsum(pmf)

    from scipy.stats import nbinom
    rng = np.random.default_rng(cm._MC_SEED)
    n = 2_000_000
    pa = r / (r + mu_a)
    pb = r / (r + mu_b)
    sample = nbinom.rvs(r, pa, size=n, random_state=rng) + nbinom.rvs(r, pb, size=n, random_state=rng)

    # compara CDF em vários pontos
    max_err = 0.0
    for k in range(0, 25):
        emp = float((sample <= k).mean())
        pred = float(cdf[k]) if k < len(cdf) else 1.0
        max_err = max(max_err, abs(emp - pred))
    assert max_err < 1e-3, f"convolução diverge da MC: erro máx CDF = {max_err:.2e}"


# ---------------------------------------------------------------------------
# Métricas: sanidade
# ---------------------------------------------------------------------------

def test_prob_over_and_cdf_consistency():
    pmf = cm._poisson_total_pmf(10.0, 80)
    # P(>8.5) = 1 - F(8)
    assert abs(cm._prob_over(pmf, 8.5) - (1 - cm._cdf_at(pmf, 8))) < 1e-12
    # linha maior ⇒ menor P(over)
    assert cm._prob_over(pmf, 10.5) < cm._prob_over(pmf, 8.5)


def test_rps_zero_when_point_mass_on_actual():
    """RPS = 0 quando toda a massa está no valor observado."""
    pmf = np.zeros(30); pmf[7] = 1.0
    assert cm._rps_trunc(pmf, 7) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Split temporal: leak-free (camada real)
# ---------------------------------------------------------------------------

@_SKIP
def test_split_no_leak_and_complete():
    """Split: interseção vazia por event_id, união = todas as 394 linhas, corte por data respeitado."""
    from src.features.counts import build_count_training_dataset
    import pandas as pd
    df = build_count_training_dataset()
    train, val = cm.get_temporal_split(df)

    # completude + partição
    assert len(train) + len(val) == len(df) == 394
    assert set(train["event_id"]).isdisjoint(set(val["event_id"]))
    assert set(train["event_id"]) | set(val["event_id"]) == set(df["event_id"])

    # corte temporal: todo treino < 2023-01-01 ≤ toda validação
    cut = pd.Timestamp(cm._SPLIT_CUTOFF)
    assert (pd.to_datetime(train["date"]) < cut).all()
    assert (pd.to_datetime(val["date"]) >= cut).all()


@_SKIP
def test_split_edition_assignment():
    """Edições nos conjuntos corretos (§7): AFCON2023 → validação; WC2022 → treino."""
    from src.features.counts import build_count_training_dataset
    df = build_count_training_dataset()
    train, val = cm.get_temporal_split(df)
    train_seasons = set(train["season"])
    val_seasons = set(val["season"])
    assert "World Cup 2022" in train_seasons
    assert "Africa Cup of Nations 2023" in val_seasons
    assert val_seasons == {"Africa Cup of Nations 2023", "EURO 2024", "Copa America 2024"}
