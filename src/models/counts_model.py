"""Modelos de contagem do Modo 2 — baseline (§5.5) + v1 (§5.6) + avaliação (§6). §10 passo 5.

Autoridade: SPEC_MARKETS.md. Nada aqui inventa parâmetro — a forma do v1 é a PINADA em §5.6.

Componentes:
  · get_temporal_split(df)  — split temporal ÚNICO (§7, corte 2023-01-01). Usado por baseline E v1.
  · baseline               — Poisson de taxa média do TREINO, por alvo (§5.5). Poisson p/ os 3 alvos
                             (inclusive cartões — baseline ingênuo único).
  · v1                     — λ_alvo(time) = ataque_shrunk(time) × defesa_shrunk(oponente) /
                             média_expansiva (§5.6). Gols/escanteios: Poisson. Cartões: NegBin com
                             r GLOBAL (momentos, no treino); TOTAL do jogo por convolução truncada.
  · avaliação              — log-loss over/under CONTRA REALIZADO por alvo × torneio de validação,
                             RPS trunc@20, PIT (§5.4/§6).

Sem rede. Não toca em counts.py/coletor/Modo 1 (só importa, read-only).

Uso:
  .venv/bin/python src/models/counts_model.py            # roda avaliação completa
  .venv/bin/python src/models/counts_model.py --skip-verify   # pula manifest --verify
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.features import counts

# ---------------------------------------------------------------------------
# Configuração — alvos, colunas, linhas de mercado (§6), split (§7)
# ---------------------------------------------------------------------------
_SPLIT_CUTOFF = "2023-01-01"   # §7: treino < cutoff ≤ validação

# alvo → (col ataque=for, col defesa=against, col target, chave da média expansiva)
_TARGETS = {
    "gols":       {"for": "goals_for",   "against": "goals_against",
                   "tgt": "goals",   "mean_key": "goals",   "dist": "poisson"},
    "escanteios": {"for": "corners_for", "against": "corners_against",
                   "tgt": "corners", "mean_key": "corners", "dist": "poisson"},
    "cartoes":    {"for": "cards_for",   "against": "cards_against",
                   "tgt": "cards",   "mean_key": "cards",   "dist": "negbin"},
}

# Linhas de over/under padrão (§6). alvo → lista de linhas.
_LINES = {
    "gols":       [2.5],
    "escanteios": [8.5, 9.5, 10.5],
    "cartoes":    [3.5, 4.5],
}

_RPS_TRUNC = 20          # §5.6 diagnóstico
_TAIL_EPS = 1e-6         # truncamento de cauda da convolução
_LL_EPS = 1e-12          # clip de probabilidade no log-loss
_MC_SEED = 20260703      # semente fixa (determinístico; sem Date.now/rand livre)


# ---------------------------------------------------------------------------
# SPLIT TEMPORAL — função ÚNICA (§7)
# ---------------------------------------------------------------------------

def get_temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal do §7: treino = jogos com date < 2023-01-01; validação = o resto.

    Corte único por data (leak-free). Retorna (train, val) — cópias, índices resetados.
    """
    date = pd.to_datetime(df["date"])
    cut = pd.Timestamp(_SPLIT_CUTOFF)
    train = df[date < cut].reset_index(drop=True).copy()
    val = df[date >= cut].reset_index(drop=True).copy()
    return train, val


# ---------------------------------------------------------------------------
# Média expansiva por-jogo (§5.6) — reusa counts._global_prior_means (leak-free)
# ---------------------------------------------------------------------------

def _expanding_means(corpus: list[dict], start_ts: int) -> dict[str, float]:
    """Média global expansiva por métrica sobre jogos com start_timestamp < start_ts.

    Mesma função que alimenta o prior do shrinkage em counts.py. Prior vazio ⇒ 0.0
    (convenção de counts.py). Retorna {'corners','cards','goals'}.
    """
    prior = [g for g in corpus if g["start_timestamp"] < start_ts and g.get("has_stats")]
    return counts._global_prior_means(prior)


# ---------------------------------------------------------------------------
# BASELINE (§5.5) — Poisson de taxa média do TREINO, por alvo (inclusive cartões)
# ---------------------------------------------------------------------------

def fit_baseline(train: pd.DataFrame) -> dict[str, float]:
    """λ por-time = média das contagens por-time no TREINO, por alvo. Um escalar por alvo."""
    lam = {}
    for alvo, c in _TARGETS.items():
        obs = pd.concat([train[f"target_{c['tgt']}_A"], train[f"target_{c['tgt']}_B"]])
        lam[alvo] = float(obs.mean())
    return lam


def baseline_total_lambda(lam: dict[str, float], alvo: str) -> float:
    """Total do jogo no baseline = 2 × λ por-time (ambos os times na média histórica)."""
    return 2.0 * lam[alvo]


# ---------------------------------------------------------------------------
# NegBin — dispersão r GLOBAL por momentos, no TREINO (§5.6)
# ---------------------------------------------------------------------------

def fit_negbin_r(train: pd.DataFrame) -> float:
    """r global (dispersão do NegBin de cartões) por MÉTODO DOS MOMENTOS no treino.

    Momentos: closed-form, sem otimizador — coerente com o 'zero parâmetro otimizado' do v1.
    Var = μ + μ²/r ⇒ r = μ²/(Var−μ). Se Var ≤ μ (não-sobredisperso) ⇒ r→∞ (degenera p/ Poisson).
    """
    obs = pd.concat([train["target_cards_A"], train["target_cards_B"]]).to_numpy(float)
    m = obs.mean()
    v = obs.var(ddof=1)
    if v <= m:
        return float("inf")  # sem sobredispersão → Poisson
    return float(m * m / (v - m))


def _nb_pmf(mu: float, r: float, kmax: int) -> np.ndarray:
    """PMF de NegBin(média=mu, dispersão=r) em 0..kmax. r=inf ⇒ Poisson(mu)."""
    k = np.arange(kmax + 1)
    if not np.isfinite(r):
        return poisson.pmf(k, mu)
    p = r / (r + mu)            # scipy: n=r (size), p; média = n(1−p)/p = mu
    return nbinom.pmf(k, r, p)


# ---------------------------------------------------------------------------
# Distribuição do TOTAL do jogo
# ---------------------------------------------------------------------------

def _poisson_total_pmf(lam_total: float, kmax: int) -> np.ndarray:
    return poisson.pmf(np.arange(kmax + 1), lam_total)


def _negbin_total_pmf(mu_a: float, mu_b: float, r: float,
                      tail_eps: float = _TAIL_EPS) -> np.ndarray:
    """PMF do TOTAL = NegBin(mu_a,r) ⊕ NegBin(mu_b,r) por CONVOLUÇÃO numérica truncada.

    A soma de dois NegBin com o MESMO r mas médias distintas NÃO é NegBin fechada — por isso
    convolui-se as PMFs. Trunca cada componente onde a cauda acumulada < tail_eps.
    """
    def _kmax_for(mu):
        # estende até a cauda superior ficar < tail_eps
        k = max(20, int(mu * 3) + 20)
        while True:
            p = _nb_pmf(mu, r, k)
            if 1.0 - p.sum() < tail_eps:
                return k
            k *= 2
    ka, kb = _kmax_for(mu_a), _kmax_for(mu_b)
    pa, pb = _nb_pmf(mu_a, r, ka), _nb_pmf(mu_b, r, kb)
    total = np.convolve(pa, pb)
    return total / total.sum()   # renormaliza o resíduo de truncamento


# ---------------------------------------------------------------------------
# v1 (§5.6) — λ multiplicativo por time a partir das features de counts.py
# ---------------------------------------------------------------------------

def v1_team_lambdas(row: pd.Series, alvo: str, mean_alvo: float) -> tuple[float, float] | None:
    """(λ_A, λ_B) do v1 para um jogo: ataque_shrunk × defesa_shrunk(oponente) / média.

    Divisor 0 (prior vazio) ⇒ None (sem previsão — nunca fabricar valor, §5.6/§7).
    """
    if mean_alvo <= 0:
        return None
    c = _TARGETS[alvo]
    atk_A, def_A = row[f"A_{c['for']}"], row[f"A_{c['against']}"]
    atk_B, def_B = row[f"B_{c['for']}"], row[f"B_{c['against']}"]
    lam_A = atk_A * def_B / mean_alvo
    lam_B = atk_B * def_A / mean_alvo
    return float(lam_A), float(lam_B)


# ---------------------------------------------------------------------------
# Métricas (§6)
# ---------------------------------------------------------------------------

def _cdf_at(pmf: np.ndarray, k: int) -> float:
    """CDF F(k)=P(X≤k) do vetor pmf (índice = valor)."""
    if k < 0:
        return 0.0
    return float(pmf[: k + 1].sum())


def _prob_over(pmf: np.ndarray, line: float) -> float:
    """P(total > line) para linha .5 (over = total ≥ ceil(line))."""
    return 1.0 - _cdf_at(pmf, int(np.floor(line)))


def _logloss(p: float, y: int) -> float:
    p = min(max(p, _LL_EPS), 1 - _LL_EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _rps_trunc(pmf: np.ndarray, actual: int, k_trunc: int = _RPS_TRUNC) -> float:
    """RPS truncado: Σ_{k=0}^{K} (F_pred(k) − 1[actual≤k])². Menor é melhor."""
    ks = np.arange(k_trunc + 1)
    fp = np.array([_cdf_at(pmf, int(k)) for k in ks])
    fo = (ks >= actual).astype(float)  # F_obs(k) = 1 se k≥actual
    return float(((fp - fo) ** 2).sum())


def _pit(pmf: np.ndarray, actual: int, rng: np.random.Generator) -> float:
    """PIT randomizado para contagem discreta: F(actual−1) + U·(F(actual)−F(actual−1))."""
    lo = _cdf_at(pmf, actual - 1)
    hi = _cdf_at(pmf, actual)
    return lo + rng.random() * (hi - lo)


# ---------------------------------------------------------------------------
# Pipeline de avaliação
# ---------------------------------------------------------------------------

def _pmf_for_row(model: str, row: pd.Series, alvo: str, lam_base: dict,
                 r_cards: float, corpus_by_ts, kmax_poisson: int = 60):
    """Retorna a PMF do TOTAL do jogo para (modelo, alvo). None se v1 sem previsão."""
    c = _TARGETS[alvo]
    if model == "baseline":
        # Baseline é Poisson de taxa média para os 3 alvos (inclusive cartões, §5.5).
        return _poisson_total_pmf(baseline_total_lambda(lam_base, alvo), kmax_poisson)
    # v1
    mean_alvo = corpus_by_ts(row["start_timestamp"])[c["mean_key"]]
    lams = v1_team_lambdas(row, alvo, mean_alvo)
    if lams is None:
        return None
    lam_A, lam_B = lams
    if alvo == "cartoes":
        return _negbin_total_pmf(lam_A, lam_B, r_cards)
    return _poisson_total_pmf(lam_A + lam_B, kmax_poisson)


def evaluate(skip_verify: bool = False) -> dict:
    """Roda split → baseline/v1 → métricas por alvo × torneio de validação. Retorna resultados."""
    if not skip_verify:
        _manifest_verify_or_die()

    df = counts.build_count_training_dataset()
    corpus = counts._load_corpus()
    # cache da média expansiva por timestamp (evita recomputar)
    _cache: dict[int, dict] = {}
    def corpus_by_ts(ts):
        if ts not in _cache:
            _cache[ts] = _expanding_means(corpus, ts)
        return _cache[ts]

    train, val = get_temporal_split(df)
    lam_base = fit_baseline(train)
    r_cards = fit_negbin_r(train)

    rng = np.random.default_rng(_MC_SEED)
    val = val.copy()
    val["_val_tourney"] = val["season"]

    rows_out = []
    pit_acc = {("v1", a): [] for a in _TARGETS}
    for _, row in val.iterrows():
        for alvo, c in _TARGETS.items():
            actual = int(row[f"target_{c['tgt']}_A"] + row[f"target_{c['tgt']}_B"])
            for model in ("baseline", "v1"):
                pmf = _pmf_for_row(model, row, alvo, lam_base, r_cards, corpus_by_ts)
                if pmf is None:
                    continue
                # log-loss médio sobre as linhas do alvo
                lls = []
                for line in _LINES[alvo]:
                    p_over = _prob_over(pmf, line)
                    y = int(actual > line)
                    lls.append(_logloss(p_over, y))
                rec = {
                    "tourney": row["season"], "alvo": alvo, "model": model,
                    "logloss": float(np.mean(lls)),
                    "rps": _rps_trunc(pmf, actual),
                    "actual": actual,
                }
                rows_out.append(rec)
                if model == "v1":
                    pit_acc[("v1", alvo)].append(_pit(pmf, actual, rng))
    res = pd.DataFrame(rows_out)
    return {"res": res, "val": val, "train": train, "lam_base": lam_base,
            "r_cards": r_cards, "pit": pit_acc}


def _manifest_verify_or_die() -> None:
    root = Path(__file__).parents[2]
    out = subprocess.run(
        [sys.executable, str(root / "src/collect/manifest.py"), "--verify"],
        capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout); print(out.stderr, file=sys.stderr)
        raise SystemExit("✗ manifest --verify FALHOU — cache mudou desde a auditoria. Abortado.")
    print("✓ manifest --verify OK (cache idêntico à auditoria)")


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def _report(out: dict) -> None:
    res, train = out["res"], out["train"]
    val = out["val"]
    print("\n" + "=" * 74)
    print("PASSO 5 — BASELINE (§5.5) vs v1 (§5.6) · validação temporal (§7)")
    print("=" * 74)
    print(f"  split: treino={len(train)} jogos  validação={len(val)} jogos")
    print(f"  torneios de validação: {sorted(val['season'].unique())}")
    print(f"  baseline λ/time (treino): " +
          "  ".join(f"{a}={v:.2f}" for a, v in out['lam_base'].items()))
    print(f"  NegBin r global (cartões, momentos/treino): {out['r_cards']:.2f}")

    tourneys = sorted(val["season"].unique())
    print("\n[LOG-LOSS over/under CONTRA REALIZADO — menor é melhor]")
    hdr = f"  {'alvo':<11} {'modelo':<9}" + "".join(f"{t[:12]:>14}" for t in tourneys) + f"{'AGREGADO':>12}"
    print(hdr); print("  " + "─" * (len(hdr) - 2))
    for alvo in _TARGETS:
        for model in ("baseline", "v1"):
            cells = []
            for t in tourneys:
                sub = res[(res.alvo == alvo) & (res.model == model) & (res.tourney == t)]
                cells.append(f"{sub.logloss.mean():>14.4f}" if len(sub) else f"{'—':>14}")
            agg = res[(res.alvo == alvo) & (res.model == model)]
            cells.append(f"{agg.logloss.mean():>12.4f}")
            print(f"  {alvo:<11} {model:<9}" + "".join(cells))
        # delta v1 − baseline (agregado)
        b = res[(res.alvo == alvo) & (res.model == "baseline")].logloss.mean()
        v = res[(res.alvo == alvo) & (res.model == "v1")].logloss.mean()
        verdict = "v1 SUPERA" if v < b else ("EMPATE" if abs(v - b) < 1e-4 else "v1 NÃO supera")
        print(f"  {'':<11} {'Δ(v1−base)':<9}{'':>{14*len(tourneys)}}{v-b:>+12.4f}  → {verdict}")
        print()

    print("[RPS truncado@20 — menor é melhor (agregado na validação)]")
    for alvo in _TARGETS:
        b = res[(res.alvo == alvo) & (res.model == "baseline")].rps.mean()
        v = res[(res.alvo == alvo) & (res.model == "v1")].rps.mean()
        print(f"  {alvo:<11} baseline={b:.3f}  v1={v:.3f}  Δ={v - b:+.3f}")

    print("\n[PIT do v1 — média≈0.50 e histograma 5-bins uniforme ⇒ calibrado (§5.4)]")
    for alvo in _TARGETS:
        vals = np.array(out["pit"][("v1", alvo)])
        if len(vals) == 0:
            print(f"  {alvo:<11} (sem previsões)"); continue
        hist, _ = np.histogram(vals, bins=5, range=(0, 1))
        print(f"  {alvo:<11} n={len(vals):>3}  média={vals.mean():.3f}  "
              f"std={vals.std():.3f}  hist(5)={hist.tolist()}")

    print("\n[LEITURA HONESTA]")
    for alvo in _TARGETS:
        b = res[(res.alvo == alvo) & (res.model == "baseline")].logloss.mean()
        v = res[(res.alvo == alvo) & (res.model == "v1")].logloss.mean()
        tag = "✓ v1 supera baseline" if v < b else "✗ v1 NÃO supera baseline"
        print(f"  {alvo:<11}: baseline={b:.4f}  v1={v:.4f}  {tag}")
    print("\n  NOTA (§6/§4.3): comparação SÓ contra realizado. Benchmark contra linha de casa de aposta")
    print("  fica PENDENTE — exige coleta de odds (não temos). Não improvisado nesta tarefa.")


def main() -> None:
    p = argparse.ArgumentParser(description="Passo 5 — baseline vs v1 de contagem (Modo 2).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Pula o manifest --verify pré-treino (default: verifica).")
    args = p.parse_args()
    out = evaluate(skip_verify=args.skip_verify)
    _report(out)


if __name__ == "__main__":
    main()
