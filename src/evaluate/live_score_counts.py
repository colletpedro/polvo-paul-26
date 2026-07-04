"""Scoring ao vivo de ESCANTEIOS (Modo 2 shadow) — análogo a live_score.py do Modo 1.

Fluxo:
  1. Lê counts_predictions_log.csv (gerado por predict_counts.py).
  2. Preenche actual_corners_total dos jogos FINISHED via cache do sofascore --live
     (home_corners + away_corners — JOGO INTEIRO, incl. prorrogação; régua §4.3, NÃO 90min).
  3. Se uma célula JÁ preenchida divergir de nova leitura da fonte, RECONCILIA e loga
     (match, antigo → novo) — nunca corrige silenciosamente (mesma disciplina do Modo 1).
  4. Calcula log-loss over/under (linhas 8.5/9.5/10.5/11.5) do MODELO (probas congeladas no
     log) vs BASELINE ingênuo (Poisson de 2× média expansiva do torneio — counts_model.py).
  5. Reporta SEPARADO por prediction_type (retro vs prereg). NENHUMA métrica mistura os dois.

Sem rede: só lê o cache local do sofascore --live e o log. (A coleta é feita à parte.)

Uso:
    .venv/bin/python src/evaluate/live_score_counts.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.models import counts_model as cm
from src.predict_counts import _LINES, _LOG_COLS, _LOG_PATH, _line_col, load_live_corpus

_KMAX = 80


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _load_log() -> list[dict]:
    if not _LOG_PATH.exists():
        return []
    with open(_LOG_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_log(rows: list[dict]) -> None:
    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLS)
        w.writeheader()
        w.writerows(rows)


def _ll(p: float, y: int) -> float:
    p = min(max(p, 1e-15), 1 - 1e-15)
    return -(math.log(p) if y else math.log(1 - p))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def run() -> None:
    print("Carregando cache vivo (sofascore --live) + corpus...")
    corpus, live = load_live_corpus()
    # índice de escanteios reais por event_id (JOGO INTEIRO)
    actual_by_id: dict[int, int] = {}
    ts_by_id: dict[int, int] = {}
    for _, r in live.iterrows():
        if r["has_stats"]:
            actual_by_id[int(r["event_id"])] = int(r["home_corners"]) + int(r["away_corners"])
            ts_by_id[int(r["event_id"])] = int(r["start_timestamp"])

    rows = _load_log()
    if not rows:
        print(f"Log vazio: {_LOG_PATH}")
        return

    # --- Preenche / reconcilia actual_corners_total ---
    reconciled: list[dict] = []
    for row in rows:
        eid = int(row["event_id"])
        actual = actual_by_id.get(eid)
        if actual is None:
            continue
        existing = (row.get("actual_corners_total") or "").strip()
        if not existing:
            row["actual_corners_total"] = str(actual)
        elif existing != str(actual):
            reconciled.append({"team_A": row["team_A"], "team_B": row["team_B"],
                               "old": existing, "new": str(actual)})
            row["actual_corners_total"] = str(actual)
    _save_log(rows)

    if reconciled:
        print(f"\n⚠ RECONCILIAÇÃO — {len(reconciled)} célula(s) corrigida(s) "
              f"(dado da fonte atualizado):")
        for r in reconciled:
            print(f"  {r['team_A']} vs {r['team_B']}: {r['old']} → {r['new']} escanteios "
                  f"(motivo: cache do sofascore atualizado)")

    # --- Log-loss por prediction_type (retro / prereg), separado ---
    print()
    for ptype in ("retro", "prereg"):
        scored = [r for r in rows if r.get("prediction_type") == ptype
                  and (r.get("actual_corners_total") or "").strip() != ""]
        print("=" * 70)
        print(f"[{ptype.upper()}] jogos pontuados: {len(scored)}")
        print("=" * 70)
        if not scored:
            pend = [r for r in rows if r.get("prediction_type") == ptype]
            print(f"  (nenhum decidido ainda — {len(pend)} aguardando resultado)\n")
            continue

        # métricas por linha: modelo vs baseline
        agg = {L: {"model": [], "base": []} for L in _LINES}
        per_game = []
        for r in scored:
            eid = int(r["event_id"])
            actual = int(r["actual_corners_total"])
            # baseline: Poisson de 2× média expansiva do torneio no ts do jogo (leak-free)
            ts = ts_by_id.get(eid)
            mean_c = cm._expanding_means(corpus, ts)["corners"] if ts is not None else 0.0
            base_pmf = cm._poisson_total_pmf(2.0 * mean_c, _KMAX)
            row_ll_m, row_ll_b = [], []
            for L in _LINES:
                y = int(actual > L)
                p_model = float(r[_line_col(L)])
                p_base = cm._prob_over(base_pmf, L)
                ll_m = _ll(p_model, y); ll_b = _ll(p_base, y)
                agg[L]["model"].append(ll_m); agg[L]["base"].append(ll_b)
                row_ll_m.append(ll_m); row_ll_b.append(ll_b)
            per_game.append((r, actual, sum(row_ll_m) / len(_LINES), sum(row_ll_b) / len(_LINES)))

        # tabela por jogo
        print(f"  {'jogo':<34} {'E[tot]':>7} {'real':>5} {'LL mod':>7} {'LL base':>8}")
        print("  " + "─" * 66)
        for r, actual, llm, llb in per_game:
            jogo = f"{r['team_A'][:15]}×{r['team_B'][:15]}"
            print(f"  {jogo:<34} {float(r['expected_total']):>7.1f} {actual:>5} "
                  f"{llm:>7.4f} {llb:>8.4f}")

        # agregado por linha
        print(f"\n  Log-loss médio por linha (modelo vs baseline) — {ptype}:")
        print(f"  {'linha':>7} {'modelo':>9} {'baseline':>9} {'Δ':>9}")
        tot_m, tot_b = [], []
        for L in _LINES:
            m = sum(agg[L]["model"]) / len(agg[L]["model"])
            b = sum(agg[L]["base"]) / len(agg[L]["base"])
            tot_m.append(m); tot_b.append(b)
            print(f"  {L:>7} {m:>9.4f} {b:>9.4f} {m - b:>+9.4f}")
        mm = sum(tot_m) / len(tot_m); bb = sum(tot_b) / len(tot_b)
        verdict = "modelo MELHOR" if mm < bb else ("empate" if abs(mm - bb) < 1e-4 else "modelo PIOR")
        print(f"  {'MÉDIA':>7} {mm:>9.4f} {bb:>9.4f} {mm - bb:>+9.4f}  → {verdict}")
        print()

    print(f"Log actualizado: {_LOG_PATH}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
