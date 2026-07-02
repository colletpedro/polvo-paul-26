"""Pontuação ao vivo do modelo vs mercado — LAST_32 WC2026.

Fluxo:
  1. Lê predictions_log.csv (gerado por src/predict.py).
  2. Busca resultados FINISHED via football-data.org (cache por padrão; --force recarrega).
  3. Converte score API → result_90min (A_win / draw / B_win), respeitando:
       - EXTRA_TIME / PENALTY_SHOOTOUT → draw nos 90 min (foram iguais).
  4. Preenche coluna actual_result no CSV e grava de volta.
  5. Calcula log-loss individual e acumulado do modelo.
  6. Calcula log-loss do mercado (probas hardcoded de RESULTS.md §6.1) nos mesmos jogos.
  7. Imprime resumo: jogos decididos, jogos pendentes, modelo vs mercado.

CLI:
    python src/evaluate/live_score.py           # usa cache de inferência (24 h)
    python src/evaluate/live_score.py --force   # recarrega da API antes de pontuar

Probabilidades de mercado (RESULTS.md §6.1):
    Midpoint das faixas das odds 1X2 de Bet365 / Pinnacle, normalizadas para somar 1.
    Hardcoded por match_id; atualizar manualmente se as odds mudarem significativamente.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from src.collect.footballdata import fetch_matches

_ROOT     = Path(__file__).parents[2]
_LOG_PATH = _ROOT / "data" / "processed" / "predictions_log.csv"
_LOG_COLS = [
    "match_id", "timestamp_predicted", "stage",
    "team_A", "team_B",
    "P_A_win", "P_draw", "P_B_win",
    "predicted_class", "entropy",
    "actual_result",
]

# ---------------------------------------------------------------------------
# Probabilidades de mercado — midpoints das faixas em RESULTS.md §6.1
# Keys: match_id (int). P_A_win + P_draw + P_B_win = 1.0 (A = time melhor rankeado).
# Fonte: odds 1X2 médias (Bet365/Pinnacle) convertidas para probabilidade implícita,
# normalizadas para remover a margem da casa.
# ---------------------------------------------------------------------------
_MARKET_PROBS: dict[int, dict[str, float]] = {
    537418: {"P_A_win": 0.31, "P_draw": 0.28, "P_B_win": 0.41},  # Morocco(A) vs Netherlands(B)
    537424: {"P_A_win": 0.34, "P_draw": 0.27, "P_B_win": 0.39},  # Norway(A) vs Ivory Coast(B)
    537416: {"P_A_win": 0.73, "P_draw": 0.17, "P_B_win": 0.10},  # France(A) vs Sweden(B)
    537425: {"P_A_win": 0.53, "P_draw": 0.26, "P_B_win": 0.21},  # Mexico(A) vs Ecuador(B)
    537426: {"P_A_win": 0.73, "P_draw": 0.17, "P_B_win": 0.10},  # England(A) vs Congo DR(B)
    537422: {"P_A_win": 0.57, "P_draw": 0.24, "P_B_win": 0.19},  # Belgium(A) vs Senegal(B)
    537421: {"P_A_win": 0.57, "P_draw": 0.25, "P_B_win": 0.18},  # United States(A) vs Bosnia(B)
    537420: {"P_A_win": 0.67, "P_draw": 0.21, "P_B_win": 0.12},  # Spain(A) vs Austria(B)
    537419: {"P_A_win": 0.57, "P_draw": 0.24, "P_B_win": 0.19},  # Portugal(A) vs Croatia(B)
    537429: {"P_A_win": 0.59, "P_draw": 0.23, "P_B_win": 0.18},  # Switzerland(A) vs Algeria(B)
    537428: {"P_A_win": 0.31, "P_draw": 0.27, "P_B_win": 0.42},  # Australia(A) vs Egypt(B)
    537427: {"P_A_win": 0.83, "P_draw": 0.11, "P_B_win": 0.06},  # Argentina(A) vs Cape Verde Is.(B)
    537430: {"P_A_win": 0.60, "P_draw": 0.24, "P_B_win": 0.16},  # Colombia(A) vs Ghana(B)
}


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _load_log() -> list[dict[str, str]]:
    if not _LOG_PATH.exists():
        return []
    with open(_LOG_PATH, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_log(rows: list[dict[str, str]]) -> None:
    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_COLS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Resolução de resultado: API → result_90min
# ---------------------------------------------------------------------------

def _raw_result_from_api(match: dict) -> str | None:
    """Extrai resultado nos 90 min a partir de um dict de jogo da API.

    Retorna "HOME_WIN", "AWAY_WIN", "draw" ou None se o jogo não está encerrado.
    Para EXTRA_TIME / PENALTY_SHOOTOUT: resultado nos 90 min é sempre empate.
    """
    if match.get("status") != "FINISHED":
        return None
    score    = match.get("score", {})
    duration = score.get("duration", "REGULAR")
    winner   = score.get("winner")  # "HOME_TEAM" | "AWAY_TEAM" | "DRAW" | null

    if duration in ("EXTRA_TIME", "PENALTY_SHOOTOUT"):
        return "draw"

    if winner == "DRAW":
        return "draw"
    if winner == "HOME_TEAM":
        return "HOME_WIN"
    if winner == "AWAY_TEAM":
        return "AWAY_WIN"
    return None


def _resolve_to_ab(raw: str, team_a: str, home_name: str, away_name: str) -> str | None:
    """Converte HOME_WIN/AWAY_WIN/draw → A_win/B_win/draw usando quem é o home team."""
    if raw == "draw":
        return "draw"

    # Comparação case-insensitive com strip para robustez
    t_a  = team_a.strip().lower()
    home = home_name.strip().lower()
    away = away_name.strip().lower()

    if t_a == home:
        a_is_home = True
    elif t_a == away:
        a_is_home = False
    else:
        return None  # nome não encontrado — avisar no chamador

    if raw == "HOME_WIN":
        return "A_win" if a_is_home else "B_win"
    if raw == "AWAY_WIN":
        return "B_win" if a_is_home else "A_win"
    return None


# ---------------------------------------------------------------------------
# Índice de jogos a partir da API
# ---------------------------------------------------------------------------

def _build_match_index(matches: list[dict]) -> dict[int, dict[str, Any]]:
    """Retorna {match_id → {status, home_name, away_name, result_90min_raw}}."""
    idx: dict[int, dict[str, Any]] = {}
    for m in matches:
        home = m["homeTeam"]
        away = m["awayTeam"]
        home_name = home["name"] if isinstance(home, dict) else str(home)
        away_name = away["name"] if isinstance(away, dict) else str(away)
        idx[int(m["id"])] = {
            "status":           m.get("status", ""),
            "home_name":        home_name,
            "away_name":        away_name,
            "result_90min_raw": _raw_result_from_api(m),
        }
    return idx


# ---------------------------------------------------------------------------
# Log-loss
# ---------------------------------------------------------------------------

def _ll(p: float) -> float:
    return -math.log(max(p, 1e-15))


# ---------------------------------------------------------------------------
# Núcleo de pontuação
# ---------------------------------------------------------------------------

def score_log(
    rows: list[dict[str, str]],
    match_index: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """Preenche actual_result nas linhas do log e retorna (rows_atualizadas, scored).

    scored: lista de dicts, um por match_id único já FINISHED, com:
      match_id, team_A, team_B, actual_result,
      P_A_win, P_draw, P_B_win (modelo),
      ll_model, ll_market (mercado), predicted_class, correct.
    """
    # Deduplica: para cada match_id, pega a primeira linha (mesmo modelo).
    seen: dict[int, dict[str, str]] = {}
    for row in rows:
        mid = int(row["match_id"])
        if mid not in seen:
            seen[mid] = row

    scored: list[dict[str, Any]] = []

    for mid, pred_row in seen.items():
        api = match_index.get(mid)
        if api is None or api["result_90min_raw"] is None:
            continue  # jogo não encerrado ou match_id não reconhecido

        team_a = pred_row["team_A"]
        result = _resolve_to_ab(
            api["result_90min_raw"], team_a,
            api["home_name"], api["away_name"],
        )
        if result is None:
            print(
                f"  AVISO: não foi possível mapear resultado para match_id={mid} "
                f"({team_a} — home={api['home_name']} / away={api['away_name']})",
                file=sys.stderr,
            )
            continue

        p_a  = float(pred_row["P_A_win"])
        p_d  = float(pred_row["P_draw"])
        p_b  = float(pred_row["P_B_win"])
        proba_map = {"A_win": p_a, "draw": p_d, "B_win": p_b}
        ll_model = _ll(proba_map[result])

        mkt = _MARKET_PROBS.get(mid)
        ll_mkt: float | None = _ll(mkt[f"P_{result}"]) if mkt else None

        scored.append({
            "match_id":        mid,
            "team_A":          team_a,
            "team_B":          pred_row["team_B"],
            "stage":           pred_row["stage"],
            "actual_result":   result,
            "P_A_win":         p_a,
            "P_draw":          p_d,
            "P_B_win":         p_b,
            "predicted_class": pred_row["predicted_class"],
            "correct":         pred_row["predicted_class"] == result,
            "ll_model":        ll_model,
            "ll_market":       ll_mkt,
            "mkt":             mkt,
        })

    # Actualiza actual_result em TODAS as linhas do log (inclui duplicados de várias runs).
    for row in rows:
        mid = int(row["match_id"])
        match_result = next(
            (s["actual_result"] for s in scored if s["match_id"] == mid), None
        )
        if match_result and not row.get("actual_result"):
            row["actual_result"] = match_result

    return rows, scored


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _stage_label(stage: str) -> str:
    return {
        "LAST_32": "LAST_32", "LAST_16": "LAST_16",
        "QUARTER_FINALS": "QF", "SEMI_FINALS": "SF",
        "THIRD_PLACE": "3P", "FINAL": "FINAL",
    }.get(stage, stage)


def _result_emoji(correct: bool) -> str:
    return "✓" if correct else "✗"


def main(force: bool = False) -> None:
    print("Carregando dados de jogos (football-data.org)...")
    matches_data = fetch_matches(use_cache=not force)
    match_index  = _build_match_index(matches_data.get("matches", []))

    rows = _load_log()
    if not rows:
        print(f"Log vazio ou não encontrado: {_LOG_PATH}")
        return

    rows, scored = score_log(rows, match_index)
    _save_log(rows)

    # Pendentes: match_ids no log que ainda não têm resultado
    scored_ids = {s["match_id"] for s in scored}
    pending_ids = {int(r["match_id"]) for r in rows} - scored_ids

    # ---- Jogos decididos ----
    print()
    if not scored:
        print("Nenhum jogo do log encerrado ainda.")
    else:
        hdr = (f"  {'time A':<22} {'time B':<22} {'resultado':>10} "
               f"  {'LL mod':>7}  {'LL mkt':>7}  {'gap':>7}  {'acerto':>6}")
        print(f"── Jogos decididos ({len(scored)}) ──────────────────────────────────────────")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for s in scored:
            gap_str = ""
            if s["ll_market"] is not None:
                gap = s["ll_model"] - s["ll_market"]
                gap_str = f"{gap:>+7.4f}"
            mkt_str = f"{s['ll_market']:>7.4f}" if s["ll_market"] is not None else "   n/a"
            print(
                f"  {s['team_A']:<22} {s['team_B']:<22} {s['actual_result']:>10}  "
                f"{s['ll_model']:>7.4f}  {mkt_str}  {gap_str}  "
                f"{_result_emoji(s['correct']):>6}"
            )

        # ---- Log-loss acumulado ----
        n     = len(scored)
        avg_m = sum(s["ll_model"] for s in scored) / n
        scored_mkt = [s for s in scored if s["ll_market"] is not None]
        print()
        print(f"── Log-loss acumulado ({n} jogos) ──────────────────────────────────────────")
        print(f"  Modelo:  {avg_m:.4f}")
        if scored_mkt:
            avg_k = sum(s["ll_market"] for s in scored_mkt) / len(scored_mkt)
            gap   = avg_m - avg_k
            sign  = "melhor" if gap < 0 else "pior"
            print(f"  Mercado: {avg_k:.4f}")
            print(f"  Gap:     {gap:+.4f}  (modelo {sign} que o mercado)")
        else:
            print("  Mercado: n/a (match_ids sem probabilidades de mercado hardcoded)")

        acertos = sum(1 for s in scored if s["correct"])
        print(f"  Acurácia ao vivo: {acertos}/{n} ({acertos/n:.1%})")

    # ---- Pendentes ----
    if pending_ids:
        print()
        print(f"── Aguardando resultado ({len(pending_ids)} jogos) ──────────────────────────")
        for row in rows:
            mid = int(row["match_id"])
            if mid in pending_ids:
                pending_ids.discard(mid)  # mostra só uma vez
                api = match_index.get(mid, {})
                status = api.get("status", "TIMED")
                print(f"  {row['team_A']:<22} vs {row['team_B']:<22}  [{status}]")

    print()
    print(f"Log actualizado: {_LOG_PATH}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pontua previsões ao vivo do LAST_32 WC2026 vs mercado."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignora cache de inferência e recarrega da API.",
    )
    args = parser.parse_args()
    main(force=args.force)
