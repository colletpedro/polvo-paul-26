"""Previsão pré-partida para o mata-mata da WC2026. SPEC.md §4.3, §4.5.

Modelo de produção: XGBoost + calibração sigmoid (§4.3).
Features: Tier 1 goals-based via build_inference_dataset() (§3.1–3.2, §2.4).

Comportamento:
  - Prevê SOMENTE jogos com status TIMED/SCHEDULED e kickoff UTC ainda no futuro,
    com os dois times já conhecidos (não-TBD).
  - Jogos FINISHED ou IN_PLAY: listados separadamente, sem previsão.
  - Jogos TBD (times indeterminados): listados separadamente.
  - Cada previsão é gravada em data/processed/predictions_log.csv por UPSERT em
    match_id: uma linha por jogo. Re-previsões atualizam as probabilidades e
    preservam actual_result se já preenchido; jogos novos são anexados.

CLI:
    python src/predict.py
    python -m src.predict
"""
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Garante que src/ está no path ao rodar como script
sys.path.insert(0, str(Path(__file__).parents[1]))

from src.collect.footballdata import fetch_matches
from src.features.tier1 import FEATURE_COLS, build_inference_dataset
from src.models.baseline import CLASSES
from src.models.calibrated import run as train_calibrated

_ROOT        = Path(__file__).parents[1]
_LOG_PATH    = _ROOT / "data" / "processed" / "predictions_log.csv"
_LOG_COLS    = [
    "match_id", "timestamp_predicted", "stage",
    "team_A", "team_B",
    "P_A_win", "P_draw", "P_B_win",
    "predicted_class", "entropy",
    "actual_result",
]

# Índices canônicos (CLASSES é lexicográfico: A_win=0, B_win=1, draw=2)
_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray) -> float:
    """Entropia de Shannon (bits base-e) de um vetor de probabilidades."""
    return float(-np.sum(p * np.log(np.clip(p, 1e-15, 1.0))))


def _stage_label(stage: str) -> str:
    return {
        "LAST_32": "Oitavas/32", "LAST_16": "Oitavas/16",
        "QUARTER_FINALS": "Quartas", "SEMI_FINALS": "Semifinal",
        "THIRD_PLACE": "3º Lugar", "FINAL": "Final",
    }.get(stage, stage)


def _load_utcdates(raw_matches: list[dict]) -> dict[int, datetime]:
    """Constrói {match_id → utcDate como datetime UTC} a partir dos dados brutos."""
    result: dict[int, datetime] = {}
    for m in raw_matches:
        try:
            result[m["id"]] = datetime.fromisoformat(
                m["utcDate"].replace("Z", "+00:00")
            )
        except (KeyError, ValueError):
            pass
    return result


# ---------------------------------------------------------------------------
# Log de previsões (upsert por match_id — uma linha por jogo)
# ---------------------------------------------------------------------------

def _upsert_log(new_rows: list[dict]) -> None:
    """Upsert das previsões no CSV de log, chaveado por match_id.

    - Se já existe linha para o match_id: atualiza as colunas de previsão
      (probabilidades, classe, entropia, timestamp, stage, times) e PRESERVA
      actual_result se já estiver preenchido.
    - Se não existe: anexa uma nova linha.
    A ordem das linhas existentes é preservada; jogos novos vão para o fim.
    """
    if not new_rows:
        return
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict] = []
    if _LOG_PATH.exists():
        with open(_LOG_PATH, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))

    # Última ocorrência vence caso o ficheiro ainda tenha duplicatas legadas.
    index = {str(r.get("match_id", "")): i for i, r in enumerate(existing)}

    for new in new_rows:
        mid = str(new["match_id"])
        if mid in index:
            cur = existing[index[mid]]
            preserved = (cur.get("actual_result") or "").strip()
            cur.update(new)
            # Nunca sobrescreve um resultado real já preenchido.
            cur["actual_result"] = preserved or new.get("actual_result", "")
        else:
            existing.append(dict(new))
            index[mid] = len(existing) - 1

    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_LOG_COLS)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, "") for k in _LOG_COLS})


# ---------------------------------------------------------------------------
# Previsão principal
# ---------------------------------------------------------------------------

def run_predictions() -> dict:
    """Treina o modelo de produção e gera previsões para jogos TIMED futuros.

    Retorna dict com: predictions (list[dict]), finished (list[dict]),
    in_play (list[dict]), tbd (list[dict]).
    """
    now_utc = datetime.now(timezone.utc)

    # --- Treino do modelo de produção ---
    print("Treinando modelo de produção (XGBoost + sigmoid)...")
    calib_result = train_calibrated()
    model = calib_result["models"]["calib_sigmoid"]
    sig_ll = calib_result["calib_sigmoid"]["log_loss"]
    print(f"  Params: {calib_result['best_params']}  "
          f"val log-loss (sigmoid)={sig_ll:.4f}\n")

    # --- Dados de inferência ---
    infer_df    = build_inference_dataset()
    raw_matches = fetch_matches().get("matches", [])
    utcdates    = _load_utcdates(raw_matches)

    # Enriquece com utcDate completo para filtro de kickoff
    infer_df["utcDatetime"] = infer_df["match_id"].map(utcdates)

    # --- Classificar cada jogo ---
    predictions: list[dict] = []
    finished:    list[dict] = []
    in_play:     list[dict] = []
    tbd:         list[dict] = []
    log_rows:    list[dict] = []

    ts_now = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    for _, row in infer_df.iterrows():
        status   = row.get("status", "")
        kickoff  = row.get("utcDatetime")
        team_a   = row.get("team_A")
        team_b   = row.get("team_B")
        teams_ok = pd.notna(team_a) and pd.notna(team_b)

        meta = {
            "match_id": int(row["match_id"]),
            "stage":    row["stage"],
            "team_A":   team_a if teams_ok else "TBD",
            "team_B":   team_b if teams_ok else "TBD",
            "match_date": str(row["match_date"])[:10],
            "kickoff_utc": kickoff.strftime("%Y-%m-%dT%H:%MZ") if kickoff else "—",
            "result_90min": row.get("result_90min"),
        }

        if status == "FINISHED":
            finished.append(meta)
            continue

        if status == "IN_PLAY":
            in_play.append(meta)
            continue

        # TIMED / SCHEDULED
        if not teams_ok:
            tbd.append(meta)
            continue

        # Verifica se kickoff ainda está no futuro
        if kickoff and kickoff <= now_utc:
            # Status desatualizado — trata como em jogo
            in_play.append({**meta, "_note": "kickoff passado, status TIMED"})
            continue

        # --- Gera previsão ---
        X = pd.DataFrame([row[FEATURE_COLS]])
        proba = model.predict_proba(X)[0]  # shape (3,) — ordem CLASSES

        p_a_win = float(proba[_IDX["A_win"]])
        p_b_win = float(proba[_IDX["B_win"]])
        p_draw  = float(proba[_IDX["draw"]])
        pred_cls = CLASSES[int(np.argmax(proba))]
        ent      = _entropy(proba)

        pred = {
            **meta,
            "P_A_win": p_a_win,
            "P_draw":  p_draw,
            "P_B_win": p_b_win,
            "predicted_class": pred_cls,
            "entropy": ent,
        }
        predictions.append(pred)

        # Acumula para upsert único no fim (uma linha por match_id).
        log_rows.append({
            "match_id":           meta["match_id"],
            "timestamp_predicted": ts_now,
            "stage":              meta["stage"],
            "team_A":             team_a,
            "team_B":             team_b,
            "P_A_win":            round(p_a_win, 4),
            "P_draw":             round(p_draw,  4),
            "P_B_win":            round(p_b_win, 4),
            "predicted_class":    pred_cls,
            "entropy":            round(ent, 4),
            "actual_result":      "",
        })

    # Upsert por match_id: atualiza probabilidades, preserva actual_result.
    _upsert_log(log_rows)

    return {
        "predictions": predictions,
        "finished":    finished,
        "in_play":     in_play,
        "tbd":         tbd,
        "timestamp":   ts_now,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _bar(p: float, width: int = 12) -> str:
    """Barra horizontal proporcional à probabilidade."""
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    result = run_predictions()
    now    = result["timestamp"]

    # --- Jogos FINISHED ---
    if result["finished"]:
        print("── Jogos já disputados (FINISHED) ──────────────────────────────")
        for m in result["finished"]:
            res = m.get("result_90min") or "?"
            print(f"  {m['match_date']}  {_stage_label(m['stage']):<12}  "
                  f"{m['team_A']:<22} vs {m['team_B']:<22}  resultado: {res}")
        print()

    # --- IN_PLAY ---
    if result["in_play"]:
        print("── Em jogo agora (IN_PLAY) ──────────────────────────────────────")
        for m in result["in_play"]:
            note = m.get("_note", "")
            print(f"  {m['kickoff_utc']}  {_stage_label(m['stage']):<12}  "
                  f"{m['team_A']:<22} vs {m['team_B']:<22}  "
                  + (f"[{note}]" if note else ""))
        print()

    # --- Previsões ---
    if result["predictions"]:
        print("── Previsões — jogos TIMED com times definidos ─────────────────")
        hdr = (f"  {'kickoff UTC':<17} {'fase':<12} {'time A':<22} {'time B':<22} "
               f"  {'P(Aw)':>7} {'P(dr)':>7} {'P(Bw)':>7}  {'favorito':<22} {'H':>5}")
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for p in sorted(result["predictions"], key=lambda x: x["kickoff_utc"]):
            winner = p["team_A"] if p["predicted_class"] == "A_win" else (
                     p["team_B"] if p["predicted_class"] == "B_win" else "Empate previsto")
            print(
                f"  {p['kickoff_utc']:<17} {_stage_label(p['stage']):<12} "
                f"{p['team_A']:<22} {p['team_B']:<22}  "
                f"{p['P_A_win']:>7.1%} {p['P_draw']:>7.1%} {p['P_B_win']:>7.1%}  "
                f"{winner:<22} {p['entropy']:>5.3f}"
            )
        print(f"\n  {len(result['predictions'])} previsões gravadas em {_LOG_PATH}")
        print(f"  timestamp_predicted: {now}")
    else:
        print("Nenhum jogo TIMED com times definidos encontrado.")

    # --- TBD ---
    if result["tbd"]:
        print(f"\n── Jogos futuros TBD (times ainda indeterminados): "
              f"{len(result['tbd'])} partidas ─────")
        stages = {}
        for m in result["tbd"]:
            stages[m["stage"]] = stages.get(m["stage"], 0) + 1
        for stage, n in sorted(stages.items()):
            print(f"  {_stage_label(stage)}: {n}")


if __name__ == "__main__":
    main()
