"""SHADOW MODE — inferência ao vivo de ESCANTEIOS na WC2026 (Modo 2). SPEC_MARKETS §5.6, §7.

ESCOPO FECHADO: só escanteios. Gols e cartões ficam FORA do vivo (corr ≈ 0, docs/RESULTS.md
Modo 2). Shadow mode = fechar o ciclo ponta-a-ponta contra a Copa real, SEM promessa de
qualidade — o sinal de escanteios é marginal (−0.0017 vs baseline) e isso está documentado.

Modelo: v1 já medido (counts_model.py), forma funcional §5.6 — λ = ataque_shrunk × defesa_shrunk
/ média_expansiva. NÃO retreina, NÃO ajusta parâmetro. Fonte: SofaScore only (§2.4).

DOIS TIPOS DE PREVISÃO, estruturalmente separados (coluna `prediction_type` obrigatória):
  - prereg : jogo ainda NÃO iniciado (ambos os times definidos) → pré-registro real.
  - retro  : jogo de mata-mata da WC2026 já FINISHED → sem valor de pré-registro, só avaliação.
  Nenhuma métrica agregada mistura os dois (o scoring reporta separado).
  "Uma vez prereg, SEMPRE prereg": se um jogo já tem linha prereg, não é reclassificado nem
  re-previsto quando roda de novo depois de começar.

Régua temporal (§4.3): escanteios do SofaScore cobrem o JOGO INTEIRO (incl. prorrogação),
consistente com o corpus de treino. NÃO extrair "escanteios de 90min" — régua diferente da
do Modo 1.

Corte temporal (§7, sem vazamento): retro é featurizado com o corpus como estava ANTES do jogo
(build_count_features usa corte estrito '<' no startTimestamp — garantido pela função canônica).

CONVENÇÃO DE PRÉ-REGISTRO (counts_predictions_log.csv é versionado — exceção no .gitignore):
    O CSV é o registro de pré-compromisso das previsões de contagem. Toda rodada de
    predict_counts.py (novas previsões) ou de live_score_counts.py (novos resultados /
    reconciliações) DEVE ser seguida de `git commit` do CSV — o histórico do git funciona como
    carimbo de tempo do pré-registro (prova de que a previsão prereg existia antes do resultado).

Uso:
    .venv/bin/python src/predict_counts.py            # coleta já feita à parte via sofascore --live
"""
from __future__ import annotations

import csv
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.collect import sofascore as sf
from src.features import counts, tier1
from src.models import counts_model as cm

_ROOT     = Path(__file__).parents[1]
_LOG_PATH = _ROOT / "data" / "processed" / "counts_predictions_log.csv"

_LINES = [8.5, 9.5, 10.5, 11.5]
_KMAX  = 80  # suporte da Poisson do total (escanteios raramente > 30)

_LOG_COLS = [
    "event_id", "timestamp_predicted", "stage", "team_A", "team_B", "prediction_type",
    "lambda_A", "lambda_B", "expected_total",
    "p_over_8_5", "p_over_9_5", "p_over_10_5", "p_over_11_5",
    "actual_corners_total", "n_games_A", "n_games_B",
]

_WC_TID  = 16
_WC_YEAR = 2026


def _line_col(line: float) -> str:
    return "p_over_" + str(line).replace(".", "_")


# ---------------------------------------------------------------------------
# Corpus vivo = histórico + WC2026 FINISHED (compartilhado com o scoring)
# ---------------------------------------------------------------------------

def load_live_corpus() -> tuple[list[dict], pd.DataFrame]:
    """Retorna (corpus_records, live_df). corpus = 394 históricos + WC2026 FINISHED."""
    hist = counts.fetch_historical_counts()
    live = sf.fetch_live_counts()
    if live.empty:
        df = hist.copy()
    else:
        df = pd.concat([hist, live], ignore_index=True)
    df = df.sort_values(["start_timestamp", "event_id"]).reset_index(drop=True)
    return df.to_dict("records"), live


# ---------------------------------------------------------------------------
# Descoberta de jogos alvo
# ---------------------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^[WL]\d+$")  # SofaScore: W87/L101 = vencedor/perdedor de match


def _is_defined(name: str) -> bool:
    return bool(name) and not _PLACEHOLDER.match(str(name).strip())


def retro_games(live: pd.DataFrame) -> list[dict]:
    """Jogos de mata-mata da WC2026 já FINISHED e com escanteios (round_name presente)."""
    if live.empty:
        return []
    ko = live[live["round_name"].notna() & live["has_stats"]]
    return ko.to_dict("records")


def fetch_prereg_fixtures(sleep_s: float = 3.0) -> list[dict]:
    """Jogos de mata-mata da WC2026 AINDA NÃO iniciados, com ambos os times definidos.

    Fonte: SofaScore /events/next. Degradação graciosa (§7): erro/HTTP != 200 → [] com aviso.
    """
    mode = sf._ensure_client()
    if mode != "curl_cffi" and not sf._COOKIE:
        print("  ⚠ curl_cffi ausente — não consigo buscar fixtures prereg. Pulando prereg.")
        return []
    seasons = sf._get_seasons(_WC_TID, sleep_s)
    season = next((s for s in seasons
                   if sf._year_of(s.get("year") or s.get("name")) == _WC_YEAR), None)
    if season is None:
        print("  ⚠ season WC2026 não exposta — sem fixtures prereg.")
        return []
    sid = season["id"]
    fixtures: list[dict] = []
    for page in range(3):  # próximas páginas de eventos futuros
        st, body = sf._get(f"unique-tournament/{_WC_TID}/season/{sid}/events/next/{page}", sleep_s)
        if st != 200:
            break
        evs = body.get("events", [])
        if not evs:
            break
        for e in evs:
            if e.get("status", {}).get("type") != "notstarted":
                continue
            ri = e.get("roundInfo") or {}
            if not ri.get("name"):  # sem fase = grupo (não é mata-mata)
                continue
            home = e.get("homeTeam", {}).get("name")
            away = e.get("awayTeam", {}).get("name")
            if not (_is_defined(home) and _is_defined(away)):
                continue  # placeholder (W87 etc.) → time indefinido, pula
            ts = e.get("startTimestamp")
            fixtures.append({
                "event_id": int(e["id"]),
                "home_team": home, "away_team": away,
                "start_timestamp": ts,
                "date": datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat() if ts else None,
                "round_name": ri.get("name"),
                "has_stats": False,  # é o alvo — não entra no corpus
            })
        if not body.get("hasNextPage"):
            break
    return fixtures


# ---------------------------------------------------------------------------
# Previsão de um jogo (v1 escanteios)
# ---------------------------------------------------------------------------

def predict_corners(match: dict, corpus: list[dict], rank_lookup: dict) -> dict | None:
    """λ_A/λ_B/total + P(over) das 4 linhas para um jogo. None se divisor 0 (degradação §7)."""
    feat = counts.build_count_features(match, corpus, rank_lookup)
    mean_c = cm._expanding_means(corpus, match["start_timestamp"])["corners"]
    lams = cm.v1_team_lambdas(feat, "escanteios", mean_c)
    if lams is None:  # divisor 0 (prior vazio) — nunca fabricar valor
        return None
    lam_A, lam_B = lams
    total = lam_A + lam_B
    pmf = cm._poisson_total_pmf(total, _KMAX)
    return {
        "team_A": feat["team_A"], "team_B": feat["team_B"],
        "lambda_A": lam_A, "lambda_B": lam_B, "expected_total": total,
        "p_over": {L: cm._prob_over(pmf, L) for L in _LINES},
        "n_games_A": feat["A_n_games"], "n_games_B": feat["B_n_games"],
    }


# ---------------------------------------------------------------------------
# CSV I/O (UPSERT por event_id)
# ---------------------------------------------------------------------------

def _load_log() -> dict[int, dict]:
    if not _LOG_PATH.exists():
        return {}
    with open(_LOG_PATH, newline="", encoding="utf-8") as fh:
        return {int(r["event_id"]): r for r in csv.DictReader(fh)}


def _save_log(by_id: dict[int, dict]) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(by_id.values(), key=lambda r: (str(r.get("prediction_type")), int(r["event_id"])))
    with open(_LOG_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLS)
        w.writeheader()
        w.writerows(rows)


def _row_from_pred(match: dict, pred: dict, ptype: str, ts_now: str, prev: dict | None) -> dict:
    row = {c: "" for c in _LOG_COLS}
    row.update({
        "event_id": int(match["event_id"]),
        "timestamp_predicted": ts_now,
        "stage": match.get("round_name") or "",
        "team_A": pred["team_A"], "team_B": pred["team_B"],
        "prediction_type": ptype,
        "lambda_A": round(pred["lambda_A"], 4),
        "lambda_B": round(pred["lambda_B"], 4),
        "expected_total": round(pred["expected_total"], 4),
        "n_games_A": pred["n_games_A"], "n_games_B": pred["n_games_B"],
    })
    for L in _LINES:
        row[_line_col(L)] = round(pred["p_over"][L], 4)
    # UPSERT preserva actual_corners_total já preenchido pelo scoring.
    row["actual_corners_total"] = (prev or {}).get("actual_corners_total", "") or ""
    return row


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def run(sleep_s: float = 3.0) -> dict:
    print("Carregando corpus vivo (histórico + WC2026 FINISHED)...")
    corpus, live = load_live_corpus()
    print(f"  corpus: {len(corpus)} jogos  |  WC2026 no cache vivo: {len(live)}")
    rank_lookup = tier1._build_rank_lookup(wc2026_only=False)

    log = _load_log()
    ts_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    n_retro = n_prereg = n_skipped = n_degraded = 0

    # --- RETRO: jogos já FINISHED ---
    retros = retro_games(live)
    print(f"\nRETRO — {len(retros)} jogos de mata-mata FINISHED:")
    for g in retros:
        eid = int(g["event_id"])
        if eid in log and log[eid].get("prediction_type") == "prereg":
            n_skipped += 1  # já era prereg → não reclassifica nem re-prevê
            continue
        pred = predict_corners(g, corpus, rank_lookup)
        if pred is None:
            print(f"  ⚠ {g['home_team']}×{g['away_team']}: divisor 0 — pulado (§7)")
            n_degraded += 1
            continue
        log[eid] = _row_from_pred(g, pred, "retro", ts_now, log.get(eid))
        n_retro += 1
        print(f"  [retro] {pred['team_A'][:16]:<16}×{pred['team_B'][:16]:<16} "
              f"E[total]={pred['expected_total']:>5.1f}  P(o9.5)={pred['p_over'][9.5]:.2f}")

    # --- PREREG: jogos futuros com times definidos ---
    fixtures = fetch_prereg_fixtures(sleep_s)
    print(f"\nPREREG — {len(fixtures)} fixtures futuros (mata-mata, times definidos):")
    for g in fixtures:
        eid = int(g["event_id"])
        if eid in log and log[eid].get("prediction_type") == "prereg":
            n_skipped += 1  # já pré-registrado — imutável
            continue
        pred = predict_corners(g, corpus, rank_lookup)
        if pred is None:
            print(f"  ⚠ {g['home_team']}×{g['away_team']}: divisor 0 — pulado (§7)")
            n_degraded += 1
            continue
        log[eid] = _row_from_pred(g, pred, "prereg", ts_now, log.get(eid))
        n_prereg += 1
        print(f"  [PREREG] {g['round_name']:<14} {pred['team_A'][:16]:<16}×{pred['team_B'][:16]:<16} "
              f"E[total]={pred['expected_total']:>5.1f}  P(o9.5)={pred['p_over'][9.5]:.2f}")

    _save_log(log)
    print(f"\n=== CONCLUÍDO === retro={n_retro} novos, prereg={n_prereg} novos, "
          f"já-prereg-preservados={n_skipped}, degradados={n_degraded}")
    print(f"Log: {_LOG_PATH}")
    return {"n_retro": n_retro, "n_prereg": n_prereg, "n_skipped": n_skipped,
            "n_degraded": n_degraded, "log": log}


def main() -> None:
    run()


if __name__ == "__main__":
    main()
