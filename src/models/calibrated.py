"""Calibração de probabilidades (§4.3) sobre o melhor XGBoost.

Aplica CalibratedClassifierCV com method='isotonic' e method='sigmoid',
usando os hiperparâmetros escolhidos pelo GridSearchCV em xgboost_model.py.

Estratégia de cv:
  cv=5 — consistente com o tuning do XGBoost.  Com 47 linhas de treino isso
  produz ~9 amostras de calibração por fold.  Para isotonic (que é não-paramétrico
  e ajusta um threshold por bin) isso é muito pouco: class draw/B_win têm ≈1-2
  positivos por fold de calibração → isotonic tende a reduzir-se ao prior ou
  overfittar ao handful de pontos.  Para sigmoid (Platt scaling — 2 parâmetros
  por classe) os ~9 pontos são suficientes para um ajuste estável.  Ambos são
  reportados; a decisão é deixada para o resultado empírico (conforme enunciado).

CLI:
    python -m src.models.calibrated
    python src/models/calibrated.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.models.baseline import (
    CLASSES,
    _build_preprocessor,
    _evaluate,
)
from src.models.xgboost_model import train as xgb_train

# cv=5: alinhado ao GridSearchCV do XGBoost (§4.2).
# Cada fold de calibração: ≈47//5 = 9 amostras — suficiente para sigmoid,
# marginal para isotonic.
_CALIB_CV = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_fresh_pipe(params: dict) -> Pipeline:
    """Pipeline XGBoost não ajustado com os melhores hiperparâmetros.

    CalibratedClassifierCV precisa de um estimador *não ajustado* — ele vai
    chamá-lo .fit() internamente em cada fold de calibração.
    """
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=42,
            verbosity=0,
        )),
    ])


# ---------------------------------------------------------------------------
# Treino + calibração + avaliação
# ---------------------------------------------------------------------------

def run() -> dict:
    """Treina XGBoost, aplica calibração isotonic e sigmoid, avalia no val temporal.

    Retorna dict com métricas de: xgb_raw, calib_isotonic, calib_sigmoid, prior.
    Inclui os modelos calibrados ajustados para uso posterior em inferência.
    """
    result      = xgb_train()
    le          = result["le"]          # LabelEncoder(CLASSES): 'A_win'→0,'B_win'→1,'draw'→2
    X_train     = result["X_train"]
    y_train     = result["y_train"]     # strings originais
    X_val       = result["X_val"]
    y_val       = result["y_val"]
    best_params = result["best_params"]

    # Labels inteiros obrigatórios para XGBoost com multi:softprob.
    y_train_enc = le.transform(y_train)

    metrics: dict[str, dict] = {}
    models:  dict[str, CalibratedClassifierCV] = {}

    # --- XGBoost cru (já disponível de xgb_train, sem re-treino) ---
    metrics["xgb_raw"] = _evaluate(y_val, result["proba"])

    # --- Calibração ---
    for method in ("isotonic", "sigmoid"):
        calib = CalibratedClassifierCV(
            estimator=_build_fresh_pipe(best_params),
            method=method,
            cv=_CALIB_CV,
        )
        calib.fit(X_train, y_train_enc)
        proba_c = calib.predict_proba(X_val)
        # calib.classes_ = [0, 1, 2]; le mapeou CLASSES[i]→i,
        # logo coluna i de predict_proba corresponde a CLASSES[i]. ✓
        metrics[f"calib_{method}"] = _evaluate(y_val, proba_c)
        models[f"calib_{method}"]  = calib

    # --- Prior histórico (referência inferior) ---
    counts = pd.Series(y_train).value_counts()
    prior  = np.array([counts.get(c, 0) / len(y_train) for c in CLASSES], dtype=float)
    metrics["baseline_prior"] = _evaluate(y_val, np.tile(prior, (len(y_val), 1)))

    return {
        "n_train":        len(y_train),
        "n_val":          len(y_val),
        "cv_calib":       _CALIB_CV,
        "best_params":    best_params,
        "prior":          {c: round(float(p), 4) for c, p in zip(CLASSES, prior)},
        "models":         models,   # calibradores ajustados (para inferência futura)
        **metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(r: dict) -> None:
    p = r["best_params"]
    n_cal = r["n_train"] // r["cv_calib"]
    print(f"\nXGBoost base: n_estimators={p['n_estimators']}  "
          f"max_depth={p['max_depth']}  lr={p['learning_rate']}")
    print(f"Calibração: CalibratedClassifierCV(cv={r['cv_calib']})  "
          f"→ ~{n_cal} amostras/fold de calibração")
    print(f"Split: treino={r['n_train']} linhas  |  validação={r['n_val']} linhas")
    print(f"Prior treino: {r['prior']}\n")

    raw_ll = r["xgb_raw"]["log_loss"]
    hdr = (f"{'modelo':34} {'log-loss':>10} {'acurácia':>10} {'brier':>10}"
           f"  {'Δ log-loss':>10}")
    print(hdr)
    print("─" * len(hdr))

    rows = [
        ("XGBoost cru §4.2",         "xgb_raw"),
        ("+ isotonic (§4.3)",         "calib_isotonic"),
        ("+ sigmoid  (§4.3)",         "calib_sigmoid"),
        ("Baseline prior",            "baseline_prior"),
    ]
    for label, key in rows:
        m     = r[key]
        delta = m["log_loss"] - raw_ll
        ds    = f"{delta:+.4f}" if key != "xgb_raw" else "      —"
        print(f"{label:34} {m['log_loss']:>10.4f} {m['accuracy']:>10.4f} "
              f"{m['brier']:>10.4f}  {ds:>10}")

    # Veredicto
    iso_ll = r["calib_isotonic"]["log_loss"]
    sig_ll = r["calib_sigmoid"]["log_loss"]
    best_ll, best_m = (iso_ll, "isotonic") if iso_ll <= sig_ll else (sig_ll, "sigmoid")
    sign = "↓ melhora" if best_ll < raw_ll else "↑ piora"
    print(f"\n  Melhor calibração: {best_m}  "
          f"log-loss={best_ll:.4f}  ({sign} vs cru={raw_ll:.4f})")
    print(f"\n  ⚠  ~{n_cal} amostras/fold: sigmoid é mais robusto neste regime; "
          f"interpretar isotonic com cautela.")


def main() -> None:
    print("Treinando XGBoost base (GridSearchCV)...")
    r = run()
    _print_report(r)


if __name__ == "__main__":
    main()
