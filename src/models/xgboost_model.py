"""Gradient boosting (XGBoost) — modelo principal Tier 1. SPEC.md §4.2, §5, §6.

Classificação multiclasse de 3 classes (A_win / draw / B_win) sobre features
de gols + contexto (Tier 1). Mesmo split temporal e pré-processamento do baseline.py.

Tuning mínimo via GridSearchCV (5-fold, neg_log_loss):
  n_estimators × max_depth × learning_rate — 27 combinações.

⚠️  PARIDADE TREINO↔INFERÊNCIA (SPEC.md §2.4): idêntica ao baseline — o scaler/
    imputers são fit exclusivamente no conjunto de treino (Pipeline.fit em X_train).

CLI:
    python -m src.models.xgboost_model
    python src/models/xgboost_model.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.features.tier1 import FEATURE_COLS, TARGET_COL, build_training_dataset
from src.models.baseline import (
    CLASSES,
    _build_preprocessor,
    _evaluate,
    _temporal_split,
    run as _run_logistic,
)

# ---------------------------------------------------------------------------
# Grid de hiperparâmetros (tuning mínimo §4.2)
# ---------------------------------------------------------------------------

# 3 × 3 × 3 = 27 combinações × 5 folds = 135 fits.
# Profundidades rasas (2–4) evitam overfitting no dataset pequeno (~47 linhas de treino).
PARAM_GRID: dict = {
    "clf__n_estimators":  [50, 100, 200],
    "clf__max_depth":     [2, 3, 4],
    "clf__learning_rate": [0.05, 0.1, 0.2],
}


# ---------------------------------------------------------------------------
# Treino + avaliação
# ---------------------------------------------------------------------------

def train() -> dict:
    """Treina XGBoost e retorna artefatos para uso externo (análise, inferência).

    Retorna dict com:
      pipeline   — Pipeline sklearn ajustado (prep + clf)
      le         — LabelEncoder(CLASSES) — converte str↔int para o clf
      X_train, y_train — dados de treino (DataFrame / ndarray de strings)
      X_val, y_val     — dados de validação
      val_df     — DataFrame completo de validação (com metadados: team_A, team_B, etc.)
      proba      — predict_proba na validação (n_val × 3, colunas = ordem de CLASSES)
      best_params, cv_log_loss
    """
    df = build_training_dataset()
    train_df, val_df = _temporal_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].to_numpy()
    X_val   = val_df[FEATURE_COLS]
    y_val   = val_df[TARGET_COL].to_numpy()

    # XGBoost >= 2.x com objective='multi:softprob' requer labels inteiros (0..n-1).
    # Fit em CLASSES (ordem lexicográfica) garante coluna i == CLASSES[i] em predict_proba.
    le = LabelEncoder()
    le.fit(CLASSES)  # 'A_win'→0, 'B_win'→1, 'draw'→2
    y_train_enc = le.transform(y_train)

    pipe = Pipeline([
        ("prep", _build_preprocessor()),
        ("clf", XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )),
    ])

    gs = GridSearchCV(
        pipe,
        PARAM_GRID,
        cv=5,
        scoring="neg_log_loss",
        refit=True,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train_enc)
    best_pipe = gs.best_estimator_
    best_params = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}

    # predict_proba: coluna i = P(CLASSES[i]) graças ao le fit em CLASSES. ✓
    proba = best_pipe.predict_proba(X_val)

    return {
        "pipeline":    best_pipe,
        "le":          le,
        "X_train":     X_train,
        "y_train":     y_train,
        "X_val":       X_val,
        "y_val":       y_val,
        "val_df":      val_df.reset_index(drop=True),
        "proba":       proba,
        "best_params": best_params,
        "cv_log_loss": round(-gs.best_score_, 4),
    }


def run() -> dict:
    """Treina XGBoost e avalia na validação temporal (interface de métricas).

    Retorna dict com métricas do XGBoost e dos dois baselines ingênuos.
    """
    result  = train()
    y_train = result["y_train"]
    y_val   = result["y_val"]
    proba   = result["proba"]

    counts = pd.Series(y_train).value_counts()
    prior = np.array([counts.get(c, 0) / len(y_train) for c in CLASSES], dtype=float)
    proba_uniform = np.full((len(y_val), len(CLASSES)), 1.0 / len(CLASSES))
    proba_prior   = np.tile(prior, (len(y_val), 1))

    return {
        "n_train":       len(y_train),
        "n_val":         len(y_val),
        "prior":         {c: round(float(p), 4) for c, p in zip(CLASSES, prior)},
        "best_params":   result["best_params"],
        "cv_log_loss":   result["cv_log_loss"],
        "xgboost":           _evaluate(y_val, proba),
        "baseline_uniforme": _evaluate(y_val, proba_uniform),
        "baseline_prior":    _evaluate(y_val, proba_prior),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(results_xgb: dict, results_log: dict) -> None:
    header = f"{'':26} {'log-loss':>10} {'acurácia':>10} {'brier':>10}"
    sep = "-" * len(header)

    print("\nValidação temporal — treino: WC2018 + Euro 2020 + AFCON 2023  |  "
          "val: WC2022 + Euro 2024 + Copa América 2024")
    print(f"(n_treino={results_xgb['n_train']}, n_val={results_xgb['n_val']}, "
          f"prior_treino={results_xgb['prior']})\n")

    print("Melhores hiperparâmetros (GridSearchCV 5-fold, neg_log_loss):")
    p = results_xgb["best_params"]
    print(f"  n_estimators={p['n_estimators']}  max_depth={p['max_depth']}  "
          f"learning_rate={p['learning_rate']}  →  CV log-loss={results_xgb['cv_log_loss']:.4f}\n")

    print(header)
    print(sep)
    rows = [
        ("XGBoost §4.2 (este)",    results_xgb["xgboost"]),
        ("Logístico §4.1",         results_log["modelo_logistico"]),
        ("Baseline uniforme",       results_xgb["baseline_uniforme"]),
        ("Baseline prior",          results_xgb["baseline_prior"]),
    ]
    for name, m in rows:
        print(f"{name:26} {m['log_loss']:>10.4f} {m['accuracy']:>10.4f} {m['brier']:>10.4f}")

    best_naive = min(results_xgb["baseline_uniforme"]["log_loss"],
                     results_xgb["baseline_prior"]["log_loss"])
    xgb_ll  = results_xgb["xgboost"]["log_loss"]
    log_ll  = results_log["modelo_logistico"]["log_loss"]
    verdict = "✓ supera" if xgb_ll < best_naive else "✗ NÃO supera"
    delta   = log_ll - xgb_ll
    sign    = "↓ melhor" if delta > 0 else "↑ pior"

    print(f"\nCritério §6.1 (log-loss): XGBoost={xgb_ll:.4f} vs melhor ingênuo="
          f"{best_naive:.4f} → {verdict} o baseline ingênuo.")
    print(f"XGBoost vs logístico: Δlog-loss={delta:+.4f} ({sign} que o baseline).")


def main() -> None:
    print("Treinando XGBoost (GridSearchCV 27 combinações × 5 folds)...")
    results_xgb = run()
    print("Treinando logístico para comparação...")
    results_log = _run_logistic()
    _print_report(results_xgb, results_log)


if __name__ == "__main__":
    main()
