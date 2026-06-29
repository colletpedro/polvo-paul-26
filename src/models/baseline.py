"""Baseline — regressão logística multinomial (Tier 1). SPEC.md §4.1, §5, §6.

Referência interpretável e de baixa variância para os resultados do mata-mata
(classificação multiclasse de 3 classes: A_win / draw / B_win).

⚠️  PARIDADE TREINO↔INFERÊNCIA (SPEC.md §2.4 — decisão mais importante do projeto):
    Este módulo usa SOMENTE build_training_dataset() (StatsBomb, Tier 1). Os dados de
    inferência da WC2026 (build_inference_dataset) NÃO são importados nem tocados aqui —
    qualquer uso deles viola §2.4. O scaler/imputers são fit APENAS no conjunto de treino
    e aplicados à validação (Pipeline.fit em X_train), nunca o contrário.

Pré-processamento (SPEC.md §3.5), montado a partir de FEATURE_COLS de tier1.py:
  - StandardScaler  → features numéricas contínuas (gols, ranking, deltas, descanso…)
  - OneHotEncoder   → A_confederation, B_confederation
  - stage_ord       → passthrough ordinal. A ordem [LAST_32 < LAST_16 < QUARTER_FINALS <
                      SEMI_FINALS < FINAL] que o §3.5 pede para `stage` JÁ foi aplicada
                      upstream em tier1.STAGE_ORD; reencodar seria duplo-encoding. Não há
                      coluna `stage` crua em FEATURE_COLS, só `stage_ord` (int 1–4).
  - recent_form     → o OrdinalEncoder [derrota < empate < vitória] do §3.5 NÃO se aplica:
                      não existe coluna `recent_form` em FEATURE_COLS (a forma já vem como
                      A_form_score/B_form_score, score contínuo). Encoder omitido por isso.
  - team_A, team_B  → NÃO estão em FEATURE_COLS (removidos em tier1.py — são metadados,
                      não features). DROP_COLS é vazio; a guarda ainda verifica cobertura total.

  Guarda dura: toda coluna de FEATURE_COLS recebe tratamento explícito. Se alguma não for
  coberta por nenhum dos casos acima, _build_preprocessor levanta ValueError nomeando-a —
  nada cai no remainder silenciosamente.

Validação temporal (SPEC.md §4.4): treina em WC2018 + Euro 2020; valida em WC2022 +
Euro 2024 + Copa América 2024.

CLI:
    python -m src.models.baseline   (ou  python src/models/baseline.py)
imprime log-loss, acurácia e brier score lado a lado para:
  (a) modelo logístico, (b) baseline uniforme, (c) baseline prior histórico (§5, §6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.tier1 import FEATURE_COLS, TARGET_COL, build_training_dataset

# ---------------------------------------------------------------------------
# Constantes de modelagem
# ---------------------------------------------------------------------------

# Ordem canônica das classes — fixa o layout das matrizes de probabilidade e dos
# vetores de prior em todo o módulo (log_loss, brier, baselines).
# IMPORTANTE: precisa ser a ordem LEXICOGRÁFICA. sklearn.metrics.log_loss ignora a
# ordem passada em labels= e ordena as classes lexicograficamente internamente; se as
# colunas de proba não seguirem essa ordem, o log-loss vem errado (visto em teste:
# previsão correta e confiante pontuava 36 em vez de ~0). LogisticRegression.classes_
# também já vem ordenada assim, então isto mantém tudo alinhado.
CLASSES: list[str] = sorted(["A_win", "draw", "B_win"])  # -> ['A_win', 'B_win', 'draw']

# Split temporal (SPEC.md §4.4) — rótulos conforme a coluna `competition` de tier1.
TRAIN_COMPETITIONS = {"FIFA World Cup 2018", "UEFA Euro 2020", "African Cup of Nations 2023"}
VAL_COMPETITIONS = {"FIFA World Cup 2022", "UEFA Euro 2024", "Copa America 2024"}

# --- Particionamento explícito de FEATURE_COLS por tipo de tratamento (SPEC.md §3.5) ---

# Features numéricas contínuas → StandardScaler (com imputação de NaN antes).
NUMERIC_COLS: list[str] = [
    "A_points", "A_goals_for", "A_goals_against", "A_goal_diff",
    "A_goals_for_pg", "A_goals_against_pg", "A_clean_sheets", "A_form_score",
    "B_points", "B_goals_for", "B_goals_against", "B_goal_diff",
    "B_goals_for_pg", "B_goals_against_pg", "B_clean_sheets", "B_form_score",
    "A_fifa_rank", "A_fifa_points",
    "B_fifa_rank", "B_fifa_points",
    "delta_rank",
    "A_days_rest", "B_days_rest", "delta_rest",
]

# Categóricas nominais → OneHotEncoder.
ONEHOT_COLS: list[str] = ["A_confederation", "B_confederation"]

# Ordinal já encodada upstream (tier1.STAGE_ORD) → passa direto, sem reencodar.
ORDINAL_PASSTHROUGH_COLS: list[str] = ["stage_ord"]

# Identificadores — team_A/team_B removidos de FEATURE_COLS em tier1.py; lista vazia
# mas mantida para o guarda de cobertura total de FEATURE_COLS não precisar de ajuste.
DROP_COLS: list[str] = []

# Encoder ordinal de forma recente do §3.5: só entra SE a coluna existir em FEATURE_COLS.
# Hoje não existe (forma vem como *_form_score contínuo), então a lista fica vazia.
ORDINAL_FORM_COLS: list[str] = [c for c in ("recent_form",) if c in FEATURE_COLS]
FORM_ORDER = ["derrota", "empate", "vitória"]


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def _build_preprocessor() -> ColumnTransformer:
    """Monta o ColumnTransformer cobrindo TODAS as colunas de FEATURE_COLS.

    Levanta ValueError se alguma coluna de FEATURE_COLS não cair em nenhum dos
    grupos de tratamento (numérico / OHE / ordinal-stage / ordinal-forma / drop) —
    impede que uma feature escape para o remainder sem tratamento (SPEC.md §3.5).
    """
    covered = (
        set(NUMERIC_COLS)
        | set(ONEHOT_COLS)
        | set(ORDINAL_PASSTHROUGH_COLS)
        | set(ORDINAL_FORM_COLS)
        | set(DROP_COLS)
    )
    uncovered = [c for c in FEATURE_COLS if c not in covered]
    if uncovered:
        raise ValueError(
            "Colunas de FEATURE_COLS sem tratamento explícito no ColumnTransformer "
            f"(SPEC.md §3.5): {uncovered}. Adicione-as a um grupo (NUMERIC_COLS, "
            "ONEHOT_COLS, ORDINAL_PASSTHROUGH_COLS, ORDINAL_FORM_COLS ou DROP_COLS) "
            "antes de treinar."
        )
    # Guarda extra: não declarar como feature algo que não está em FEATURE_COLS.
    declared = [c for c in covered if c not in FEATURE_COLS]
    if declared:
        raise ValueError(
            f"Colunas declaradas no preprocessor mas ausentes de FEATURE_COLS: "
            f"{sorted(declared)}."
        )

    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    # Confederação ausente vira categoria explícita 'UNKNOWN' — imputação explícita +
    # serve de flag de ausência (SPEC.md §3.5: "imputação explícita + flag").
    onehot_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("numeric", numeric_pipe, NUMERIC_COLS),
        ("onehot", onehot_pipe, ONEHOT_COLS),
        ("stage", "passthrough", ORDINAL_PASSTHROUGH_COLS),
    ]
    if DROP_COLS:
        transformers.append(("ident_drop", "drop", DROP_COLS))
    if ORDINAL_FORM_COLS:
        from sklearn.preprocessing import OrdinalEncoder
        form_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(
                categories=[FORM_ORDER] * len(ORDINAL_FORM_COLS),
                handle_unknown="use_encoded_value", unknown_value=-1,
            )),
        ])
        transformers.append(("recent_form", form_pipe, ORDINAL_FORM_COLS))

    # remainder='drop': como a guarda acima garante cobertura total de FEATURE_COLS,
    # o remainder fica necessariamente vazio (nada escapa sem tratamento).
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Métricas (SPEC.md §5.1–5.2)
# ---------------------------------------------------------------------------

def _multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score multiclasse: média de Σ_k (p_k − y_k)². Faixa [0, 2]."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    onehot = np.zeros_like(proba, dtype=float)
    for row, label in enumerate(y_true):
        onehot[row, idx[label]] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def _evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Computa log-loss, acurácia e brier para previsões já alinhadas a CLASSES."""
    preds = [CLASSES[i] for i in proba.argmax(axis=1)]
    return {
        "log_loss": float(log_loss(y_true, proba, labels=CLASSES)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "brier": _multiclass_brier(y_true, proba),
    }


# ---------------------------------------------------------------------------
# Treino + avaliação
# ---------------------------------------------------------------------------

def _temporal_split(df: pd.DataFrame):
    """Particiona o dataset de treino por competição (validação temporal §4.4)."""
    train_df = df[df["competition"].isin(TRAIN_COMPETITIONS)].reset_index(drop=True)
    val_df = df[df["competition"].isin(VAL_COMPETITIONS)].reset_index(drop=True)

    unknown = set(df["competition"].unique()) - TRAIN_COMPETITIONS - VAL_COMPETITIONS
    if unknown:
        raise ValueError(
            f"Competições fora do split temporal definido (§4.4): {sorted(unknown)}. "
            "Atualize TRAIN_COMPETITIONS/VAL_COMPETITIONS."
        )
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Split temporal vazio: treino={len(train_df)} val={len(val_df)}."
        )
    return train_df, val_df


def run() -> dict:
    """Treina o baseline logístico e avalia contra os baselines ingênuos.

    Retorna dict {modelo_logistico, baseline_uniforme, baseline_prior} com as métricas.
    """
    df = build_training_dataset()
    train_df, val_df = _temporal_split(df)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].to_numpy()
    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL].to_numpy()

    # --- (a) Modelo logístico multinomial (§4.1) ---
    # solver='lbfgs' faz classificação multiclasse nativamente; multi_class é deprecated
    # em sklearn >= 1.5 e por isso NÃO é passado.
    clf = Pipeline([
        ("prep", _build_preprocessor()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000)),
    ])
    clf.fit(X_train, y_train)  # fit do scaler/imputers só no treino (§2.4)

    # Alinha as colunas de predict_proba à ordem de CLASSES.
    order = [list(clf.classes_).index(c) for c in CLASSES]
    proba_model = clf.predict_proba(X_val)[:, order]

    # --- (b) Baseline uniforme: [1/3, 1/3, 1/3] (§6.1) ---
    proba_uniform = np.full((len(y_val), len(CLASSES)), 1.0 / len(CLASSES))

    # --- (c) Baseline prior histórico: frequência das classes no TREINO (§6.1) ---
    counts = pd.Series(y_train).value_counts()
    prior = np.array([counts.get(c, 0) / len(y_train) for c in CLASSES], dtype=float)
    proba_prior = np.tile(prior, (len(y_val), 1))

    return {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "prior": {c: round(float(p), 4) for c, p in zip(CLASSES, prior)},
        "modelo_logistico": _evaluate(y_val, proba_model),
        "baseline_uniforme": _evaluate(y_val, proba_uniform),
        "baseline_prior": _evaluate(y_val, proba_prior),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_report(results: dict) -> None:
    rows = [
        ("Modelo logístico", results["modelo_logistico"]),
        ("Baseline uniforme", results["baseline_uniforme"]),
        ("Baseline prior", results["baseline_prior"]),
    ]
    header = f"{'':22} {'log-loss':>10} {'acurácia':>10} {'brier':>10}"
    print("\nValidação temporal — treino: WC2018 + Euro 2020 + AFCON 2023  |  "
          f"val: WC2022 + Euro 2024 + Copa América 2024")
    print(f"(n_treino={results['n_train']}, n_val={results['n_val']}, "
          f"prior_treino={results['prior']})\n")
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:22} {m['log_loss']:>10.4f} {m['accuracy']:>10.4f} {m['brier']:>10.4f}")

    # Critério de sucesso §6: log-loss do modelo < melhor baseline ingênuo.
    best_naive = min(results["baseline_uniforme"]["log_loss"],
                     results["baseline_prior"]["log_loss"])
    model_ll = results["modelo_logistico"]["log_loss"]
    verdict = "✓ supera" if model_ll < best_naive else "✗ NÃO supera"
    print(f"\nCritério §6.1 (log-loss): modelo={model_ll:.4f} vs melhor baseline="
          f"{best_naive:.4f} → {verdict} o baseline ingênuo.")


def main() -> None:
    _print_report(run())


if __name__ == "__main__":
    main()
