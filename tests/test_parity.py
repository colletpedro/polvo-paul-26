"""Gate de paridade treino↔inferência — SPEC.md §2.4.

Verifica que build_training_dataset() e build_inference_dataset() produzem
exatamente as mesmas feature columns (nome e ordem), conforme o checklist:

    'A lista de colunas do dataset de treino é idêntica à do dataset de
     inferência (mesmo nome, mesma ordem). Assertar em código.'
"""
import pytest

from src.features.tier1 import (
    FEATURE_COLS,
    TARGET_COL,
    build_inference_dataset,
    build_training_dataset,
)


@pytest.fixture(scope="module")
def train_df():
    return build_training_dataset()


@pytest.fixture(scope="module")
def infer_df():
    return build_inference_dataset()


# ---------------------------------------------------------------------------
# 1. Feature columns — nome e ordem idênticos
# ---------------------------------------------------------------------------

def test_feature_columns_identical(train_df, infer_df):
    """SPEC.md §2.4 — colunas de feature iguais em nome e ordem nos dois datasets."""
    train_feat = [c for c in train_df.columns if c in FEATURE_COLS]
    infer_feat = [c for c in infer_df.columns if c in FEATURE_COLS]

    if train_feat != infer_feat:
        only_train = sorted(set(train_feat) - set(infer_feat))
        only_infer = sorted(set(infer_feat) - set(train_feat))
        common_order_diff = [
            (i, t, iv)
            for i, (t, iv) in enumerate(zip(train_feat, infer_feat))
            if t != iv
        ]
        msg = (
            "\n\nParidade treino↔inferência violada (SPEC.md §2.4):"
            f"\n  Só no treino  : {only_train or 'nenhuma'}"
            f"\n  Só inferência : {only_infer or 'nenhuma'}"
            f"\n  Ordem diverge : {common_order_diff or 'nenhuma'}"
        )
        pytest.fail(msg)


def test_feature_columns_match_canonical(train_df, infer_df):
    """Ambos os datasets produzem exatamente FEATURE_COLS na ordem canônica."""
    train_feat = [c for c in train_df.columns if c in FEATURE_COLS]
    infer_feat = [c for c in infer_df.columns if c in FEATURE_COLS]

    if train_feat != FEATURE_COLS:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(train_feat, FEATURE_COLS)) if a != b]
        missing = sorted(set(FEATURE_COLS) - set(train_feat))
        extra   = sorted(set(train_feat)  - set(FEATURE_COLS))
        pytest.fail(
            f"\nTreino diverge de FEATURE_COLS:"
            f"\n  missing={missing}, extra={extra}, ordem_diff={diff}"
        )

    if infer_feat != FEATURE_COLS:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(infer_feat, FEATURE_COLS)) if a != b]
        missing = sorted(set(FEATURE_COLS) - set(infer_feat))
        extra   = sorted(set(infer_feat)  - set(FEATURE_COLS))
        pytest.fail(
            f"\nInferência diverge de FEATURE_COLS:"
            f"\n  missing={missing}, extra={extra}, ordem_diff={diff}"
        )


# ---------------------------------------------------------------------------
# 2. Target presente em ambos
# ---------------------------------------------------------------------------

def test_target_column_present(train_df, infer_df):
    """result_90min existe nos dois datasets."""
    assert TARGET_COL in train_df.columns,  f"'{TARGET_COL}' ausente no treino"
    assert TARGET_COL in infer_df.columns,  f"'{TARGET_COL}' ausente na inferência"


def test_target_not_in_feature_cols():
    """Target não contamina o vetor de features (leakage guard)."""
    assert TARGET_COL not in FEATURE_COLS, (
        f"'{TARGET_COL}' está em FEATURE_COLS — isso causa leakage no modelo."
    )


# ---------------------------------------------------------------------------
# 3. Sanidade básica dos dados
# ---------------------------------------------------------------------------

def test_training_has_rows(train_df):
    assert len(train_df) > 0, "build_training_dataset() retornou DataFrame vazio"


def test_inference_has_rows(infer_df):
    assert len(infer_df) > 0, "build_inference_dataset() retornou DataFrame vazio"


def test_training_target_no_nulls(train_df):
    """Partidas históricas sempre têm resultado conhecido."""
    nulls = train_df[TARGET_COL].isna().sum()
    assert nulls == 0, f"{nulls} linhas do treino com result_90min nulo"


def test_inference_target_nullable(infer_df):
    """Inferência pode ter NaN no target (partidas futuras) — mas a coluna deve existir."""
    assert TARGET_COL in infer_df.columns
    # Partidas FINISHED devem ter target preenchido
    finished = infer_df[infer_df["status"] == "FINISHED"]
    if len(finished) > 0:
        nulls = finished[TARGET_COL].isna().sum()
        assert nulls == 0, f"{nulls} partidas FINISHED sem result_90min"
