"""Análise de erros do XGBoost no conjunto de validação. SPEC.md §5.2.

1. Matriz de confusão 3×3 (predito vs real)
2. Distribuição das probabilidades preditas por classe (confiante ou espalhado?)
3. Reliability diagram — salvo como PNG em data/processed/reliability_diagram.png
4. Top-10 partidas com maior log-loss individual (onde o modelo mais errou)

CLI:
    python -m src.evaluate.error_analysis
    python src/evaluate/error_analysis.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — sem display necessário
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from src.models.baseline import CLASSES
from src.models.xgboost_model import train as xgb_train

_ROOT          = Path(__file__).parents[2]
_PROCESSED_DIR = _ROOT / "data" / "processed"
_RELIABILITY_PNG = _PROCESSED_DIR / "reliability_diagram.png"

# Índice canônico de cada classe (CLASSES é lexicográfico: A_win=0, B_win=1, draw=2)
_CLS_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# 1. Matriz de confusão
# ---------------------------------------------------------------------------

def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Imprime matriz de confusão 3×3 em texto."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)

    col_w = max(len(c) for c in CLASSES) + 2  # largura de coluna uniforme
    print("\n── 1. Matriz de confusão (linhas = real, colunas = predito) ──")
    header = " " * col_w + "".join(f"{c:>{col_w}}" for c in CLASSES)
    print(header)
    print("─" * len(header))
    for i, row_cls in enumerate(CLASSES):
        cells = "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(CLASSES)))
        print(f"{row_cls:>{col_w}}{cells}")

    total   = int(cm.sum())
    correct = int(cm.diagonal().sum())
    print(f"\n  Acurácia: {correct}/{total} = {correct/total:.1%}")
    for cls in CLASSES:
        i = _CLS_IDX[cls]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec  = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"  {cls:<8}  precision={prec:.2f}  recall={rec:.2f}  "
              f"support={int(cm[i, :].sum())}")


# ---------------------------------------------------------------------------
# 2. Distribuição das probabilidades preditas por classe
# ---------------------------------------------------------------------------

def print_prob_distribution(proba: np.ndarray) -> None:
    """Mostra resumo estatístico e histograma de texto das probabilidades por classe."""
    print("\n── 2. Distribuição das probabilidades preditas ──")
    bins = [0.0, 1/3, 2/3, 1.0]
    bin_labels = ["baixa (0–33%)", "média (33–67%)", "alta (67–100%)"]

    col_w = 16
    header = (f"{'classe':<9} {'n':>3}  {'média':>6}  {'mediana':>7}  {'std':>5}  "
              + "  ".join(f"{b:<{col_w}}" for b in bin_labels))
    print(header)
    print("─" * len(header))
    for cls in CLASSES:
        i   = _CLS_IDX[cls]
        p   = proba[:, i]
        cnts, _ = np.histogram(p, bins=bins)
        buckets = "  ".join(f"{c:>2} ({c/len(p):5.1%})" for c in cnts)
        print(f"{cls:<9} {len(p):>3}  {p.mean():>6.3f}  {np.median(p):>7.3f}  "
              f"{p.std():>5.3f}  {buckets}")

    mean_entropy = -np.sum(proba * np.log(np.clip(proba, 1e-15, 1)), axis=1).mean()
    print(f"\n  Entropia média das previsões: {mean_entropy:.4f} "
          f"(max={math.log(len(CLASSES)):.4f}  →  "
          f"{'modelo incerto' if mean_entropy > 0.9 else 'alguma confiança'})")


# ---------------------------------------------------------------------------
# 3. Reliability diagram
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    y_true: np.ndarray,
    proba:  np.ndarray,
    path:   Path = _RELIABILITY_PNG,
) -> None:
    """Gera e salva o reliability diagram (one-vs-rest, uma coluna por classe).

    Pontos acima da diagonal: modelo subestima a probabilidade (subconfiante).
    Pontos abaixo da diagonal: modelo superestima (superconfiante).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.suptitle(
        "Reliability Diagram — XGBoost Tier 1\n"
        "val: WC2022 + Euro 2024 + Copa América 2024 (n=39)",
        fontsize=11,
    )

    colors = {"A_win": "#2196F3", "B_win": "#F44336", "draw": "#4CAF50"}

    for ax, cls in zip(axes, CLASSES):
        y_bin  = (y_true == cls).astype(int)
        y_prob = proba[:, _CLS_IDX[cls]]
        n_pos  = y_bin.sum()

        # Referência perfeita
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfeito")

        if n_pos >= 2:
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_bin, y_prob, n_bins=5, strategy="uniform"
                )
                ax.plot(mean_pred, frac_pos, "o-", color=colors[cls],
                        lw=2, ms=6, label=f"XGBoost (n_pos={n_pos})")
            except ValueError:
                pass
        else:
            ax.text(0.5, 0.5, f"n_pos={n_pos}\n(insuficiente)", ha="center",
                    va="center", fontsize=9, color="gray")

        # Rug plot das probabilidades individuais
        ax.scatter(y_prob, np.full_like(y_prob, -0.04),
                   marker="|", s=60, alpha=0.5, color=colors[cls])

        ax.set_title(cls, fontsize=10, fontweight="bold")
        ax.set_xlabel("Probabilidade predita", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Fração de positivos", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n── 3. Reliability diagram ──")
    print(f"  Salvo em: {path}")
    print("  Leitura: curva acima da diagonal = modelo subestima (conservador);")
    print("           curva abaixo = superestima (superconfiante).")


# ---------------------------------------------------------------------------
# 4. Top-N partidas com maior log-loss individual
# ---------------------------------------------------------------------------

def print_worst_predictions(
    val_df: pd.DataFrame,
    y_val:  np.ndarray,
    proba:  np.ndarray,
    top_n:  int = 10,
) -> None:
    """Lista as `top_n` partidas onde o modelo teve maior penalidade de log-loss."""
    individual_ll = np.array([
        -math.log(max(proba[i, _CLS_IDX[y_val[i]]], 1e-15))
        for i in range(len(y_val))
    ])
    worst_idx = np.argsort(individual_ll)[::-1][:top_n]

    print(f"\n── 4. Top-{top_n} partidas com maior log-loss individual ──")
    hdr = (f"{'#':>2}  {'data':<11} {'fase':<10} "
           f"{'time A':<20} {'time B':<20} "
           f"{'real':<8}  {'P(Aw)':>6} {'P(Bw)':>6} {'P(dr)':>6}  log-loss")
    print(hdr)
    print("─" * len(hdr))

    stage_abbr = {
        "Round of 16": "R16", "Quarter-finals": "QF", "Semi-finals": "SF",
        "Final": "F", "3rd Place Final": "3rd",
        # football-data.org (val set vem do StatsBomb — estágios em inglês)
        "LAST_16": "R16", "QUARTER_FINALS": "QF", "SEMI_FINALS": "SF",
        "FINAL": "F", "THIRD_PLACE": "3rd",
    }

    for rank, idx in enumerate(worst_idx, 1):
        row   = val_df.iloc[idx]
        p     = proba[idx]
        ll    = individual_ll[idx]
        stage = stage_abbr.get(str(row["stage"]), str(row["stage"])[:8])
        date  = str(row["match_date"])[:10]
        print(
            f"{rank:>2}  {date:<11} {stage:<10} "
            f"{str(row['team_A']):<20} {str(row['team_B']):<20} "
            f"{y_val[idx]:<8}  "
            f"{p[_CLS_IDX['A_win']]:>6.3f} {p[_CLS_IDX['B_win']]:>6.3f} "
            f"{p[_CLS_IDX['draw']]:>6.3f}  {ll:>7.4f}"
        )

    print(f"\n  Log-loss médio total: {individual_ll.mean():.4f}")
    print(f"  Log-loss mediano:      {float(np.median(individual_ll)):.4f}")
    print(f"  Log-loss máximo:       {individual_ll.max():.4f}  "
          f"(jogo #{np.argmax(individual_ll)+1} acima)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("Treinando XGBoost (GridSearchCV)...")
    result  = xgb_train()
    y_val   = result["y_val"]
    proba   = result["proba"]
    val_df  = result["val_df"]
    y_pred  = np.array([CLASSES[i] for i in proba.argmax(axis=1)])

    p = result["best_params"]
    print(f"  params: n_estimators={p['n_estimators']} max_depth={p['max_depth']} "
          f"lr={p['learning_rate']}  CV log-loss={result['cv_log_loss']:.4f}")

    print_confusion_matrix(y_val, y_pred)
    print_prob_distribution(proba)
    plot_reliability_diagram(y_val, proba)
    print_worst_predictions(val_df, y_val, proba)


if __name__ == "__main__":
    main()
