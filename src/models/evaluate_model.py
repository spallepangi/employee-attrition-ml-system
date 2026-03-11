"""
Model evaluation: confusion matrix, ROC curve, precision-recall curve, feature importance.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)
from src.config.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix",
) -> np.ndarray:
    """
    Plot and optionally save confusion matrix.

    Returns:
        Confusion matrix array.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["No (0)", "Yes (1)"],
        yticklabels=["No (0)", "Yes (1)"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved confusion matrix to %s", save_path)
    else:
        plt.close()
    return cm


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "ROC Curve",
) -> float:
    """
    Plot ROC curve. y_prob is probability of positive class (1).

    Returns:
        ROC AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved ROC curve to %s", save_path)
    else:
        plt.close()
    return float(roc_auc)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Precision-Recall Curve",
) -> None:
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="green", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved precision-recall curve to %s", save_path)
    plt.close()


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Extract feature importance from tree-based model or logistic regression coefficients.

    Returns:
        Dict mapping feature_name -> importance (absolute for linear).
    """
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Logistic regression: use absolute coefficient
        imp = np.abs(model.coef_).flatten()
    else:
        return {}

    if len(imp) != len(feature_names):
        logger.warning(
            "Feature importance length %d != feature names length %d",
            len(imp),
            len(feature_names),
        )
        feature_names = [f"f{i}" for i in range(len(imp))]
    return dict(zip(feature_names, imp.tolist()))


def plot_feature_importance(
    importance_dict: Dict[str, float],
    save_path: Optional[Path] = None,
    top_n: int = 30,
    title: str = "Feature Importance",
) -> None:
    """Plot horizontal bar chart: most important at top, bars extend right (many features visible)."""
    if not importance_dict:
        logger.warning("No feature importance to plot.")
        return
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    n = len(names)
    # Tall figure so all feature names visible; horizontal bars (value on x-axis)
    fig, ax = plt.subplots(figsize=(9, max(10, n * 0.45)))
    y_pos = np.arange(n)
    ax.barh(y_pos, values, align="center", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    # Most important at top (index 0 at top)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved feature importance to %s", save_path)
    plt.close()


def evaluate_model(
    model: Any,
    X: np.ndarray,
    y_true: np.ndarray,
    feature_names: List[str],
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run full evaluation: confusion matrix, ROC, PR curve, feature importance.
    Optionally save all plots to output_dir.

    Returns:
        Dict of metrics (accuracy, precision, recall, f1, roc_auc).
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = y_pred.astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    logger.info("Evaluation metrics: %s", metrics)

    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(y_true, y_pred, save_path=output_dir / "confusion_matrix.png")
        plot_roc_curve(y_true, y_prob, save_path=output_dir / "roc_curve.png")
        plot_precision_recall_curve(y_true, y_prob, save_path=output_dir / "precision_recall_curve.png")
        imp = get_feature_importance(model, feature_names)
        if imp:
            plot_feature_importance(imp, save_path=output_dir / "feature_importance.png")

    return metrics
