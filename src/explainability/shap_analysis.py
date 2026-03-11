"""
SHAP-based explainability: global importance, local explanations, summary and dependence plots.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_explainer(
    model: Any,
    X_background: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> shap.Explainer:
    """
    Create SHAP explainer. Uses TreeExplainer for tree models, else KernelExplainer with a sample.

    Args:
        model: Fitted model (predict_proba required for KernelExplainer).
        X_background: Background dataset for SHAP (e.g. X_train); can be a sample.
        feature_names: Optional feature names for plots.
    """
    if hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model, X_background, feature_names=feature_names)
            logger.info("Using TreeExplainer")
            return explainer
        except Exception as e:
            logger.warning("TreeExplainer failed (%s), falling back to KernelExplainer", e)

    # KernelExplainer needs a small background
    if len(X_background) > 100:
        idx = np.random.RandomState(42).choice(len(X_background), 100, replace=False)
        X_background = X_background[idx]
    explainer = shap.KernelExplainer(
        model.predict_proba,
        X_background,
    )
    logger.info("Using KernelExplainer")
    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
) -> np.ndarray:
    """
    Compute SHAP values for X. For binary classification returns values for positive class.
    """
    raw = explainer(X)
    if isinstance(raw, list):
        raw = raw[1]  # positive class
    # KernelExplainer returns Explanation; extract .values for numpy
    if hasattr(raw, "values"):
        return np.array(raw.values)
    return np.array(raw)


def plot_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    max_display: int = 20,
) -> None:
    """Generate SHAP summary plot (beeswarm). Saves with layout that fits dashboards."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_fig
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=max_display,
    )
    plt_fig.tight_layout(pad=1.2)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt_fig.savefig(save_path, dpi=120, bbox_inches="tight", pad_inches=0.4)
        plt_fig.close()
        logger.info("Saved SHAP summary plot to %s", save_path)


def plot_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    feature_index: Union[int, str],
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    SHAP dependence plot for one feature.
    feature_index: column index (int) or feature name (str).
    """
    if isinstance(feature_index, str):
        if feature_index not in feature_names:
            logger.warning("Feature %s not in feature_names", feature_index)
            return
        feature_index = feature_names.index(feature_index)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_fig
    shap.dependence_plot(
        feature_index,
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
    )
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt_fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt_fig.close()
        logger.info("Saved SHAP dependence plot to %s", save_path)


def run_shap_analysis(
    model: Any,
    X_train: np.ndarray,
    X_explain: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    top_dependence: int = 3,
) -> np.ndarray:
    """
    Run full SHAP analysis: explainer, summary plot, and dependence plots for top features.

    Args:
        model: Fitted model.
        X_train: Training data (used as background).
        X_explain: Data to explain (default: X_train, optionally sampled).
        feature_names: Feature names for plots.
        output_dir: Directory to save plots.
        top_dependence: Number of top features to plot dependence for.

    Returns:
        SHAP values for X_explain.
    """
    if X_explain is None:
        X_explain = X_train
    if len(X_explain) > 500:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_explain), 500, replace=False)
        X_explain = X_explain[idx]

    explainer = get_explainer(model, X_train, feature_names=feature_names)
    shap_values = compute_shap_values(explainer, X_explain)

    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_summary(shap_values, X_explain, feature_names=feature_names, save_path=output_dir / "shap_summary.png")

        # Top features by mean |SHAP|
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:top_dependence]
        for i, idx in enumerate(top_indices):
            name = feature_names[idx] if feature_names else str(idx)
            plot_dependence(
                shap_values,
                X_explain,
                feature_names or [str(j) for j in range(shap_values.shape[1])],
                idx,
                save_path=output_dir / f"shap_dependence_{name.replace(' ', '_')}.png",
            )

    return shap_values


def get_local_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    sample_index: int = 0,
    top_k: int = 10,
) -> List[dict]:
    """
    Get local explanation for one sample: top features by |SHAP| and their values.
    """
    row = shap_values[sample_index]
    indices = np.argsort(np.abs(row))[::-1][:top_k]
    return [
        {"feature": feature_names[i], "shap_value": float(row[i])}
        for i in indices
    ]
