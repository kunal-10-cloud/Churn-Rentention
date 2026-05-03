"""Per-customer churn explainability using SHAP.

Exposes two helpers used by the agent and the Streamlit UI:
    - explain_prediction(...): returns the top-k contributing features for a
      single customer as plain dicts (feature, value, shap_value, direction).
    - generate_shap_plot(...): returns a matplotlib Figure (waterfall) for UI.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np

try:
    import shap
except ImportError:
    shap = None

import matplotlib.pyplot as plt


def _ensure_shap():
    if shap is None:
        raise ImportError(
            "shap is not installed. Add `shap` to requirements.txt and install."
        )


def _build_explainer(model, background: Optional[np.ndarray], feature_names: List[str]):
    """Pick a SHAP explainer appropriate for the given model."""
    _ensure_shap()

    if background is None or len(background) == 0:
        background = np.zeros((1, len(feature_names)))

    if hasattr(model, "estimators_") or model.__class__.__name__ in {
        "RandomForestClassifier", "GradientBoostingClassifier",
        "XGBClassifier", "LGBMClassifier", "DecisionTreeClassifier",
    }:
        return shap.TreeExplainer(model, background)

    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, background)

    predict_fn = getattr(model, "predict_proba", model.predict)
    return shap.KernelExplainer(predict_fn, background[:100])


def _extract_positive_class_values(shap_values) -> np.ndarray:
    """Return a 1-D array of SHAP values for the positive (churn=1) class."""
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[0, :, 1] if values.shape[-1] > 1 else values[0, :, 0]
    elif values.ndim == 2:
        if values.shape[0] == 1:
            values = values[0]
        else:
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]
    return values


def explain_prediction(
    model,
    customer_row: np.ndarray,
    feature_names: List[str],
    background: Optional[np.ndarray] = None,
    top_k: int = 5,
) -> List[Dict]:
    """Return the top-k SHAP contributors for a single customer.

    Args:
        model: fitted sklearn-style classifier.
        customer_row: 1-D array in the same feature space the model was trained on.
        feature_names: list of feature names aligned with customer_row.
        background: optional background matrix for the explainer.
        top_k: number of features to return.

    Returns:
        List of dicts with keys: feature, value, shap_value, direction.
        Sorted by |shap_value| descending.
    """
    _ensure_shap()

    row = np.asarray(customer_row).reshape(1, -1)
    explainer = _build_explainer(model, background, feature_names)
    sv = explainer.shap_values(row) if not callable(getattr(explainer, "__call__", None)) else explainer(row)
    values = _extract_positive_class_values(sv)

    order = np.argsort(np.abs(values))[::-1][:top_k]
    out = []
    for idx in order:
        shap_val = float(values[idx])
        out.append({
            "feature": feature_names[idx],
            "value": float(row[0, idx]),
            "shap_value": shap_val,
            "direction": "increases churn risk" if shap_val > 0 else "decreases churn risk",
        })
    return out


def generate_shap_plot(
    model,
    customer_row: np.ndarray,
    feature_names: List[str],
    background: Optional[np.ndarray] = None,
    top_k: int = 10,
):
    """Return a matplotlib Figure summarising SHAP contributions for one row."""
    _ensure_shap()

    row = np.asarray(customer_row).reshape(1, -1)
    explainer = _build_explainer(model, background, feature_names)
    sv = explainer.shap_values(row) if not callable(getattr(explainer, "__call__", None)) else explainer(row)
    values = _extract_positive_class_values(sv)

    order = np.argsort(np.abs(values))[::-1][:top_k]
    names = [feature_names[i] for i in order][::-1]
    vals = [values[i] for i in order][::-1]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(vals) + 1))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title("Per-customer feature attribution")
    fig.tight_layout()
    return fig


def format_shap_for_prompt(top_factors: List[Dict]) -> str:
    """Render SHAP factor list as a compact string for LLM prompts."""
    if not top_factors:
        return "No SHAP explanation available."
    lines = []
    for f in top_factors:
        lines.append(
            f"- {f['feature']} (value={f['value']:.3g}, "
            f"shap={f['shap_value']:+.3f}, {f['direction']})"
        )
    return "\n".join(lines)
