"""
explainer.py
============
Explainability helpers for LightGBM and Prophet forecasters.

LightGBM
--------
We expose three views of "importance":
    1. Gain      : standard LightGBM gain importance (sum of split gains).
    2. Split     : how often a feature is used to split.
    3. SHAP      : Shapley value attributions, both global (mean |shap|) and
                   local (per-row, used for waterfall / force-plot style
                   explanations of a single forecast).

For tree models, SHAP is computed exactly via TreeSHAP — fast and theoretically
sound.

Prophet
-------
Prophet decomposes its forecast into trend + seasonality + holidays + extra
regressors. We expose the decomposition as a tidy DataFrame ready for Plotly
charts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import shap
except ImportError:                              # pragma: no cover
    shap = None


# ---------------------------------------------------------------------------
# LightGBM importance / SHAP
# ---------------------------------------------------------------------------
class LGBMExplainer:
    """
    Wraps a fitted LightGBM Booster and exposes consistent global / local
    explanation outputs.

    Usage:
        ex = LGBMExplainer(booster, feature_names)
        importance_df = ex.importance(kind="gain")
        shap_df = ex.shap_summary(X_sample)            # global mean |shap|
        local = ex.shap_local(X_sample, row_index=0)   # one-row attributions
    """

    def __init__(self, booster: lgb.Booster, feature_names: Optional[Sequence[str]] = None):
        self.booster = booster
        self.feature_names = list(feature_names or booster.feature_name())
        self._explainer = None

    # ---- gain / split importance ------------------------------------------
    def importance(self, kind: str = "gain", normalize: bool = True) -> pd.DataFrame:
        if kind not in ("gain", "split"):
            raise ValueError("kind must be 'gain' or 'split'")
        vals = self.booster.feature_importance(importance_type=kind)
        df = pd.DataFrame({"feature": self.feature_names, "importance": vals})
        if normalize and df["importance"].sum() > 0:
            df["importance_pct"] = df["importance"] / df["importance"].sum() * 100
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    # ---- SHAP --------------------------------------------------------------
    def _ensure_shap(self):
        if shap is None:
            raise ImportError("Install shap:  pip install shap")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.booster)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw SHAP values (n_rows, n_features)."""
        self._ensure_shap()
        # LightGBM TreeExplainer returns ndarray for single-output regression
        sv = self._explainer.shap_values(X[self.feature_names])
        return np.asarray(sv)

    def shap_summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature: global importance with sign-aware view."""
        sv = self.shap_values(X)
        return (pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
            "mean_shap":     sv.mean(axis=0),                  # signed
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True))

    def shap_local(self, X: pd.DataFrame, row_index: int = 0) -> pd.DataFrame:
        """Per-feature contribution to a single prediction."""
        sv = self.shap_values(X.iloc[[row_index]])
        feature_values = X.iloc[row_index][self.feature_names].values
        # Expected value (base prediction)
        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).ravel()[0])
        df = pd.DataFrame({
            "feature": self.feature_names,
            "value": feature_values,
            "shap": sv.ravel(),
        }).sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)
        df.attrs["base_value"] = base
        df.attrs["prediction"] = base + sv.sum()
        return df

    # ---- Partial dependence (1-D), via shap dependence ---------------------
    def shap_dependence(self, X: pd.DataFrame, feature: str) -> pd.DataFrame:
        sv = self.shap_values(X)
        idx = self.feature_names.index(feature)
        return pd.DataFrame({
            feature: X[feature].values,
            "shap": sv[:, idx],
        })


# ---------------------------------------------------------------------------
# Prophet decomposition
# ---------------------------------------------------------------------------
def prophet_decomposition(prophet_model, key_label: str = "") -> pd.DataFrame:
    """
    Pull a tidy decomposition out of a fitted Prophet model. Includes the
    in-sample fit and a small "tail" of fitted values; pair with .predict() for
    out-of-sample components.
    """
    history = prophet_model.history.copy()
    forecast = prophet_model.predict(history[["ds"] + [c for c in history.columns
                                              if c not in ("ds", "y", "floor")]])
    cols = [c for c in ["ds", "trend", "yearly", "weekly", "daily",
                        "holidays", "additive_terms", "yhat"] if c in forecast.columns]
    out = forecast[cols].copy()
    out["y"] = history["y"].values
    if key_label:
        out["key"] = key_label
    return out
