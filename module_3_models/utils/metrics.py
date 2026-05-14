"""
metrics.py
==========
Forecasting metrics. All functions accept numpy arrays / pandas Series and are
zero-protection-aware so they don't blow up on intermittent demand series.

Recommended pairings
--------------------
Smooth / continuous demand     :  RMSE, MAE, MAPE
Erratic demand                 :  RMSE, MAE, WAPE
Intermittent demand (many 0s)  :  WAPE, MAE, RMSE, bias            (avoid MAPE)
Lumpy demand                   :  WAPE, MAE, Tweedie deviance      (avoid MAPE)
Cross-key aggregation          :  WAPE, MASE, RMSSE                (scale-free)
"""
from __future__ import annotations

from typing import Optional, Union
import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series, list]


def _to_arrays(y_true: ArrayLike, y_pred: ArrayLike):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Point-error metrics
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Classic MAPE. Undefined when y_true == 0; we mask zero rows."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    """Symmetric MAPE. Bounded in [0, 200]. Treats over- and under-prediction
    asymmetrically when y is small; useful but not perfect."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def wape(y_true, y_pred, eps: float = 1e-8) -> float:
    """
    Weighted Absolute Percentage Error  =  sum|err| / sum|y|.

    The default "go-to" intermittent-demand metric: it is well-defined when
    individual y values are zero, scale-free, and aggregable across keys
    weighted by volume.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    if denom < eps:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def bias(y_true, y_pred) -> float:
    """Mean error (signed). Positive => systematic under-forecast."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    return float(np.mean(y_true - y_pred))


def mean_pct_bias(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = _to_arrays(y_true, y_pred)
    denom = np.sum(np.abs(y_true))
    if denom < eps:
        return np.nan
    return float(np.sum(y_true - y_pred) / denom * 100)


# ---------------------------------------------------------------------------
# Scale-free metrics that need an in-sample reference (training history)
# ---------------------------------------------------------------------------
def mase(y_true, y_pred, y_train: ArrayLike, season: int = 1, eps: float = 1e-8) -> float:
    """
    Mean Absolute Scaled Error (Hyndman & Koehler 2006).
    Scales the test MAE by the in-sample MAE of a seasonal naive forecast.
    Comparable across series and well-defined for intermittent demand.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if len(y_train) <= season:
        return np.nan
    naive = np.abs(y_train[season:] - y_train[:-season]).mean()
    if naive < eps:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / naive)


def rmsse(y_true, y_pred, y_train: ArrayLike, season: int = 1, eps: float = 1e-8) -> float:
    """Root Mean Squared Scaled Error  (used in the M5 competition)."""
    y_true, y_pred = _to_arrays(y_true, y_pred)
    y_train = np.asarray(y_train, dtype=float).ravel()
    if len(y_train) <= season:
        return np.nan
    naive = np.mean((y_train[season:] - y_train[:-season]) ** 2)
    if naive < eps:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / naive))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------
def pinball_loss(y_true, y_pred_quantile, q: float) -> float:
    """
    Pinball / quantile loss. y_pred_quantile is the model's q-th quantile
    forecast; q in (0,1).  Average pinball over all quantiles  =  CRPS estimator.
    """
    y_true, y_pred = _to_arrays(y_true, y_pred_quantile)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(y_true, y_lower, y_upper) -> float:
    """Empirical coverage of a prediction interval [y_lower, y_upper]."""
    y_true = np.asarray(y_true).ravel()
    y_lower = np.asarray(y_lower).ravel()
    y_upper = np.asarray(y_upper).ravel()
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


# ---------------------------------------------------------------------------
# Convenience: full metric report
# ---------------------------------------------------------------------------
def metric_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: Optional[ArrayLike] = None,
    season: int = 1,
) -> pd.Series:
    """One-shot summary that handles intermittent series gracefully."""
    out = {
        "MAE":   mae(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "WAPE%": wape(y_true, y_pred),
        "sMAPE%": smape(y_true, y_pred),
        "MAPE%": mape(y_true, y_pred),
        "Bias":  bias(y_true, y_pred),
        "PctBias%": mean_pct_bias(y_true, y_pred),
    }
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, season=season)
        out["RMSSE"] = rmsse(y_true, y_pred, y_train, season=season)
    return pd.Series(out)


def metric_report_by_key(
    df: pd.DataFrame,
    key_cols,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    """Per-forecast-key metric report (handy for diagnosing bad performers)."""
    rows = []
    for keys, g in df.groupby(list(key_cols)):
        row = metric_report(g[y_true_col].values, g[y_pred_col].values).to_dict()
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, v in zip(key_cols, keys):
            row[c] = v
        rows.append(row)
    cols = list(key_cols) + [c for c in rows[0].keys() if c not in key_cols]
    return pd.DataFrame(rows)[cols]
