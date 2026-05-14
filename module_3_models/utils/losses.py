"""
losses.py
=========
Loss-function helpers and explanations for forecasting.

LightGBM ships with several built-in objectives we care about:
    'regression' / 'regression_l2'  -> MSE
    'regression_l1'                 -> MAE
    'huber'                         -> Huber (robust)
    'poisson'                       -> Poisson (count, integer non-negative)
    'tweedie'                       -> Tweedie (zero-inflated, with variance power)
    'quantile'                      -> Pinball loss for a single quantile

This module documents *why* you'd pick each one, and provides a recommendation
helper given a series' demand pattern (Smooth / Erratic / Intermittent / Lumpy).

It also exposes a custom RMSLE objective + metric pair, since LightGBM does not
ship with RMSLE out of the box and it is occasionally useful for non-negative
right-skewed demand.
"""
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Educational catalog of losses
# ---------------------------------------------------------------------------
LOSS_CATALOG: Dict[str, Dict[str, str]] = {
    "regression_l2": {
        "name": "MSE / RMSE",
        "lightgbm_objective": "regression",
        "domain": "y in R",
        "use_when": "Smooth / continuous demand. Symmetric, penalises large errors quadratically.",
        "watch_out": "Very sensitive to outliers; can drive predictions far from zero on lumpy data.",
    },
    "regression_l1": {
        "name": "MAE",
        "lightgbm_objective": "regression_l1",
        "domain": "y in R",
        "use_when": "Robust regression; minimises the median, not the mean. Good when you have heavy-tailed errors.",
        "watch_out": "Gradient is sign(error) — slower convergence than MSE in the central region.",
    },
    "huber": {
        "name": "Huber",
        "lightgbm_objective": "huber",
        "domain": "y in R",
        "use_when": "Compromise between MSE and MAE; quadratic near zero, linear beyond delta.",
        "watch_out": "Requires tuning the alpha (huber delta) parameter.",
    },
    "poisson": {
        "name": "Poisson",
        "lightgbm_objective": "poisson",
        "domain": "y in {0, 1, 2, ...}",
        "use_when": "Count data with mean ~ variance. Predictions are positive by construction (model outputs log-mean).",
        "watch_out": "Overdispersion (variance > mean) is common in retail; switch to Tweedie or Negative Binomial.",
    },
    "tweedie": {
        "name": "Tweedie",
        "lightgbm_objective": "tweedie",
        "domain": "y >= 0",
        "use_when": "Zero-inflated, right-skewed demand (intermittent / lumpy). Tweedie is *the* default for retail forecasting at scale (Walmart M5 winners, Amazon DeepAR alternatives).",
        "watch_out": "Tune `tweedie_variance_power` in (1, 2). 1.0 -> Poisson, 2.0 -> Gamma. Typical retail values: 1.1 - 1.5.",
    },
    "quantile": {
        "name": "Quantile (Pinball)",
        "lightgbm_objective": "quantile",
        "domain": "y in R",
        "use_when": "You want a *probabilistic* forecast. Train one model per quantile (e.g. 0.1, 0.5, 0.9) to get prediction intervals.",
        "watch_out": "Quantile crossing can occur (q=0.9 prediction below q=0.5); post-process by sorting if needed.",
    },
}


def recommend_loss(pattern: str) -> str:
    """Return a recommended LightGBM objective given a Syntetos-Boylan pattern."""
    pattern = pattern.lower()
    if pattern == "smooth":
        return "regression"        # MSE
    if pattern == "erratic":
        return "huber"
    if pattern == "intermittent":
        return "tweedie"
    if pattern == "lumpy":
        return "tweedie"
    raise ValueError(f"Unknown pattern: {pattern}")


# ---------------------------------------------------------------------------
# Custom LightGBM objective: RMSLE
# ---------------------------------------------------------------------------
# LightGBM custom objective signature:
#     def objective(y_true, y_pred) -> (grad, hess)
# where y_pred is the raw score (no link applied).
#
# RMSLE (Root Mean Squared Logarithmic Error) measures relative error and
# punishes under-prediction more than over-prediction. For non-negative
# right-skewed demand it is sometimes preferable to RMSE.
#
# Working in log-space:
#   loss_i = 0.5 * (log(1+y_true_i) - log(1+y_pred_i))**2
# Gradient w.r.t. y_pred:
#   grad_i = -(log1p(y_true) - log1p(y_pred)) / (1 + y_pred)
#   hess_i is approximated as 1/(1+y_pred)**2 (good enough for tree boosting).
# We clip to keep things numerically stable.
def rmsle_objective(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    LightGBM-compatible custom objective for RMSLE.
    Plug into LightGBM:
        params = {'objective': rmsle_objective, ...}
        model = lgb.train(params, train_set, ...)
    """
    y_pred = np.maximum(y_pred, -1 + 1e-6)  # keep 1+y_pred positive
    y_true = np.maximum(y_true, 0.0)
    log_y_pred = np.log1p(y_pred)
    log_y_true = np.log1p(y_true)
    grad = -(log_y_true - log_y_pred) / (1.0 + y_pred)
    hess = 1.0 / (1.0 + y_pred) ** 2
    return grad, hess


def rmsle_eval(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """LightGBM-compatible custom metric for RMSLE."""
    y_pred = np.maximum(y_pred, 0.0)
    y_true = np.maximum(y_true, 0.0)
    val = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    # signature: (eval_name, eval_result, is_higher_better)
    return "rmsle", float(val), False


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def print_loss_catalog() -> None:
    """Print a human-readable summary of supported losses."""
    for key, info in LOSS_CATALOG.items():
        print(f"\n{info['name']}  (LightGBM: '{info['lightgbm_objective']}')")
        print(f"  domain    : {info['domain']}")
        print(f"  use when  : {info['use_when']}")
        print(f"  watch out : {info['watch_out']}")
