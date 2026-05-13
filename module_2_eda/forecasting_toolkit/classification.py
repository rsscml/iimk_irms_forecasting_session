"""
Demand classification (Syntetos–Boylan) for forecasting datasets.

The Syntetos–Boylan scheme partitions every time series into one of four
quadrants based on two summary numbers:

- ADI (Average Demand Interval): the average gap, in periods, between
  non-zero observations. Series with frequent demand have ADI ≈ 1; series
  with sporadic demand have ADI » 1.

- CV² (squared Coefficient of Variation of non-zero demand): how variable
  the *size* of demand is, conditional on it being non-zero.

The four quadrants:

           CV² < 0.49        CV² ≥ 0.49
ADI<1.32  Smooth            Erratic
ADI≥1.32  Intermittent      Lumpy

Forecasting choice depends heavily on this classification — Croston-type
methods for intermittent/lumpy demand, ARIMA / ETS-type methods for
smooth/erratic demand.

Reference: Syntetos, A.A. & Boylan, J.E. (2005). "The accuracy of
intermittent demand estimates." *International Journal of Forecasting*,
21(2), 303–314.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

# Standard Syntetos–Boylan thresholds (the "SBC" cut-offs)
ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49

CATEGORIES = ["smooth", "intermittent", "erratic", "lumpy"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def compute_adi(series: pd.Series) -> float:
    """
    Average Demand Interval: total periods divided by the number of non-zero
    periods.

    Returns ``np.inf`` if the series has no non-zero observations and
    ``np.nan`` if the series itself is empty.
    """
    s = series.dropna()
    n_total = len(s)
    n_nonzero = int((s != 0).sum())
    if n_total == 0:
        return np.nan
    if n_nonzero == 0:
        return np.inf
    return float(n_total / n_nonzero)


def compute_cv2(series: pd.Series) -> float:
    """
    Squared Coefficient of Variation of *non-zero* demand only.

    Using only non-zero values isolates demand-size variability from
    demand-occurrence variability (which ADI already captures).
    """
    s = series.dropna()
    nz = s[s != 0]
    if len(nz) < 2 or nz.mean() == 0:
        return np.nan
    return float((nz.std() / nz.mean()) ** 2)


def sb_classify(adi: float, cv2: float,
                adi_threshold: float = ADI_THRESHOLD,
                cv2_threshold: float = CV2_THRESHOLD) -> str:
    """
    Map an (ADI, CV²) pair to one of {smooth, intermittent, erratic, lumpy}.

    Returns ``'undefined'`` if either input is NaN, and ``'no_demand'`` if
    ADI is infinite (the series is all zeros).
    """
    if pd.isna(adi) or pd.isna(cv2):
        return "undefined"
    if np.isinf(adi):
        return "no_demand"
    if adi < adi_threshold and cv2 < cv2_threshold:
        return "smooth"
    if adi >= adi_threshold and cv2 < cv2_threshold:
        return "intermittent"
    if adi < adi_threshold and cv2 >= cv2_threshold:
        return "erratic"
    return "lumpy"


# ---------------------------------------------------------------------------
# Bulk classification
# ---------------------------------------------------------------------------

def classify_all_series(
    df: pd.DataFrame,
    spec: Dict,
    adi_threshold: float = ADI_THRESHOLD,
    cv2_threshold: float = CV2_THRESHOLD,
) -> pd.DataFrame:
    """
    Classify every forecast key in the dataset.

    Returns
    -------
    pd.DataFrame
        Columns: the spec's ``key_cols``, plus ``adi``, ``cv2``, ``n_obs``,
        ``zero_share``, and ``sb_class``.
    """
    target = spec["target_col"]
    key_cols = spec["key_cols"]

    rows: List[Dict] = []
    for keys, sub in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = sub[target]
        adi = compute_adi(s)
        cv2 = compute_cv2(s)
        rows.append({
            **dict(zip(key_cols, keys)),
            "n_obs": int(s.notna().sum()),
            "zero_share": float((s == 0).mean()) if len(s) else np.nan,
            "adi": adi,
            "cv2": cv2,
            "sb_class": sb_classify(adi, cv2, adi_threshold, cv2_threshold),
        })
    return pd.DataFrame(rows)


def classification_summary(classification_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts and percentages of each Syntetos–Boylan category.
    """
    counts = (
        classification_df["sb_class"]
        .value_counts(dropna=False)
        .rename_axis("sb_class")
        .reset_index(name="n_series")
    )
    counts["pct"] = (counts["n_series"] / counts["n_series"].sum() * 100).round(2)
    return counts


def recommend_strategy(sb_class: str) -> str:
    """
    Plain-language recommendation for which family of forecasting models
    typically suits each Syntetos–Boylan category. Educational only.
    """
    table = {
        "smooth":
            "Classical methods work well: ETS, ARIMA, regression with trend "
            "and seasonality. Standard error metrics (MAE/RMSE) are reliable.",
        "intermittent":
            "Use Croston, SBA (Syntetos–Boylan Approximation), TSB or "
            "compound Poisson models. Avoid plain MAPE — many actuals are zero.",
        "erratic":
            "High demand-size variance with regular occurrence: try robust "
            "regression, log/Box-Cox transforms before ETS/ARIMA, or quantile "
            "forecasting to protect against outliers.",
        "lumpy":
            "The hardest category — irregular timing AND irregular size. "
            "Croston/SBA is the safe baseline; consider hierarchical "
            "aggregation upward to a smoother level for forecasting.",
        "no_demand":
            "Series has only zero values. Forecast = 0 unless external "
            "information (new product launch, planned promotion) suggests "
            "otherwise.",
        "undefined":
            "Insufficient non-zero observations to classify reliably.",
    }
    return table.get(sb_class, "Unknown class.")
