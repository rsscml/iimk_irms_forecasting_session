"""
Data quality checks for forecasting datasets.

This module finds problems BEFORE preprocessing fixes them:

- missing values (per column / per series)
- gaps in the time index (the most common silent error in forecasting)
- duplicate rows for the same (key, date)
- outliers via IQR, Z-score, modified Z-score
- target sign and zero patterns
- dtype inconsistencies

Each function returns a dataframe so the issue list can be filtered, sorted
and turned into a Plotly chart.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value report, sorted by missing count descending."""
    rep = pd.DataFrame({
        "column": df.columns,
        "missing": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(2).values,
        "non_null": df.notna().sum().values,
        "dtype": df.dtypes.astype(str).values,
    })
    return rep.sort_values("missing", ascending=False).reset_index(drop=True)


def missing_by_series(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Per-forecast-key missing count and percentage for the target column.
    Highlights series that are sparse vs. series that are dense.
    """
    target = spec["target_col"]
    key_cols = spec["key_cols"]
    g = df.groupby(key_cols, dropna=False)[target]
    out = pd.DataFrame({
        "n_obs": g.count() + g.apply(lambda s: s.isna().sum()),
        "missing": g.apply(lambda s: int(s.isna().sum())),
    })
    out["missing_pct"] = (out["missing"] / out["n_obs"] * 100).round(2)
    return out.reset_index().sort_values("missing_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time gaps
# ---------------------------------------------------------------------------

def detect_time_gaps(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Identify forecast keys whose date index has gaps relative to the spec's
    declared frequency.

    Returns
    -------
    pd.DataFrame
        One row per key with columns:
        ``[*key_cols, n_obs, expected_obs, missing_dates, first_gap, last_gap, gap_pct]``.

    Notes
    -----
    Frequency strings follow pandas offset aliases: 'D' (daily), 'B'
    (business day), 'W' (weekly), 'MS' (month-start), 'H' (hourly), etc.
    """
    date_col = spec["date_col"]
    key_cols = spec["key_cols"]
    freq = spec["frequency"]

    rows = []
    for keys, sub in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        dates = pd.to_datetime(sub[date_col]).sort_values()
        if dates.empty:
            continue
        full = pd.date_range(dates.min(), dates.max(), freq=freq)
        missing_dates = full.difference(pd.DatetimeIndex(dates))

        rows.append({
            **dict(zip(key_cols, keys)),
            "n_obs": int(len(dates)),
            "expected_obs": int(len(full)),
            "missing_dates": int(len(missing_dates)),
            "gap_pct": round(len(missing_dates) / max(len(full), 1) * 100, 2),
            "first_gap": missing_dates[0] if len(missing_dates) else pd.NaT,
            "last_gap": missing_dates[-1] if len(missing_dates) else pd.NaT,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("missing_dates", ascending=False)
        .reset_index(drop=True)
    )


def gap_examples(df: pd.DataFrame, spec: Dict, key_values, max_examples: int = 20) -> pd.DataFrame:
    """
    For a single forecast key, return the actual missing dates (up to
    ``max_examples``). Helps confirm whether gaps are random or structural
    (e.g. weekends, holidays, year-end).
    """
    from .data_io import get_series

    sub = get_series(df, spec, key_values)
    dates = pd.to_datetime(sub[spec["date_col"]]).sort_values()
    if dates.empty:
        return pd.DataFrame(columns=["missing_date", "weekday"])
    full = pd.date_range(dates.min(), dates.max(), freq=spec["frequency"])
    missing = full.difference(pd.DatetimeIndex(dates))[:max_examples]
    return pd.DataFrame({
        "missing_date": missing,
        "weekday": missing.day_name(),
    })


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def detect_duplicates(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Rows that share the same (forecast key, date). Such rows make the time
    series ill-defined and must be resolved (drop, sum, or average) before
    modelling.
    """
    cols = list(spec["key_cols"]) + [spec["date_col"]]
    mask = df.duplicated(subset=cols, keep=False)
    return df.loc[mask].sort_values(cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Boolean Series — True where the value is outside ``[Q1 - k·IQR, Q3 + k·IQR]``.
    The classic Tukey rule. Robust because it relies on quartiles, not mean/std.
    """
    s = series.astype(float)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (s < lower) | (s > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Boolean Series — True where ``|z| > threshold``. Use cautiously: mean
    and std are themselves pulled by extreme values, so this can under-flag
    in heavy-tailed series.
    """
    s = series.astype(float)
    mu, sigma = s.mean(), s.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(False, index=s.index)
    return ((s - mu).abs() / sigma) > threshold


def detect_outliers_modified_zscore(series: pd.Series, threshold: float = 3.5) -> pd.Series:
    """
    Iglewicz–Hoaglin modified Z-score using median and MAD.

    More robust than the standard Z-score and recommended for skewed
    sales/demand distributions.
    """
    s = series.astype(float)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or pd.isna(mad):
        return pd.Series(False, index=s.index)
    modified_z = 0.6745 * (s - med) / mad
    return modified_z.abs() > threshold


def outlier_summary(
    df: pd.DataFrame,
    spec: Dict,
    method: str = "iqr",
    **method_kwargs,
) -> pd.DataFrame:
    """
    Per-forecast-key outlier counts using the chosen method.

    ``method`` must be one of ``'iqr'``, ``'zscore'``, ``'modified_zscore'``.
    Outlier rules are computed *within each series*, which is the right
    granularity for multi-series forecasting datasets.
    """
    detectors = {
        "iqr": detect_outliers_iqr,
        "zscore": detect_outliers_zscore,
        "modified_zscore": detect_outliers_modified_zscore,
    }
    if method not in detectors:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(detectors)}.")
    fn = detectors[method]

    target = spec["target_col"]
    key_cols = spec["key_cols"]

    rows = []
    for keys, sub in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        flags = fn(sub[target], **method_kwargs)
        rows.append({
            **dict(zip(key_cols, keys)),
            "n_obs": int(len(sub)),
            "n_outliers": int(flags.sum()),
            "outlier_pct": round(float(flags.mean()) * 100, 2),
        })
    return pd.DataFrame(rows).sort_values("n_outliers", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target signal checks
# ---------------------------------------------------------------------------

def target_sign_check(df: pd.DataFrame, spec: Dict) -> Dict:
    """
    Counts of negative, zero, and positive values for the target.

    Negative sales/demand usually indicate returns, refunds or data errors.
    Catching them up front prevents log-transforms and Box-Cox from blowing
    up downstream.
    """
    s = df[spec["target_col"]].dropna()
    return {
        "n_total": int(len(s)),
        "n_negative": int((s < 0).sum()),
        "n_zero": int((s == 0).sum()),
        "n_positive": int((s > 0).sum()),
        "pct_negative": round(float((s < 0).mean()) * 100, 2),
        "pct_zero": round(float((s == 0).mean()) * 100, 2),
    }


# ---------------------------------------------------------------------------
# Static-feature consistency
# ---------------------------------------------------------------------------

def static_feature_consistency(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Verify that each declared static column is in fact constant within
    every forecast key. Returns one row per static column with the count of
    keys where the column is *not* constant — these need attention before
    forward/backward filling.
    """
    static_cols = spec.get("static_cols") or []
    key_cols = spec["key_cols"]
    if not static_cols:
        return pd.DataFrame(columns=["column", "n_keys", "n_inconsistent_keys", "inconsistent_pct"])

    rows = []
    for col in static_cols:
        if col not in df.columns:
            continue
        nun = df.groupby(key_cols, dropna=False)[col].nunique(dropna=True)
        n_keys = len(nun)
        n_bad = int((nun > 1).sum())
        rows.append({
            "column": col,
            "n_keys": n_keys,
            "n_inconsistent_keys": n_bad,
            "inconsistent_pct": round(n_bad / max(n_keys, 1) * 100, 2),
        })
    return pd.DataFrame(rows)
