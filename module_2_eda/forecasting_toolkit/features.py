"""
Feature engineering and transformation utilities.

Functions in this module build the predictor matrix that downstream
forecasting models consume. They cover:

- date / calendar features
- lag and rolling-window features
- numeric transformations (log, Box-Cox, Yeo-Johnson)
- scaling (Standard, MinMax, Robust)
- encoding (label, ordinal, one-hot, frequency, target)

Every leak-prone operation (target encoding, scaling) supports a fit/transform
split so the same parameters can be reused on the validation/test sets.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Calendar / date features
# ---------------------------------------------------------------------------

def add_date_features(
    df: pd.DataFrame,
    spec: Dict,
    features: Optional[List[str]] = None,
    cyclical: bool = True,
) -> pd.DataFrame:
    """
    Add calendar attributes derived from the spec's date column.

    Parameters
    ----------
    features : list, optional
        Subset of {'year', 'quarter', 'month', 'week', 'day',
        'dayofweek', 'dayofyear', 'is_weekend', 'is_month_start',
        'is_month_end', 'is_quarter_start', 'is_quarter_end',
        'is_year_start', 'is_year_end'}. ``None`` adds them all.
    cyclical : bool, default True
        If True, also encodes month / dayofweek / day-of-year as
        ``sin/cos`` pairs so models see December and January as adjacent.
    """
    available = [
        "year", "quarter", "month", "week", "day",
        "dayofweek", "dayofyear",
        "is_weekend", "is_month_start", "is_month_end",
        "is_quarter_start", "is_quarter_end",
        "is_year_start", "is_year_end",
    ]
    features = features or available

    out = df.copy()
    dt = pd.to_datetime(out[spec["date_col"]])

    builders = {
        "year": dt.dt.year,
        "quarter": dt.dt.quarter,
        "month": dt.dt.month,
        "week": dt.dt.isocalendar().week.astype(int),
        "day": dt.dt.day,
        "dayofweek": dt.dt.dayofweek,
        "dayofyear": dt.dt.dayofyear,
        "is_weekend": dt.dt.dayofweek.isin([5, 6]).astype(int),
        "is_month_start": dt.dt.is_month_start.astype(int),
        "is_month_end": dt.dt.is_month_end.astype(int),
        "is_quarter_start": dt.dt.is_quarter_start.astype(int),
        "is_quarter_end": dt.dt.is_quarter_end.astype(int),
        "is_year_start": dt.dt.is_year_start.astype(int),
        "is_year_end": dt.dt.is_year_end.astype(int),
    }
    for feat in features:
        if feat in builders:
            out[feat] = builders[feat].values

    if cyclical:
        for col, period in [("month", 12), ("dayofweek", 7), ("dayofyear", 365)]:
            if col in out.columns:
                out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
                out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


# ---------------------------------------------------------------------------
# Lag and rolling features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    spec: Dict,
    column: Optional[str] = None,
    lags: Iterable[int] = (1, 7, 14, 28),
) -> pd.DataFrame:
    """
    Add lagged copies of a column, computed within each forecast key.

    Lag features are the bedrock of supervised-style ML for time series:
    a model predicting demand at *t* can use demand at *t-1*, *t-7*, etc.
    as inputs.
    """
    column = column or spec["target_col"]
    out = df.copy().sort_values(list(spec["key_cols"]) + [spec["date_col"]])
    g = out.groupby(spec["key_cols"], dropna=False)[column]
    for L in lags:
        out[f"{column}_lag_{L}"] = g.shift(L)
    return out.reset_index(drop=True)


def add_rolling_features(
    df: pd.DataFrame,
    spec: Dict,
    column: Optional[str] = None,
    windows: Iterable[int] = (7, 14, 28),
    stats: Iterable[str] = ("mean", "std", "min", "max"),
    shift: int = 1,
) -> pd.DataFrame:
    """
    Add rolling-window statistics computed within each forecast key.

    The window is shifted by ``shift`` periods (default 1) BEFORE being
    aggregated, so the value at time *t* is computed from window
    ``[t-shift-W+1, t-shift]``. This avoids using the current observation
    as a feature for itself (target leakage).
    """
    column = column or spec["target_col"]
    out = df.copy().sort_values(list(spec["key_cols"]) + [spec["date_col"]])
    g = out.groupby(spec["key_cols"], dropna=False)[column]
    shifted = g.shift(shift)

    for W in windows:
        roll = shifted.rolling(W, min_periods=1)
        for stat in stats:
            colname = f"{column}_roll_{stat}_{W}"
            if stat == "mean":
                out[colname] = roll.mean().values
            elif stat == "std":
                out[colname] = roll.std().values
            elif stat == "min":
                out[colname] = roll.min().values
            elif stat == "max":
                out[colname] = roll.max().values
            elif stat == "median":
                out[colname] = roll.median().values
            else:
                raise ValueError(f"Unknown stat: {stat}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Numeric transformations
# ---------------------------------------------------------------------------

def log_transform(s: pd.Series, plus_one: bool = True) -> pd.Series:
    """
    log(x) or log(1+x). ``plus_one=True`` is the standard choice when zero
    is a valid value, since log(0) is undefined.
    """
    if plus_one:
        return np.log1p(s)
    return np.log(s)


def boxcox_transform(s: pd.Series) -> Tuple[pd.Series, float]:
    """
    Box–Cox transform on *strictly positive* data. Returns ``(transformed, lambda)``.
    Use ``inverse_boxcox`` with the same lambda to undo it.
    """
    from scipy.stats import boxcox
    s = s.astype(float)
    if (s <= 0).any():
        raise ValueError("Box–Cox requires all values > 0. "
                         "Use Yeo–Johnson (yj_transform) for non-positive data.")
    transformed, lam = boxcox(s.values)
    return pd.Series(transformed, index=s.index), float(lam)


def inverse_boxcox(s: pd.Series, lam: float) -> pd.Series:
    """Inverse of Box–Cox given the lambda parameter."""
    if abs(lam) < 1e-12:
        return np.exp(s)
    return (s * lam + 1) ** (1.0 / lam)


def yj_transform(s: pd.Series) -> Tuple[pd.Series, float]:
    """Yeo–Johnson transform — works with zero and negative values too."""
    from sklearn.preprocessing import PowerTransformer
    pt = PowerTransformer(method="yeo-johnson", standardize=False)
    arr = pt.fit_transform(s.values.reshape(-1, 1)).ravel()
    return pd.Series(arr, index=s.index), float(pt.lambdas_[0])


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scaler(values: np.ndarray, method: str = "standard"):
    """
    Fit a scaler on a 1-D or 2-D array and return the fitted object.

    Methods
    -------
    'standard'  — zero mean, unit variance
    'minmax'    — rescaled to [0, 1]
    'robust'    — uses median and IQR; resistant to outliers
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    scalers = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }
    if method not in scalers:
        raise ValueError(f"Unknown scaler '{method}'. Choose from {list(scalers)}.")
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    sc = scalers[method]()
    sc.fit(arr)
    return sc


def apply_scaler(values: np.ndarray, scaler) -> np.ndarray:
    """Apply a previously-fitted scaler. Always returns a 1-D array."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return scaler.transform(arr).ravel()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def label_encode(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
    """
    Map each unique value in each column to an integer code.

    Returns
    -------
    (encoded_df, mappings)
        ``mappings[col] = {value: code}`` so encodings can be reapplied later
        on the validation/test split.
    """
    out = df.copy()
    mappings: Dict[str, Dict] = {}
    for c in cols:
        cats = pd.Series(out[c].astype("object").unique()).dropna()
        mapping = {v: i for i, v in enumerate(sorted(cats.tolist(), key=str))}
        mappings[c] = mapping
        out[c] = out[c].map(mapping).astype("Int64")
    return out, mappings


def apply_label_encoding(df: pd.DataFrame, mappings: Dict[str, Dict],
                         unknown: int = -1) -> pd.DataFrame:
    """
    Apply previously-fitted label encodings. Unknown categories are mapped
    to ``unknown`` (default -1) so the schema stays stable on new data.
    """
    out = df.copy()
    for c, mapping in mappings.items():
        if c in out.columns:
            out[c] = out[c].map(mapping).fillna(unknown).astype("Int64")
    return out


def one_hot_encode(df: pd.DataFrame, cols: List[str],
                   drop_first: bool = False) -> pd.DataFrame:
    """
    Standard one-hot expansion. ``drop_first=True`` drops the first level
    of each variable to avoid the dummy-variable trap in linear models.
    """
    return pd.get_dummies(df, columns=cols, drop_first=drop_first, dtype=int)


def frequency_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Replace each category with the number of times it appears in the column.

    Cheap, leak-free, and often surprisingly effective for tree-based models.
    """
    out = df.copy()
    for c in cols:
        counts = out[c].value_counts(dropna=False)
        out[f"{c}_freq"] = out[c].map(counts).astype("Int64")
    return out


def target_encode(
    train_df: pd.DataFrame,
    cols: List[str],
    target: str,
    smoothing: float = 10.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """
    Target (mean) encoding with smoothing toward the global mean.

    Important
    ---------
    Fit ONLY on the training split. Then call :func:`apply_target_encoding`
    on the validation and test splits using the returned ``encodings``.
    Otherwise label leakage will inflate validation accuracy.
    """
    out = train_df.copy()
    global_mean = out[target].mean()
    encodings: Dict[str, pd.Series] = {}
    for c in cols:
        agg = out.groupby(c, dropna=False)[target].agg(["mean", "count"])
        smoothed = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        encodings[c] = smoothed
        out[f"{c}_te"] = out[c].map(smoothed)
    encodings["__global_mean__"] = pd.Series([global_mean])
    return out, encodings


def apply_target_encoding(df: pd.DataFrame, encodings: Dict[str, pd.Series]) -> pd.DataFrame:
    """Apply target encodings fit on the training split."""
    out = df.copy()
    global_mean = float(encodings["__global_mean__"].iloc[0])
    for c, enc in encodings.items():
        if c == "__global_mean__":
            continue
        if c in out.columns:
            out[f"{c}_te"] = out[c].map(enc).fillna(global_mean)
    return out
