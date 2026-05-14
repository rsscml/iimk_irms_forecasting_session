"""
feature_engineering.py
======================
Causal feature engineering for panel time-series forecasting.

Causality
---------
A "causal" feature at time t is computable using only information from times
< t (or <= t for exogenous variables that are known in advance, like a holiday
flag). When you want a *purely causal* model (one that could have run live in
production at time t), every feature must be causal.

The lag and rolling helpers here are written specifically so that the value at
row t never depends on row t itself. We always shift by `min_lag` (default 1)
*before* computing rolling stats. This is the single most common source of
data leakage in ML forecasting tutorials, so we treat it explicitly.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    lags: Iterable[int],
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    Append y_lag_{k} columns. Within each forecast key the lag is computed
    independently, so we never leak across keys.
    """
    df = df.copy()
    prefix = prefix or f"{target_col}_lag_"
    grp = df.groupby(list(group_cols), sort=False)[target_col]
    for k in lags:
        df[f"{prefix}{k}"] = grp.shift(k)
    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------
def add_rolling_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    windows: Iterable[int],
    stats: Sequence[str] = ("mean", "std"),
    min_lag: int = 1,
    min_periods: Optional[int] = None,
) -> pd.DataFrame:
    """
    Append rolling-window statistics computed over a *lagged* series, so the
    statistic at row t excludes row t itself.

    stats : any of  mean, std, min, max, median, sum, var, skew, kurt
    """
    df = df.copy()
    grp = df.groupby(list(group_cols), sort=False)[target_col]
    shifted = grp.shift(min_lag)

    for w in windows:
        roller = shifted.rolling(window=w, min_periods=min_periods or 1)
        for stat in stats:
            col = f"{target_col}_roll_{stat}_{w}"
            df[col] = getattr(roller, stat)()
    return df


def add_ewm_features(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    target_col: str,
    halflives: Iterable[float],
    min_lag: int = 1,
) -> pd.DataFrame:
    """Exponentially-weighted moving averages over the lagged target."""
    df = df.copy()
    grp = df.groupby(list(group_cols), sort=False)[target_col]
    shifted = grp.shift(min_lag)
    for hl in halflives:
        df[f"{target_col}_ewm_hl{hl}"] = shifted.ewm(halflife=hl, adjust=False).mean()
    return df


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------
def add_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    cyclical: bool = True,
) -> pd.DataFrame:
    """
    Add date-derived features. Calendar features are causal because they are
    deterministic functions of the date and known arbitrarily far in advance.

    If cyclical=True, day-of-week / month / day-of-year are also encoded as
    sin/cos pairs so tree models can exploit smooth seasonality if needed
    (mostly useful for linear models; trees handle integer encodings fine but
    sin/cos rarely hurt).
    """
    df = df.copy()
    d = df[date_col]
    df["year"] = d.dt.year
    df["quarter"] = d.dt.quarter
    df["month"] = d.dt.month
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["day"] = d.dt.day
    df["day_of_week"] = d.dt.dayofweek
    df["day_of_year"] = d.dt.dayofyear
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    df["is_quarter_end"] = d.dt.is_quarter_end.astype(int)

    if cyclical:
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    return df


# ---------------------------------------------------------------------------
# Categorical handling
# ---------------------------------------------------------------------------
def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    method: str = "category",
) -> pd.DataFrame:
    """
    LightGBM accepts pandas 'category' dtype natively and uses an optimised
    split algorithm; this is almost always what you want. We expose a method
    arg only for completeness.
    """
    df = df.copy()
    if method == "category":
        for c in cat_cols:
            df[c] = df[c].astype("category")
    elif method == "label":
        for c in cat_cols:
            df[c], _ = pd.factorize(df[c])
    else:
        raise ValueError(f"Unknown method: {method}")
    return df


# ---------------------------------------------------------------------------
# All-in-one builder
# ---------------------------------------------------------------------------
class FeatureEngineer:
    """
    A reusable, parameterised feature builder. Call .transform(df) once on
    train and again on test using the *same instance* so that the configuration
    is locked in.
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        key_cols: Sequence[str],
        lags: Sequence[int] = (1, 7, 14, 28),
        rolling_windows: Sequence[int] = (7, 14, 28),
        rolling_stats: Sequence[str] = ("mean", "std"),
        ewm_halflives: Sequence[float] = (7.0, 28.0),
        rolling_min_lag: int = 1,
        cyclical_calendar: bool = True,
        encode_keys_as_category: bool = True,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.key_cols = list(key_cols)
        self.lags = list(lags)
        self.rolling_windows = list(rolling_windows)
        self.rolling_stats = list(rolling_stats)
        self.ewm_halflives = list(ewm_halflives)
        self.rolling_min_lag = rolling_min_lag
        self.cyclical_calendar = cyclical_calendar
        self.encode_keys_as_category = encode_keys_as_category
        self._feature_names: Optional[List[str]] = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = add_calendar_features(out, self.date_col, cyclical=self.cyclical_calendar)
        if self.lags:
            out = add_lag_features(out, self.key_cols, self.target_col, self.lags)
        if self.rolling_windows:
            out = add_rolling_features(
                out, self.key_cols, self.target_col,
                self.rolling_windows, self.rolling_stats,
                min_lag=self.rolling_min_lag,
            )
        if self.ewm_halflives:
            out = add_ewm_features(out, self.key_cols, self.target_col, self.ewm_halflives,
                                   min_lag=self.rolling_min_lag)
        if self.encode_keys_as_category:
            out = encode_categoricals(out, self.key_cols, method="category")

        # Lock in feature list on first transform
        if self._feature_names is None:
            self._feature_names = [c for c in out.columns
                                   if c not in (self.date_col, self.target_col)]
        return out

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise RuntimeError("Call .transform first.")
        return self._feature_names
