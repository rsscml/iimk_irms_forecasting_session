"""
Data preprocessing for forecasting datasets.

The order of operations matters and is encoded in this module:

    1. fill_time_gaps  — reindex each series onto a complete date range
    2. ffill_static / bfill_static — propagate constant per-key features
    3. impute_missing — fill numeric gaps in the target / dynamic features
    4. treat_outliers — winsorize, cap, or replace extreme values

Every function returns a new dataframe — none modifies the input in place.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Time-gap filling
# ---------------------------------------------------------------------------

def fill_time_gaps(
    df: pd.DataFrame,
    spec: Dict,
    fill_value=None,
    add_indicator: bool = True,
) -> pd.DataFrame:
    """
    Reindex each forecast key onto a complete, evenly-spaced date range.

    For each key the date range runs from that key's first observation to its
    last, at the spec's frequency. Newly-inserted rows have NaN for all
    non-key columns; an optional ``_was_imputed`` indicator lets downstream
    steps tell originals from inserted rows.

    Parameters
    ----------
    fill_value : optional
        If given, the target column is set to this value for newly inserted
        rows (e.g. ``0`` for true zero-demand periods). All other non-key
        columns remain NaN.
    add_indicator : bool, default True
        If True, adds a boolean column ``_was_imputed`` marking rows that
        were inserted by gap-filling.
    """
    date_col = spec["date_col"]
    key_cols = spec["key_cols"]
    target_col = spec["target_col"]
    freq = spec["frequency"]

    pieces: List[pd.DataFrame] = []
    for keys, sub in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        sub = sub.sort_values(date_col).copy()
        full = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq=freq)

        s_indexed = sub.set_index(date_col)
        reindexed = s_indexed.reindex(full)
        reindexed.index.name = date_col

        # Restore key column values everywhere (they were lost on reindex)
        for col, val in zip(key_cols, keys):
            reindexed[col] = val

        if add_indicator:
            reindexed["_was_imputed"] = reindexed[target_col].isna()
        if fill_value is not None:
            reindexed[target_col] = reindexed[target_col].fillna(fill_value)

        pieces.append(reindexed.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    cols = [date_col] + [c for c in df.columns if c != date_col]
    if add_indicator and "_was_imputed" not in cols:
        cols = cols + ["_was_imputed"]
    out = out[[c for c in cols if c in out.columns]]
    return out.sort_values(list(key_cols) + [date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Static-feature propagation
# ---------------------------------------------------------------------------

def ffill_static(df: pd.DataFrame, spec: Dict, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Forward-fill static columns *within each forecast key*.

    Use after ``fill_time_gaps``: a column like ``store_type`` should be the
    same value on every date for a given store, but reindexing inserts NaNs.
    Forward-fill within the group restores them.
    """
    return _groupwise_fill(df, spec, cols, method="ffill")


def bfill_static(df: pd.DataFrame, spec: Dict, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Backward-fill static columns *within each forecast key*."""
    return _groupwise_fill(df, spec, cols, method="bfill")


def ffill_bfill_static(df: pd.DataFrame, spec: Dict, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convenience: forward-fill, then backward-fill within each key. Handles the
    common case where the very first row of a key is also missing the static
    column (because the original first row was missing).
    """
    out = _groupwise_fill(df, spec, cols, method="ffill")
    out = _groupwise_fill(out, spec, cols, method="bfill")
    return out


def _groupwise_fill(df: pd.DataFrame, spec: Dict,
                    cols: Optional[List[str]], method: str) -> pd.DataFrame:
    if cols is None:
        cols = spec.get("static_cols") or []
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return df.copy()

    out = df.copy()
    g = out.groupby(spec["key_cols"], dropna=False)[cols]
    if method == "ffill":
        out[cols] = g.ffill()
    elif method == "bfill":
        out[cols] = g.bfill()
    else:
        raise ValueError("method must be 'ffill' or 'bfill'")
    return out


# ---------------------------------------------------------------------------
# 3. Numeric imputation
# ---------------------------------------------------------------------------

def impute_missing(
    df: pd.DataFrame,
    spec: Dict,
    cols: Union[str, List[str]],
    method: str = "linear",
    group: bool = True,
    constant_value=None,
    rolling_window: int = 7,
) -> pd.DataFrame:
    """
    Impute NaN values in one or more numeric columns.

    Parameters
    ----------
    cols : str or list of str
        Column(s) to impute.
    method : {'mean', 'median', 'mode', 'zero', 'constant',
              'ffill', 'bfill', 'linear', 'time', 'rolling_mean',
              'rolling_median', 'seasonal_naive_7'}
        Strategy. ``'time'`` is linear interpolation that respects unequal
        time spacing. ``'seasonal_naive_7'`` fills value at *t* with the
        value at *t-7* (good for daily-with-weekly-pattern data).
    group : bool, default True
        If True, apply the imputation independently within each forecast key.
        Almost always what you want.
    constant_value :
        Required when ``method='constant'``.
    rolling_window : int, default 7
        Window length for ``rolling_mean`` / ``rolling_median``.

    Returns
    -------
    pd.DataFrame
        New dataframe with imputed values.
    """
    if isinstance(cols, str):
        cols = [cols]
    out = df.copy()

    if method == "constant" and constant_value is None:
        raise ValueError("method='constant' requires constant_value to be set.")

    def _impute_one(s: pd.Series) -> pd.Series:
        if method == "mean":
            return s.fillna(s.mean())
        if method == "median":
            return s.fillna(s.median())
        if method == "mode":
            mode = s.mode(dropna=True)
            return s.fillna(mode.iloc[0]) if len(mode) else s
        if method == "zero":
            return s.fillna(0)
        if method == "constant":
            return s.fillna(constant_value)
        if method == "ffill":
            return s.ffill()
        if method == "bfill":
            return s.bfill()
        if method == "linear":
            return s.interpolate(method="linear", limit_direction="both")
        if method == "time":
            return s.interpolate(method="time", limit_direction="both")
        if method == "rolling_mean":
            roll = s.rolling(rolling_window, min_periods=1).mean()
            return s.fillna(roll)
        if method == "rolling_median":
            roll = s.rolling(rolling_window, min_periods=1).median()
            return s.fillna(roll)
        if method == "seasonal_naive_7":
            return s.fillna(s.shift(7)).ffill().bfill()
        raise ValueError(f"Unknown method: {method}")

    date_col = spec["date_col"]
    key_cols = spec["key_cols"]

    if method == "time":
        # 'time' interpolation requires a DatetimeIndex
        out = out.sort_values(list(key_cols) + [date_col])
        if group:
            for col in cols:
                out[col] = out.groupby(key_cols, dropna=False, group_keys=False).apply(
                    lambda sub: sub.set_index(date_col)[col]
                                   .interpolate(method="time", limit_direction="both")
                                   .reset_index(drop=True)
                ).reset_index(drop=True).values
        else:
            tmp = out.set_index(date_col)
            for col in cols:
                tmp[col] = tmp[col].interpolate(method="time", limit_direction="both")
            out = tmp.reset_index()
        return out

    if group:
        for col in cols:
            out[col] = out.groupby(key_cols, dropna=False, group_keys=False)[col].apply(_impute_one)
    else:
        for col in cols:
            out[col] = _impute_one(out[col])
    return out


# ---------------------------------------------------------------------------
# 4. Outlier treatment
# ---------------------------------------------------------------------------

def winsorize(
    df: pd.DataFrame,
    spec: Dict,
    col: str,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    group: bool = True,
) -> pd.DataFrame:
    """
    Cap values in ``col`` at given lower and upper percentiles.

    Default 1%/99% protects against extreme outliers without over-distorting
    the distribution. Set ``group=True`` to compute percentiles per series.
    """
    out = df.copy()

    def _cap(s: pd.Series) -> pd.Series:
        lo = s.quantile(lower_pct)
        hi = s.quantile(upper_pct)
        return s.clip(lower=lo, upper=hi)

    if group:
        out[col] = out.groupby(spec["key_cols"], dropna=False, group_keys=False)[col].apply(_cap)
    else:
        out[col] = _cap(out[col])
    return out


def replace_outliers_with(
    df: pd.DataFrame,
    spec: Dict,
    col: str,
    detector: Callable[[pd.Series], pd.Series],
    replace_with: str = "median",
    group: bool = True,
) -> pd.DataFrame:
    """
    Replace outliers identified by ``detector`` with a summary statistic.

    Parameters
    ----------
    detector : callable
        Function taking a Series and returning a boolean Series of the same
        length (e.g. ``forecasting_toolkit.quality.detect_outliers_iqr``).
    replace_with : {'median', 'mean', 'nan'}
        ``'nan'`` lets a downstream :func:`impute_missing` step decide.
    """
    out = df.copy()

    def _replace(s: pd.Series) -> pd.Series:
        flags = detector(s)
        s2 = s.copy()
        if replace_with == "median":
            s2[flags] = s.median()
        elif replace_with == "mean":
            s2[flags] = s.mean()
        elif replace_with == "nan":
            s2[flags] = np.nan
        else:
            raise ValueError("replace_with must be 'median', 'mean' or 'nan'")
        return s2

    if group:
        out[col] = out.groupby(spec["key_cols"], dropna=False, group_keys=False)[col].apply(_replace)
    else:
        out[col] = _replace(out[col])
    return out
