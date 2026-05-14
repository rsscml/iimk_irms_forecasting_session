"""
data_utils.py
=============
Core data utilities for panel / hierarchical time-series forecasting.

Conventions
-----------
A "forecast key" identifies a single series (e.g., (store_id, item_id)).
A panel data frame has columns:
    [date_col] + key_cols + [target_col] + (optional exogenous columns)

All helpers here are forecast-key aware: they operate per-key without leaking
information across keys.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_forecasting_data_old(
    path: str,
    date_col: str,
    target_col: str,
    key_cols: Sequence[str],
    parse_dates: bool = True,
    sort: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a panel forecasting dataset from a CSV / Parquet file.

    Parameters
    ----------
    path : str
        File path. CSV (any delimiter via read_csv_kwargs) or Parquet (.parquet).
    date_col : str
        Name of the date / timestamp column.
    target_col : str
        Name of the numeric target (e.g., 'sales', 'units_sold').
    key_cols : Sequence[str]
        Columns that together identify a single forecast key.
    parse_dates : bool
        Convert date_col to pandas datetime.
    sort : bool
        Sort by key + date (recommended for downstream feature engineering).

    Returns
    -------
    pd.DataFrame
    """
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, **read_csv_kwargs)

    missing = [c for c in [date_col, target_col, *key_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}")

    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])

    if sort:
        df = df.sort_values([*key_cols, date_col]).reset_index(drop=True)

    return df

def load_forecasting_data(
    path: str,
    date_col: str,
    target_col: str,
    key_cols: Sequence[str],
    parse_dates: bool = True,
    sort: bool = True,
    exclude_cols: Optional[Sequence[str]] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a panel forecasting dataset from a CSV / Parquet file.
 
    Parameters
    ----------
    path : str
        File path. CSV (any delimiter via read_csv_kwargs) or Parquet (.parquet).
    date_col : str
        Name of the date / timestamp column.
    target_col : str
        Name of the numeric target (e.g., 'sales', 'units_sold').
    key_cols : Sequence[str]
        Columns that together identify a single forecast key.
    parse_dates : bool
        Convert date_col to pandas datetime.
    sort : bool
        Sort by key + date (recommended for downstream feature engineering).
    exclude_cols : Sequence[str], optional
        Columns to drop immediately after loading, *before* any downstream
        processing sees them. Useful for stowaways like row IDs, free-text
        product names, audit fields, or any column you don't want a model
        to train on. Two forms are supported, mixed freely:
 
          * exact names         e.g. "id", "product_name"
          * prefix wildcards    e.g. "event_*", "audit_*"
 
        The wildcard match is anchored at the start of the column name and
        only accepts a trailing "*"; this is intentional (no regex / suffix
        matching) to keep the API obvious. The date column, target column,
        and key columns are protected — naming them here raises an error
        rather than silently dropping pipeline-critical fields.
 
    Returns
    -------
    pd.DataFrame
    """
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, **read_csv_kwargs)
 
    missing = [c for c in [date_col, target_col, *key_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from data: {missing}")
 
    # --- Apply exclude_cols -------------------------------------------------
    if exclude_cols:
        protected = {date_col, target_col, *key_cols}
        to_drop: List[str] = []
        unresolved: List[str] = []
        for pattern in exclude_cols:
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                matched = [c for c in df.columns if c.startswith(prefix)]
                if not matched:
                    unresolved.append(pattern)
                to_drop.extend(matched)
            else:
                if pattern in df.columns:
                    to_drop.append(pattern)
                else:
                    unresolved.append(pattern)
 
        conflicts = [c for c in to_drop if c in protected]
        if conflicts:
            raise ValueError(
                f"exclude_cols cannot reference the date column, target column, "
                f"or any key column. Conflicting entries: {conflicts}"
            )
 
        if unresolved:
            warnings.warn(
                f"exclude_cols entries not found in data and ignored: {unresolved}",
                stacklevel=2,
            )
 
        seen = set()
        to_drop_unique = [c for c in to_drop if not (c in seen or seen.add(c))]
        if to_drop_unique:
            df = df.drop(columns=to_drop_unique)
 
    if parse_dates:
        df[date_col] = pd.to_datetime(df[date_col])
 
    if sort:
        df = df.sort_values([*key_cols, date_col]).reset_index(drop=True)
 
    return df


# ---------------------------------------------------------------------------
# Panel validation & completion
# ---------------------------------------------------------------------------
def validate_panel(
    df: pd.DataFrame,
    date_col: str,
    key_cols: Sequence[str],
    target_col: str,
    freq: str = "D",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Audit a panel dataset.

    Returns a small summary DataFrame with one row per forecast key:
        n_obs, first_date, last_date, missing_dates, n_zeros, n_negatives, n_nans
    """
    rows = []
    for keys, g in df.groupby(list(key_cols)):
        full_idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq=freq)
        rows.append({
            **(dict(zip(key_cols, keys)) if isinstance(keys, tuple) else {key_cols[0]: keys}),
            "n_obs": len(g),
            "first_date": g[date_col].min(),
            "last_date": g[date_col].max(),
            "missing_dates": len(full_idx) - len(g),
            "n_zeros": int((g[target_col] == 0).sum()),
            "n_negatives": int((g[target_col] < 0).sum()),
            "n_nans": int(g[target_col].isna().sum()),
        })
    summary = pd.DataFrame(rows)
    if verbose:
        print(f"Panel summary: {len(summary)} forecast keys, freq='{freq}'")
        print(f"  Total missing date rows : {summary['missing_dates'].sum():,}")
        print(f"  Total zero observations : {summary['n_zeros'].sum():,}")
        print(f"  Total negatives         : {summary['n_negatives'].sum():,}")
        print(f"  Total NaNs in target    : {summary['n_nans'].sum():,}")
    return summary


def complete_panel(
    df: pd.DataFrame,
    date_col: str,
    key_cols: Sequence[str],
    target_col: str,
    freq: str = "D",
    fill_value: Union[float, str, None] = 0.0,
) -> pd.DataFrame:
    """
    Ensure every (key, date) combination exists between each key's first and
    last observation. Missing target rows are filled with `fill_value`.

    For intermittent demand it is *critical* to insert these zero-demand rows;
    otherwise lag/rolling features silently skip days with no sales.
    """
    pieces = []
    key_cols = list(key_cols)
    for keys, g in df.groupby(key_cols):
        idx = pd.date_range(g[date_col].min(), g[date_col].max(), freq=freq)
        full = pd.DataFrame({date_col: idx})
        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, v in zip(key_cols, keys):
            full[c] = v
        merged = full.merge(g, on=[date_col, *key_cols], how="left")
        if fill_value is not None:
            merged[target_col] = merged[target_col].fillna(fill_value)
        pieces.append(merged)
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values([*key_cols, date_col]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Time-based splits
# ---------------------------------------------------------------------------
def time_based_split(
    df: pd.DataFrame,
    date_col: str,
    cutoff: Union[str, pd.Timestamp],
    holdout_end: Optional[Union[str, pd.Timestamp]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a panel into (train, test) by a single calendar cutoff.
        train = rows where date < cutoff
        test  = rows where cutoff <= date <= holdout_end (or end of series)
    """
    cutoff = pd.Timestamp(cutoff)
    train = df.loc[df[date_col] < cutoff].copy()
    test = df.loc[df[date_col] >= cutoff].copy()
    if holdout_end is not None:
        test = test.loc[test[date_col] <= pd.Timestamp(holdout_end)].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demand classification (ADI / CV²)
# ---------------------------------------------------------------------------
def classify_demand(
    df: pd.DataFrame,
    date_col: str,
    key_cols: Sequence[str],
    target_col: str,
) -> pd.DataFrame:
    """
    Classify each forecast key by the Syntetos-Boylan demand pattern matrix.

    ADI  = Average Demand Interval  = (#periods) / (#non-zero periods)
    CV²  = (std / mean) ** 2 over non-zero demand sizes

    Pattern rules of thumb (Syntetos, Boylan, Croston 2005):
        Smooth        : ADI < 1.32  and  CV² < 0.49
        Erratic       : ADI < 1.32  and  CV² >= 0.49
        Intermittent  : ADI >= 1.32 and  CV² < 0.49
        Lumpy         : ADI >= 1.32 and  CV² >= 0.49

    Lumpy & intermittent series are the ones that need Tweedie / Poisson losses
    rather than vanilla MSE.
    """
    rows = []
    for keys, g in df.groupby(list(key_cols)):
        y = g[target_col].values
        n = len(y)
        nz = y[y > 0]
        if len(nz) == 0:
            adi = np.inf
            cv2 = 0.0
        else:
            adi = n / len(nz)
            cv2 = (nz.std(ddof=0) / nz.mean()) ** 2 if nz.mean() > 0 else 0.0

        if adi < 1.32 and cv2 < 0.49:
            pattern = "Smooth"
        elif adi < 1.32 and cv2 >= 0.49:
            pattern = "Erratic"
        elif adi >= 1.32 and cv2 < 0.49:
            pattern = "Intermittent"
        else:
            pattern = "Lumpy"

        rec = dict(zip(key_cols, keys)) if isinstance(keys, tuple) else {key_cols[0]: keys}
        rec.update({"ADI": adi, "CV2": cv2, "pattern": pattern,
                    "mean_demand": float(y.mean()), "zero_share": float((y == 0).mean())})
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def summarize_split(train: pd.DataFrame, test: pd.DataFrame, date_col: str) -> None:
    """Print a one-shot summary of a train/test split."""
    print(f"  train: {len(train):>8,} rows  |  {train[date_col].min().date()} → {train[date_col].max().date()}")
    print(f"  test : {len(test):>8,} rows  |  {test[date_col].min().date()} → {test[date_col].max().date()}")
