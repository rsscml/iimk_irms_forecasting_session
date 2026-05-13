"""
Data I/O and inspection utilities for forecasting datasets.

This module provides functions for loading CSV/parquet datasets, validating
required columns, and producing structured summaries of the data shape,
forecast keys, and time coverage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataset specification helper
# ---------------------------------------------------------------------------

def make_spec(
    date_col: str,
    target_col: str,
    key_cols: Union[str, List[str]],
    static_cols: Optional[List[str]] = None,
    dynamic_cols: Optional[List[str]] = None,
    frequency: str = "D",
) -> Dict:
    """
    Build a 'DatasetSpec' dict describing the schema of a forecasting dataset.

    The spec is passed to most toolkit functions so they know which column
    is the timestamp, which is the target, what uniquely identifies each
    series, and what the expected sampling frequency is.

    Parameters
    ----------
    date_col : str
        Name of the column holding timestamps.
    target_col : str
        Name of the column being forecast (e.g., sales, demand).
    key_cols : str or list of str
        One or more columns whose combination uniquely identifies a single
        time series (e.g., ['store_id', 'item_id']).
    static_cols : list of str, optional
        Columns that are CONSTANT within a forecast key (e.g., store type,
        item category). These typically have a single value per series.
    dynamic_cols : list of str, optional
        Columns that VARY over time within a forecast key (e.g., promotion
        flag, price, weather).
    frequency : str, default 'D'
        Pandas offset alias describing the expected sampling frequency:
        'D' (daily), 'W' (weekly), 'MS' (month-start), 'H' (hourly), etc.

    Returns
    -------
    dict
        The spec, with ``key_cols`` always normalised to a list.
    """
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    return {
        "date_col": date_col,
        "target_col": target_col,
        "key_cols": list(key_cols),
        "static_cols": list(static_cols) if static_cols else [],
        "dynamic_cols": list(dynamic_cols) if dynamic_cols else [],
        "frequency": frequency,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_data(
    path: Union[str, Path],
    spec: Dict,
    file_format: str = "auto",
    **read_kwargs,
) -> pd.DataFrame:
    """
    Load a forecasting dataset from CSV or Parquet, with date parsing,
    column validation and stable sorting by ``key_cols + [date_col]``.

    Parameters
    ----------
    path : str or Path
        File location.
    spec : dict
        DatasetSpec produced by :func:`make_spec`.
    file_format : {'auto', 'csv', 'parquet'}
        How to read the file. ``'auto'`` infers from the file extension.
    **read_kwargs
        Forwarded to ``pd.read_csv`` / ``pd.read_parquet``.

    Returns
    -------
    pd.DataFrame
        Loaded, date-parsed, and sorted dataframe.
    """
    path = Path(path)
    if file_format == "auto":
        ext = path.suffix.lower()
        file_format = "parquet" if ext in {".parquet", ".pq"} else "csv"

    if file_format == "csv":
        df = pd.read_csv(path, **read_kwargs)
    elif file_format == "parquet":
        df = pd.read_parquet(path, **read_kwargs)
    else:
        raise ValueError(f"Unsupported file_format: {file_format}")

    #date_col = spec["date_col"]
    #if date_col in df.columns and not np.issubdtype(df[date_col].dtype, np.datetime64):
    #    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    date_col = spec["date_col"]
    if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    _validate_columns(df, spec)

    sort_cols = list(spec["key_cols"]) + [date_col]
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def _validate_columns(df: pd.DataFrame, spec: Dict) -> None:
    """Raise if any column declared in the spec is missing from the dataframe."""
    declared: List[str] = (
        [spec["date_col"], spec["target_col"]]
        + list(spec["key_cols"])
        + list(spec.get("static_cols") or [])
        + list(spec.get("dynamic_cols") or [])
    )
    missing = [c for c in declared if c not in df.columns]
    if missing:
        raise ValueError(
            f"Spec references columns not present in the dataframe: {missing}"
        )


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-row-per-column overview of the dataframe.

    Returns a tidy table covering dtype, missing counts, missing %,
    unique counts, and a small example value. Useful as the very first
    look at any dataset.
    """
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        example = s.dropna().iloc[0] if s.notna().any() else None
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(s.isna().mean() * 100, 2),
            "n_unique": int(s.nunique(dropna=True)),
            "unique_pct": round(s.nunique(dropna=True) / max(n, 1) * 100, 2),
            "example": example,
        })
    return pd.DataFrame(rows)


def summarize_keys(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Per-forecast-key summary: row count, date span, target stats, zero/missing
    counts. Returned sorted by ``n_obs`` descending.

    Use this to identify short series, sparse series and outlier series at a
    glance.
    """
    date_col, target_col = spec["date_col"], spec["target_col"]
    key_cols = list(spec["key_cols"])

    grouped = df.groupby(key_cols, dropna=False)
    summary = grouped.agg(
        n_obs=(date_col, "count"),
        date_min=(date_col, "min"),
        date_max=(date_col, "max"),
        target_mean=(target_col, "mean"),
        target_std=(target_col, "std"),
        target_min=(target_col, "min"),
        target_max=(target_col, "max"),
        target_sum=(target_col, "sum"),
        n_zeros=(target_col, lambda x: int((x == 0).sum())),
        n_missing=(target_col, lambda x: int(x.isna().sum())),
    ).reset_index()

    summary["span_days"] = (summary["date_max"] - summary["date_min"]).dt.days
    summary["zero_pct"] = (summary["n_zeros"] / summary["n_obs"] * 100).round(2)
    summary["missing_pct"] = (summary["n_missing"] / summary["n_obs"] * 100).round(2)

    return summary.sort_values("n_obs", ascending=False).reset_index(drop=True)


def coverage_report(df: pd.DataFrame, spec: Dict) -> Dict:
    """
    Top-level coverage stats for the whole dataset.

    Useful as a one-glance health check before any deeper EDA.
    """
    date_col = spec["date_col"]
    key_cols = spec["key_cols"]

    n_keys = df.groupby(key_cols, dropna=False).ngroups
    return {
        "rows": int(len(df)),
        "n_forecast_keys": int(n_keys),
        "min_date": df[date_col].min(),
        "max_date": df[date_col].max(),
        "total_days": int((df[date_col].max() - df[date_col].min()).days) + 1,
        "avg_rows_per_key": round(len(df) / max(n_keys, 1), 2),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def get_series(df: pd.DataFrame, spec: Dict, key_values: Union[Dict, tuple]) -> pd.DataFrame:
    """
    Return the subset of rows belonging to a single forecast key, sorted by date.

    Parameters
    ----------
    key_values : dict or tuple
        If a dict, must map ``key_col -> value``. If a tuple, must align with
        the ordering of ``spec['key_cols']``.
    """
    key_cols = spec["key_cols"]
    if isinstance(key_values, dict):
        mask = pd.Series(True, index=df.index)
        for col, val in key_values.items():
            mask &= (df[col] == val)
    else:
        if len(key_values) != len(key_cols):
            raise ValueError("key_values length must match spec['key_cols']")
        mask = pd.Series(True, index=df.index)
        for col, val in zip(key_cols, key_values):
            mask &= (df[col] == val)

    sub = df.loc[mask].sort_values(spec["date_col"]).reset_index(drop=True)
    return sub
