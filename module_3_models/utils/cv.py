"""
cv.py
=====
Time-series cross-validation utilities.

The standard k-fold or shuffled CV is *wrong* for forecasting because it lets
information leak from the future into the past. Use these splitters instead.

Two flavours are provided:
    expanding_window_split  : training set grows, validation always last k days
    sliding_window_split    : fixed-size training window slides forward

Both are panel-aware: every fold contains rows from every forecast key, split
by the same calendar dates. This mirrors how a production model is retrained.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp     # inclusive
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp     # inclusive
    train_idx: np.ndarray
    valid_idx: np.ndarray


def expanding_window_split(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int = 4,
    horizon_days: int = 28,
    gap_days: int = 0,
    initial_train_days: Optional[int] = None,
) -> List[TimeSeriesFold]:
    """
    Build expanding-window folds.

       fold 1: train on [start, t1)             -> validate [t1+gap, t1+gap+H)
       fold 2: train on [start, t2)             -> validate [t2+gap, t2+gap+H)
       ...
    where t_i are evenly spaced.

    Parameters
    ----------
    horizon_days : int
        Length of each validation window in days.
    gap_days : int
        Optional gap between train end and validation start. Use this to
        simulate the time it takes to refresh features in production
        (e.g., 7 days if your features include a 7-day lag).
    initial_train_days : Optional[int]
        Force the first training window to span at least this many days. If
        None, we leave folds * horizon_days at the end for validation.
    """
    dates = pd.to_datetime(df[date_col])
    start, end = dates.min().normalize(), dates.max().normalize()

    total_days = (end - start).days + 1
    needed = n_folds * horizon_days + gap_days
    if total_days <= needed:
        raise ValueError(
            f"Not enough history: have {total_days} days, need > {needed}."
        )
    if initial_train_days is None:
        initial_train_days = total_days - needed

    folds: List[TimeSeriesFold] = []
    for i in range(n_folds):
        train_end = start + pd.Timedelta(days=initial_train_days + i * horizon_days - 1)
        valid_start = train_end + pd.Timedelta(days=1 + gap_days)
        valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
        if valid_end > end:
            break
        train_mask = dates <= train_end
        valid_mask = (dates >= valid_start) & (dates <= valid_end)
        folds.append(TimeSeriesFold(
            fold=i,
            train_start=start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            train_idx=np.where(train_mask)[0],
            valid_idx=np.where(valid_mask)[0],
        ))
    return folds


def sliding_window_split(
    df: pd.DataFrame,
    date_col: str,
    n_folds: int = 4,
    train_days: int = 365,
    horizon_days: int = 28,
    gap_days: int = 0,
) -> List[TimeSeriesFold]:
    """
    Sliding-window folds. Useful when older history is no longer representative
    (e.g., post-pandemic retail) and you want to evaluate what a fixed-window
    retrain would look like.
    """
    dates = pd.to_datetime(df[date_col])
    start, end = dates.min().normalize(), dates.max().normalize()
    total_days = (end - start).days + 1
    needed = train_days + gap_days + horizon_days
    if total_days < needed + (n_folds - 1) * horizon_days:
        raise ValueError("Not enough history for the requested sliding folds.")

    folds: List[TimeSeriesFold] = []
    for i in range(n_folds):
        offset = i * horizon_days
        train_start = start + pd.Timedelta(days=offset)
        train_end = train_start + pd.Timedelta(days=train_days - 1)
        valid_start = train_end + pd.Timedelta(days=1 + gap_days)
        valid_end = valid_start + pd.Timedelta(days=horizon_days - 1)
        if valid_end > end:
            break
        train_mask = (dates >= train_start) & (dates <= train_end)
        valid_mask = (dates >= valid_start) & (dates <= valid_end)
        folds.append(TimeSeriesFold(
            fold=i,
            train_start=train_start, train_end=train_end,
            valid_start=valid_start, valid_end=valid_end,
            train_idx=np.where(train_mask)[0],
            valid_idx=np.where(valid_mask)[0],
        ))
    return folds


def describe_folds(folds: Sequence[TimeSeriesFold]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": f.fold,
        "train_start": f.train_start.date(),
        "train_end": f.train_end.date(),
        "valid_start": f.valid_start.date(),
        "valid_end": f.valid_end.date(),
        "n_train": len(f.train_idx),
        "n_valid": len(f.valid_idx),
    } for f in folds])
