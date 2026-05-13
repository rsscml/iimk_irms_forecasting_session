"""
Time-aware splitting for forecasting datasets.

For time series, ALWAYS split by time, never randomly. The validation and
test sets must come *after* the training set on the calendar so that the
training-time evaluation mimics how the model will be used in production.

This module implements four canonical schemes:

1. :func:`time_based_split` — single chronological split into train/val/test.
2. :func:`holdout_horizon_split` — keep the last *H* periods as test.
3. :func:`expanding_window_split` — walk-forward with growing training set.
4. :func:`rolling_window_split` — walk-forward with fixed-size training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """Container with the three split parts plus the cut-off dates that produced them."""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> pd.DataFrame:
        """Tabular summary of row counts, date ranges and shares."""
        total = len(self.train) + len(self.val) + len(self.test)
        rows = [
            ("train", self.train, self.train_end),
            ("val",   self.val,   self.val_end),
            ("test",  self.test,  self.test_end),
        ]
        out = []
        for name, part, end in rows:
            out.append({
                "split": name,
                "n_rows": len(part),
                "pct": round(len(part) / max(total, 1) * 100, 2),
                "start": part.iloc[:, 0].min() if len(part) else None,
                "end_cutoff": end,
            })
        return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Single split
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    spec: Dict,
    train_end: str,
    val_end: str,
    test_end: Optional[str] = None,
) -> SplitResult:
    """
    Single chronological split.

    Rows with ``date <= train_end`` go to *train*.
    Rows with ``train_end < date <= val_end`` go to *val*.
    Rows with ``val_end < date <= test_end`` go to *test*
    (if ``test_end`` is None, all remaining rows go to *test*).

    All three subsets keep every forecast key — only the time axis is split.
    """
    date_col = spec["date_col"]
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end) if test_end else df[date_col].max()

    if not (train_end_ts < val_end_ts <= test_end_ts):
        raise ValueError(
            f"Cut-offs must satisfy train_end ({train_end_ts}) < "
            f"val_end ({val_end_ts}) <= test_end ({test_end_ts})."
        )

    train = df[df[date_col] <= train_end_ts].copy()
    val = df[(df[date_col] > train_end_ts) & (df[date_col] <= val_end_ts)].copy()
    test = df[(df[date_col] > val_end_ts) & (df[date_col] <= test_end_ts)].copy()
    return SplitResult(train, val, test, train_end_ts, val_end_ts, test_end_ts)


def holdout_horizon_split(
    df: pd.DataFrame,
    spec: Dict,
    test_horizon: int,
    val_horizon: Optional[int] = None,
) -> SplitResult:
    """
    Hold out the last ``test_horizon`` periods (in the spec's frequency) as
    *test*, and optionally the ``val_horizon`` periods before that as *val*.

    Convenient when you want to forecast a fixed-length future horizon
    (e.g. last 28 days of a daily dataset).
    """
    date_col = spec["date_col"]
    freq = spec["frequency"]
    end_date = df[date_col].max()
    offset = pd.tseries.frequencies.to_offset(freq)

    test_start = end_date - offset * (test_horizon - 1)
    if val_horizon:
        val_start = test_start - offset * val_horizon
        train_end = val_start - offset
    else:
        val_start = test_start
        train_end = test_start - offset

    val_end = test_start - offset
    return time_based_split(
        df,
        spec,
        train_end=train_end.strftime("%Y-%m-%d"),
        val_end=val_end.strftime("%Y-%m-%d") if val_horizon else train_end.strftime("%Y-%m-%d"),
        test_end=end_date.strftime("%Y-%m-%d"),
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def expanding_window_split(
    df: pd.DataFrame,
    spec: Dict,
    initial_train_end: str,
    val_horizon: int,
    n_folds: int,
    step: Optional[int] = None,
) -> List[SplitResult]:
    """
    Walk-forward with an EXPANDING training window.

    Each fold uses everything from the dataset start up to a moving cut-off
    as training, then the next ``val_horizon`` periods as validation. Realistic
    for production retraining where you keep all available history.
    """
    date_col = spec["date_col"]
    freq = spec["frequency"]
    offset = pd.tseries.frequencies.to_offset(freq)
    step = step or val_horizon

    folds: List[SplitResult] = []
    train_end = pd.Timestamp(initial_train_end)
    end_date = df[date_col].max()

    for _ in range(n_folds):
        val_start = train_end + offset
        val_end = val_start + offset * (val_horizon - 1)
        if val_end > end_date:
            break
        fold = time_based_split(
            df,
            spec,
            train_end=train_end.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
            test_end=val_end.strftime("%Y-%m-%d"),
        )
        # In walk-forward, the "test" slot above is a duplicate of "val".
        # Empty it so consumers don't accidentally reuse it.
        fold.test = fold.test.iloc[0:0]
        folds.append(fold)
        train_end = train_end + offset * step
    return folds


def rolling_window_split(
    df: pd.DataFrame,
    spec: Dict,
    train_size: int,
    val_horizon: int,
    n_folds: int,
    step: Optional[int] = None,
) -> List[SplitResult]:
    """
    Walk-forward with a FIXED-SIZE training window.

    Useful when distant history is no longer representative and you only
    want the model to learn from recent behaviour.
    """
    date_col = spec["date_col"]
    freq = spec["frequency"]
    offset = pd.tseries.frequencies.to_offset(freq)
    step = step or val_horizon

    folds: List[SplitResult] = []
    start_date = df[date_col].min()
    end_date = df[date_col].max()

    train_start = start_date
    train_end = train_start + offset * (train_size - 1)

    for _ in range(n_folds):
        val_start = train_end + offset
        val_end = val_start + offset * (val_horizon - 1)
        if val_end > end_date:
            break

        train = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)].copy()
        val = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)].copy()
        test = df.iloc[0:0].copy()
        folds.append(SplitResult(train, val, test, train_end, val_end, val_end))

        train_start = train_start + offset * step
        train_end = train_end + offset * step
    return folds


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

def assert_no_leakage(split: SplitResult, spec: Dict) -> None:
    """
    Raise AssertionError if any train row is dated *after* a val/test row,
    or any val row is dated after a test row. Useful as a tutorial guard-rail.
    """
    date_col = spec["date_col"]
    if len(split.train) and len(split.val):
        assert split.train[date_col].max() <= split.val[date_col].min(), (
            "Leakage: a training row is dated after the earliest validation row."
        )
    if len(split.val) and len(split.test):
        assert split.val[date_col].max() <= split.test[date_col].min(), (
            "Leakage: a validation row is dated after the earliest test row."
        )
