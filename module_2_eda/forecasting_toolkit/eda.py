"""
Exploratory Data Analysis utilities for time series forecasting.

Functions here compute descriptive and time-series-specific statistics:
moments, dispersion, autocorrelation, partial autocorrelation, seasonal
strength and trend strength. They never plot — plotting helpers live in
``forecasting_toolkit.plotting``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# statsmodels is heavy; import lazily inside functions where needed
# so that "import forecasting_toolkit" stays cheap.


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------

def describe_numeric(df: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extended descriptive statistics for numeric columns.

    Goes beyond ``df.describe()`` by adding skewness, kurtosis, coefficient
    of variation, IQR, and the share of zero values — all of which matter
    for forecasting decisions (e.g. low-mean intermittent series often have
    high CV and many zeros).
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()

    rows = []
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        rows.append({
            "column": c,
            "count": int(s.size),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": q1,
            "median": s.median(),
            "p75": q3,
            "max": s.max(),
            "iqr": q3 - q1,
            "skewness": s.skew(),
            "kurtosis": s.kurt(),
            "cv": s.std() / s.mean() if s.mean() else np.nan,
            "zero_pct": round((s == 0).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, cols: Optional[List[str]] = None,
                         top_n: int = 5) -> pd.DataFrame:
    """
    Per-column statistics for categorical / object columns: cardinality, mode,
    mode share, and the top-N value distribution.
    """
    if cols is None:
        cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    rows = []
    for c in cols:
        s = df[c]
        vc = s.value_counts(dropna=False)
        top = vc.head(top_n)
        rows.append({
            "column": c,
            "n_unique": int(s.nunique(dropna=True)),
            "missing": int(s.isna().sum()),
            "mode": vc.index[0] if len(vc) else None,
            "mode_share_pct": round(vc.iloc[0] / len(s) * 100, 2) if len(vc) else 0,
            "top_values": ", ".join([f"{v}({c_})" for v, c_ in top.items()]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time series statistics
# ---------------------------------------------------------------------------

def coefficient_of_variation(series: pd.Series, only_nonzero: bool = False) -> float:
    """Coefficient of variation: std/mean. Returns NaN if mean is zero."""
    s = series.dropna()
    if only_nonzero:
        s = s[s != 0]
    if len(s) == 0 or s.mean() == 0:
        return np.nan
    return float(s.std() / s.mean())


def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
    alpha: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Autocorrelation and partial autocorrelation values up to ``nlags``.

    Returns a dict with keys ``acf``, ``pacf``, ``acf_ci``, ``pacf_ci`` —
    where the ``*_ci`` arrays are 2-column (lower/upper) confidence bounds.

    Notes
    -----
    ACF / PACF are the canonical tools for diagnosing autocorrelation
    structure: ACF spikes at seasonal lags suggest seasonality; a slow
    ACF decay suggests a trend or non-stationarity; a sharp PACF cut-off
    at lag *p* suggests an AR(p) signal.
    """
    from statsmodels.tsa.stattools import acf, pacf

    s = series.dropna().astype(float).values
    nlags = min(nlags, len(s) - 1)
    if nlags < 1:
        empty = np.array([])
        return {"acf": empty, "pacf": empty,
                "acf_ci": empty.reshape(0, 2), "pacf_ci": empty.reshape(0, 2)}

    acf_vals, acf_ci = acf(s, nlags=nlags, alpha=alpha, fft=True)
    pacf_vals, pacf_ci = pacf(s, nlags=nlags, alpha=alpha, method="ywm")
    return {"acf": acf_vals, "pacf": pacf_vals,
            "acf_ci": acf_ci, "pacf_ci": pacf_ci}


def adf_test(series: pd.Series) -> Dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns a dict with the test statistic, p-value, used lag, n observations,
    critical values, and a human-readable verdict at the 5% level.
    """
    from statsmodels.tsa.stattools import adfuller

    s = series.dropna()
    if len(s) < 10:
        return {"error": "Series too short for ADF (need ≥10 observations)."}
    stat, pvalue, used_lag, nobs, crit, _ = adfuller(s, autolag="AIC")
    return {
        "test_statistic": float(stat),
        "p_value": float(pvalue),
        "used_lag": int(used_lag),
        "n_obs": int(nobs),
        "crit_values": {k: float(v) for k, v in crit.items()},
        "is_stationary_5pct": bool(pvalue < 0.05),
    }


def seasonal_decompose(
    series: pd.Series,
    period: int,
    model: str = "additive",
):
    """
    Classical seasonal decomposition into trend, seasonal and residual.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series. NaNs are forward-filled then back-filled prior
        to decomposition (statsmodels does not accept missing values).
    period : int
        Number of observations per season (7 for daily-with-weekly-pattern,
        12 for monthly-with-yearly-pattern, etc.).
    model : {'additive', 'multiplicative'}
    """
    from statsmodels.tsa.seasonal import seasonal_decompose as _decompose
    s = series.copy().ffill().bfill()
    return _decompose(s, period=period, model=model, extrapolate_trend="freq")


def seasonal_strength(decomposition) -> Dict[str, float]:
    """
    Quantify trend and seasonal strength on a 0–1 scale (Wang/Hyndman/Smith).

    A series with strong seasonality has ``Fs`` close to 1; a series with
    strong trend has ``Ft`` close to 1.
    """
    resid = decomposition.resid.dropna()
    seasonal = decomposition.seasonal.loc[resid.index]
    trend = decomposition.trend.loc[resid.index]

    var_resid = resid.var()
    var_seas_resid = (seasonal + resid).var()
    var_trend_resid = (trend + resid).var()

    fs = max(0.0, 1 - var_resid / var_seas_resid) if var_seas_resid else 0.0
    ft = max(0.0, 1 - var_resid / var_trend_resid) if var_trend_resid else 0.0
    return {"trend_strength": float(ft), "seasonal_strength": float(fs)}

def stl_decompose(
    series: pd.Series,
    period: int,
    seasonal: int = 7,
    trend: Optional[int] = None,
    robust: bool = False,
):
    """
    STL (Seasonal-Trend decomposition using LOESS) into trend, seasonal
    and residual components.

    Parameters
    ----------
    series : pd.Series
        Time-indexed series. NaNs are forward-filled then back-filled prior
        to decomposition (statsmodels' STL does not accept missing values).
    period : int
        Number of observations per season (7 for daily-with-weekly-pattern,
        12 for monthly-with-yearly-pattern, etc.).
    seasonal : int, default 7
        Length of the seasonal smoother. Must be an odd integer >= 7.
        Larger values produce a smoother seasonal component.
    trend : int, optional
        Length of the trend smoother. Must be an odd integer strictly
        greater than `period`. If None, statsmodels picks a sensible
        default (smallest odd integer >= 1.5 * period / (1 - 1.5/seasonal)).
    robust : bool, default False
        If True, uses a robust fitting procedure that down-weights outliers.
        Slower but more reliable on series with shocks or heavy tails.

    Notes
    -----
    STL is additive only. For multiplicative seasonality, log-transform
    the series first and exponentiate the components afterwards.
    """
    from statsmodels.tsa.seasonal import STL

    s = series.copy().ffill().bfill()
    return STL(s, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()

def per_series_stats(df: pd.DataFrame, spec: Dict) -> pd.DataFrame:
    """
    Per-forecast-key time-series statistics.

    For each unique key combination, computes mean, std, CV, CV of non-zero
    demand, and the share of zero values. This is the input that
    :func:`forecasting_toolkit.classification.classify_all_series` then
    consumes.
    """
    from .classification import compute_adi, compute_cv2

    target = spec["target_col"]
    key_cols = spec["key_cols"]

    out = []
    for keys, sub in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        s = sub[target]
        out.append({
            **dict(zip(key_cols, keys)),
            "n_obs": int(len(s)),
            "mean": s.mean(),
            "std": s.std(),
            "cv": coefficient_of_variation(s),
            "cv_nonzero": coefficient_of_variation(s, only_nonzero=True),
            "zero_share": float((s == 0).mean()) if len(s) else np.nan,
            "adi": compute_adi(s),
            "cv2": compute_cv2(s),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Pairwise correlation matrix for numeric columns.

    ``method`` accepts ``'pearson'`` (linear), ``'spearman'`` (rank), or
    ``'kendall'``. Spearman is more robust to outliers and non-linearity.
    """
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    return df[cols].corr(method=method)
