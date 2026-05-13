"""
tsf_utils.py
============

Reusable utilities for the *Statistical Time Series Forecasting* tutorial.

All visualisations use **Plotly** so plots are interactive in Colab/Jupyter
(hover tooltips, zoom, pan, legend-click to hide/show series, double-click to
isolate, etc.).

Every helper is designed to work with **any** univariate time series provided
as a `pandas.Series` with a `DatetimeIndex` (or a regular `RangeIndex` where
indicated). Multivariate helpers accept a `pandas.DataFrame`.

Typical Colab usage
-------------------
1. Upload `tsf_utils.py` to your Colab session (left sidebar -> Files -> upload).
2. Or, if the file lives in your Drive:
       from google.colab import drive
       drive.mount('/content/drive')
       import sys; sys.path.append('/content/drive/MyDrive/<your_folder>')
3. Then in any notebook cell:
       import tsf_utils as tsf
       series = tsf.load_timeseries('your_file.csv', date_col='Date', value_col='Sales')
       tsf.plot_timeseries(series, title='My Series').show()

Every plotting function returns a `plotly.graph_objects.Figure` — call `.show()`
to display, or chain `.update_layout(...)` to customise.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Tuple, Dict, Union, Callable, List

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Default theme for the whole tutorial. Override in any notebook with:
#   pio.templates.default = 'plotly_dark'   (etc.)
pio.templates.default = "plotly_white"

# Consistent palette so the same model gets the same colour across notebooks.
PALETTE = {
    "actual":     "#111111",
    "train":      "#7f8fa6",
    "test":       "#000000",
    "forecast":   "#e74c3c",
    "trend":      "#2980b9",
    "seasonal":   "#16a085",
    "residual":   "#7d3c98",
    "level":      "#27ae60",
    "ci":         "rgba(231, 76, 60, 0.18)",
    "naive":      "#95a5a6",
    "snaive":     "#34495e",
}

DEFAULT_HEIGHT = 380   # pixels — comfortable for a single-row plot


# ---------------------------------------------------------------------------
# 1. Data loading & preparation
# ---------------------------------------------------------------------------

def load_timeseries(
    path: str,
    date_col: str,
    value_col: Optional[str] = None,
    freq: Optional[str] = None,
    parse_dates_kwargs: Optional[dict] = None,
    fill_method: Optional[str] = None,
    sep: str = ",",
) -> Union[pd.Series, pd.DataFrame]:
    """Load a CSV/Excel file as a time-indexed Series or DataFrame."""
    parse_dates_kwargs = parse_dates_kwargs or {}

    if path.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=sep)

    df[date_col] = pd.to_datetime(df[date_col], **parse_dates_kwargs)
    df = df.sort_values(date_col).set_index(date_col)

    if freq is not None:
        df = df.asfreq(freq)

    if fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "interpolate":
        df = df.interpolate(method="time")

    if value_col is not None:
        return df[value_col].rename(value_col)
    return df


def infer_seasonal_period(series: pd.Series) -> int:
    """Suggest a seasonal period from the inferred frequency."""
    freq = series.index.freqstr or pd.infer_freq(series.index)
    if freq is None:
        return 1
    f = freq.upper()
    if f.startswith("M"): return 12
    if f.startswith("Q"): return 4
    if f.startswith("W"): return 52
    if f.startswith("D"): return 7
    if f.startswith("H"): return 24
    if f.startswith("A") or f.startswith("Y"): return 1
    return 1


# ---------------------------------------------------------------------------
# 2. Visualisation helpers (Plotly)
# ---------------------------------------------------------------------------

def _base_layout(title: str = "", xlabel: str = "Date", ylabel: str = "",
                 height: int = DEFAULT_HEIGHT) -> dict:
    return dict(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )


def plot_timeseries(series: pd.Series, title: str = "", ylabel: str = "",
                    color: str = None, height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Plot a single time series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        name=series.name or "value",
        line=dict(color=color or "#1f77b4", width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=title or series.name or "Time series",
        ylabel=ylabel or series.name or "Value",
    ))
    return fig


def plot_rolling_statistics(series: pd.Series, window: int = 12,
                            title: str = "Rolling mean & std",
                            height: int = DEFAULT_HEIGHT) -> go.Figure:
    """Overlay rolling mean & std."""
    rolling_mean = series.rolling(window).mean()
    rolling_std  = series.rolling(window).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines",
                             name="Observed",
                             line=dict(color="#3498db", width=1.2),
                             opacity=0.6))
    fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean.values,
                             mode="lines", name=f"Rolling mean (w={window})",
                             line=dict(color="#e74c3c", width=2)))
    fig.add_trace(go.Scatter(x=rolling_std.index, y=rolling_std.values,
                             mode="lines", name=f"Rolling std (w={window})",
                             line=dict(color="#27ae60", width=2)))
    fig.update_layout(**_base_layout(title=title, ylabel="Value", height=height))
    return fig


def plot_seasonal_view(series: pd.Series, period: Optional[int] = None,
                       height: int = DEFAULT_HEIGHT) -> go.Figure:
    """One line per seasonal cycle."""
    if period is None:
        period = infer_seasonal_period(series)
    if period <= 1:
        raise ValueError("Seasonal period must be > 1 for a seasonal view.")

    df = series.to_frame("value").copy()
    df["cycle"]    = np.arange(len(df)) // period
    df["position"] = np.arange(len(df)) % period
    pivot = df.pivot(index="position", columns="cycle", values="value")

    fig = go.Figure()
    n_cycles = pivot.shape[1]
    palette = px.colors.sequential.Viridis
    for i, col in enumerate(pivot.columns):
        col_idx = int(i / max(n_cycles - 1, 1) * (len(palette) - 1))
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[col].values, mode="lines",
            name=str(col),
            line=dict(width=1.5, color=palette[col_idx]),
            opacity=0.8,
            hovertemplate=f"Cycle {col}<br>Position %{{x}}<br>%{{y:.2f}}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        title=f"Seasonal view (period={period}) — one line per cycle",
        xlabel="Position within cycle",
        ylabel=series.name or "Value",
        height=height,
    ))
    fig.update_layout(hovermode="closest")
    return fig


def plot_lag(series: pd.Series, lags: Iterable[int] = (1, 2, 3, 4),
             height: int = 600) -> go.Figure:
    """Scatter plots of y(t) vs y(t-k)."""
    lags = list(lags)
    n    = len(lags)
    cols = 2
    rows = (n + 1) // cols

    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Lag {k}" for k in lags])
    for i, lag in enumerate(lags):
        r, c = i // cols + 1, i % cols + 1
        x = series.shift(lag)
        fig.add_trace(go.Scatter(
            x=x.values, y=series.values, mode="markers",
            marker=dict(size=5, opacity=0.5, color="#3498db"),
            name=f"Lag {lag}", showlegend=False,
            hovertemplate=f"y(t-{lag})=%{{x:.2f}}<br>y(t)=%{{y:.2f}}<extra></extra>",
        ), row=r, col=c)
        fig.update_xaxes(title_text=f"y(t-{lag})", row=r, col=c)
        fig.update_yaxes(title_text="y(t)", row=r, col=c)
    fig.update_layout(height=height, title_text="Lag plots",
                      margin=dict(l=60, r=30, t=80, b=50))
    return fig


def decompose_and_plot(series: pd.Series, model: str = "additive",
                       period: Optional[int] = None, method: str = "classical",
                       height: int = 700):
    """Decompose & plot. Returns (statsmodels_result, plotly_figure)."""
    from statsmodels.tsa.seasonal import seasonal_decompose, STL

    if period is None:
        period = infer_seasonal_period(series)
    if period <= 1:
        raise ValueError("Cannot decompose: seasonal period <= 1. "
                         "Pass `period=` explicitly.")

    series = series.dropna()

    if method == "classical":
        result = seasonal_decompose(series, model=model, period=period)
    elif method == "stl":
        result = STL(series, period=period, robust=True).fit()
    else:
        raise ValueError("method must be 'classical' or 'stl'")

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.06,
    )
    panels = [
        ("Observed", result.observed, "#1f77b4"),
        ("Trend",    result.trend,    PALETTE["trend"]),
        ("Seasonal", result.seasonal, PALETTE["seasonal"]),
        ("Residual", result.resid,    PALETTE["residual"]),
    ]
    for i, (name, comp, color) in enumerate(panels, start=1):
        fig.add_trace(go.Scatter(
            x=comp.index, y=comp.values, mode="lines",
            name=name, line=dict(color=color, width=1.4),
            showlegend=False,
        ), row=i, col=1)
        if name == "Residual":
            fig.add_hline(y=0 if model == "additive" else 1,
                          line=dict(color="grey", dash="dot", width=1),
                          row=i, col=1)
    fig.update_layout(
        height=height,
        title_text=f"{method.upper()} decomposition (model={model}, period={period})",
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    return result, fig


def _acf_values(series: pd.Series, lags: int):
    from statsmodels.tsa.stattools import acf
    series = series.dropna()
    n = len(series)
    safe_lags = max(1, min(lags, n - 1))
    return acf(series, nlags=safe_lags, fft=True), n


def _acf_pacf_values(series: pd.Series, lags: int):
    """Compute both ACF and PACF, clamping lags to what each function allows.

    PACF requires nlags < n/2; ACF allows up to n-1. We clamp to the stricter
    of the two so both arrays line up.
    """
    from statsmodels.tsa.stattools import acf, pacf
    series = series.dropna()
    n = len(series)
    safe_lags = max(1, min(lags, n // 2 - 1))
    acf_v  = acf(series, nlags=safe_lags, fft=True)
    pacf_v = pacf(series, nlags=safe_lags, method="ywm")
    return acf_v, pacf_v, n


def plot_acf_pacf(series: pd.Series, lags: int = 40,
                  height: int = 380) -> go.Figure:
    """Side-by-side ACF and PACF with white-noise CI band."""
    acf_v, pacf_v, n = _acf_pacf_values(series, lags=lags)
    ci = 1.96 / np.sqrt(n)
    xs = np.arange(len(acf_v))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    for col_idx, vals, name in [(1, acf_v, "ACF"), (2, pacf_v, "PACF")]:
        # CI band first (lowest layer)
        fig.add_hrect(y0=-ci, y1=ci, fillcolor="rgba(52,152,219,0.15)",
                      line_width=0, row=1, col=col_idx)
        # Stems
        for x, v in zip(xs, vals):
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=v,
                          line=dict(color="#34495e", width=1.5),
                          row=1, col=col_idx)
        # Markers (so hover works)
        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="markers",
            marker=dict(color="#2c3e50", size=6),
            name=name, showlegend=False,
            hovertemplate=f"{name} lag %{{x}}<br>%{{y:.3f}}<extra></extra>",
        ), row=1, col=col_idx)
        fig.add_hline(y=0, line=dict(color="black", width=0.6),
                      row=1, col=col_idx)

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Correlation", row=1, col=1)
    fig.update_layout(height=height, margin=dict(l=60, r=30, t=60, b=50),
                      showlegend=False)
    return fig


def plot_forecast(train: pd.Series, test: Optional[pd.Series],
                  forecast: pd.Series, conf_int: Optional[pd.DataFrame] = None,
                  title: str = "Forecast vs actual",
                  height: int = 420) -> go.Figure:
    """Plot history, hold-out actual, forecast, and (optional) CI band."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train.index, y=train.values, mode="lines",
        name="Train", line=dict(color=PALETTE["train"], width=1.2),
        opacity=0.85,
    ))
    if test is not None:
        fig.add_trace(go.Scatter(
            x=test.index, y=test.values, mode="lines",
            name="Actual (test)", line=dict(color=PALETTE["test"], width=1.6),
        ))

    if conf_int is not None:
        lo, hi = conf_int.iloc[:, 0], conf_int.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=forecast.index, y=hi.values, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast.index, y=lo.values, mode="lines",
            line=dict(width=0), fill="tonexty", fillcolor=PALETTE["ci"],
            name="Confidence interval", hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values, mode="lines",
        name="Forecast", line=dict(color=PALETTE["forecast"], width=2),
    ))
    fig.update_layout(**_base_layout(title=title, ylabel="Value", height=height))
    return fig


def plot_residual_diagnostics(residuals: pd.Series, lags: int = 30,
                              height: int = 700) -> go.Figure:
    """Four-panel diagnostic plot for model residuals."""
    from scipy import stats
    res = residuals.dropna()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Residuals over time", "Histogram + Normal pdf",
                        "Q-Q plot", "Residual ACF"),
        vertical_spacing=0.15, horizontal_spacing=0.10,
    )

    # 1) residuals over time
    fig.add_trace(go.Scatter(
        x=res.index, y=res.values, mode="lines",
        line=dict(color="#34495e", width=1.2), showlegend=False,
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="grey", dash="dot"), row=1, col=1)

    # 2) histogram + normal pdf
    fig.add_trace(go.Histogram(
        x=res.values, nbinsx=30, histnorm="probability density",
        marker_color="#3498db", opacity=0.65, showlegend=False,
    ), row=1, col=2)
    xs = np.linspace(res.min(), res.max(), 200)
    pdf = stats.norm.pdf(xs, res.mean(), res.std())
    fig.add_trace(go.Scatter(
        x=xs, y=pdf, mode="lines", line=dict(color="#e74c3c", width=2),
        showlegend=False,
    ), row=1, col=2)

    # 3) Q-Q plot
    (osm, osr), (slope, intercept, _) = stats.probplot(res, dist="norm")
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color="#2c3e50", size=6), showlegend=False,
    ), row=2, col=1)
    line_x = np.array([osm.min(), osm.max()])
    fig.add_trace(go.Scatter(
        x=line_x, y=slope * line_x + intercept, mode="lines",
        line=dict(color="#e74c3c", width=2, dash="dash"), showlegend=False,
    ), row=2, col=1)

    # 4) ACF on residuals
    acf_v, n = _acf_values(res, lags=lags)
    ci = 1.96 / np.sqrt(n)
    xs2 = np.arange(len(acf_v))
    fig.add_hrect(y0=-ci, y1=ci, fillcolor="rgba(52,152,219,0.15)",
                  line_width=0, row=2, col=2)
    for x, v in zip(xs2, acf_v):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=v,
                      line=dict(color="#34495e", width=1.5),
                      row=2, col=2)
    fig.add_trace(go.Scatter(
        x=xs2, y=acf_v, mode="markers",
        marker=dict(color="#2c3e50", size=6), showlegend=False,
    ), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="black", width=0.6), row=2, col=2)

    fig.update_xaxes(title_text="Date",          row=1, col=1)
    fig.update_yaxes(title_text="Residual",      row=1, col=1)
    fig.update_xaxes(title_text="Residual",      row=1, col=2)
    fig.update_yaxes(title_text="Density",       row=1, col=2)
    fig.update_xaxes(title_text="Theoretical Q", row=2, col=1)
    fig.update_yaxes(title_text="Sample Q",      row=2, col=1)
    fig.update_xaxes(title_text="Lag",           row=2, col=2)
    fig.update_yaxes(title_text="Correlation",   row=2, col=2)
    fig.update_layout(height=height, showlegend=False,
                      margin=dict(l=60, r=30, t=80, b=50))
    return fig


def plot_distribution(series: pd.Series, height: int = 380) -> go.Figure:
    """Histogram + boxplot for outlier and shape inspection."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Histogram", "Boxplot"),
                        column_widths=[0.7, 0.3])
    fig.add_trace(go.Histogram(x=series.dropna().values, nbinsx=30,
                               marker_color="#3498db", opacity=0.75,
                               showlegend=False), row=1, col=1)
    fig.add_trace(go.Box(y=series.dropna().values, boxpoints="suspectedoutliers",
                         marker_color="#3498db", showlegend=False,
                         name=series.name or "value"), row=1, col=2)
    fig.update_layout(height=height, margin=dict(l=60, r=30, t=60, b=50))
    return fig


def plot_multi_forecast(train: pd.Series, test: pd.Series,
                        forecasts: Dict[str, pd.Series],
                        title: str = "Forecasts vs actuals",
                        height: int = 460) -> go.Figure:
    """Compare several named forecasts on one figure (legend-toggleable)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines",
                             name="Train", line=dict(color="#bdc3c7", width=1.1)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines",
                             name="Actual", line=dict(color="black", width=1.8)))
    for name, fc in forecasts.items():
        fig.add_trace(go.Scatter(
            x=fc.index, y=fc.values, mode="lines",
            name=name, line=dict(width=1.4, dash="dash"),
        ))
    fig.update_layout(**_base_layout(title=title, ylabel="Value", height=height))
    return fig


def plot_compare_pair(series_a: pd.Series, series_b: pd.Series,
                      titles: Tuple[str, str] = ("Original", "Transformed"),
                      height: int = 500) -> go.Figure:
    """Two stacked panels — original vs transformed view."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=titles, vertical_spacing=0.12)
    fig.add_trace(go.Scatter(x=series_a.index, y=series_a.values, mode="lines",
                             line=dict(color="#1f77b4", width=1.4),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=series_b.index, y=series_b.values, mode="lines",
                             line=dict(color="#e67e22", width=1.4),
                             showlegend=False), row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=60, r=30, t=60, b=50),
                      hovermode="x unified")
    return fig


# ---------------------------------------------------------------------------
# 3. Statistical tests
# ---------------------------------------------------------------------------

def adf_test(series: pd.Series, regression: str = "c", verbose: bool = True
             ) -> Dict[str, float]:
    """Augmented Dickey-Fuller. H0: unit root (non-stationary)."""
    from statsmodels.tsa.stattools import adfuller
    res = adfuller(series.dropna(), regression=regression, autolag="AIC")
    out = {"statistic": res[0], "p_value": res[1],
           "n_lags": res[2], "n_obs": res[3], "critical_values": res[4]}
    if verbose:
        print("Augmented Dickey-Fuller (ADF) test")
        print("-" * 38)
        print(f"  Test statistic : {out['statistic']:.4f}")
        print(f"  p-value        : {out['p_value']:.4f}")
        print(f"  Lags used      : {out['n_lags']}")
        print(f"  N observations : {out['n_obs']}")
        for k, v in out["critical_values"].items():
            print(f"  Critical {k:>4}: {v:.4f}")
        verdict = "STATIONARY" if out["p_value"] < 0.05 else "NON-STATIONARY"
        print(f"  -> At alpha=0.05, series is {verdict}")
    return out


def kpss_test(series: pd.Series, regression: str = "c", verbose: bool = True
              ) -> Dict[str, float]:
    """KPSS test. H0: stationary."""
    from statsmodels.tsa.stattools import kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = kpss(series.dropna(), regression=regression, nlags="auto")
    out = {"statistic": res[0], "p_value": res[1],
           "n_lags": res[2], "critical_values": res[3]}
    if verbose:
        print("KPSS test")
        print("-" * 38)
        print(f"  Test statistic : {out['statistic']:.4f}")
        print(f"  p-value        : {out['p_value']:.4f}")
        print(f"  Lags used      : {out['n_lags']}")
        for k, v in out["critical_values"].items():
            print(f"  Critical {k:>4}: {v:.4f}")
        verdict = "NON-STATIONARY" if out["p_value"] < 0.05 else "STATIONARY"
        print(f"  -> At alpha=0.05, series is {verdict}")
    return out


def stationarity_report(series: pd.Series) -> None:
    """Run ADF + KPSS and print a combined verdict."""
    print("=" * 50)
    adf = adf_test(series, verbose=True);  print()
    kp  = kpss_test(series, verbose=True); print()
    adf_stat  = adf["p_value"] < 0.05
    kpss_stat = kp["p_value"] >= 0.05
    if adf_stat and kpss_stat:
        verdict = "Both tests agree: STATIONARY"
    elif (not adf_stat) and (not kpss_stat):
        verdict = "Both tests agree: NON-STATIONARY (consider differencing)"
    elif adf_stat and not kpss_stat:
        verdict = "Conflict: possibly DIFFERENCE-STATIONARY"
    else:
        verdict = "Conflict: possibly TREND-STATIONARY (try detrending)"
    print(">>> Combined verdict:", verdict)
    print("=" * 50)


def ljung_box_test(residuals: pd.Series, lags: int = 10, verbose: bool = True
                   ) -> pd.DataFrame:
    """Ljung-Box test. H0: residuals are independent (white noise)."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    out = acorr_ljungbox(residuals.dropna(), lags=lags, return_df=True)
    if verbose:
        print("Ljung-Box test (H0: no autocorrelation)")
        print(out.to_string())
        if (out["lb_pvalue"] > 0.05).all():
            print(">>> All p-values > 0.05 — residuals look like white noise.")
        else:
            print(">>> Some p-values <= 0.05 — residual autocorrelation present.")
    return out


# ---------------------------------------------------------------------------
# 4. Train/test split & cross-validation
# ---------------------------------------------------------------------------

def train_test_split_ts(series: pd.Series, test_size: Union[int, float] = 0.2
                        ) -> Tuple[pd.Series, pd.Series]:
    """Chronological split (NO shuffling)."""
    n = len(series)
    n_test = int(np.ceil(n * test_size)) if isinstance(test_size, float) else int(test_size)
    if n_test <= 0 or n_test >= n:
        raise ValueError("test_size must leave at least one row in each split.")
    return series.iloc[:-n_test].copy(), series.iloc[-n_test:].copy()


def walk_forward_validation(series: pd.Series,
                            forecast_func: Callable[[pd.Series, int], pd.Series],
                            n_test: int, step: int = 1, horizon: int = 1,
                            verbose: bool = False) -> pd.DataFrame:
    """Rolling-origin / walk-forward validation."""
    if n_test <= 0 or n_test >= len(series):
        raise ValueError("Invalid n_test.")

    cutoff = len(series) - n_test
    actual_records, pred_records = [], []
    i = 0
    while cutoff + i + horizon <= len(series):
        history    = series.iloc[: cutoff + i]
        future_idx = series.index[cutoff + i : cutoff + i + horizon]
        try:
            yhat = forecast_func(history, horizon)
        except Exception as e:
            if verbose:
                print(f"  step {i}: failed -> {e}")
            i += step
            continue
        if not isinstance(yhat, pd.Series):
            yhat = pd.Series(np.asarray(yhat).ravel(), index=future_idx)
        else:
            yhat.index = future_idx
        for ts in future_idx:
            actual_records.append((ts, series.loc[ts]))
            pred_records.append((ts, yhat.loc[ts]))
        i += step

    actual = pd.Series(dict(actual_records)).sort_index()
    pred   = pd.Series(dict(pred_records)).sort_index()
    return pd.DataFrame({"actual": actual, "predicted": pred})


# ---------------------------------------------------------------------------
# 5. Forecasting metrics
# ---------------------------------------------------------------------------

def _to_array(x): return np.asarray(x).astype(float).ravel()

def mae(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    if mask.sum() == 0: return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)

def mase(y_true, y_pred, y_train, seasonality: int = 1):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    y_train = _to_array(y_train)
    if len(y_train) <= seasonality: return float("nan")
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0: return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def me(y_true, y_pred):
    y_true, y_pred = _to_array(y_true), _to_array(y_pred)
    return float(np.mean(y_pred - y_true))


def forecast_metrics(y_true, y_pred, y_train: Optional[pd.Series] = None,
                     seasonality: int = 1) -> Dict[str, float]:
    out = {"MAE": mae(y_true, y_pred), "MSE": mse(y_true, y_pred),
           "RMSE": rmse(y_true, y_pred), "MAPE": mape(y_true, y_pred),
           "sMAPE": smape(y_true, y_pred), "ME": me(y_true, y_pred)}
    if y_train is not None:
        out["MASE"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    return out


def metrics_dataframe(metrics_by_model: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(metrics_by_model).T.round(4)


def compare_models(y_train: pd.Series, y_test: pd.Series,
                   forecasts: Dict[str, pd.Series],
                   seasonality: int = 1) -> pd.DataFrame:
    rows = {}
    for name, fc in forecasts.items():
        fc_aligned = fc.reindex(y_test.index)
        rows[name] = forecast_metrics(y_test.values, fc_aligned.values,
                                      y_train=y_train, seasonality=seasonality)
    return metrics_dataframe(rows)


# ---------------------------------------------------------------------------
# 6. Simple baseline forecasters
# ---------------------------------------------------------------------------

def _future_index(history: pd.Series, horizon: int) -> pd.DatetimeIndex:
    freq = history.index.freq or pd.infer_freq(history.index)
    return pd.date_range(start=history.index[-1], periods=horizon + 1, freq=freq)[1:]


def naive_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series([history.iloc[-1]] * horizon, index=_future_index(history, horizon))


def seasonal_naive_forecast(history: pd.Series, horizon: int,
                            season_length: int) -> pd.Series:
    if len(history) < season_length:
        raise ValueError("History shorter than one seasonal cycle.")
    last = history.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    fc   = np.tile(last, reps)[:horizon]
    return pd.Series(fc, index=_future_index(history, horizon))


def mean_forecast(history: pd.Series, horizon: int) -> pd.Series:
    return pd.Series([history.mean()] * horizon, index=_future_index(history, horizon))


def drift_forecast(history: pd.Series, horizon: int) -> pd.Series:
    n = len(history)
    if n < 2: return naive_forecast(history, horizon)
    slope = (history.iloc[-1] - history.iloc[0]) / (n - 1)
    last  = history.iloc[-1]
    fc = [last + slope * (h + 1) for h in range(horizon)]
    return pd.Series(fc, index=_future_index(history, horizon))


# ---------------------------------------------------------------------------
# 7. Intermittent / lumpy demand: classification and Croston-family methods
# ---------------------------------------------------------------------------

def classify_demand_pattern(series: pd.Series, verbose: bool = True) -> Dict:
    """Syntetos-Boylan-Croston (SBC) classification of demand patterns.

    Two summary statistics drive the classification:

    - **ADI** (Average Demand Interval) = N_obs / N_nonzero
        Larger ADI ⇒ more zeros between non-zero events.
    - **CV²** (squared coefficient of variation of *non-zero* demands)
        Larger CV² ⇒ more variable demand sizes when demand occurs.

    Categories (using Syntetos-Boylan-Croston thresholds ADI=1.32, CV²=0.49):

    +-----------------+-------------+-----------------------+
    |                 | CV² < 0.49  | CV² ≥ 0.49            |
    +=================+=============+=======================+
    | **ADI < 1.32**  | smooth      | erratic               |
    +-----------------+-------------+-----------------------+
    | **ADI ≥ 1.32**  | intermittent| lumpy                 |
    +-----------------+-------------+-----------------------+
    """
    y = np.asarray(series.dropna().values).astype(float)
    if (y < 0).any():
        raise ValueError("Demand series must be non-negative.")

    nonzero = y[y > 0]
    n_nonzero = len(nonzero)
    n_obs = len(y)

    if n_nonzero < 2:
        out = {"category": "insufficient_data", "ADI": np.inf,
               "CV2": np.nan, "n_obs": n_obs, "n_nonzero": n_nonzero,
               "pct_zero": (1 - n_nonzero / n_obs) * 100 if n_obs else np.nan}
        if verbose:
            print(f"Only {n_nonzero} non-zero values — cannot classify.")
        return out

    ADI = n_obs / n_nonzero
    CV2 = (nonzero.std(ddof=1) / nonzero.mean()) ** 2

    if ADI < 1.32 and CV2 < 0.49:
        cat = "smooth"
    elif ADI < 1.32 and CV2 >= 0.49:
        cat = "erratic"
    elif ADI >= 1.32 and CV2 < 0.49:
        cat = "intermittent"
    else:
        cat = "lumpy"

    out = {"category": cat, "ADI": ADI, "CV2": CV2,
           "n_obs": n_obs, "n_nonzero": n_nonzero,
           "pct_zero": (1 - n_nonzero / n_obs) * 100}

    if verbose:
        print(f"  Total observations    : {n_obs}")
        print(f"  Non-zero observations : {n_nonzero}  ({n_nonzero/n_obs*100:.1f}%)")
        print(f"  Zero observations     : {n_obs - n_nonzero}  ({(1-n_nonzero/n_obs)*100:.1f}%)")
        print(f"  ADI (avg interval)    : {ADI:.3f}     "
              f"{'⇒ intermittent timing' if ADI >= 1.32 else '⇒ regular timing'}")
        print(f"  CV² of non-zero sizes : {CV2:.3f}     "
              f"{'⇒ variable sizes' if CV2 >= 0.49 else '⇒ stable sizes'}")
        print(f"  → Classification: {cat.upper()}")
        suggestion = {
            "smooth":       "Standard ES / ARIMA work well.",
            "erratic":      "Try ES variants or median-based methods.",
            "intermittent": "Use Croston's method (classic or SBA).",
            "lumpy":        "Hardest case. Try Croston-SBA or TSB; "
                            "also consider zero-inflated regression.",
            "insufficient_data": "Need more non-zero observations.",
        }
        print(f"  → Suggested approach: {suggestion[cat]}")
    return out


def croston(history: pd.Series, horizon: int,
            alpha: float = 0.1, variant: str = "classic") -> pd.Series:
    """Croston's method for intermittent demand.

    The series is decomposed into:
      - **demand sizes** z_t (the non-zero values)
      - **inter-demand intervals** p_t (gaps between non-zero events)
    Each is smoothed via simple exponential smoothing with the same α.
    The forecast is the rate of demand per period:  ŷ = z / p (constant).

    Parameters
    ----------
    history : pd.Series of non-negative values.
    horizon : forecast horizon (a constant rate is repeated `horizon` times).
    alpha   : smoothing parameter in (0, 1).
    variant : {"classic", "sba"}
        - "classic" — original Croston (1972). Biased upward in expectation.
        - "sba"     — Syntetos-Boylan Approximation. Multiplies the rate by
                      (1 - α/2) to correct the bias. Recommended in most cases.
    """
    if variant not in {"classic", "sba"}:
        raise ValueError("variant must be 'classic' or 'sba' (see croston_tsb for TSB).")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")

    y = np.asarray(history.values).astype(float)
    n = len(y)
    future = _future_index(history, horizon)

    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) == 0:
        return pd.Series([0.0] * horizon, index=future)
    if len(nonzero_idx) == 1:
        rate = y[nonzero_idx[0]] / n
        return pd.Series([rate] * horizon, index=future)

    # Initialize on first observed demand
    z = y[nonzero_idx[0]]
    p = float(nonzero_idx[0] + 1)   # interval from "virtual t=0" to first demand
    last = nonzero_idx[0]

    for t in range(nonzero_idx[0] + 1, n):
        if y[t] > 0:
            q = t - last
            z = alpha * y[t] + (1 - alpha) * z
            p = alpha * q + (1 - alpha) * p
            last = t

    rate = z / p
    if variant == "sba":
        rate *= (1 - alpha / 2)

    return pd.Series([rate] * horizon, index=future)


def croston_tsb(history: pd.Series, horizon: int,
                alpha: float = 0.1, beta: float = 0.1) -> pd.Series:
    """Teunter-Syntetos-Babai (TSB) method for intermittent demand.

    Instead of tracking the inter-demand *interval*, TSB tracks the
    **probability of demand** in any given period. This makes it the right
    choice when **obsolescence** is a concern — if demand has stopped, the
    probability decays toward zero and so does the forecast (Croston's
    classical / SBA forecasts stay flat regardless).

    Parameters
    ----------
    alpha : smoothing for demand size.
    beta  : smoothing for demand probability.
    """
    if not 0 < alpha < 1 or not 0 < beta < 1:
        raise ValueError("alpha and beta must lie in (0, 1).")

    y = np.asarray(history.values).astype(float)
    n = len(y)
    future = _future_index(history, horizon)

    nonzero_idx = np.where(y > 0)[0]
    if len(nonzero_idx) == 0:
        return pd.Series([0.0] * horizon, index=future)

    z = y[nonzero_idx[0]]
    p = 1.0 / (nonzero_idx[0] + 1)   # initial demand probability

    for t in range(nonzero_idx[0] + 1, n):
        if y[t] > 0:
            z = alpha * y[t] + (1 - alpha) * z
            p = beta * 1.0 + (1 - beta) * p
        else:
            p = beta * 0.0 + (1 - beta) * p

    rate = z * p
    return pd.Series([rate] * horizon, index=future)


def plot_intermittent_series(series: pd.Series, title: str = "",
                             height: int = 380) -> go.Figure:
    """Bar plot of an intermittent demand series with zero/non-zero coloring."""
    y = series.values
    idx = series.index
    colors = ["#e74c3c" if v > 0 else "#ecf0f1" for v in y]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=idx, y=y, marker_color=colors, marker_line=dict(width=0),
        hovertemplate="%{x|%Y-%m-%d}<br>demand: %{y}<extra></extra>",
    ))

    n_nonzero = int((y > 0).sum())
    pct = n_nonzero / len(y) * 100 if len(y) else 0
    annotation = f"{n_nonzero}/{len(y)} non-zero periods ({pct:.0f}%)"

    fig.update_layout(
        title=title or "Intermittent demand",
        xaxis_title="Date", yaxis_title=series.name or "Demand",
        height=height, margin=dict(l=60, r=30, t=60, b=50),
        annotations=[dict(text=annotation, xref="paper", yref="paper",
                          x=0.02, y=0.95, showarrow=False,
                          bgcolor="rgba(255,255,255,0.85)",
                          bordercolor="#bdc3c7", borderwidth=1)],
        showlegend=False,
    )
    return fig


def simulate_intermittent_demand(n_periods: int = 120,
                                 demand_prob: float = 0.25,
                                 mean_size: float = 8.0,
                                 size_cv: float = 0.35,
                                 obsolescence_start: Optional[int] = None,
                                 freq: str = "MS",
                                 start: str = "2018-01-01",
                                 seed: int = 42) -> pd.Series:
    """Simulate a realistic intermittent demand series for teaching examples.

    Parameters
    ----------
    n_periods         : length of the series.
    demand_prob       : per-period probability of demand occurring.
    mean_size         : mean of non-zero demand sizes.
    size_cv           : coefficient of variation of demand sizes (controls CV²).
    obsolescence_start: if set, demand probability decays to 0 after this index
                        — useful for demonstrating TSB.
    """
    rng = np.random.default_rng(seed)
    occurs = np.zeros(n_periods, dtype=int)

    for t in range(n_periods):
        prob = demand_prob
        if obsolescence_start is not None and t >= obsolescence_start:
            decay = (t - obsolescence_start) / max(1, n_periods - obsolescence_start)
            prob = demand_prob * max(0.0, 1.0 - decay * 1.5)
        occurs[t] = rng.binomial(1, prob)

    # Lognormal sizes with target mean and CV
    sigma2 = np.log(1 + size_cv ** 2)
    mu = np.log(mean_size) - sigma2 / 2
    sizes = rng.lognormal(mean=mu, sigma=np.sqrt(sigma2), size=n_periods)
    sizes = np.round(sizes).astype(int)
    sizes[sizes < 1] = 1

    y = occurs * sizes
    idx = pd.date_range(start=start, periods=n_periods, freq=freq)
    return pd.Series(y, index=idx, name="demand")
