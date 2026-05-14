"""
viz.py
======
Plotly-based visualisation helpers for forecasting workflows.

All functions return a `plotly.graph_objects.Figure` so the caller can either
display (`fig.show()`) or compose into a dashboard / subplot.

Theming
-------
We use a single shared light theme so every chart in the tutorial looks the
same. Override at runtime via `set_theme(...)`.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
DEFAULT_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=40, r=20, t=60, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(size=12),
)

PALETTE = dict(
    actual="#1f2937",       # near-black
    forecast="#2563eb",     # blue
    forecast2="#dc2626",    # red
    forecast3="#059669",    # green
    interval="rgba(37, 99, 235, 0.18)",   # translucent blue
    history="#9ca3af",      # gray
    grid="#e5e7eb",
)


def _apply_theme(fig: go.Figure, **overrides) -> go.Figure:
    fig.update_layout(**{**DEFAULT_LAYOUT, **overrides})
    return fig


# ---------------------------------------------------------------------------
# Time-series exploration
# ---------------------------------------------------------------------------
def plot_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    title: str = "Time series",
    height: int = 380,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[date_col], y=df[target_col],
        mode="lines", name=target_col,
        line=dict(color=PALETTE["actual"], width=1.5),
    ))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="Date", yaxis_title=target_col)


def plot_multi_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    key_cols: Sequence[str],
    n_keys: int = 8,
    title: str = "Sample of forecast keys",
    height: int = 480,
) -> go.Figure:
    """Faceted line chart for a sample of forecast keys."""
    keys_df = df[list(key_cols)].drop_duplicates().head(n_keys)
    sample = df.merge(keys_df, on=list(key_cols), how="inner")
    sample["__key"] = sample[list(key_cols)].astype(str).agg(" | ".join, axis=1)
    fig = px.line(sample, x=date_col, y=target_col, color="__key", facet_col="__key",
                  facet_col_wrap=4, facet_col_spacing=0.05, facet_row_spacing=0.12,
                  height=height)
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_layout(showlegend=False, title=title)
    return _apply_theme(fig, title=title, height=height)


def plot_demand_classification(
    classification: pd.DataFrame,
    title: str = "Syntetos–Boylan demand classification",
    height: int = 480,
) -> go.Figure:
    """Scatter of CV² vs ADI with quadrant boundaries at (1.32, 0.49)."""
    fig = px.scatter(
        classification, x="ADI", y="CV2", color="pattern",
        hover_data=[c for c in classification.columns if c not in ("ADI", "CV2", "pattern")],
        log_x=True, height=height,
        color_discrete_map={
            "Smooth": "#10b981", "Erratic": "#f59e0b",
            "Intermittent": "#3b82f6", "Lumpy": "#dc2626",
        },
    )
    fig.add_vline(x=1.32, line_dash="dash", line_color=PALETTE["history"])
    fig.add_hline(y=0.49, line_dash="dash", line_color=PALETTE["history"])
    fig.add_annotation(x=1.0,  y=0.1,  text="Smooth",       showarrow=False)
    fig.add_annotation(x=1.0,  y=2.0,  text="Erratic",      showarrow=False)
    fig.add_annotation(x=4.0,  y=0.1,  text="Intermittent", showarrow=False)
    fig.add_annotation(x=4.0,  y=2.0,  text="Lumpy",        showarrow=False)
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="ADI (log scale)", yaxis_title="CV²")


# ---------------------------------------------------------------------------
# Loss-function shapes
# ---------------------------------------------------------------------------
def plot_loss_shapes(
    y_true: float = 5.0,
    y_pred_grid: Optional[np.ndarray] = None,
    title: str = "Loss as a function of prediction (y_true=5)",
    height: int = 420,
) -> go.Figure:
    """
    Visualise how MSE, MAE, Poisson and Tweedie penalise different prediction
    values for a fixed y_true. Helps students see why Tweedie's curve is
    steeper for under-prediction at low counts (zero-inflation friendly).
    """
    if y_pred_grid is None:
        y_pred_grid = np.linspace(0.01, 15, 400)
    yt = max(y_true, 1e-9)
    yp = y_pred_grid

    mse_loss   = (yt - yp) ** 2
    mae_loss   = np.abs(yt - yp)
    # Poisson NLL (up to constant): yp - yt*log(yp)
    pois_loss  = yp - yt * np.log(yp)
    pois_loss  = pois_loss - pois_loss.min()
    # Tweedie deviance for power p in (1,2)
    p = 1.5
    tw = 2 * (yt ** (2 - p) / ((1 - p) * (2 - p))
              - yt * yp ** (1 - p) / (1 - p)
              + yp ** (2 - p) / (2 - p))
    tw = tw - tw.min()

    fig = go.Figure()
    for name, loss, color in [("MSE", mse_loss, "#1f77b4"),
                              ("MAE", mae_loss, "#9467bd"),
                              ("Poisson NLL", pois_loss, "#2ca02c"),
                              ("Tweedie p=1.5", tw, "#d62728")]:
        fig.add_trace(go.Scatter(x=yp, y=loss, mode="lines", name=name, line=dict(width=2)))
    fig.add_vline(x=yt, line_dash="dash", line_color="gray",
                  annotation_text=f"y_true = {yt}", annotation_position="top right")
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="prediction y_pred", yaxis_title="loss (zero-shifted)")


# ---------------------------------------------------------------------------
# Forecast plots
# ---------------------------------------------------------------------------
def plot_forecast(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    date_col: str,
    target_col: str,
    yhat_col: str = "yhat",
    yhat_lower_col: Optional[str] = None,
    yhat_upper_col: Optional[str] = None,
    history: Optional[pd.DataFrame] = None,
    title: str = "Forecast vs actual",
    height: int = 420,
) -> go.Figure:
    """
    Plot a single-key forecast with optional history and prediction interval.
    Pass `history` (the training window) to show context before the forecast.
    """
    fig = go.Figure()
    if history is not None:
        fig.add_trace(go.Scatter(
            x=history[date_col], y=history[target_col],
            mode="lines", name="history",
            line=dict(color=PALETTE["history"], width=1.2),
        ))
    fig.add_trace(go.Scatter(
        x=actual[date_col], y=actual[target_col],
        mode="lines+markers", name="actual",
        line=dict(color=PALETTE["actual"], width=2),
        marker=dict(size=4),
    ))
    if yhat_lower_col is not None and yhat_upper_col is not None:
        fig.add_trace(go.Scatter(
            x=predicted[date_col], y=predicted[yhat_upper_col],
            line=dict(width=0), mode="lines", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=predicted[date_col], y=predicted[yhat_lower_col],
            fill="tonexty", fillcolor=PALETTE["interval"],
            line=dict(width=0), mode="lines", name="prediction interval",
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=predicted[date_col], y=predicted[yhat_col],
        mode="lines", name="forecast",
        line=dict(color=PALETTE["forecast"], width=2, dash="solid"),
    ))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="Date", yaxis_title=target_col)


def plot_forecast_grid(
    forecasts: Dict[str, pd.DataFrame],         # name -> df with date_col, yhat
    actual: pd.DataFrame,
    date_col: str,
    target_col: str,
    history: Optional[pd.DataFrame] = None,
    title: str = "Model comparison",
    height: int = 460,
) -> go.Figure:
    """Overlay multiple model forecasts against the same actual series."""
    fig = go.Figure()
    if history is not None:
        fig.add_trace(go.Scatter(x=history[date_col], y=history[target_col],
                                 mode="lines", name="history",
                                 line=dict(color=PALETTE["history"], width=1.2)))
    fig.add_trace(go.Scatter(x=actual[date_col], y=actual[target_col],
                             mode="lines+markers", name="actual",
                             line=dict(color=PALETTE["actual"], width=2),
                             marker=dict(size=4)))
    palette = px.colors.qualitative.Set1
    for i, (name, df) in enumerate(forecasts.items()):
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df["yhat"], mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="Date", yaxis_title=target_col)


# ---------------------------------------------------------------------------
# Residual diagnostics
# ---------------------------------------------------------------------------
def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual diagnostics",
    height: int = 420,
) -> go.Figure:
    res = np.asarray(y_true) - np.asarray(y_pred)
    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Predicted vs actual", "Residuals over index", "Residual histogram"))
    fig.add_trace(go.Scatter(x=y_pred, y=y_true, mode="markers",
                             marker=dict(size=4, opacity=0.5, color=PALETTE["forecast"]),
                             name=""), row=1, col=1)
    lim = float(max(np.max(y_true), np.max(y_pred)))
    fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                             line=dict(color="gray", dash="dash"),
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=res, mode="markers",
                             marker=dict(size=4, opacity=0.5, color=PALETTE["forecast2"]),
                             name=""), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_trace(go.Histogram(x=res, nbinsx=40, marker=dict(color=PALETTE["forecast3"]),
                               name=""), row=1, col=3)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="y_pred", row=1, col=1)
    fig.update_yaxes(title_text="y_true", row=1, col=1)
    fig.update_xaxes(title_text="row index", row=1, col=2)
    fig.update_yaxes(title_text="residual", row=1, col=2)
    fig.update_xaxes(title_text="residual", row=1, col=3)
    return _apply_theme(fig, title=title, height=height)


# ---------------------------------------------------------------------------
# CV folds
# ---------------------------------------------------------------------------
def plot_cv_folds(folds, title: str = "CV folds", height: int = 360) -> go.Figure:
    """Gantt-style chart of train/valid windows per fold."""
    rows = []
    for f in folds:
        rows.append({"fold": f"fold {f.fold}", "phase": "train",
                     "start": f.train_start, "end": f.train_end})
        rows.append({"fold": f"fold {f.fold}", "phase": "valid",
                     "start": f.valid_start, "end": f.valid_end})
    df = pd.DataFrame(rows)
    fig = px.timeline(df, x_start="start", x_end="end", y="fold", color="phase",
                      color_discrete_map={"train": "#3b82f6", "valid": "#dc2626"})
    fig.update_yaxes(autorange="reversed")
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="Date", yaxis_title="")


# ---------------------------------------------------------------------------
# Importance / SHAP
# ---------------------------------------------------------------------------
def plot_importance(
    importance_df: pd.DataFrame,
    top_n: int = 25,
    value_col: str = "importance",
    feature_col: str = "feature",
    title: str = "Feature importance",
    height: int = 520,
) -> go.Figure:
    df = importance_df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df[value_col], y=df[feature_col], orientation="h",
        marker=dict(color=PALETTE["forecast"]),
    ))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title=value_col, yaxis_title="")


def plot_shap_summary(
    shap_df: pd.DataFrame,
    top_n: int = 25,
    title: str = "SHAP global importance (mean |shap|)",
    height: int = 540,
) -> go.Figure:
    df = shap_df.head(top_n).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["mean_abs_shap"], y=df["feature"], orientation="h",
        marker=dict(color=df["mean_shap"],
                    colorscale="RdBu", cmid=0, showscale=True,
                    colorbar=dict(title="mean SHAP<br>(signed)")),
    ))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="mean |SHAP|", yaxis_title="")


def plot_shap_waterfall(
    local_df: pd.DataFrame,
    top_n: int = 15,
    title: str = "Per-prediction SHAP waterfall",
    height: int = 480,
) -> go.Figure:
    """Plotly waterfall for a single prediction's SHAP attributions."""
    base = local_df.attrs.get("base_value", 0.0)
    pred = local_df.attrs.get("prediction", base + local_df["shap"].sum())
    df = local_df.head(top_n).iloc[::-1]
    labels = [f"{r.feature} = {r.value}" for r in df.itertuples()]
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(df),
        x=df["shap"], y=labels,
        connector=dict(line=dict(color="rgb(180,180,180)")),
    ))
    fig.add_annotation(x=0, y=-1, text=f"base value = {base:.3f}",
                       showarrow=False, xref="x", yref="y")
    fig.add_annotation(x=0, y=len(df), text=f"prediction = {pred:.3f}",
                       showarrow=False, xref="x", yref="y")
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="SHAP contribution", yaxis_title="")


def plot_shap_dependence(dependence_df: pd.DataFrame, feature: str,
                         title: Optional[str] = None, height: int = 380) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=dependence_df[feature], y=dependence_df["shap"], mode="markers",
        marker=dict(size=5, opacity=0.5, color=PALETTE["forecast"]),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    return _apply_theme(fig,
                        title=title or f"SHAP dependence: {feature}",
                        height=height,
                        xaxis_title=feature, yaxis_title="SHAP value")


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------
def plot_optuna_history(history: pd.DataFrame, title: str = "Optuna optimization history",
                        height: int = 380) -> go.Figure:
    h = history.dropna(subset=["value"]).copy()
    h["best_so_far"] = h["value"].cummin()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["number"], y=h["value"], mode="markers",
                             marker=dict(size=6, color=PALETTE["forecast"]),
                             name="trial value"))
    fig.add_trace(go.Scatter(x=h["number"], y=h["best_so_far"], mode="lines",
                             line=dict(color=PALETTE["forecast2"], width=2),
                             name="best so far"))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="trial #", yaxis_title="objective")


# ---------------------------------------------------------------------------
# Prophet decomposition
# ---------------------------------------------------------------------------
def plot_prophet_components(decomp: pd.DataFrame, title: str = "Prophet components",
                            height: int = 720) -> go.Figure:
    cols = [c for c in ["trend", "yearly", "weekly", "daily", "holidays"] if c in decomp.columns]
    fig = make_subplots(rows=len(cols) + 1, cols=1, shared_xaxes=True,
                        subplot_titles=["Fit (yhat vs y)"] + cols, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=decomp["ds"], y=decomp["y"], mode="lines",
                             line=dict(color=PALETTE["actual"], width=1.3),
                             name="y"), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomp["ds"], y=decomp["yhat"], mode="lines",
                             line=dict(color=PALETTE["forecast"], width=1.3),
                             name="yhat"), row=1, col=1)
    for i, c in enumerate(cols, start=2):
        fig.add_trace(go.Scatter(x=decomp["ds"], y=decomp[c], mode="lines",
                                 name=c), row=i, col=1)
    return _apply_theme(fig, title=title, height=height, showlegend=False)


# ---------------------------------------------------------------------------
# Probabilistic forecast (quantile)
# ---------------------------------------------------------------------------
def plot_quantile_forecast(
    quantile_df: pd.DataFrame,
    actual: pd.DataFrame,
    date_col: str,
    target_col: str,
    history: Optional[pd.DataFrame] = None,
    title: str = "Probabilistic forecast (quantile fan)",
    height: int = 460,
) -> go.Figure:
    """Render multiple quantile bands as nested transparent fills."""
    qcols = sorted([c for c in quantile_df.columns if c.startswith("q")])
    n = len(qcols)
    fig = go.Figure()
    if history is not None:
        fig.add_trace(go.Scatter(x=history[date_col], y=history[target_col],
                                 mode="lines", name="history",
                                 line=dict(color=PALETTE["history"], width=1.2)))
    # Pair outermost quantiles to form bands
    for i in range(n // 2):
        lo, hi = qcols[i], qcols[n - 1 - i]
        fig.add_trace(go.Scatter(x=quantile_df[date_col], y=quantile_df[hi],
                                 line=dict(width=0), mode="lines",
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=quantile_df[date_col], y=quantile_df[lo],
                                 fill="tonexty",
                                 fillcolor=f"rgba(37, 99, 235, {0.15 + i * 0.10:.2f})",
                                 line=dict(width=0), mode="lines",
                                 name=f"{lo} – {hi}", hoverinfo="skip"))
    median = qcols[n // 2]
    fig.add_trace(go.Scatter(x=quantile_df[date_col], y=quantile_df[median],
                             mode="lines", name=median,
                             line=dict(color=PALETTE["forecast"], width=2)))
    fig.add_trace(go.Scatter(x=actual[date_col], y=actual[target_col],
                             mode="lines+markers", name="actual",
                             line=dict(color=PALETTE["actual"], width=2),
                             marker=dict(size=4)))
    return _apply_theme(fig, title=title, height=height,
                        xaxis_title="Date", yaxis_title=target_col)


# ---------------------------------------------------------------------------
# Metric tables
# ---------------------------------------------------------------------------
def metric_bar(metric_table: pd.DataFrame, metric: str = "WAPE%",
               group_col: str = "model",
               title: Optional[str] = None,
               height: int = 360) -> go.Figure:
    fig = px.bar(metric_table, x=group_col, y=metric, color=group_col, text=metric)
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return _apply_theme(fig, title=title or f"{metric} by {group_col}",
                        height=height, showlegend=False)
