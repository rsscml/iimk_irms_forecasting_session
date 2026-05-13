"""
Plotly-based plotting utilities for forecasting EDA.

Every function returns a ``plotly.graph_objects.Figure`` so the caller can
display it (``fig.show()``), tweak the layout, or export it. Nothing here
calls ``fig.show()`` itself — that decision belongs to the notebook.

A consistent template, colour palette and figure size are applied across
all plots so the resulting tutorial materials feel coherent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

PALETTE = ["#2E86AB", "#E63946", "#F4A261", "#2A9D8F", "#8E44AD",
           "#264653", "#F77F00", "#06A77D", "#D62828", "#5A189A"]

SB_COLORS = {
    "smooth":       "#2A9D8F",
    "intermittent": "#F4A261",
    "erratic":      "#E76F51",
    "lumpy":        "#9D4EDD",
    "no_demand":    "#6c757d",
    "undefined":    "#adb5bd",
}


def _apply_layout(fig: go.Figure, title: str = "", height: int = 460) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=70, b=50),
        legend=dict(bgcolor="rgba(255,255,255,0.7)"),
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def plot_time_series(
    df: pd.DataFrame,
    spec: Dict,
    keys: Optional[Sequence] = None,
    max_keys: int = 5,
    title: str = "Time Series",
) -> go.Figure:
    """
    Line plot of the target column over time, with one trace per forecast key.

    Parameters
    ----------
    keys : list of dict or list of tuple, optional
        Specific keys to plot. If None, plots the first ``max_keys`` keys.
    """
    date_col = spec["date_col"]
    target = spec["target_col"]
    key_cols = spec["key_cols"]

    fig = go.Figure()

    if keys is None:
        unique_keys = df[key_cols].drop_duplicates().head(max_keys).values
    else:
        unique_keys = []
        for k in keys:
            if isinstance(k, dict):
                unique_keys.append([k[c] for c in key_cols])
            else:
                unique_keys.append(list(k))

    for i, kvals in enumerate(unique_keys):
        mask = pd.Series(True, index=df.index)
        for col, val in zip(key_cols, kvals):
            mask &= (df[col] == val)
        sub = df.loc[mask].sort_values(date_col)
        if sub.empty:
            continue
        label = " | ".join(f"{c}={v}" for c, v in zip(key_cols, kvals))
        fig.add_trace(go.Scatter(
            x=sub[date_col], y=sub[target],
            mode="lines", name=label,
            line=dict(color=PALETTE[i % len(PALETTE)], width=1.6),
        ))

    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=target)
    return _apply_layout(fig, title)


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 50,
    log_y: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    """Histogram + box-plot for a single numeric column."""
    title = title or f"Distribution of '{column}'"

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.04,
    )
    s = df[column].dropna()
    fig.add_trace(go.Histogram(
        x=s, nbinsx=bins, marker_color=PALETTE[0],
        marker_line_color="white", marker_line_width=1,
        name="count", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Box(
        x=s, marker_color=PALETTE[1], boxmean=True,
        showlegend=False, name="box",
    ), row=2, col=1)
    fig.update_yaxes(title="Count", type="log" if log_y else "linear", row=1, col=1)
    fig.update_xaxes(title=column, row=2, col=1)
    return _apply_layout(fig, title, height=520)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def plot_missing_heatmap(df: pd.DataFrame, max_cols: int = 40,
                         title: str = "Missing-value pattern") -> go.Figure:
    """
    Heat-map of missingness: rows are sampled rows of the dataframe, columns
    are dataframe columns, light = present, dark = missing.

    Trying to render every row makes the figure unreadable, so for big
    dataframes the rows are uniformly sampled down to ~1500.
    """
    sub = df.iloc[:, :max_cols]
    if len(sub) > 1500:
        idx = np.linspace(0, len(sub) - 1, 1500, dtype=int)
        sub = sub.iloc[idx]
    z = sub.isna().astype(int).values
    fig = go.Figure(go.Heatmap(
        z=z, x=sub.columns.tolist(), y=list(range(len(sub))),
        colorscale=[[0, "#e9ecef"], [1, "#d62828"]],
        showscale=True, colorbar=dict(title="missing", tickvals=[0, 1]),
    ))
    fig.update_yaxes(title="row index", autorange="reversed")
    fig.update_xaxes(title="", tickangle=-30)
    return _apply_layout(fig, title, height=520)


def plot_missing_bar(missing_report: pd.DataFrame,
                     title: str = "Missing values per column") -> go.Figure:
    """Horizontal bar of missing %, columns ordered by missingness."""
    rep = missing_report[missing_report["missing"] > 0].sort_values("missing_pct")
    if rep.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing values 🎉",
                           x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=18))
        return _apply_layout(fig, title)

    fig = go.Figure(go.Bar(
        x=rep["missing_pct"], y=rep["column"], orientation="h",
        marker_color=PALETTE[1], text=rep["missing_pct"].round(2),
        texttemplate="%{text}%", textposition="outside",
    ))
    fig.update_xaxes(title="% missing")
    fig.update_yaxes(title="")
    return _apply_layout(fig, title, height=max(360, 24 * len(rep) + 120))


# ---------------------------------------------------------------------------
# ACF / PACF / decomposition
# ---------------------------------------------------------------------------

def plot_acf_pacf(acf_pacf: Dict, nlags: Optional[int] = None,
                  title: str = "Autocorrelation diagnostics") -> go.Figure:
    """Side-by-side ACF and PACF stem-and-band plot."""
    acf, pacf = acf_pacf["acf"], acf_pacf["pacf"]
    acf_ci, pacf_ci = acf_pacf["acf_ci"], acf_pacf["pacf_ci"]
    if nlags is None:
        nlags = len(acf) - 1

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    for col, vals, ci, name in [
        (1, acf, acf_ci, "ACF"),
        (2, pacf, pacf_ci, "PACF"),
    ]:
        lags = np.arange(len(vals))
        # Stems
        for L, v in zip(lags, vals):
            fig.add_trace(go.Scatter(
                x=[L, L], y=[0, v], mode="lines",
                line=dict(color=PALETTE[0], width=2),
                showlegend=False,
            ), row=1, col=col)
        # Markers
        fig.add_trace(go.Scatter(
            x=lags, y=vals, mode="markers",
            marker=dict(color=PALETTE[0], size=7),
            name=name, showlegend=False,
        ), row=1, col=col)
        # Zero line
        fig.add_hline(y=0, line=dict(color="black", width=1), row=1, col=col)
        # 95% band (CI is provided as raw [lower, upper] around acf value;
        # subtract the values to get +/- bounds)
        if len(ci):
            band = (ci[:, 1] - vals)
            fig.add_trace(go.Scatter(
                x=np.concatenate([lags, lags[::-1]]),
                y=np.concatenate([band, -band[::-1]]),
                fill="toself", fillcolor="rgba(46,134,171,0.12)",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

        fig.update_xaxes(title="lag", row=1, col=col)
        fig.update_yaxes(title="correlation", row=1, col=col, range=[-1.05, 1.05])
    return _apply_layout(fig, title, height=380)


def plot_decomposition(decomp, title: str = "Seasonal decomposition") -> go.Figure:
    """Four-panel plot: observed, trend, seasonal, residual."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.05,
    )
    parts = [
        decomp.observed, decomp.trend, decomp.seasonal, decomp.resid,
    ]
    for i, part in enumerate(parts, start=1):
        s = part.dropna()
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            line=dict(color=PALETTE[(i - 1) % len(PALETTE)], width=1.4),
            showlegend=False,
        ), row=i, col=1)
    return _apply_layout(fig, title, height=720)


# ---------------------------------------------------------------------------
# Syntetos–Boylan classification
# ---------------------------------------------------------------------------

def plot_sb_classification(
    classification_df: pd.DataFrame,
    adi_threshold: float = 1.32,
    cv2_threshold: float = 0.49,
    title: str = "Syntetos–Boylan demand classification",
    log_axes: bool = False,
) -> go.Figure:
    """
    Quadrant scatter plot of ADI vs CV² coloured by SB class.

    The two grey reference lines mark the standard SB cut-offs (1.32 and
    0.49). Hover labels show the forecast key and the underlying numbers.
    """
    df = classification_df.copy()
    df = df[np.isfinite(df["adi"]) & df["cv2"].notna()]

    # Build a hover label that includes whatever key columns happen to be present
    key_cols = [c for c in df.columns if c not in
                {"adi", "cv2", "n_obs", "zero_share", "sb_class"}]
    df["__label__"] = df[key_cols].astype(str).agg(" | ".join, axis=1)

    fig = go.Figure()
    for cls in df["sb_class"].unique():
        sub = df[df["sb_class"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["adi"], y=sub["cv2"],
            mode="markers", name=cls,
            marker=dict(color=SB_COLORS.get(cls, "#777"),
                        size=8, line=dict(color="white", width=0.5),
                        opacity=0.8),
            text=sub["__label__"],
            customdata=sub[["n_obs", "zero_share"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "ADI=%{x:.2f}<br>CV²=%{y:.3f}<br>"
                "n_obs=%{customdata[0]}<br>zero_share=%{customdata[1]:.2%}<extra></extra>"
            ),
        ))

    # Threshold lines
    fig.add_vline(x=adi_threshold, line=dict(color="#444", dash="dash"))
    fig.add_hline(y=cv2_threshold, line=dict(color="#444", dash="dash"))

    fig.update_xaxes(title="ADI (avg demand interval)",
                     type="log" if log_axes else "linear")
    fig.update_yaxes(title="CV² of non-zero demand",
                     type="log" if log_axes else "linear")
    return _apply_layout(fig, title, height=540)


def plot_sb_summary(summary_df: pd.DataFrame,
                    title: str = "Distribution of SB classes") -> go.Figure:
    """Bar chart of how many series fall into each SB class."""
    fig = go.Figure(go.Bar(
        x=summary_df["sb_class"], y=summary_df["n_series"],
        text=summary_df["pct"].apply(lambda v: f"{v}%"),
        textposition="outside",
        marker_color=[SB_COLORS.get(c, "#777") for c in summary_df["sb_class"]],
    ))
    fig.update_xaxes(title="")
    fig.update_yaxes(title="number of series")
    return _apply_layout(fig, title)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def plot_split(split, spec: Dict, sample_keys: int = 4,
               title: str = "Train / validation / test split") -> go.Figure:
    """
    Visualise a single SplitResult by overlaying train/val/test on the
    target time series for a few sample keys.
    """
    date_col, target = spec["date_col"], spec["target_col"]
    key_cols = spec["key_cols"]

    parts_full = pd.concat(
        [split.train.assign(__split__="train"),
         split.val.assign(__split__="val"),
         split.test.assign(__split__="test")],
        ignore_index=True,
    )

    chosen = parts_full[key_cols].drop_duplicates().head(sample_keys).values

    fig = go.Figure()
    colours = {"train": PALETTE[0], "val": PALETTE[2], "test": PALETTE[1]}

    for kvals in chosen:
        mask = pd.Series(True, index=parts_full.index)
        for col, val in zip(key_cols, kvals):
            mask &= (parts_full[col] == val)
        sub = parts_full.loc[mask].sort_values(date_col)
        for split_name, colour in colours.items():
            part = sub[sub["__split__"] == split_name]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part[date_col], y=part[target],
                mode="lines",
                line=dict(color=colour, width=1.6),
                name=f"{split_name} | {' '.join(map(str, kvals))}",
                legendgroup=split_name,
            ))

    # add_vline + annotation with pandas Timestamp can fail in some plotly
    # versions, so add the line and the annotation as separate calls.
    for cut, label in [(split.train_end, "train_end"), (split.val_end, "val_end")]:
        cut_str = pd.Timestamp(cut).isoformat()
        fig.add_shape(
            type="line", x0=cut_str, x1=cut_str, xref="x",
            y0=0, y1=1, yref="paper",
            line=dict(color="#222", dash="dot", width=1),
        )
        fig.add_annotation(
            x=cut_str, xref="x", y=1.02, yref="paper",
            text=label, showarrow=False,
            font=dict(size=11, color="#222"),
        )

    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=target)
    return _apply_layout(fig, title)


def plot_walk_forward(folds, spec: Dict,
                      title: str = "Walk-forward folds") -> go.Figure:
    """
    Gantt-style overview of a list of SplitResults from
    :func:`expanding_window_split` or :func:`rolling_window_split`.
    """
    date_col = spec["date_col"]
    fig = go.Figure()

    for i, fold in enumerate(folds):
        if len(fold.train):
            fig.add_trace(go.Bar(
                base=[fold.train[date_col].min()],
                x=[fold.train[date_col].max() - fold.train[date_col].min()],
                y=[f"fold {i+1}"], orientation="h",
                marker_color=PALETTE[0], name="train",
                showlegend=(i == 0),
            ))
        if len(fold.val):
            fig.add_trace(go.Bar(
                base=[fold.val[date_col].min()],
                x=[fold.val[date_col].max() - fold.val[date_col].min()],
                y=[f"fold {i+1}"], orientation="h",
                marker_color=PALETTE[2], name="val",
                showlegend=(i == 0),
            ))

    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Date", type="date")
    return _apply_layout(fig, title, height=80 + 40 * max(len(folds), 1))


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(corr: pd.DataFrame,
                             title: str = "Correlation matrix") -> go.Figure:
    """Diverging heatmap centered at 0 with annotated values."""
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1,
        colorscale="RdBu_r",
        text=corr.round(2).values, texttemplate="%{text}",
        hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(tickangle=-30)
    return _apply_layout(fig, title, height=520)


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def plot_outliers_overlay(
    series: pd.Series,
    flags: pd.Series,
    dates: Optional[pd.Series] = None,
    title: str = "Outliers highlighted",
) -> go.Figure:
    """Line plot of a series with outlier points highlighted in red."""
    x = dates if dates is not None else pd.Series(np.arange(len(series)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=series, mode="lines", line=dict(color=PALETTE[0], width=1.3),
        name="series",
    ))
    fig.add_trace(go.Scatter(
        x=x[flags.values], y=series[flags.values],
        mode="markers", marker=dict(color=PALETTE[1], size=8,
                                    line=dict(color="white", width=0.7)),
        name="outliers",
    ))
    fig.update_xaxes(title="time")
    fig.update_yaxes(title="value")
    return _apply_layout(fig, title)


def plot_calendar_heatmap(
    df: pd.DataFrame,
    spec: Dict,
    key_values=None,
    aggregate: str = "mean",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Year × month-of-year heatmap of an aggregated target.

    If ``key_values`` is given, restricts to that single forecast key;
    otherwise aggregates across all keys.
    """
    date_col, target = spec["date_col"], spec["target_col"]
    sub = df
    if key_values is not None:
        from .data_io import get_series
        sub = get_series(df, spec, key_values)

    sub = sub.copy()
    sub["__year__"] = pd.to_datetime(sub[date_col]).dt.year
    sub["__month__"] = pd.to_datetime(sub[date_col]).dt.month

    pivot = sub.pivot_table(
        index="__year__", columns="__month__",
        values=target, aggfunc=aggregate,
    )
    pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale="Viridis",
        hovertemplate="%{y} %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title=aggregate),
    ))
    fig.update_yaxes(title="Year", autorange="reversed")
    fig.update_xaxes(title="")
    return _apply_layout(fig, title or f"{aggregate.title()} of {target} by year × month",
                         height=420)
