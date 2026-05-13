# Statistical Time Series Forecasting — Hands-On Tutorial

A self-contained, interactive, ten-notebook tutorial on classical statistical
forecasting, designed for **business management students** working in
**Google Colab**. All visualisations are **interactive Plotly charts** —
hover for values, drag to zoom, click legend entries to toggle series,
double-click to reset.

You bring **your own univariate time-series dataset** (CSV or Excel with one
date column and at least one numeric value column). Everything else — the
analyses, models, plots, and tests — is provided by the shared `tsf_utils.py`
module.

---

## Contents

| # | Notebook | What you'll learn |
|---|----------|-------------------|
| 1 | `01_introduction_and_eda.ipynb` | Time-series concepts, components, EDA, distributional checks, lag plots |
| 2 | `02_decomposition.ipynb` | Additive vs multiplicative, classical decomposition, STL, strength scores |
| 3 | `03_stationarity_and_tests.ipynb` | Stationarity intuition, random walks, ADF, KPSS, differencing, transformations |
| 4 | `04_autocorrelation.ipynb` | ACF, PACF, reading them for ARIMA orders, Ljung-Box |
| 5 | `05_metrics_and_validation.ipynb` | MAE/RMSE/MAPE/sMAPE/MASE/ME, chronological splits, walk-forward, residual diagnostics |
| 6 | `06_baseline_methods.ipynb` | Naive, mean, drift, seasonal naive — the bar every model must beat |
| 7 | `07_exponential_smoothing.ipynb` | SES, Holt linear/damped, Holt-Winters, the unified ETS framework |
| 8 | `08_arima.ipynb` | Box-Jenkins methodology, manual + auto ARIMA, AIC/BIC selection |
| 9 | `09_sarima_sarimax.ipynb` | SARIMA, auto-SARIMA, exogenous regressors, business-grade interpretation |
| 10 | `10_end_to_end_workflow.ipynb` | A reusable template — load → diagnose → split → model zoo → forecast |
| 11 | `11_croston_intermittent_demand.ipynb` | Intermittent & lumpy demand: SBC classification, Croston, SBA, TSB |

Plus the shared module:
- `tsf_utils.py` — every reusable helper (loaders, plotters, tests, metrics, baselines).

---

## Getting started in Google Colab

1. **Upload the project folder** to Colab (or your Google Drive). Easiest way:
   - Click the folder icon in the left sidebar in any Colab notebook.
   - Drag the whole `time_series_tutorial` folder into the upload area.

2. **Open `01_introduction_and_eda.ipynb`** to begin. The notebooks build on
   each other — start at 1 and work through in order.

3. In each notebook, find the **` LOAD YOUR DATASET`** cell and change four
   lines:
   ```python
   DATA_PATH = "your_dataset.csv"
   DATE_COL  = "Date"
   VALUE_COL = "Sales"
   FREQ      = "MS"          # 'D','W','MS','M','Q','A','H' or None
   ```

4. Run cells top to bottom. Each plot is interactive — try hovering, zooming,
   and clicking legend entries.

> **If plots don't render in Colab**, uncomment this line in the setup cell:
> ```python
> # pio.renderers.default = "colab"
> ```

---

## Required packages

Colab already has `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`,
and `plotly`. The notebooks auto-install `pmdarima` on first use.

If you ever need to install everything explicitly:
```bash
pip install numpy pandas scipy statsmodels scikit-learn plotly pmdarima
```

---

## What's in `tsf_utils.py`

A single import (`import tsf_utils as tsf`) gives you:

**Data loading**
- `load_timeseries(path, date_col, value_col, freq, fill_method)`
- `infer_seasonal_period(series)`

**Plotly visualisations** (every function returns a `plotly.graph_objects.Figure`)
- `plot_timeseries`, `plot_rolling_statistics`, `plot_seasonal_view`, `plot_lag`
- `plot_distribution`, `plot_compare_pair`
- `decompose_and_plot` (returns `(result, figure)`)
- `plot_acf_pacf`
- `plot_forecast` (with optional confidence-interval band)
- `plot_residual_diagnostics` (4-panel: time / histogram / Q-Q / ACF)
- `plot_multi_forecast` (compare several models on one chart)

**Statistical tests**
- `adf_test`, `kpss_test`, `stationarity_report`, `ljung_box_test`

**Validation**
- `train_test_split_ts`, `walk_forward_validation`

**Metrics**
- `mae`, `mse`, `rmse`, `mape`, `smape`, `mase`, `me`, `forecast_metrics`,
  `metrics_dataframe`, `compare_models`

**Baselines**
- `naive_forecast`, `seasonal_naive_forecast`, `mean_forecast`, `drift_forecast`

**Intermittent demand**
- `classify_demand_pattern` (Syntetos-Boylan-Croston with ADI / CV²)
- `croston` (`variant="classic"` or `"sba"`), `croston_tsb`
- `plot_intermittent_series`, `simulate_intermittent_demand`

Every function has a docstring — `help(tsf.plot_forecast)` or
`tsf.plot_forecast?` in a Colab cell shows it.

---

## Plotly tips for these notebooks

- **Hover** any line/marker to see exact `(date, value)` tuples.
- **Click** a legend entry to hide/show that series.
- **Double-click** a legend entry to isolate it (everything else hidden).
- **Click-and-drag** on a chart area to zoom into that range.
- **Double-click** anywhere on the chart to reset zoom.
- The toolbar in the top-right of each plot has download-as-PNG, pan, and
  reset buttons.

---

## Dataset suggestions

If you don't yet have a dataset, examples that work well:
- **Monthly sales** of a single SKU over 3+ years
- **Daily store traffic** with weekly seasonality
- **Quarterly revenue** for a public company
- **Hourly energy demand** for a region
- **Weekly active users** of a product
- **Open datasets from UCI and Kaggle**
The longer the history relative to the seasonal period, the better.
A rough rule of thumb: aim for **at least four full seasonal cycles** of data.

---

## Further reading

- *Forecasting: Principles and Practice* (Hyndman & Athanasopoulos) —
  the free, friendly classic: <https://otexts.com/fpp3/>
- *Demand Forecasting for Executives and Professionals (Stephen Kolassa et al.)* https://dfep.netlify.app/