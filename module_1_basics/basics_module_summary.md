# Statistical Time Series Forecasting — Complete Module Summary

A consolidated reference covering every concept, method, test, and metric introduced across the 11-notebook tutorial, the tests reference document, and the `tsf_utils.py` helper module.

Use this as:
-  A **study guide** before exams or interviews
-  A **navigation map** to locate concepts across notebooks
-  A **quick reference** during real forecasting projects
-  A **teaching aid** for instructors running the workshop

---

## Module overview

**Audience:** Business management students with basic Python and statistics.
**Format:** 11 hands-on Jupyter notebooks + reference docs + shared utility module.
**Environment:** Google Colab (or any Jupyter runtime).
**Visualization:** Plotly throughout — every chart is interactive.
**Data:** Bring your own univariate time series, or use a suggested public dataset.

### Deliverables

| File | Type | Purpose |
|------|------|---------|
| `README.md` | Markdown | Project intro, setup instructions, Colab loading |
| `tsf_utils.py` | Python module | All reusable functions — load, plot, test, model, evaluate |
| `tests_reference.md` | Markdown | Copy-paste-able descriptions of 15 statistical tests/transforms |
| `01_introduction_and_eda.ipynb` | Notebook | Time series basics + EDA |
| `02_decomposition.ipynb` | Notebook | Classical & STL decomposition |
| `03_stationarity_and_tests.ipynb` | Notebook | ADF, KPSS, differencing, transformations |
| `04_autocorrelation.ipynb` | Notebook | ACF, PACF, Ljung-Box |
| `05_metrics_and_validation.ipynb` | Notebook | Forecast metrics, validation, residuals |
| `06_baseline_methods.ipynb` | Notebook | Naive, mean, drift, seasonal naive |
| `07_exponential_smoothing.ipynb` | Notebook | SES, Holt, Holt-Winters, ETS |
| `08_arima.ipynb` | Notebook | Box-Jenkins, manual + auto ARIMA |
| `09_sarima_sarimax.ipynb` | Notebook | SARIMA, exogenous regressors |
| `10_end_to_end_workflow.ipynb` | Notebook | The complete pipeline template |
| `11_croston_intermittent_demand.ipynb` | Notebook | SBC classification, Croston, SBA, TSB |

---

## Part 1 — Notebook-by-notebook concept summary

### Notebook 1 — Introduction & EDA

**Goal:** Build intuition for what makes time series different from cross-sectional data.

**Core concepts:**
- **What is a time series:** Sequence of observations in time order. *Order matters* — shuffling destroys information.
- **The four classical components:**
  - **Trend** $T_t$ — long-term direction
  - **Seasonality** $S_t$ — repeating pattern of fixed period
  - **Cyclic** $C_t$ — repeating pattern with no fixed period
  - **Irregular** $\varepsilon_t$ — random noise / one-off events
- **Additive vs multiplicative models:**
  - Additive: $y_t = T_t + S_t + \varepsilon_t$
  - Multiplicative: $y_t = T_t \times S_t \times \varepsilon_t$
- **The first habit:** always plot before doing anything else.

**Diagnostic techniques introduced:**
- Single-series plot for trend & seasonality detection
- Summary statistics
- Distribution & outlier detection (histogram + boxplot)
- Rolling mean & std (visual stationarity check)
- Seasonal view — one line per cycle (exposes seasonality)
- Lag plots (visual autocorrelation)
- Resampling to different frequencies

---

### Notebook 2 — Decomposition

**Goal:** Split a series into interpretable trend, seasonality, and residual components.

**Core concepts:**
- **Why decompose:** Diagnostic, not predictive. Exposes structure, checks assumptions, guides modeling choices.
- **Classical decomposition (`seasonal_decompose`):**
  - Moving-average based
  - Edge NaNs (length of MA window)
  - Fixed seasonality across cycles
  - Supports additive AND multiplicative natively
- **STL (Seasonal-Trend decomposition using Loess):**
  - Iterative LOESS smoothing
  - No edge NaNs
  - Seasonality can evolve over time
  - Robust mode handles outliers
  - Additive only (log-transform for multiplicative)
- **Component strength metrics** (Hyndman & Athanasopoulos):
$$F_T = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(T_t + R_t)}\right),\quad F_S = \max\!\left(0,\; 1 - \frac{\text{Var}(R_t)}{\text{Var}(S_t + R_t)}\right)$$
- **Seasonally-adjusted series:** $y_t - S_t$ — exposes underlying trend changes.

**Decision criterion (additive vs multiplicative):**
- Constant seasonal swings → additive
- Swings grow with level → multiplicative
- Quantitative checks: rolling std vs mean correlation, Box-Cox $\lambda$, residual whiteness

---

### Notebook 3 — Stationarity & Differencing

**Goal:** Understand and test for stationarity (the foundation of ARIMA-family models).

**Core concepts:**
- **Weak stationarity:** Constant mean, variance, and autocovariance (depends only on lag, not on time).
- **Random walk:** $y_t = y_{t-1} + \varepsilon_t$ — canonical non-stationary process. Mean is constant but variance grows linearly with $t$.
- **Three flavors of non-stationarity:**
  | Type | Example | Fix |
  |------|---------|-----|
  | Trend (deterministic) | $y_t = \alpha + \beta t + \varepsilon_t$ | Detrend |
  | Stochastic trend (unit root) | Random walk | First-difference |
  | Seasonality | Strong cycle | Seasonal-difference (lag $s$) |

**Tests covered:**
- **ADF (Augmented Dickey-Fuller):** H₀ = unit root. Reject ⇒ stationary.
- **KPSS:** H₀ = stationary. Reject ⇒ non-stationary.
- **Combined ADF + KPSS verdict** — strongest evidence comes from agreement.

**Operations covered:**
- First differencing: $\nabla y_t = y_t - y_{t-1}$
- Seasonal differencing: $\nabla_s y_t = y_t - y_{t-s}$
- Combined differencing: both applied
- Log transform: for series with positive values and exponential growth
- Box-Cox transform: parameterized variance stabilizer
- Auto-determination: `pmdarima.ndiffs`, `pmdarima.nsdiffs`

---

### Notebook 4 — Autocorrelation

**Goal:** Read ACF and PACF fluently — they drive ARIMA model selection.

**Core concepts:**
- **ACF (Autocorrelation Function):** Correlation between $y_t$ and $y_{t-k}$ — total dependence, including indirect effects.
- **PACF (Partial Autocorrelation Function):** Correlation between $y_t$ and $y_{t-k}$ after removing the influence of lags 1 through $k-1$ — direct effect only.
- **White-noise CI band:** $\pm 1.96/\sqrt{n}$. Bars inside the band are statistically indistinguishable from zero.
- **Always read ACF/PACF on the stationary series** (not the raw one).

**Reading guide for ARIMA orders:**

| Pattern | ACF | PACF | Suggests |
|---------|-----|------|----------|
| Tails off slowly | Slow decay | Cuts off after lag $p$ | **AR($p$)** |
| Cuts off after lag $q$ | Cuts off | Slow decay | **MA($q$)** |
| Both decay | Decay | Decay | **ARMA($p$,$q$)** |
| Spike at lag $s$ | Spike at $s$ | Spike at $s$ | **Seasonal** component |

**Test introduced:**
- **Ljung-Box:** H₀ = independence (white noise). Used on residuals — want high p-values here!

---

### Notebook 5 — Metrics & Validation

**Goal:** Validate forecasts honestly (chronologically) and pick the right error metric.

**Core concepts:**
- **Why random splits break time series:** Shuffling leaks future into training, inflates validation accuracy, hides true forecasting ability.
- **Chronological train/test split:** Last fraction of the series becomes hold-out.
- **Walk-forward validation:** Rolling-origin refits — many one-step-ahead forecasts, more honest than a single split.
- **TimeSeriesSplit:** k-fold version where each fold's training set precedes its test set.

**Metrics covered:**

| Metric | Formula | Properties |
|--------|---------|------------|
| **MAE** | $\frac{1}{n}\sum |y - \hat y|$ | Same units, robust to outliers |
| **MSE** | $\frac{1}{n}\sum(y-\hat y)^2$ | Squared units, sensitive to large errors |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same units, penalizes large errors |
| **MAPE** | $\frac{100}{n}\sum \frac{|y-\hat y|}{|y|}$ | Percentage, undefined at $y=0$ |
| **sMAPE** | $\frac{100}{n}\sum \frac{|y-\hat y|}{(|y|+|\hat y|)/2}$ | Range [0, 200], handles zeros |
| **MASE** | $\frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$ | <1 beats naive baseline |
| **ME** | $\frac{1}{n}\sum(\hat y - y)$ | Measures bias |

**Residual diagnostics — a good model produces residuals that:**
1. Have mean ≈ 0 (no bias)
2. Have constant variance (homoskedastic)
3. Have no autocorrelation (flat ACF, high Ljung-Box p-value)
4. Are approximately normal (for valid prediction intervals)

---

### Notebook 6 — Baselines

**Goal:** Establish the bar every "sophisticated" model must beat.

**Methods covered:**

| Method | Formula | Hard to beat when |
|--------|---------|-------------------|
| **Naive** | $\hat y_{t+h} = y_t$ | Random walks (stock prices) |
| **Mean** | $\hat y_{t+h} = \bar y$ | Mean-reverting, no trend |
| **Drift** | Naive + avg per-step change | Constant linear trend |
| **Seasonal naive** | $\hat y_{t+h} = y_{t+h-s}$ | Strongly seasonal, stable level |

**Key takeaway:** **MASE < 1** means you've beaten the naive baseline — the cleanest "is my model worth it?" check.

---

### Notebook 7 — Exponential Smoothing

**Goal:** Master the ETS family with appropriate parameter intuition.

**Core idea:** Recent observations weighted more than distant ones, with exponentially decaying weights:
$$\hat y_{t+1} = \alpha y_t + \alpha(1-\alpha) y_{t-1} + \alpha(1-\alpha)^2 y_{t-2} + \dots$$

**Smoothing parameters:**
- $\alpha$ — level smoothing
- $\beta$ — trend smoothing
- $\gamma$ — seasonal smoothing
- $\phi$ — trend damping (0 < $\phi$ < 1)

**Models covered:**

| Model | Components | When to use |
|-------|-----------|-------------|
| **SES** | Level only | No trend, no seasonality |
| **Holt** | Level + trend | Trend, no seasonality |
| **Damped Holt** | Level + damped trend | Trend that should flatten long-term |
| **Holt-Winters additive** | All three (constant seasonal amplitude) | Stable seasonality |
| **Holt-Winters multiplicative** | All three (growing seasonal amplitude) | Seasonality scales with level |
| **ETS framework** | Unified (Error, Trend, Seasonal) interface | Any of the above + state-space inference |

---

### Notebook 8 — ARIMA

**Goal:** The Box-Jenkins methodology for autoregressive integrated moving average models.

**ARIMA(p, d, q):**
- **p** = AR order (depends on previous $p$ values)
- **d** = differencing order (to achieve stationarity)
- **q** = MA order (depends on previous $q$ forecast errors)

Model equation after $d$ differences:
$$\phi(B)\, \nabla^d y_t = \theta(B)\, \varepsilon_t$$

**Box-Jenkins methodology:**
1. **Identify** — Make stationary (choose $d$). Read ACF/PACF for $p$, $q$.
2. **Estimate** — Fit candidate models.
3. **Diagnose** — Verify residuals are white noise.
4. **Forecast** — With point estimates + confidence intervals.

**Model selection:**
- **AIC** (Akaike Information Criterion) — penalizes complexity, lower is better
- **BIC** (Bayesian Information Criterion) — heavier penalty than AIC
- **Grid search** via `itertools.product` over (p, d, q) combinations
- **`auto_arima`** (pmdarima) — stepwise search

---

### Notebook 9 — SARIMA & SARIMAX

**Goal:** Extend ARIMA for seasonality and exogenous drivers.

**SARIMA(p, d, q)(P, D, Q, s):**
- Non-seasonal: $(p, d, q)$ as in ARIMA
- Seasonal:
  - **P** = seasonal AR
  - **D** = seasonal differencing
  - **Q** = seasonal MA
  - **s** = seasonal period (12, 4, 7, 52, ...)

**Reading ACF/PACF for seasonal orders:**
- Focus on lags **around** the seasonal lag $s$
- Spike at $s$ in PACF, decay around it in ACF → seasonal AR(1)
- Spike at $s$ in ACF, decay around it in PACF → seasonal MA(1)

**SARIMAX — the "X" = eXogenous regressors:**
$$y_t = \beta^\top x_t + \text{SARIMA error process}$$

Lets `y` depend on external time series (promotions, holidays, weather, prices, marketing spend).

**Critical caveat:** To forecast `y` into the future, you need **future values of `x`**:
- Use known-in-advance regressors (calendar effects, scheduled promos), OR
- Forecast `x` separately first (adds error), OR
- Limit yourself to one-step-ahead use cases.

**Coefficient interpretation:** Linear-regression-style — sign, magnitude, p-value all interpretable.

---

### Notebook 10 — End-to-End Workflow

**Goal:** A reusable template that ties everything together.

**The recommended pipeline:**
```
1. EDA              → Plot, decompose, infer frequency & seasonality
2. Stationarise     → ADF/KPSS, transform & difference as needed
3. Identify         → ACF/PACF → candidate orders
4. Split            → Chronological train/test split
5. Baselines        → Naive, seasonal naive, mean, drift
6. Models           → ETS family, ARIMA, SARIMA, (SARIMAX)
7. Diagnostics      → Residuals, Ljung-Box, AIC/BIC
8. Compare          → Metrics table, MASE vs naive
9. Forecast         → Refit on full data, project forward with CI
```

**Common pitfalls collected:**
1. Random train/test split (leaks future)
2. Reporting only MAPE on near-zero series
3. Ignoring residual autocorrelation
4. Forecasting multiplicative seasonality with additive model
5. Forgetting to refit on full data before final forecast
6. No baseline comparison
7. Treating SARIMAX exogenous forecasts as free
8. Overfitting by tuning on the test set
9. Ignoring regime shifts (pandemic, structural break, regulation)

---

### Notebook 11 — Croston for Intermittent Demand

**Goal:** Forecast series with lots of zeros — where ETS/ARIMA fail silently.

**SBC classification (Syntetos-Boylan-Croston):**
- **ADI** (Average Demand Interval) = $\frac{N_{\text{obs}}}{N_{\text{nonzero}}}$
- **CV²** = squared coefficient of variation of non-zero demands

|              | CV² < 0.49 | CV² ≥ 0.49 |
|--------------|------------|------------|
| **ADI < 1.32** | Smooth | Erratic |
| **ADI ≥ 1.32** | **Intermittent** | **Lumpy** |

**Why ETS fails:** SES averages over zeros and gives a flat low forecast that hides what the business needs (demand size when it occurs + rate of occurrence).

**Croston's classical method (1972):**
Decomposes the series into:
- **Demand size** $z_t$ (smoothed via SES)
- **Inter-demand interval** $p_t$ (smoothed via SES)

Forecast = rate of demand:
$$\hat y_{t+h} = \frac{z_t}{p_t}$$

**SBA (Syntetos-Boylan Approximation):**
Bias correction — multiplies the rate by $(1 - \alpha/2)$. **The recommended default.**

**TSB (Teunter-Syntetos-Babai):**
Replaces interval with **probability of demand**. Probability decays during long zero-runs → forecast naturally drops to zero for obsolete products.

**Metrics gotchas for intermittent demand:**
- MAPE is undefined (zeros in denominator)
- MAE/RMSE work but absolute scale matters
- **MASE** and **CFE (Cumulative Forecast Error)** are most informative
- **Predicting all zeros can "win" on MAE** — but guarantees stockouts. *The metric that wins must be the one that drives correct business decisions.*

---

## Part 2 — Concept cross-reference index

Quick lookup for "where is concept X explained or used?"

| Concept | Introduced | Reinforced/Used |
|---------|-----------|-----------------|
| **Time series components** | NB 1 | NB 2 (decomposition) |
| **Additive vs multiplicative** | NB 1 | NB 2, 7 (HW models) |
| **Trend** | NB 1 | NB 2, 7 (Holt), 10 |
| **Seasonality** | NB 1 | NB 2, 7 (HW), 9 (SARIMA) |
| **Stationarity** | NB 3 | NB 8 (ARIMA prep), 9 |
| **Random walk** | NB 3 | NB 6 (naive baseline) |
| **Differencing** | NB 3 | NB 8, 9 |
| **Box-Cox / log transform** | NB 3 | NB 7 (multiplicative HW) |
| **ACF / PACF** | NB 4 | NB 5 (residuals), 8, 9 |
| **White noise** | NB 4 | NB 5 (residual diagnostics) |
| **Train/test split** | NB 5 | NB 6, 7, 8, 9, 10, 11 |
| **Walk-forward validation** | NB 5 | NB 10 |
| **Residual diagnostics** | NB 5 | NB 7, 8, 9, 10 |
| **Metrics (MAE, MAPE, etc.)** | NB 5 | NB 6, 7, 8, 9, 10, 11 |
| **MASE** | NB 5 | NB 6, 10, 11 |
| **Baseline models** | NB 6 | NB 7, 8, 9, 10 (comparison) |
| **Exponential smoothing** | NB 7 | NB 10 (model zoo) |
| **ETS framework** | NB 7 | NB 10 |
| **Box-Jenkins methodology** | NB 8 | NB 9, 10 |
| **AIC / BIC** | NB 8 | NB 9 (auto-SARIMA) |
| **auto_arima** | NB 8 | NB 9, 10 |
| **Exogenous regressors** | NB 9 | (template only) |
| **SBC classification** | NB 11 | — |
| **Croston family** | NB 11 | — |
| **CFE (cumulative forecast error)** | NB 11 | — |

---

## Part 3 — All models at a glance

| Model | Captures Trend | Captures Seasonality | Exogenous | Best For | Notebook |
|-------|:--:|:--:|:--:|----------|:--:|
| Naive | ❌ | ❌ | ❌ | Random walks | 6 |
| Mean | ❌ | ❌ | ❌ | Stationary, no trend | 6 |
| Drift | ✅ linear | ❌ | ❌ | Constant linear trend | 6 |
| Seasonal naive | ❌ | ✅ rigid | ❌ | Strong stable seasonality | 6 |
| **SES** | ❌ | ❌ | ❌ | Level-only, no patterns | 7 |
| **Holt** | ✅ | ❌ | ❌ | Trend, no seasonality | 7 |
| **Damped Holt** | ✅ damped | ❌ | ❌ | Trend that flattens long-term | 7 |
| **Holt-Winters (add)** | ✅ | ✅ constant amp. | ❌ | Stable seasonal swings | 7 |
| **Holt-Winters (mul)** | ✅ | ✅ growing amp. | ❌ | Seasonality scales with level | 7 |
| **ETS** | ✅ | ✅ | ❌ | State-space view of all of the above | 7 |
| **ARIMA(p,d,q)** | via $d$ | ❌ | ❌ | Trended series with autocorrelation | 8 |
| **SARIMA(p,d,q)(P,D,Q,s)** | via $d$ | ✅ via $D$ | ❌ | Seasonal + trended series | 9 |
| **SARIMAX** | via $d$ | ✅ via $D$ | ✅ | Adds promos/weather/holidays/prices | 9 |
| **Croston (classical)** | ❌ | ❌ | ❌ | Intermittent demand | 11 |
| **Croston SBA** | ❌ | ❌ | ❌ | Intermittent — bias-corrected default | 11 |
| **Croston TSB** | ❌ | ❌ | ❌ | Intermittent with obsolescence risk | 11 |

---

## Part 4 — All statistical tests at a glance

(See `tests_reference.md` for full descriptions, hypotheses, and code snippets.)

### Stationarity
| Test | H₀ | Reject ⇒ | Used in |
|------|-----|----------|---------|
| **ADF** | Unit root (non-stationary) | Stationary | NB 3, 8 |
| **KPSS** | Stationary | Non-stationary | NB 3 |
| **Phillips-Perron** | Unit root | Stationary | Reference only |
| **Zivot-Andrews** | Unit root with break | Stationary around break | Reference only |

### Autocorrelation
| Test | H₀ | Reject ⇒ | Used in |
|------|-----|----------|---------|
| **Ljung-Box** | No autocorrelation (white noise) | Autocorrelation present | NB 4, 5, 7, 8, 9 |
| **Box-Pierce** | No autocorrelation | Autocorrelation present | Reference only |
| **Durbin-Watson** | No first-order AC | (~2 = no AC) | Reference only |
| **Breusch-Godfrey** | No AC up to lag $p$ | AC present | Reference only |

### Normality
| Test | H₀ | Reject ⇒ | Used in |
|------|-----|----------|---------|
| **Jarque-Bera** | Normal | Non-normal | NB 5 (auto in summaries) |
| **Shapiro-Wilk** | Normal (small n) | Non-normal | Reference only |

### Heteroskedasticity
| Test | H₀ | Reject ⇒ | Used in |
|------|-----|----------|---------|
| **ARCH-LM (Engle)** | Homoskedastic | Volatility clustering | Reference only |
| **White's test** | Homoskedastic | Heteroskedastic | Reference only |

### Transformations (not tests, but variance stabilizers)
| Transform | When to use |
|-----------|-------------|
| **Log** | Strictly positive, exponential growth, multiplicative seasonality |
| **Box-Cox** | Strictly positive, parameterized variance correction |
| **Yeo-Johnson** | Series with zeros or negative values |

---

## Part 5 — All metrics at a glance

| Metric | Formula | Range | Notes |
|--------|---------|:-----:|-------|
| **MAE** | $\tfrac{1}{n}\sum |e_i|$ | $[0,\infty)$ | Same units as data |
| **MSE** | $\tfrac{1}{n}\sum e_i^2$ | $[0,\infty)$ | Squared units |
| **RMSE** | $\sqrt{\text{MSE}}$ | $[0,\infty)$ | Same units; penalizes large errors |
| **MAPE** | $\tfrac{100}{n}\sum \tfrac{|e_i|}{|y_i|}$ | $[0, \infty)$ | %; **undefined at $y=0$** |
| **sMAPE** | $\tfrac{100}{n}\sum \tfrac{|e_i|}{(|y_i|+|\hat y_i|)/2}$ | $[0, 200]$ | %; handles zeros |
| **MASE** | $\tfrac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{naive}}}$ | $[0,\infty)$ | <1 beats naive; unitless |
| **ME / Bias** | $\tfrac{1}{n}\sum e_i$ | $(-\infty,\infty)$ | Signed; measures bias |
| **CFE** | $\sum e_i$ | $(-\infty,\infty)$ | Cumulative bias — critical for inventory |

where $e_i = \hat y_i - y_i$.

**Quick selection guide:**
- Single series with intuitive units → MAE + RMSE
- Comparing across series of different scales → MASE or sMAPE
- Business reporting → MAPE (when safe — no zeros)
- Inventory / supply chain → CFE in addition to error metrics
- Detecting systematic bias → ME or CFE
- Intermittent demand → MASE + CFE (avoid MAPE entirely)

---

## Part 6 — `tsf_utils.py` function reference

### Data loading
- `load_timeseries(path, date_col, value_col, freq, fill_method)` — CSV/Excel → indexed Series
- `infer_seasonal_period(series)` — Suggests period from frequency

### Plotly visualizations (return `go.Figure`)
- `plot_timeseries` — basic line plot
- `plot_rolling_statistics` — rolling mean & std overlay
- `plot_seasonal_view` — one line per cycle
- `plot_lag` — y(t) vs y(t-k) scatters
- `plot_distribution` — histogram + boxplot
- `plot_compare_pair` — stacked original vs transformed
- `decompose_and_plot` — returns `(result, figure)` tuple
- `plot_acf_pacf` — side-by-side ACF and PACF
- `plot_forecast` — train + test + forecast + optional CI band
- `plot_residual_diagnostics` — 4-panel (time / hist / Q-Q / ACF)
- `plot_multi_forecast` — multiple models on one chart
- `plot_intermittent_series` — bar plot with zero coloring

### Statistical tests
- `adf_test(series)` — Augmented Dickey-Fuller
- `kpss_test(series)` — KPSS test
- `stationarity_report(series)` — Combined ADF + KPSS verdict
- `ljung_box_test(residuals, lags)` — White noise test

### Validation
- `train_test_split_ts(series, test_size)` — chronological split
- `walk_forward_validation(series, forecast_func, n_test, step, horizon)` — rolling-origin

### Metrics
- `mae`, `mse`, `rmse`, `mape`, `smape`, `mase`, `me`
- `forecast_metrics(y_true, y_pred, y_train, seasonality)` — returns dict
- `metrics_dataframe(metrics_by_model)` — combine into DataFrame
- `compare_models(y_train, y_test, forecasts, seasonality)` — full comparison table

### Baseline forecasters
- `naive_forecast(history, horizon)`
- `seasonal_naive_forecast(history, horizon, season_length)`
- `mean_forecast(history, horizon)`
- `drift_forecast(history, horizon)`

### Intermittent demand (Notebook 11)
- `classify_demand_pattern(series)` — SBC classification (ADI, CV²)
- `croston(history, horizon, alpha, variant)` — variant: `"classic"` or `"sba"`
- `croston_tsb(history, horizon, alpha, beta)` — probability-tracking variant
- `simulate_intermittent_demand(...)` — generates teaching examples with optional obsolescence

---

## 🎓 Part 7 — Suggested study path

### For self-study (~15 hours total)

| Session | Notebooks | Focus | Time |
|---------|-----------|-------|------|
| 1 | NB 1, 2 | EDA + decomposition — build visual intuition | 2 hrs |
| 2 | NB 3 | Stationarity (cornerstone concept) | 2 hrs |
| 3 | NB 4 | ACF / PACF reading | 1.5 hrs |
| 4 | NB 5 | Metrics + validation discipline | 1.5 hrs |
| 5 | NB 6, 7 | Baselines + exponential smoothing | 2 hrs |
| 6 | NB 8 | ARIMA + Box-Jenkins | 2 hrs |
| 7 | NB 9 | SARIMA + SARIMAX | 2 hrs |
| 8 | NB 10 | End-to-end workflow on real data | 1.5 hrs |
| 9 | NB 11 | Intermittent demand | 1 hr |

### For a 2-day intensive workshop

**Day 1 (Foundations):**
- Morning: NB 1, 2, 3 — what & why of time series, stationarity
- Afternoon: NB 4, 5 — ACF/PACF, metrics & validation

**Day 2 (Methods):**
- Morning: NB 6, 7 — baselines & exponential smoothing
- Afternoon: NB 8, 9, 10 — ARIMA family + end-to-end
- Optional: NB 11 if relevant to participants' data

### For an MBA elective (4-week module)

| Week | Topics |
|------|--------|
| 1 | NB 1-3 — concepts, decomposition, stationarity |
| 2 | NB 4-6 — ACF/PACF, validation, baselines |
| 3 | NB 7-8 — exponential smoothing, ARIMA |
| 4 | NB 9-11 — SARIMA/SARIMAX, workflow, intermittent demand |

Each week: 1 lecture (concepts) + 1 lab (notebook) + 1 take-home (own dataset).

---

## Part 8 — Common pitfalls collected from across the module

1. **Random train/test split on time series** — leaks future into training (NB 5).
2. **Reporting MAPE on a near-zero series** — explodes or breaks; use sMAPE / MASE (NB 5, 11).
3. **Ignoring residual autocorrelation** — your prediction intervals are too tight (NB 5, 8).
4. **Multiplicative seasonality with additive model** — under/overshoots peaks (NB 2, 7).
5. **Forgetting to refit on full data** — final forecast comes from a stale model (NB 10).
6. **No baseline comparison** — can't claim a model is "good" without beating `seasonal_naive` (NB 6).
7. **Treating SARIMAX exogenous forecasts as free** — you still need future regressors (NB 9).
8. **Tuning on the test set** — keep a third validation slice for extensive tuning (NB 5).
9. **Ignoring regime shifts** — pandemic, structural break, regulation change. May require retraining on recent data only (NB 10).
10. **Using ETS/ARIMA on intermittent demand** — silently produces flat low forecasts (NB 11).
11. **Predicting all zeros wins MAE on sparse series** — but guarantees stockouts. Use CFE + service-level simulation (NB 11).
12. **Reading ACF/PACF on the raw (non-stationary) series** — always work on the stationary version (NB 4).
13. **Trusting auto_arima blindly** — always check residuals before deployment (NB 8).
14. **Skipping the visual inspection step** — most modeling decisions follow from what you see (NB 1).
15. **Choosing decomposition method by habit** — STL is usually better than classical for real data (NB 2).

---

## Part 9 — Further reading

**Books**
- Hyndman & Athanasopoulos — *Forecasting: Principles and Practice* (3rd ed.) — free at https://otexts.com/fpp3/ — the friendly modern standard
- Box, Jenkins, Reinsel, Ljung — *Time Series Analysis: Forecasting and Control* — the formal Box-Jenkins reference
- Brockwell & Davis — *Introduction to Time Series and Forecasting* — more mathematical

**Software libraries (beyond what this tutorial uses)**
- **`statsforecast`** (Nixtla) — production-grade implementations of Croston family, ARIMA, ETS, with parallel fitting
- **`prophet`** (Meta) — additive-decomposition models with built-in holiday/calendar effects
- **`neuralprophet`** — NN-augmented Prophet
- **`darts`** — unified API across statistical, ML, and deep learning models
- **`sktime`** — sklearn-compatible time series ML
- **`gluonts`** (AWS) — DeepAR, N-BEATS, Temporal Fusion Transformer

**Methods beyond classical**
- **Tree-based regressors with engineered lag features** — XGBoost/LightGBM
- **Deep learning** — DeepAR, N-BEATS, TFT, Informer
- **Probabilistic forecasting** — quantile regression, distributional models
- **Hierarchical forecasting** — reconciling forecasts across an aggregation hierarchy

---

## The big picture in one paragraph

A forecasting project is really an answer to four questions: *What does the data look like* (EDA + decomposition), *what assumptions are appropriate* (stationarity tests, additive vs multiplicative), *what model captures the structure* (baselines → ETS → ARIMA → SARIMA → SARIMAX → Croston for the intermittent case), and *how do we know it works* (chronological validation, residual diagnostics, metrics that match the business decision). Every notebook in this module is a deeper answer to one of those four questions. Get those right, and the choice of fitting library or Python package becomes a footnote.

**You have now covered every concept in the module.** Time to forecast.
