"""
ML Forecasting Tutorial - Utility Package
==========================================

A collection of reusable modules for building, tuning, and interpreting
ML forecasting models on panel/hierarchical time series data.

Modules
-------
data_utils          : Data loading, validation, panel completion, train/test splits
feature_engineering : Causal lag/rolling/calendar feature builders
metrics             : Forecasting metrics (RMSE, WAPE, MASE, RMSSE, bias, ...)
losses              : Loss-function helpers, intermittent demand classification (ADI/CV²)
cv                  : Time-series cross-validation splitters
lgbm_forecaster     : Direct, Recursive, and Global LightGBM forecasters
prophet_forecaster  : Multi-key Prophet wrapper with probabilistic intervals
tuning              : Grid Search + Optuna (Bayesian) hyper-parameter tuning
explainer           : SHAP-based explanations for LightGBM, Prophet decompositions
viz                 : Plotly-based visualizations for forecasts and diagnostics
"""

__version__ = "1.0.0"
