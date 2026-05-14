"""
prophet_forecaster.py
=====================
A thin wrapper around facebook-prophet (`prophet` package) that fits a separate
Prophet model per forecast key and returns probabilistic forecasts (mean +
prediction intervals derived from posterior samples).

Why Prophet for this tutorial?
------------------------------
Prophet is a generative additive model:
    y(t) = trend(t) + seasonality(t) + holidays(t) + epsilon(t)

It exposes its uncertainty natively — `mcmc_samples > 0` gives full Bayesian
posteriors, while the default MAP fit returns approximate intervals via a
parametric noise model. For business-management students, the appeal is the
interpretability: each component can be visualised independently.

Limitations
-----------
- One model per key: scales linearly with the number of keys.
- Cannot directly use lag features the way LightGBM does; instead it relies
  on its decomposition.  Prophet's regressors (add_regressor) are how you
  inject exogenous variables.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
import pandas as pd

try:
    from prophet import Prophet                # newer (>=1.1)
except ImportError:                            # pragma: no cover
    from fbprophet import Prophet              # older

# Silence Prophet's chatty stan output
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Per-key Prophet wrapper
# ---------------------------------------------------------------------------
class MultiKeyProphet:
    """
    Fit one Prophet model per forecast key.

    Inputs
    ------
    date_col, target_col, key_cols : column names in the panel
    prophet_kwargs                 : forwarded to Prophet(**prophet_kwargs)
    extra_regressors               : list of (name, prior_scale, mode) tuples
                                     for Prophet.add_regressor.

    Probabilistic outputs
    ---------------------
    Prophet's `predict()` returns yhat plus yhat_lower / yhat_upper for the
    configured `interval_width` (default 0.80). Set mcmc_samples > 0 in
    prophet_kwargs to draw full posterior samples (slower but more accurate).
    """

    def __init__(
        self,
        date_col: str,
        target_col: str,
        key_cols: Sequence[str],
        prophet_kwargs: Optional[Dict] = None,
        extra_regressors: Iterable[Tuple[str, float, str]] = (),
        country_holidays: Optional[str] = None,
        floor_zero: bool = True,
    ):
        self.date_col = date_col
        self.target_col = target_col
        self.key_cols = list(key_cols)
        self.prophet_kwargs = prophet_kwargs or {}
        self.extra_regressors = list(extra_regressors)
        self.country_holidays = country_holidays
        self.floor_zero = floor_zero
        self.models: Dict[Tuple, Prophet] = {}
        self._regressor_names: List[str] = [r[0] for r in self.extra_regressors]

    # ---- helpers -----------------------------------------------------------
    def _to_prophet_frame(self, g: pd.DataFrame) -> pd.DataFrame:
        cols = {self.date_col: "ds", self.target_col: "y"}
        ph = g.rename(columns=cols)[["ds", "y", *self._regressor_names]].copy()
        if self.floor_zero:
            ph["floor"] = 0.0
        return ph

    # ---- API ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame, verbose: bool = False) -> "MultiKeyProphet":
        for keys, g in df.groupby(self.key_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ph_df = self._to_prophet_frame(g)
            kwargs = dict(self.prophet_kwargs)
            # Auto-enable saturating-zero floor only if requested (forces logistic growth)
            m = Prophet(**kwargs)
            for name, prior_scale, mode in self.extra_regressors:
                m.add_regressor(name, prior_scale=prior_scale, mode=mode)
            if self.country_holidays is not None:
                m.add_country_holidays(country_name=self.country_holidays)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(ph_df)
            self.models[keys] = m
            if verbose:
                print(f"  fit Prophet for key={keys}  (n={len(ph_df)})")
        return self

    def predict(
        self,
        future_dates: Union[pd.DataFrame, Sequence[pd.Timestamp]],
        exogenous_future: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Predict for `future_dates`. If exogenous_future is provided, it must
        contain `date_col`, all key_cols, and all extra regressor columns.
        """
        if isinstance(future_dates, pd.DataFrame):
            dates = pd.to_datetime(future_dates[self.date_col]).unique()
        else:
            dates = pd.to_datetime(list(future_dates))
        out: List[pd.DataFrame] = []

        for keys, model in self.models.items():
            future = pd.DataFrame({"ds": dates})
            if self._regressor_names:
                if exogenous_future is None:
                    raise ValueError("extra regressors configured but exogenous_future not provided")
                key_dict = dict(zip(self.key_cols, keys))
                mask = np.ones(len(exogenous_future), dtype=bool)
                for c, v in key_dict.items():
                    mask &= (exogenous_future[c] == v)
                fut_exo = exogenous_future.loc[mask, [self.date_col, *self._regressor_names]]
                fut_exo = fut_exo.rename(columns={self.date_col: "ds"})
                future = future.merge(fut_exo, on="ds", how="left")
            if self.floor_zero:
                future["floor"] = 0.0
            f = model.predict(future)
            f = f[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": self.date_col})
            for c, v in zip(self.key_cols, keys):
                f[c] = v
            out.append(f)

        return pd.concat(out, ignore_index=True)[
            [self.date_col, *self.key_cols, "yhat", "yhat_lower", "yhat_upper"]
        ]

    # ---- diagnostics -------------------------------------------------------
    def get_components(self, key: Tuple, periods: int = 60, freq: str = "D") -> pd.DataFrame:
        """Return a DataFrame of trend/seasonality components for a single key."""
        if key not in self.models:
            raise KeyError(f"key {key} not found")
        m = self.models[key]
        future = m.make_future_dataframe(periods=periods, freq=freq, include_history=True)
        if self.floor_zero:
            future["floor"] = 0.0
        # Components require regressor columns at zero if missing - fill them
        for r in self._regressor_names:
            if r not in future.columns:
                future[r] = 0.0
        forecast = m.predict(future)
        comp_cols = [c for c in ["ds", "trend", "yearly", "weekly", "daily",
                                 "holidays", "additive_terms", "yhat"] if c in forecast.columns]
        return forecast[comp_cols]
