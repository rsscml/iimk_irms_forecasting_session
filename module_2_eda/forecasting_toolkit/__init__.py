"""
Forecasting Toolkit
===================

A teaching/working toolkit for hands-on time-series forecasting preparation.
Covers data loading, EDA, demand classification (Syntetos-Boylan), data quality
checks, gap-filling, imputation, feature engineering, encoding, and
time-aware train/validation/test splitting.

Usage
-----
    from forecasting_toolkit import data_io, eda, classification
    from forecasting_toolkit import quality, preprocessing, features
    from forecasting_toolkit import splitting, plotting

    # Or import specific functions:
    from forecasting_toolkit.preprocessing import fill_time_gaps, impute_missing

The toolkit is designed around a small "DatasetSpec" idea: a dictionary that
declares which columns are dates, targets, forecast keys, static features,
and dynamic features. Most functions accept these column names so the toolkit
works on ANY tabular forecasting dataset (M5, Rossmann, Walmart, etc.).
"""

from . import data_io
from . import eda
from . import classification
from . import quality
from . import preprocessing
from . import features
from . import splitting
from . import plotting

__version__ = "1.0.0"
__all__ = [
    "data_io", "eda", "classification", "quality",
    "preprocessing", "features", "splitting", "plotting",
]
