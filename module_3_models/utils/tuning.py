"""
tuning.py
=========
Hyper-parameter optimisation utilities for LightGBM forecasters.

We expose two flavours:

1. grid_search_lgbm
   - Exhaustive cartesian product over a small grid.
   - Useful pedagogically and for confirming Optuna's results.

2. optuna_tune_lgbm
   - Bayesian search via Optuna's TPE sampler with optional pruning.
   - The recommended approach in practice.

Both work with TimeSeriesFold objects from utils.cv so that information never
leaks across folds. Each candidate parameter combination is scored as the mean
metric across all folds (lower is better by default).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    optuna = None


# ---------------------------------------------------------------------------
# Shared scoring helper
# ---------------------------------------------------------------------------
def _score_one_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_engineer,
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    categorical_features: Optional[Sequence[str]],
) -> float:
    fe = feature_engineer
    feats_tr = fe.transform(train_df).dropna(subset=[fe.target_col])
    feats_va = fe.transform(valid_df).dropna(subset=[fe.target_col])

    non_features = {fe.date_col, fe.target_col}
    feature_cols = [c for c in feats_tr.columns if c not in non_features]

    X_tr, y_tr = feats_tr[feature_cols], feats_tr[fe.target_col]
    X_va, y_va = feats_va[feature_cols], feats_va[fe.target_col]

    cats = list(categorical_features or fe.key_cols)
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cats, free_raw_data=False)
    valid_set = lgb.Dataset(X_va, label=y_va, categorical_feature=cats,
                            reference=train_set, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params, train_set,
        num_boost_round=num_boost_round,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    yhat = booster.predict(X_va)
    return metric_fn(y_va.values, yhat)


# ---------------------------------------------------------------------------
# 1. Grid search
# ---------------------------------------------------------------------------
def grid_search_lgbm(
    df: pd.DataFrame,
    feature_engineer,
    folds,                                          # List[TimeSeriesFold]
    base_params: Dict,
    param_grid: Dict[str, Sequence],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    categorical_features: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Exhaustive search over the cartesian product of `param_grid`.

    Returns a DataFrame ordered by mean fold-score ascending. The best params
    are in the first row.
    """
    keys = list(param_grid.keys())
    values_list = [param_grid[k] for k in keys]
    combos = list(product(*values_list))

    rows = []
    for i, combo in enumerate(combos):
        params = {**base_params, **dict(zip(keys, combo))}
        fold_scores = []
        for f in folds:
            train_df = df.iloc[f.train_idx]
            valid_df = df.iloc[f.valid_idx]
            score = _score_one_fold(
                train_df, valid_df, feature_engineer, params,
                num_boost_round, early_stopping_rounds, metric_fn, categorical_features,
            )
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        rows.append({"trial": i, **dict(zip(keys, combo)),
                     "mean_score": mean_score, "std_score": std_score,
                     **{f"fold{j}": s for j, s in enumerate(fold_scores)}})
        if verbose:
            print(f"  trial {i+1:>3}/{len(combos)}  {dict(zip(keys, combo))}  -> {mean_score:.4f}")
    return pd.DataFrame(rows).sort_values("mean_score").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Optuna (Bayesian)
# ---------------------------------------------------------------------------
@dataclass
class OptunaSearchSpace:
    """Declarative search space. Each entry is a dict with keys:
        type: 'int' | 'float' | 'log_float' | 'categorical'
        low, high (numeric)  OR  choices (categorical)
        step (optional, int)
    Example:
        space = OptunaSearchSpace({
            'num_leaves':       {'type':'int', 'low':16, 'high':256, 'step':16},
            'learning_rate':    {'type':'log_float', 'low':1e-3, 'high':3e-1},
            'feature_fraction': {'type':'float', 'low':0.5, 'high':1.0},
            'tweedie_variance_power': {'type':'float','low':1.05,'high':1.95},
        })
    """
    space: Dict[str, Dict[str, Any]]

    def sample(self, trial) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, spec in self.space.items():
            t = spec["type"]
            if t == "int":
                out[name] = trial.suggest_int(name, spec["low"], spec["high"],
                                              step=spec.get("step", 1))
            elif t == "float":
                out[name] = trial.suggest_float(name, spec["low"], spec["high"])
            elif t == "log_float":
                out[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
            elif t == "categorical":
                out[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown type {t}")
        return out


def optuna_tune_lgbm(
    df: pd.DataFrame,
    feature_engineer,
    folds,
    base_params: Dict,
    search_space: OptunaSearchSpace,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_trials: int = 50,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    categorical_features: Optional[Sequence[str]] = None,
    direction: str = "minimize",
    seed: int = 42,
    timeout: Optional[float] = None,
    show_progress_bar: bool = True,
):
    """
    Run Optuna with the TPE sampler. Returns (study, best_params, history_df).
    Each trial averages the metric across all CV folds.

    Pruning: we use MedianPruner by default; trials are reported after each
    fold so under-performing combinations can be pruned early.
    """
    if optuna is None:
        raise ImportError("Install optuna:  pip install optuna")

    def objective(trial):
        sampled = search_space.sample(trial)
        params = {**base_params, **sampled}
        fold_scores = []
        for j, f in enumerate(folds):
            train_df = df.iloc[f.train_idx]
            valid_df = df.iloc[f.valid_idx]
            score = _score_one_fold(
                train_df, valid_df, feature_engineer, params,
                num_boost_round, early_stopping_rounds, metric_fn, categorical_features,
            )
            fold_scores.append(score)
            trial.report(np.mean(fold_scores), step=j)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=show_progress_bar)

    best_params = {**base_params, **study.best_params}
    history = study.trials_dataframe(attrs=("number", "value", "params", "state", "duration"))
    return study, best_params, history
