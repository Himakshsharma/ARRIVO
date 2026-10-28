"""Shared modelling utilities: preprocessing, candidate models, search spaces and run tracking.

XGBoost, LightGBM, Optuna and MLflow are optional. When one is missing, a
scikit-learn or built-in equivalent is used and labelled as such.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Callable

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

log = logging.getLogger(__name__)
SEED = config.RANDOM_SEED


def _try_import(name):
    try:
        return __import__(name)
    except Exception:      # noqa: BLE001 - ImportError or broken native lib
        return None


xgb = _try_import("xgboost")
lgb = _try_import("lightgbm")
optuna = _try_import("optuna")
mlflow = _try_import("mlflow")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Scale numeric columns (linear models only) and one-hot encode categoricals.

    Unseen categories are encoded as all zeros.
    """
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [("num", num, config.NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
          config.CATEGORICAL_FEATURES)],
        verbose_feature_names_out=True,
    )


def make_pipeline(estimator, linear: bool) -> Pipeline:
    return Pipeline([("pre", make_preprocessor(scale_numeric=linear)), ("model", estimator)])


# ---------------------------------------------------------------------------
# Candidate models, baseline first
# ---------------------------------------------------------------------------
def regression_candidates() -> dict[str, tuple[Callable, bool]]:
    """name -> (factory(**params), is_linear)"""
    c = {
        "LinearRegression": (lambda **p: LinearRegression(**p), True),
        "RandomForest": (lambda **p: RandomForestRegressor(
            **{"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": -1, "random_state": SEED, **p}), False),
    }
    if xgb is not None:
        c["XGBoost"] = (lambda **p: xgb.XGBRegressor(
            **{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 6, "subsample": 0.8,
               "colsample_bytree": 0.8, "random_state": SEED, "n_jobs": -1, **p}), False)
    if lgb is not None:
        c["LightGBM"] = (lambda **p: lgb.LGBMRegressor(
            **{"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8, "random_state": SEED, "verbose": -1, **p}), False)
    if xgb is None or lgb is None:
        c["GradientBoosting (sklearn)"] = (lambda **p: GradientBoostingRegressor(
            **{"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
               "random_state": SEED, **p}), False)
    return c


def classification_candidates(pos_weight: float) -> dict[str, tuple[Callable, bool]]:
    """Classifiers configured for class imbalance (class weights or ``scale_pos_weight``)."""
    c = {
        "LogisticRegression": (lambda **p: LogisticRegression(
            **{"max_iter": 2000, "class_weight": "balanced", **p}), True),
        "RandomForest": (lambda **p: RandomForestClassifier(
            **{"n_estimators": 300, "min_samples_leaf": 3, "class_weight": "balanced_subsample",
               "n_jobs": -1, "random_state": SEED, **p}), False),
    }
    if xgb is not None:
        c["XGBoost"] = (lambda **p: xgb.XGBClassifier(
            **{"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "subsample": 0.8,
               "colsample_bytree": 0.8, "scale_pos_weight": pos_weight, "eval_metric": "auc",
               "random_state": SEED, "n_jobs": -1, **p}), False)
    if lgb is not None:
        c["LightGBM"] = (lambda **p: lgb.LGBMClassifier(
            **{"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 31, "subsample": 0.8,
               "subsample_freq": 1, "colsample_bytree": 0.8, "is_unbalance": True,
               "random_state": SEED, "verbose": -1, **p}), False)
    if xgb is None or lgb is None:
        c["GradientBoosting (sklearn)"] = (lambda **p: GradientBoostingClassifier(
            **{"n_estimators": 400, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
               "random_state": SEED, **p}), False)
    return c


def needs_sample_weight(name: str) -> bool:
    """scikit-learn's GradientBoosting has no ``class_weight``; balance via sample weights."""
    return name.startswith("GradientBoosting")


def balanced_sample_weight(y) -> np.ndarray:
    y = np.asarray(y)
    pos = y.mean()
    return np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos))


# ---------------------------------------------------------------------------
# Hyper-parameter search spaces (shared by Optuna and the random-search fallback)
# ---------------------------------------------------------------------------
def search_space(name: str):
    """Return ``trial -> params`` for the model; accepts an Optuna trial or ``_RandomTrial``."""
    def rf(t):
        return {"n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
                "max_depth": t.suggest_int("max_depth", 6, 30),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 10),
                "max_features": t.suggest_float("max_features", 0.3, 1.0)}

    def xg(t):
        return {"n_estimators": t.suggest_int("n_estimators", 200, 1200, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 9),
                "min_child_weight": t.suggest_float("min_child_weight", 1, 10, log=True),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}

    def lg(t):
        return {"n_estimators": t.suggest_int("n_estimators", 200, 1500, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": t.suggest_int("num_leaves", 15, 127),
                "min_child_samples": t.suggest_int("min_child_samples", 5, 80),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}

    def gb(t):
        return {"n_estimators": t.suggest_int("n_estimators", 200, 700, step=100),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.2, log=True),
                "max_depth": t.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 40),
                "subsample": t.suggest_float("subsample", 0.6, 1.0)}

    def logreg(t):
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}

    return {"RandomForest": rf, "XGBoost": xg, "LightGBM": lg, "LogisticRegression": logreg,
            "GradientBoosting (sklearn)": gb}.get(name)


class _RandomTrial:
    """Minimal Optuna-trial interface used for random search when Optuna is unavailable."""
    def __init__(self, rng):
        self.rng, self.params = rng, {}

    def suggest_int(self, name, lo, hi, step=1):
        v = int(self.rng.choice(np.arange(lo, hi + 1, step))); self.params[name] = v; return v

    def suggest_float(self, name, lo, hi, log=False):
        v = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))) if log else self.rng.uniform(lo, hi))
        self.params[name] = v; return v


def tune(name: str, objective: Callable[[dict], float], n_trials: int, direction: str) -> tuple[dict, float, str]:
    """Tune hyper-parameters against ``objective(params) -> validation score``.

    Uses Optuna's TPE sampler when available, otherwise random search over the
    same space. Returns ``(best_params, best_score, method)``.
    """
    space = search_space(name)
    if space is None:
        return {}, objective({}), "none"
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=direction,
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(lambda t: objective(space(t)), n_trials=n_trials, show_progress_bar=False)
        return study.best_params, study.best_value, "optuna-tpe"
    rng = np.random.default_rng(SEED)
    best_p, best_v = {}, None
    for _ in range(n_trials):
        t = _RandomTrial(rng); params = space(t); v = objective(params)
        better = best_v is None or (v > best_v if direction == "maximize" else v < best_v)
        if better:
            best_p, best_v = params, v
    return best_p, best_v, "random-search"


# ---------------------------------------------------------------------------
# Experiment tracking (MLflow, or a JSON-lines log when MLflow is missing)
# ---------------------------------------------------------------------------
class Tracker:
    def __init__(self, experiment: str = config.MLFLOW_EXPERIMENT):
        self.use_mlflow = mlflow is not None
        config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
        self.fallback_file = config.METRICS_DIR / "runs.jsonl"
        if self.use_mlflow:
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(experiment)

    def log_run(self, run_name: str, params: dict, metrics: dict, tags: dict | None = None):
        clean_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating))}
        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
                mlflow.log_metrics(clean_metrics)
                if tags:
                    mlflow.set_tags(tags)
        with open(self.fallback_file, "a") as f:
            f.write(json.dumps({"run": run_name, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "params": {k: str(v) for k, v in params.items()},
                                "metrics": clean_metrics, "tags": tags or {}}) + "\n")


def backend_summary() -> dict:
    return {"xgboost": getattr(xgb, "__version__", None), "lightgbm": getattr(lgb, "__version__", None),
            "optuna": getattr(optuna, "__version__", None), "mlflow": getattr(mlflow, "__version__", None)}
