"""Train and select the ETA (transit hours) regressor.

Candidates are fitted on the training split and compared on validation MAE.
The best one is tuned, the test split is scored once, and the pipeline is saved.
"""
from __future__ import annotations

import logging
import time

import joblib
import pandas as pd

from src import config
from src.explain.shap_utils import set_linear_background
from src.features.build_features import feature_matrix
from src.models.common import Tracker, make_pipeline, regression_candidates, tune
from src.models.evaluate import plot_eta, regression_metrics

log = logging.getLogger(__name__)


def _fit(name, factory, linear, params, X, y):
    pipe = make_pipeline(factory(**params), linear)
    return pipe.fit(X, y)


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, tracker: Tracker,
        n_trials: int = 30) -> dict:
    Xtr, ytr = feature_matrix(train), train[config.ETA_TARGET]
    Xva, yva = feature_matrix(val), val[config.ETA_TARGET]
    Xte, yte = feature_matrix(test), test[config.ETA_TARGET]

    # Benchmark: the planner's own ETA
    rows = [{"model": "Planner's ETA (no ML)", "stage": "benchmark",
             **{f"val_{k}": v for k, v in regression_metrics(yva, val.planned_hours).items()},
             **{f"test_{k}": v for k, v in regression_metrics(yte, test.planned_hours).items()}}]

    cands = regression_candidates()
    fitted = {}
    for name, (factory, linear) in cands.items():
        t0 = time.time()
        pipe = _fit(name, factory, linear, {}, Xtr, ytr)
        mv, mt = regression_metrics(yva, pipe.predict(Xva)), regression_metrics(yte, pipe.predict(Xte))
        fitted[name] = pipe
        rows.append({"model": name, "stage": "default", **{f"val_{k}": v for k, v in mv.items()},
                     **{f"test_{k}": v for k, v in mt.items()}, "fit_seconds": round(time.time() - t0, 1)})
        tracker.log_run(f"eta/{name}", {"stage": "default"}, {f"val_{k}": v for k, v in mv.items()},
                        {"task": "eta"})
        log.info("ETA %-28s val MAE %.2f h", name, mv["MAE"])

    table = pd.DataFrame(rows)
    ml_rows = table[table.stage == "default"]
    best_name = ml_rows.sort_values("val_MAE").iloc[0]["model"]
    factory, linear = cands[best_name]

    def objective(params):
        return regression_metrics(yva, _fit(best_name, factory, linear, params, Xtr, ytr).predict(Xva))["MAE"]

    best_params, best_val, method = tune(best_name, objective, n_trials, "minimize")
    final = _fit(best_name, factory, linear, best_params, Xtr, ytr)
    mv, mt = regression_metrics(yva, final.predict(Xva)), regression_metrics(yte, final.predict(Xte))
    table = pd.concat([table, pd.DataFrame([{"model": f"{best_name} (tuned)", "stage": f"tuned:{method}",
                                             **{f"val_{k}": v for k, v in mv.items()},
                                             **{f"test_{k}": v for k, v in mt.items()}}])], ignore_index=True)
    tracker.log_run(f"eta/{best_name}-tuned", {"stage": "tuned", "method": method, **best_params},
                    {**{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}},
                    {"task": "eta", "final": "true"})

    # Keep the tuned model only if it beats the default one on validation
    default_val = float(ml_rows.loc[ml_rows.model == best_name, "val_MAE"].iloc[0])
    if mv["MAE"] > default_val:
        log.info("Tuning did not help on validation -> keeping default parameters")
        final, best_params = fitted[best_name], {}
        mt = regression_metrics(yte, final.predict(Xte))

    set_linear_background(final, Xtr)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_eta(yte, final.predict(Xte), config.FIGURES_DIR / "eta_test_performance.png",
             f"ETA model ({best_name}) – test set")
    artifact = {"pipeline": final, "model_name": best_name, "params": best_params,
                "tuning_method": method, "test_metrics": mt, "features": config.ALL_FEATURES}
    joblib.dump(artifact, config.ETA_MODEL_FILE)
    table.to_csv(config.METRICS_DIR / "eta_model_comparison.csv", index=False)
    return {"table": table, "best_model": best_name, "test_metrics": mt, "params": best_params,
            "tuning_method": method}
