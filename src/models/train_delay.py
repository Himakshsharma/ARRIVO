"""
Delay classification: probability that a shipment arrives late.

Class imbalance (~1 late for every 2 on-time):
  * every model is trained with class weights / scale_pos_weight, so the
    minority "late" class is not ignored;
  * the decision threshold is tuned on validation to maximise F1 instead of
    blindly using 0.5;
  * class weighting distorts raw probabilities, so we re-calibrate them on the
    validation set (Platt / sigmoid scaling). A "70% risk" then really means
    roughly 7 in 10 such shipments are late - important for the risk bands.
"""
from __future__ import annotations

import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src import config
from src.explain.shap_utils import set_linear_background
from src.features.build_features import feature_matrix
from src.models.common import (Tracker, balanced_sample_weight, classification_candidates, make_pipeline,
                               needs_sample_weight, tune)
from src.models.evaluate import best_f1_threshold, classification_metrics, plot_classifier

try:                                    # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:                     # pragma: no cover
    FrozenEstimator = None

log = logging.getLogger(__name__)


def _fit(name, factory, linear, params, X, y):
    pipe = make_pipeline(factory(**params), linear)
    fit_kw = {"model__sample_weight": balanced_sample_weight(y)} if needs_sample_weight(name) else {}
    return pipe.fit(X, y, **fit_kw)


def calibrate(pipe, X_val, y_val):
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(FrozenEstimator(pipe), method="sigmoid")
    else:                               # older scikit-learn
        cal = CalibratedClassifierCV(pipe, method="sigmoid", cv="prefit")
    return cal.fit(X_val, y_val)


def run(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, tracker: Tracker,
        n_trials: int = 30) -> dict:
    Xtr, ytr = feature_matrix(train), train[config.DELAY_TARGET]
    Xva, yva = feature_matrix(val), val[config.DELAY_TARGET]
    Xte, yte = feature_matrix(test), test[config.DELAY_TARGET]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    rows = []
    cands = classification_candidates(pos_weight)
    fitted = {}
    for name, (factory, linear) in cands.items():
        t0 = time.time()
        pipe = _fit(name, factory, linear, {}, Xtr, ytr)
        pv, pt = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, pv)
        mv, mt = classification_metrics(yva, pv, thr), classification_metrics(yte, pt, thr)
        fitted[name] = pipe
        rows.append({"model": name, "stage": "default", **{f"val_{k}": v for k, v in mv.items()},
                     **{f"test_{k}": v for k, v in mt.items()}, "fit_seconds": round(time.time() - t0, 1)})
        tracker.log_run(f"delay/{name}", {"stage": "default", "pos_weight": round(pos_weight, 3)},
                        {f"val_{k}": v for k, v in mv.items()}, {"task": "delay"})
        log.info("DELAY %-28s val AUC %.3f F1 %.3f", name, mv["ROC_AUC"], mv["F1"])

    table = pd.DataFrame(rows)
    best_name = table.sort_values("val_ROC_AUC", ascending=False).iloc[0]["model"]
    factory, linear = cands[best_name]

    def objective(params):
        p = _fit(best_name, factory, linear, params, Xtr, ytr).predict_proba(Xva)[:, 1]
        return classification_metrics(yva, p)["ROC_AUC"]

    best_params, _, method = tune(best_name, objective, n_trials, "maximize")
    final = _fit(best_name, factory, linear, best_params, Xtr, ytr)
    default_auc = float(table.loc[table.model == best_name, "val_ROC_AUC"].iloc[0])
    if classification_metrics(yva, final.predict_proba(Xva)[:, 1])["ROC_AUC"] < default_auc:
        log.info("Tuning did not help on validation -> keeping default parameters")
        final, best_params = fitted[best_name], {}

    set_linear_background(final, Xtr)   # needed to explain a linear model

    # Calibrate probabilities + choose threshold, both on VALIDATION only
    calibrated = calibrate(final, Xva, yva)
    pv = calibrated.predict_proba(Xva)[:, 1]
    threshold = best_f1_threshold(yva, pv)
    pt = calibrated.predict_proba(Xte)[:, 1]
    mv, mt = classification_metrics(yva, pv, threshold), classification_metrics(yte, pt, threshold)
    table = pd.concat([table, pd.DataFrame([{"model": f"{best_name} (tuned + calibrated)",
                                             "stage": f"tuned:{method}",
                                             **{f"val_{k}": v for k, v in mv.items()},
                                             **{f"test_{k}": v for k, v in mt.items()}}])], ignore_index=True)
    tracker.log_run(f"delay/{best_name}-tuned", {"stage": "tuned", "method": method, **best_params},
                    {**{f"val_{k}": v for k, v in mv.items()}, **{f"test_{k}": v for k, v in mt.items()}},
                    {"task": "delay", "final": "true"})

    # Calibration check: mean predicted risk vs actual late rate per risk band
    bands = pd.cut(pt, [b[0] for b in config.RISK_BANDS.values()] + [1.01],
                   labels=list(config.RISK_BANDS), right=False)
    band_table = (pd.DataFrame({"band": bands, "pred": pt, "late": yte.values})
                  .groupby("band", observed=False).agg(shipments=("late", "size"),
                                                       mean_predicted_risk=("pred", "mean"),
                                                       actual_late_rate=("late", "mean")).reset_index())
    band_table.to_csv(config.METRICS_DIR / "risk_band_calibration.csv", index=False)

    plot_classifier(yte, pt, threshold, config.FIGURES_DIR / "delay_test_performance.png",
                    f"Delay model ({best_name}) – test set")
    artifact = {"pipeline": final, "calibrated": calibrated, "model_name": best_name,
                "params": best_params, "threshold": threshold, "tuning_method": method,
                "test_metrics": mt, "pos_weight": pos_weight, "features": config.ALL_FEATURES}
    joblib.dump(artifact, config.DELAY_MODEL_FILE)
    table.to_csv(config.METRICS_DIR / "delay_model_comparison.csv", index=False)
    return {"table": table, "best_model": best_name, "test_metrics": mt, "params": best_params,
            "threshold": threshold, "tuning_method": method, "band_table": band_table,
            "test_proba": np.asarray(pt)}
