"""
End-to-end pipeline: raw data -> clean -> enrich -> features -> models -> explanations.

Usage:
    python -m src.pipeline                 # full run (30 Optuna trials per model)
    python -m src.pipeline --trials 10     # quicker
    python -m src.pipeline --synthetic     # ignore any Kaggle file in data/raw
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.clean import clean
from src.data.enrich import enrich
from src.data.load import load_canonical
from src.eda import run_all as run_eda, write_insights
from src.explain.shap_utils import explain, plot_beeswarm, plot_importance, plot_waterfall
from src.features.build_features import base_features, build_model_frames, feature_matrix
from src.models import train_delay, train_eta
from src.models.common import Tracker, backend_summary

log = logging.getLogger("arrivo")


def set_seeds(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed)


def run(n_trials: int = 30, force_synthetic: bool = False) -> dict:
    t_start = time.time()
    set_seeds()
    for d in (config.INTERIM_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR, config.METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1-2: load + clean -------------------------------------------------
    canonical, source = load_canonical(force_synthetic)
    df, clean_report = clean(canonical)
    df.to_csv(config.CLEAN_FILE, index=False)
    log.info("Data source: %s | clean rows: %d | delay rate %.1f%%", source, len(df), 100 * df.is_delayed.mean())

    # ---- Phase 3: enrichment + features -------------------------------------------
    live = source != "synthetic"           # synthetic world -> synthetic (climatology) weather
    enriched = enrich(df, live=live)
    write_insights(run_eda(canonical, df, enriched, clean_report))   # 11 EDA plots + insights
    full = base_features(enriched)
    full.to_csv(config.FEATURES_FILE, index=False)
    train, val, test, hist_train = build_model_frames(enriched)
    log.info("Split sizes  train %d | val %d | test %d", len(train), len(val), len(test))

    # ---- Phase 4: models --------------------------------------------------------------
    tracker = Tracker()
    eta_res = train_eta.run(train, val, test, tracker, n_trials)
    delay_res = train_delay.run(train, val, test, tracker, n_trials)
    joblib.dump(hist_train, config.ROUTE_STATS_FILE)

    # ---- Phase 5: explainability (global, on the TEST set) -------------------------
    eta_art, del_art = joblib.load(config.ETA_MODEL_FILE), joblib.load(config.DELAY_MODEL_FILE)
    Xte = feature_matrix(test)
    eta_c, eta_base, eta_method = explain(eta_art["pipeline"], Xte, "regression")
    del_c, del_base, del_method = explain(del_art["pipeline"], Xte, "classification")
    # additivity check: base + sum(contributions) must equal the model output
    eta_pred = eta_art["pipeline"].predict(Xte)
    additivity_err = float(np.max(np.abs(eta_base + eta_c.sum(axis=1).values - eta_pred)))
    fig = config.FIGURES_DIR
    eta_imp = plot_importance(eta_c, fig / "shap_eta_importance.png", "What drives transit time (ETA model)", "hours")
    del_unit = "probability" if del_method == "saabas" and "RandomForest" in del_art["model_name"] else "log-odds"
    del_imp = plot_importance(del_c, fig / "shap_delay_importance.png", "What drives delay risk", del_unit)
    plot_beeswarm(eta_c, Xte, fig / "shap_eta_summary.png", "SHAP summary – ETA model", "hours")
    plot_beeswarm(del_c, Xte, fig / "shap_delay_summary.png", "SHAP summary – delay model", del_unit)
    # example shipment: a typical high-risk case (probability closest to 0.85), not the most extreme one
    risky = int(np.argmin(np.abs(delay_res["test_proba"] - 0.85)))
    idx = Xte.index[risky]
    plot_waterfall(eta_c.loc[idx], eta_base, Xte.loc[idx], fig / "shap_eta_waterfall_example.png",
                   f"Why this shipment's ETA is {eta_pred[risky]:.1f} hrs", "hours")
    plot_waterfall(del_c.loc[idx], del_base, Xte.loc[idx], fig / "shap_delay_waterfall_example.png",
                   "Why this shipment is high-risk", del_unit)

    # ---- model card: everything the API / dashboard needs to know -----------------
    tr_val = pd.concat([train, val])
    slope, intercept = np.polyfit(tr_val.road_distance_km, tr_val.planned_hours, 1)
    hav = tr_val.haversine_km.clip(lower=1)
    card = {
        "project": "Arrivo", "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": source, "rows_clean": int(len(df)),
        "split": {"train": [str(train.dispatch_time.min()), str(train.dispatch_time.max()), len(train)],
                  "validation": [str(val.dispatch_time.min()), str(val.dispatch_time.max()), len(val)],
                  "test": [str(test.dispatch_time.min()), str(test.dispatch_time.max()), len(test)]},
        "delay_definition": f"actual_hours > planned_hours + {config.DELAY_GRACE_HOURS} h",
        "delay_rate": {"train": float(train.is_delayed.mean()), "val": float(val.is_delayed.mean()),
                       "test": float(test.is_delayed.mean())},
        "eta_model": {"name": eta_res["best_model"], "params": eta_res["params"],
                      "tuning": eta_res["tuning_method"], "test_metrics": eta_res["test_metrics"],
                      "explainer": eta_method, "shap_additivity_max_error": additivity_err},
        "delay_model": {"name": delay_res["best_model"], "params": delay_res["params"],
                        "tuning": delay_res["tuning_method"], "threshold": delay_res["threshold"],
                        "test_metrics": delay_res["test_metrics"], "explainer": del_method,
                        "calibration": "sigmoid (Platt) on validation set"},
        "risk_bands": config.RISK_BANDS,
        "top_eta_drivers": {k: round(float(v), 3) for k, v in eta_imp.head(10).items()},
        "top_delay_drivers": {k: round(float(v), 4) for k, v in del_imp.head(10).items()},
        "median_planned_speed_kmph": float(1 / slope), "median_planned_buffer_hours": float(intercept),
        "median_detour_factor": float((tr_val.road_distance_km / hav).median()),
        "vehicle_types": sorted(df.vehicle_type.unique().tolist()),
        "material_types": sorted(df.material_type.unique().tolist()),
        "suppliers": sorted(df.supplier_id.unique().tolist()),
        "clean_report": clean_report, "backends": backend_summary(),
        "features": config.ALL_FEATURES,
    }
    config.MODEL_CARD_FILE.write_text(json.dumps(card, indent=2, default=str))

    # A sample batch file for the dashboard/API demo (taken from the test period)
    sample = test.sample(40, random_state=config.RANDOM_SEED)
    sample[["booking_id", "origin_city", "destination_city", "dispatch_time", "vehicle_type", "material_type",
            "supplier_id", "planned_hours", "road_distance_km"]].rename(
        columns={"booking_id": "shipment_id"}).to_csv(config.PROCESSED_DIR / "sample_batch.csv", index=False)

    summary = {"source": source, "seconds": round(time.time() - t_start, 1), "clean_report": clean_report,
               "eta": eta_res["test_metrics"], "delay": delay_res["test_metrics"],
               "eta_model": eta_res["best_model"], "delay_model": delay_res["best_model"]}
    (config.METRICS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.trials, a.synthetic), indent=2, default=str))
