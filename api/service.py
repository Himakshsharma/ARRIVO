"""
Business logic behind the API, kept separate from FastAPI so it can be unit
tested directly and reused by the dashboard or a batch job.
"""
from __future__ import annotations

import io
import json
from functools import lru_cache

import pandas as pd
from pydantic import ValidationError

from api.schemas import ShipmentIn
from src import config
from src.data.geo import HUBS
from src.llm.alerts import make_alert
from src.models.predict import ArrivoPredictor

MAX_BATCH_ROWS = 5000
REQUIRED_BATCH_COLUMNS = {"origin_city", "destination_city", "dispatch_time"}


class ModelNotReady(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_predictor() -> ArrivoPredictor:
    """Load the models once per process (they're ~MBs; loading per request would be slow)."""
    if not config.ETA_MODEL_FILE.exists():
        raise ModelNotReady("Models not trained yet. Run: python -m src.pipeline")
    return ArrivoPredictor()


def health() -> dict:
    try:
        p = get_predictor()
        return {"status": "ok", "models_loaded": True, "eta_model": p.card["eta_model"]["name"],
                "delay_model": p.card["delay_model"]["name"], "trained_at": p.card["trained_at"],
                "data_source": p.card["data_source"], "live_enrichment": p.live}
    except ModelNotReady as e:
        return {"status": "degraded", "models_loaded": False, "detail": str(e)}


def meta() -> dict:
    card = get_predictor().card
    return {"cities": {c: {"lat": v[0], "lon": v[1], "region": v[2]} for c, v in sorted(HUBS.items())},
            "vehicle_types": card["vehicle_types"], "material_types": card["material_types"],
            "suppliers": card["suppliers"], "risk_bands": config.RISK_BANDS}


def predict_single(shipment: ShipmentIn) -> dict:
    return get_predictor().predict_one(shipment.to_record())


def _summary(results: list[dict], errors: list[dict]) -> dict:
    results = sorted(results, key=lambda r: r["delay_probability"], reverse=True)
    count = lambda lvl: sum(r["risk_level"] == lvl for r in results)  # noqa: E731
    return {"count": len(results), "high_risk": count("High"), "medium_risk": count("Medium"),
            "low_risk": count("Low"), "results": results, "errors": errors}


def predict_batch_csv(content: bytes) -> dict:
    """Validate every row; predict the valid ones in a single vectorised call; report bad rows."""
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:                                 # noqa: BLE001
        raise ValueError(f"Could not read CSV: {e}") from e
    missing = REQUIRED_BATCH_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError("CSV has no rows")
    if len(df) > MAX_BATCH_ROWS:
        raise ValueError(f"Batch too large ({len(df)} rows); limit is {MAX_BATCH_ROWS}")
    df = df.astype(object).where(df.notna(), None)
    records, errors = [], []
    for i, row in enumerate(df.to_dict("records")):
        try:
            records.append(ShipmentIn(**row).to_record())
        except ValidationError as e:
            msg = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'row'}: {err['msg']}" for err in e.errors())
            errors.append({"row": i + 2, "shipment_id": row.get("shipment_id"), "error": msg})  # +2: header + 1-based
    results = get_predictor().predict_frame(records) if records else []
    return _summary(results, errors)


def alert(shipment: ShipmentIn, customer_name: str, tone: str, use_llm: bool | None) -> dict:
    pred = predict_single(shipment)
    return {**make_alert(pred, customer_name, tone, use_llm), "prediction": pred}


def model_metrics() -> dict:
    card = get_predictor().card
    out = {k: card[k] for k in ("data_source", "trained_at", "split", "delay_definition", "delay_rate",
                                "eta_model", "delay_model", "top_eta_drivers", "top_delay_drivers", "backends")}
    for name in ("eta_model_comparison", "delay_model_comparison", "risk_band_calibration"):
        path = config.METRICS_DIR / f"{name}.csv"
        if path.exists():
            out[name] = json.loads(pd.read_csv(path).round(4).to_json(orient="records"))
    return out
