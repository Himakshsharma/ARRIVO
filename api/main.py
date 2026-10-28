"""
Arrivo REST API.

Run locally:   uvicorn api.main:app --reload --port 8000
Swagger docs:  http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api import service
from api.schemas import AlertIn, AlertOut, BatchOut, PredictionOut, ShipmentIn

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Arrivo – Shipment ETA & Delay Risk API",
    version="1.0.0",
    description=("Predicts transit time (ETA), the probability a shipment arrives late, and the top "
                 "reasons behind each prediction. Built for logistics operations teams."),
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("ARRIVO_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024   # 5 MB


def _models_or_503():
    try:
        return service.get_predictor()
    except service.ModelNotReady as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/health", tags=["system"])
def health():
    """Liveness/readiness check used by Docker, Render and the dashboard."""
    return service.health()


@app.get("/meta", tags=["system"])
def meta():
    """Supported cities, vehicle types, cargo types and risk bands (for building input forms)."""
    _models_or_503()
    return service.meta()


@app.get("/model/metrics", tags=["system"])
def model_metrics():
    """Model card: which models are live, their test metrics and the model comparison tables."""
    _models_or_503()
    return service.model_metrics()


@app.post("/predict", response_model=PredictionOut, tags=["prediction"])
def predict(shipment: ShipmentIn):
    """Predict ETA, delay probability, risk level and top-3 reasons for ONE shipment."""
    _models_or_503()
    try:
        return service.predict_single(shipment)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/predict/batch", response_model=BatchOut, tags=["prediction"])
async def predict_batch(file: UploadFile = File(..., description="CSV with columns origin_city, "
                                                "destination_city, dispatch_time (+ optional fields)")):
    """Score a CSV of shipments. Results are sorted by delay risk (highest first); invalid rows are
    reported in `errors` instead of failing the whole batch."""
    _models_or_503()
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Please upload a .csv file")
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 5 MB")
    try:
        return service.predict_batch_csv(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/alert", response_model=AlertOut, tags=["alerts"])
def alert(req: AlertIn):
    """Customer-facing message for a shipment (LLM if configured, otherwise a template)."""
    _models_or_503()
    try:
        return service.alert(req.shipment, req.customer_name, req.tone, req.use_llm)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
