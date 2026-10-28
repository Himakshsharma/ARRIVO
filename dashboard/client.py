"""Dashboard client for the Arrivo API.

Calls the HTTP API at ``ARRIVO_API_URL``. If it is unreachable (or
``ARRIVO_EMBEDDED=1``), the same service functions are called in-process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_URL = os.getenv("ARRIVO_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 60


class ApiError(RuntimeError):
    pass


class ArrivoClient:
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
        self.mode = "api" if self._api_alive() else "embedded"

    def _api_alive(self) -> bool:
        if os.getenv("ARRIVO_EMBEDDED") == "1":
            return False
        try:
            return requests.get(f"{self.base_url}/health", timeout=3).ok
        except requests.RequestException:
            return False

    # ------------------------------------------------------------------
    def _post(self, path, **kw):
        r = requests.post(f"{self.base_url}{path}", timeout=TIMEOUT, **kw)
        if not r.ok:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        return r.json()

    def _get(self, path):
        r = requests.get(f"{self.base_url}{path}", timeout=TIMEOUT)
        if not r.ok:
            raise ApiError(f"{r.status_code}: {r.text}")
        return r.json()

    # ------------------------------------------------------------------
    def health(self) -> dict:
        if self.mode == "api":
            return self._get("/health")
        from api import service
        return service.health()

    def meta(self) -> dict:
        if self.mode == "api":
            return self._get("/meta")
        from api import service
        return service.meta()

    def metrics(self) -> dict:
        if self.mode == "api":
            return self._get("/model/metrics")
        from api import service
        return service.model_metrics()

    def predict(self, shipment: dict) -> dict:
        if self.mode == "api":
            return self._post("/predict", json=shipment)
        from pydantic import ValidationError

        from api import service
        from api.schemas import ShipmentIn
        try:
            return service.predict_single(ShipmentIn(**shipment))
        except (ValidationError, ValueError) as e:
            raise ApiError(str(e)) from e

    def predict_batch(self, csv_bytes: bytes, filename: str = "batch.csv") -> dict:
        if self.mode == "api":
            return self._post("/predict/batch", files={"file": (filename, csv_bytes, "text/csv")})
        from api import service
        try:
            return service.predict_batch_csv(csv_bytes)
        except ValueError as e:
            raise ApiError(str(e)) from e

    def alert(self, shipment: dict, customer_name: str, tone: str) -> dict:
        payload = {"shipment": shipment, "customer_name": customer_name, "tone": tone}
        if self.mode == "api":
            return self._post("/alert", json=payload)
        from api import service
        from api.schemas import ShipmentIn
        return service.alert(ShipmentIn(**shipment), customer_name, tone, None)
