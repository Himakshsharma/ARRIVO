"""HTTP-level tests (need fastapi + httpx installed: pip install -r requirements.txt)."""
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402
from tests.helpers import SAMPLE  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] in {"Low", "Medium", "High"} and len(body["top_reasons"]) == 3


def test_predict_validation_error():
    r = client.post("/predict", json={**SAMPLE, "destination_city": "Mumbai"})
    assert r.status_code == 422


def test_batch_upload():
    csv = b"origin_city,destination_city,dispatch_time\nDelhi,Jaipur,2025-08-01 10:00\nPune,Chennai,2025-09-01 08:00\n"
    r = client.post("/predict/batch", files={"file": ("ships.csv", csv, "text/csv")})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_batch_wrong_type():
    r = client.post("/predict/batch", files={"file": ("ships.txt", b"x", "text/plain")})
    assert r.status_code == 415


def test_alert():
    r = client.post("/alert", json={"shipment": SAMPLE, "customer_name": "Priya", "use_llm": False})
    assert r.status_code == 200 and r.json()["mode"] == "template"
