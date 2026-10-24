import pytest
from pydantic import ValidationError

from api import service
from api.schemas import ShipmentIn
from src.llm.alerts import make_alert
from tests.helpers import SAMPLE

CSV_OK = (b"shipment_id,origin_city,destination_city,dispatch_time,vehicle_type\n"
          b"A1,Delhi,Jaipur,2025-08-01 10:00,20 FT Container\n"
          b"A2,Mumbai,Kolkata,2025-07-20 18:30,32 FT Multi-Axle\n"
          b"A3,Delhi,Delhi,2025-07-20 18:30,32 FT Multi-Axle\n"
          b"A4,Delhi,Atlantis,2025-07-20 18:30,\n")


def test_schema_validation():
    assert ShipmentIn(**SAMPLE).origin_city == "Mumbai"
    with pytest.raises(ValidationError):
        ShipmentIn(**{**SAMPLE, "destination_city": "Mumbai"})
    with pytest.raises(ValidationError):
        ShipmentIn(**{**SAMPLE, "dispatch_time": "not a date"})
    with pytest.raises(ValidationError):
        ShipmentIn(**{**SAMPLE, "dest_rain_mm": -5})


def test_batch_reports_bad_rows_and_sorts_by_risk():
    out = service.predict_batch_csv(CSV_OK)
    assert out["count"] == 2 and len(out["errors"]) == 2
    assert {e["row"] for e in out["errors"]} == {4, 5}
    probs = [r["delay_probability"] for r in out["results"]]
    assert probs == sorted(probs, reverse=True)


def test_batch_rejects_missing_columns():
    with pytest.raises(ValueError):
        service.predict_batch_csv(b"origin_city,dispatch_time\nDelhi,2025-01-01\n")


def test_health_and_metrics():
    assert service.health()["status"] == "ok"
    m = service.model_metrics()
    assert "eta_model_comparison" in m and "delay_model_comparison" in m


def test_template_alert():
    pred = service.predict_single(ShipmentIn(**SAMPLE))
    a = make_alert(pred, "Priya", use_llm=False)
    assert a["mode"] == "template" and "Priya" in a["message"] and "Kolkata" in a["message"]
