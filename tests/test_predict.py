from datetime import datetime, timedelta

import pytest

from src.models.predict import risk_level
from tests.helpers import SAMPLE, predictor


def test_risk_levels():
    assert risk_level(0.1) == "Low" and risk_level(0.5) == "Medium" and risk_level(0.9) == "High"
    assert risk_level(0.35) == "Medium" and risk_level(1.0) == "High"


def test_predict_one_contract():
    r = predictor().predict_one(SAMPLE)
    assert r["eta_hours"] > 0 and 0 <= r["delay_probability"] <= 1
    assert r["risk_level"] in {"Low", "Medium", "High"}
    assert len(r["top_reasons"]) == 3
    arrival = datetime.fromisoformat(r["expected_arrival"])
    dispatch = datetime.fromisoformat(r["dispatch_time"])
    assert abs((arrival - dispatch) - timedelta(hours=r["eta_hours"])) < timedelta(minutes=1)


def test_heavy_rain_raises_risk_and_eta():
    dry = predictor().predict_one({**SAMPLE, "origin_rain_mm": 0, "dest_rain_mm": 0})
    wet = predictor().predict_one({**SAMPLE, "origin_rain_mm": 60, "dest_rain_mm": 90})
    assert wet["eta_hours"] > dry["eta_hours"]
    assert wet["delay_probability"] > dry["delay_probability"]


def test_unknown_city_needs_coordinates():
    with pytest.raises(ValueError):
        predictor().predict_one({**SAMPLE, "origin_city": "Atlantis"})
    r = predictor().predict_one({**SAMPLE, "origin_city": "Nashik", "origin_lat": 20.0, "origin_lon": 73.79})
    assert r["eta_hours"] > 0


def test_batch_matches_single():
    single = predictor().predict_one(SAMPLE)
    batch = predictor().predict_frame([SAMPLE, {**SAMPLE, "destination_city": "Pune"}])
    assert batch[0]["eta_hours"] == pytest.approx(single["eta_hours"])
