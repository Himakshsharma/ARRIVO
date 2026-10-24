import numpy as np

from src import config
from src.data.geo import climatology_weather
from src.data.load import generate_synthetic
from tests.helpers import small_clean


def test_generator_is_reproducible():
    a = generate_synthetic(n_trips=200, seed=1)
    b = generate_synthetic(n_trips=200, seed=1)
    assert a.equals(b)


def test_cleaning_removes_duplicates_and_impossible_trips():
    df, rep = small_clean()
    assert rep["duplicates_removed"] > 0
    assert df.booking_id.is_unique
    assert (df.actual_hours > 0).all()
    speed = df.road_distance_km / df.actual_hours
    assert speed.between(5, 90).all()
    assert df.vehicle_type.notna().all() and df.road_distance_km.notna().all()


def test_delay_label_matches_definition():
    df, _ = small_clean()
    expected = (df.actual_hours - df.planned_hours > config.DELAY_GRACE_HOURS).astype(int)
    assert (df.is_delayed == expected).all()
    assert 0.15 < df.is_delayed.mean() < 0.6


def test_climatology_weather_is_deterministic_and_seasonal():
    from datetime import date
    w1 = climatology_weather(19.08, 72.88, "West", 0, date(2024, 7, 15))
    assert w1 == climatology_weather(19.08, 72.88, "West", 0, date(2024, 7, 15))
    july = np.mean([climatology_weather(19.08, 72.88, "West", 0, date(2024, 7, d))["rain_mm"] for d in range(1, 31)])
    jan = np.mean([climatology_weather(19.08, 72.88, "West", 0, date(2024, 1, d))["rain_mm"] for d in range(1, 31)])
    assert july > jan
