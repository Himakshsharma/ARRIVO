"""Cached objects shared by several test files (built once per test session)."""
from functools import lru_cache

from src.data.clean import clean
from src.data.enrich import enrich
from src.data.load import generate_synthetic, standardise_truck_raw


@lru_cache(maxsize=1)
def small_clean():
    df, report = clean(standardise_truck_raw(generate_synthetic(n_trips=1500, seed=7)))
    return df, report


@lru_cache(maxsize=1)
def small_enriched():
    return enrich(small_clean()[0], live=False)


@lru_cache(maxsize=1)
def predictor():
    from src.models.predict import ArrivoPredictor
    return ArrivoPredictor()


SAMPLE = {"shipment_id": "T-1", "origin_city": "Mumbai", "destination_city": "Kolkata",
          "dispatch_time": "2025-07-20T18:30:00", "vehicle_type": "32 FT Multi-Axle",
          "material_type": "Cement & Steel", "supplier_id": "SUP007"}
