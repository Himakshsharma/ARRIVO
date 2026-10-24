"""
Cleaning + target creation.

Each rule is written down (and counted in a report) because "why did you drop
those rows?" is a guaranteed viva question.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src import config
from src.data.geo import haversine_km

log = logging.getLogger(__name__)

MIN_SPEED_KMPH = 5      # slower than this over a whole trip = data error, not a delay
MAX_SPEED_KMPH = 90     # Indian trucks are speed-governed at 80 km/h; faster = bad timestamps
MAX_TRIP_HOURS = 240    # 10 days; anything longer is a stuck/abandoned booking


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean dataframe with targets, report of what was removed/fixed)."""
    rep = {"rows_in": len(df)}
    df = df.copy()

    # 1. Duplicates: same booking reported twice by the GPS provider
    before = len(df)
    df = df.drop_duplicates(subset="booking_id", keep="first")
    rep["duplicates_removed"] = before - len(df)

    # 2. Rows without the timestamps we need cannot have a target
    before = len(df)
    df = df.dropna(subset=["dispatch_time", "trip_end_time", "planned_hours"])
    rep["missing_timestamps_removed"] = before - len(df)

    # 3. Missing distance -> estimate from coordinates x median detour factor
    hav = haversine_km(df.origin_lat, df.origin_lon, df.dest_lat, df.dest_lon)
    detour = (df.road_distance_km / hav).replace([np.inf, -np.inf], np.nan)
    median_detour = float(detour.median()) if detour.notna().any() else 1.25
    missing_dist = df.road_distance_km.isna()
    df.loc[missing_dist, "road_distance_km"] = hav[missing_dist] * median_detour
    rep["distance_imputed"] = int(missing_dist.sum())
    rep["median_detour_factor"] = round(median_detour, 3)

    # 4. Missing categoricals -> explicit "Unknown" (the model can learn from it)
    rep["vehicle_type_imputed"] = int(df.vehicle_type.isna().sum())
    df["vehicle_type"] = df.vehicle_type.fillna("Unknown")
    df["material_type"] = df.material_type.fillna("Unknown")
    df["supplier_id"] = df.supplier_id.fillna("UNKNOWN")

    # 5. Targets
    df["actual_hours"] = (df.trip_end_time - df.dispatch_time).dt.total_seconds() / 3600
    df["delay_hours"] = df.actual_hours - df.planned_hours
    df["is_delayed"] = (df.delay_hours > config.DELAY_GRACE_HOURS).astype(int)

    # 6. Impossible / corrupted trips (negative duration, impossible speeds, >10 days)
    speed = df.road_distance_km / df.actual_hours
    bad = ((df.actual_hours <= 0) | (df.actual_hours > MAX_TRIP_HOURS)
           | (speed < MIN_SPEED_KMPH) | (speed > MAX_SPEED_KMPH) | (df.planned_hours <= 0))
    rep["invalid_trips_removed"] = int(bad.sum())
    df = df[~bad]

    df = df.sort_values("dispatch_time").reset_index(drop=True)
    rep["rows_out"] = len(df)
    rep["delay_rate"] = round(float(df.is_delayed.mean()), 4)
    return df, rep
