"""Inference: shipment details in, ETA, delay risk and top reasons out.

Uses the same feature code as training (``src.features`` and ``src.data.enrich``).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data.enrich import add_calendar, route_info, weather_for_location
from src.data.geo import HUBS, haversine_km, nearest_region
from src.explain.shap_utils import explain, top_reasons
from src.features.build_features import base_features, feature_matrix

log = logging.getLogger(__name__)

WEATHER_FIELDS = ["origin_rain_mm", "dest_rain_mm", "origin_temp_c", "dest_temp_c", "max_wind_kmph"]


def risk_level(p: float) -> str:
    for name, (lo, hi) in config.RISK_BANDS.items():
        if lo <= p < hi:
            return name
    return "High"


class ArrivoPredictor:
    def __init__(self, models_dir=config.MODELS_DIR):
        self.eta = joblib.load(models_dir / config.ETA_MODEL_FILE.name)
        self.delay = joblib.load(models_dir / config.DELAY_MODEL_FILE.name)
        self.history = joblib.load(models_dir / config.ROUTE_STATS_FILE.name)
        self.card = json.loads((models_dir / config.MODEL_CARD_FILE.name).read_text())
        # Models trained on synthetic data expect climatology weather, not live observations.
        default_live = self.card.get("data_source", "synthetic") != "synthetic"
        self.live = os.getenv("ARRIVO_LIVE_ENRICHMENT", str(int(default_live))) == "1"

    # ------------------------------------------------------------------
    def _locate(self, rec: dict, side: str):
        city = rec.get(f"{side}_city")
        lat, lon = rec.get(f"{side}_lat"), rec.get(f"{side}_lon")
        if city in HUBS and (lat is None or lon is None):
            lat, lon = HUBS[city][0], HUBS[city][1]
        if lat is None or lon is None:
            raise ValueError(f"Unknown {side} city '{city}'. Use a supported hub or pass {side}_lat/{side}_lon.")
        if city in HUBS:
            region, hilly = HUBS[city][2], HUBS[city][3]
        else:
            region, hilly = nearest_region(lat, lon), int(lat > 29.5 and lon > 76.5)
        return float(lat), float(lon), region, int(hilly)

    def build_frame(self, records: list[dict]) -> pd.DataFrame:
        """Build the model feature frame for raw request records."""
        rows = []
        for rec in records:
            olat, olon, oreg, ohill = self._locate(rec, "origin")
            dlat, dlon, dreg, dhill = self._locate(rec, "destination")
            t0 = pd.Timestamp(rec["dispatch_time"]).tz_localize(None)
            km = rec.get("road_distance_km")
            route = route_info(olat, olon, dlat, dlon, known_km=km, live=self.live)
            if km is None:
                hav = float(haversine_km(olat, olon, dlat, dlon))
                km = route["osrm_km"] if route["route_source"] == "osrm" else hav * self.card["median_detour_factor"]
            planned = rec.get("planned_hours")
            if planned is None:     # estimate from the training-set planning rate
                planned = km / self.card["median_planned_speed_kmph"] + self.card["median_planned_buffer_hours"]
            row = {"shipment_id": rec.get("shipment_id") or f"REQ-{len(rows) + 1}",
                   "dispatch_time": t0, "origin_city": rec.get("origin_city", "Custom"),
                   "destination_city": rec.get("destination_city", "Custom"),
                   "origin_lat": olat, "origin_lon": olon, "dest_lat": dlat, "dest_lon": dlon,
                   "origin_region": oreg, "dest_region": dreg, "origin_is_hilly": ohill, "dest_is_hilly": dhill,
                   "road_distance_km": float(km), "planned_hours": float(planned),
                   "osrm_duration_hrs": route["osrm_duration_hrs"],
                   "vehicle_type": rec.get("vehicle_type") or "Unknown",
                   "material_type": rec.get("material_type") or "Unknown",
                   "supplier_id": rec.get("supplier_id") or "UNKNOWN"}
            # Caller-supplied weather overrides the lookup
            ow = weather_for_location(olat, olon, oreg, ohill, [t0.date()], self.live).iloc[0]
            dw = weather_for_location(dlat, dlon, dreg, dhill, [t0.date()], self.live).iloc[0]
            auto = {"origin_rain_mm": ow.rain_mm, "dest_rain_mm": dw.rain_mm, "origin_temp_c": ow.temp_c,
                    "dest_temp_c": dw.temp_c, "max_wind_kmph": max(ow.wind_kmph, dw.wind_kmph)}
            for f in WEATHER_FIELDS:
                row[f] = float(rec[f]) if rec.get(f) is not None else float(auto[f])
            row["weather_source"] = "user" if any(rec.get(f) is not None for f in WEATHER_FIELDS) else ow.weather_source
            row["is_heavy_rain"] = int(max(row["origin_rain_mm"], row["dest_rain_mm"]) >= config.HEAVY_RAIN_MM)
            rows.append(row)
        df = add_calendar(pd.DataFrame(rows))
        return self.history.transform(base_features(df))

    # ------------------------------------------------------------------
    def predict_frame(self, records: list[dict], explain_top_k: int = 3) -> list[dict]:
        df = self.build_frame(records)
        X = feature_matrix(df)
        eta = np.clip(self.eta["pipeline"].predict(X), 0.5, None)
        proba = self.delay["calibrated"].predict_proba(X)[:, 1]
        eta_c, _, _ = explain(self.eta["pipeline"], X, "regression")
        del_c, _, _ = explain(self.delay["pipeline"], X, "classification")
        out = []
        for i, idx in enumerate(df.index):
            r = df.loc[idx]
            arrival = r.dispatch_time + timedelta(hours=float(eta[i]))
            planned_arrival = r.dispatch_time + timedelta(hours=float(r.planned_hours))
            out.append({
                "shipment_id": r.shipment_id, "origin_city": r.origin_city,
                "destination_city": r.destination_city,
                "dispatch_time": r.dispatch_time.isoformat(),
                "eta_hours": round(float(eta[i]), 2),
                "expected_arrival": arrival.isoformat(timespec="minutes"),
                "planned_hours": round(float(r.planned_hours), 2),
                "planned_arrival": planned_arrival.isoformat(timespec="minutes"),
                "expected_delay_hours": round(float(eta[i] - r.planned_hours), 2),
                "delay_probability": round(float(proba[i]), 4),
                "risk_level": risk_level(float(proba[i])),
                "flagged_late": bool(proba[i] >= self.delay["threshold"]),
                "top_reasons": top_reasons(eta_c.loc[idx], del_c.loc[idx], X.loc[idx], k=explain_top_k,
                                           prefer="lower" if risk_level(float(proba[i])) == "Low" else "raise"),
                "route": {"origin": [r.origin_lat, r.origin_lon], "destination": [r.dest_lat, r.dest_lon],
                          "road_distance_km": round(float(r.road_distance_km), 1)},
                "context": {"origin_rain_mm": r.origin_rain_mm, "dest_rain_mm": r.dest_rain_mm,
                            "is_festival_window": int(r.is_festival_window), "nearest_festival": r.nearest_festival,
                            "is_monsoon": int(r.is_monsoon), "weather_source": r.weather_source,
                            "route_hist_delay_rate": round(float(r.route_hist_delay_rate), 3),
                            "route_hist_trips": int(r.route_hist_trips)},
            })
        return out

    def predict_one(self, record: dict) -> dict:
        return self.predict_frame([record])[0]
