"""
External enrichment: weather (Open-Meteo), road route (OSRM) and calendar
(holidays + Indian festivals).

Design rules
------------
* Every API result is cached on disk -> we never call the same request twice,
  re-runs are fast, and we stay polite to free public APIs.
* Every API has an offline fallback -> the pipeline and the live API never
  crash because a third-party service is down.
* Set ARRIVO_OFFLINE=1 to skip network calls entirely (CI, exams, no Wi-Fi).
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime

import numpy as np
import pandas as pd

from src import config
from src.data.geo import (FESTIVAL_WINDOW_DAYS, NATIONAL_HOLIDAYS_FALLBACK, climatology_weather,
                          festival_dates, haversine_km)

log = logging.getLogger(__name__)

try:                                   # optional dependency
    import requests
except ImportError:                    # pragma: no cover
    requests = None

try:
    import holidays as _holidays_lib   # optional dependency
except ImportError:                    # pragma: no cover
    _holidays_lib = None

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
DAILY_VARS = "precipitation_sum,temperature_2m_mean,wind_speed_10m_max"

_network_ok: bool | None = None        # remembered after the first failure


def _online() -> bool:
    global _network_ok
    if os.getenv("ARRIVO_OFFLINE") == "1" or requests is None:
        return False
    if _network_ok is None:
        try:
            requests.get("https://api.open-meteo.com", timeout=4)
            _network_ok = True
        except Exception:              # noqa: BLE001 - any failure means "offline"
            log.warning("Open-Meteo unreachable -> using climatology fallback")
            _network_ok = False
    return _network_ok


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
def _cache_path(lat: float, lon: float):
    config.WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.WEATHER_CACHE_DIR / f"{lat:.2f}_{lon:.2f}.csv"


def _fetch_open_meteo(lat: float, lon: float, start: date, end: date) -> pd.DataFrame | None:
    """Fetch daily weather for one location over a date range (one request)."""
    today = date.today()
    url = OPEN_METEO_ARCHIVE if end < today else OPEN_METEO_FORECAST
    params = {"latitude": lat, "longitude": lon, "daily": DAILY_VARS, "timezone": "Asia/Kolkata",
              "start_date": start.isoformat(), "end_date": end.isoformat()}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = r.json()["daily"]
        return pd.DataFrame({"date": d["time"], "rain_mm": d["precipitation_sum"],
                             "temp_c": d["temperature_2m_mean"], "wind_kmph": d["wind_speed_10m_max"]})
    except Exception as exc:           # noqa: BLE001
        log.warning("Open-Meteo request failed (%s) -> fallback", exc)
        return None


def weather_for_location(lat, lon, region, hilly, days: list[date], live: bool) -> pd.DataFrame:
    """Daily weather for one location for the given days (cached, with fallback)."""
    lat, lon = round(float(lat), 2), round(float(lon), 2)
    days = sorted(set(days))
    frame = None
    if live and _online():
        path = _cache_path(lat, lon)
        cached = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date"])
        have = set(cached["date"].astype(str))
        missing = [d for d in days if d.isoformat() not in have]
        if missing:
            fetched = _fetch_open_meteo(lat, lon, missing[0], missing[-1])
            if fetched is not None:
                cached = pd.concat([cached, fetched]).drop_duplicates("date", keep="last")
                cached.to_csv(path, index=False)
        frame = cached
    rows = []
    lookup = frame.set_index(frame["date"].astype(str)) if frame is not None and len(frame) else None
    for d in days:
        rec = None
        if lookup is not None and d.isoformat() in lookup.index:
            r = lookup.loc[d.isoformat()]
            if not pd.isna(r["rain_mm"]):
                rec = {"rain_mm": float(r["rain_mm"]), "temp_c": float(r["temp_c"]),
                       "wind_kmph": float(r["wind_kmph"]), "weather_source": "open-meteo"}
        if rec is None:
            rec = {**climatology_weather(lat, lon, region, hilly, d), "weather_source": "climatology"}
        rows.append({"date": d, **rec})
    return pd.DataFrame(rows)


def add_weather(df: pd.DataFrame, live: bool) -> pd.DataFrame:
    """Add origin/destination weather on the dispatch date."""
    df = df.copy()
    df["_day"] = df.dispatch_time.dt.date
    for side, lat, lon, reg, hill in (("origin", "origin_lat", "origin_lon", "origin_region", "origin_is_hilly"),
                                      ("dest", "dest_lat", "dest_lon", "dest_region", "dest_is_hilly")):
        parts = []
        for (la, lo, rg, hl), grp in df.groupby([lat, lon, reg, hill]):
            w = weather_for_location(la, lo, rg, hl, list(grp["_day"]), live)
            w = w.rename(columns={"date": "_day", "rain_mm": f"{side}_rain_mm", "temp_c": f"{side}_temp_c",
                                  "wind_kmph": f"{side}_wind_kmph", "weather_source": f"{side}_weather_source"})
            parts.append(grp[["_day"]].reset_index().merge(w, on="_day", how="left").set_index("index"))
        merged = pd.concat(parts).drop(columns="_day")
        df = df.join(merged)
    df["max_wind_kmph"] = df[["origin_wind_kmph", "dest_wind_kmph"]].max(axis=1)
    df["is_heavy_rain"] = ((df.origin_rain_mm >= config.HEAVY_RAIN_MM)
                           | (df.dest_rain_mm >= config.HEAVY_RAIN_MM)).astype(int)
    return df.drop(columns=["_day", "origin_wind_kmph", "dest_wind_kmph"])


# ---------------------------------------------------------------------------
# Route (OSRM)
# ---------------------------------------------------------------------------
def _osrm(lat1, lon1, lat2, lon2):
    try:
        r = requests.get(OSRM_URL.format(lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2),
                         params={"overview": "false"}, timeout=15)
        r.raise_for_status()
        route = r.json()["routes"][0]
        return route["distance"] / 1000, route["duration"] / 3600, "osrm"
    except Exception as exc:           # noqa: BLE001
        log.warning("OSRM failed (%s) -> fallback", exc)
        return None


def route_info(lat1, lon1, lat2, lon2, known_km: float | None = None, live: bool = True) -> dict:
    """Road distance (km) and free-flow driving time (hrs) for one origin/destination pair."""
    key = f"{lat1:.3f},{lon1:.3f},{lat2:.3f},{lon2:.3f}"
    cache = pd.read_csv(config.ROUTE_CACHE_FILE) if config.ROUTE_CACHE_FILE.exists() else None
    if cache is not None and key in set(cache["key"]):
        r = cache[cache["key"] == key].iloc[0]
        return {"osrm_km": r["km"], "osrm_duration_hrs": r["hrs"], "route_source": r["source"]}
    res = _osrm(lat1, lon1, lat2, lon2) if (live and _online()) else None
    if res is None:
        hav = float(haversine_km(lat1, lon1, lat2, lon2))
        km = known_km if known_km and not np.isnan(known_km) else hav * 1.25
        res = (km, km / 55.0, "estimate")   # 55 km/h = typical free-flow highway speed
    else:
        config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([{"key": key, "km": res[0], "hrs": res[1], "source": res[2]}])
        row.to_csv(config.ROUTE_CACHE_FILE, mode="a", index=False, header=cache is None)
    return {"osrm_km": res[0], "osrm_duration_hrs": res[1], "route_source": res[2]}


def add_route(df: pd.DataFrame, live: bool) -> pd.DataFrame:
    df = df.copy()
    pairs = df[["origin_lat", "origin_lon", "dest_lat", "dest_lon"]].drop_duplicates()
    info = {}
    for r in pairs.itertuples(index=False):
        sub = df[(df.origin_lat == r.origin_lat) & (df.origin_lon == r.origin_lon)
                 & (df.dest_lat == r.dest_lat) & (df.dest_lon == r.dest_lon)]
        info[tuple(r)] = route_info(*r, known_km=float(sub.road_distance_km.median()), live=live)
    keys = list(zip(df.origin_lat, df.origin_lon, df.dest_lat, df.dest_lon))
    df["osrm_duration_hrs"] = [info[k]["osrm_duration_hrs"] for k in keys]
    df["route_source"] = [info[k]["route_source"] for k in keys]
    return df


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def _is_national_holiday(d: date) -> int:
    if _holidays_lib is not None:
        return int(d in _holidays_lib.India(years=d.year))
    return int(d.strftime("%m-%d") in NATIONAL_HOLIDAYS_FALLBACK)


def calendar_features(ts: datetime) -> dict:
    d = ts.date() if isinstance(ts, (datetime, pd.Timestamp)) else ts
    fest = festival_dates()
    gaps = [((fd - d).days, name) for name, fd in fest]
    nearest = min(gaps, key=lambda g: abs(g[0]))
    in_window = any(abs(g) <= FESTIVAL_WINDOW_DAYS[n] for g, n in gaps)
    return {"is_national_holiday": _is_national_holiday(d),
            "is_festival_window": int(in_window),
            "days_to_festival": int(min(abs(nearest[0]), 30)),     # capped at 30 days
            "nearest_festival": nearest[1],
            "is_monsoon": int(d.month in (6, 7, 8, 9))}


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cal = pd.DataFrame([calendar_features(t) for t in df.dispatch_time], index=df.index)
    return df.join(cal)


def enrich(df: pd.DataFrame, live: bool) -> pd.DataFrame:
    """Run all enrichment steps. `live=False` uses only offline sources."""
    return add_calendar(add_route(add_weather(df, live), live))
