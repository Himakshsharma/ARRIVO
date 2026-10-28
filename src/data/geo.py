"""Static reference data: logistics hubs, regional climate normals and the festival calendar."""
from __future__ import annotations

import hashlib
import math
from datetime import date

import numpy as np

# ---------------------------------------------------------------------------
# Logistics hubs: name -> (lat, lon, region, is_hilly, traffic_weight)
# traffic_weight: relative freight volume, used by the synthetic generator
# ---------------------------------------------------------------------------
HUBS: dict[str, tuple[float, float, str, int, int]] = {
    "Delhi": (28.61, 77.21, "North", 0, 10),
    "Gurugram": (28.46, 77.03, "North", 0, 6),
    "Mumbai": (19.08, 72.88, "West", 0, 10),
    "Bhiwandi": (19.30, 73.06, "West", 0, 5),
    "Bengaluru": (12.97, 77.59, "South", 0, 9),
    "Chennai": (13.08, 80.27, "South", 0, 8),
    "Kolkata": (22.57, 88.36, "East", 0, 7),
    "Hyderabad": (17.39, 78.49, "South", 0, 8),
    "Pune": (18.52, 73.86, "West", 0, 7),
    "Ahmedabad": (23.02, 72.57, "West", 0, 7),
    "Surat": (21.17, 72.83, "West", 0, 5),
    "Jaipur": (26.91, 75.79, "North", 0, 5),
    "Lucknow": (26.85, 80.95, "North", 0, 5),
    "Chandigarh": (30.73, 76.78, "North", 0, 4),
    "Ludhiana": (30.90, 75.85, "North", 0, 4),
    "Indore": (22.72, 75.86, "Central", 0, 5),
    "Nagpur": (21.15, 79.09, "Central", 0, 5),
    "Bhopal": (23.26, 77.41, "Central", 0, 3),
    "Raipur": (21.25, 81.63, "Central", 0, 2),
    "Coimbatore": (11.02, 76.96, "South", 0, 4),
    "Kochi": (9.93, 76.27, "South", 0, 4),
    "Visakhapatnam": (17.69, 83.22, "South", 0, 3),
    "Bhubaneswar": (20.30, 85.82, "East", 0, 3),
    "Patna": (25.59, 85.14, "East", 0, 3),
    "Guwahati": (26.14, 91.74, "Northeast", 1, 3),
    "Shimla": (31.10, 77.17, "North", 1, 2),
    "Dehradun": (30.32, 78.03, "North", 1, 2),
}

REGIONS = ["North", "West", "South", "East", "Central", "Northeast"]

# Approximate monthly rainfall normals (mm per month), Jan..Dec, per region.
# Rounded from IMD climatological normals for a representative city per region.
MONTHLY_RAIN_MM = {
    "North":     [19, 20, 15, 10, 25, 70, 210, 230, 125, 15, 5, 8],
    "West":      [1, 0, 0, 1, 15, 500, 850, 550, 330, 80, 15, 3],
    "South":     [5, 5, 10, 40, 110, 90, 120, 140, 170, 190, 110, 40],
    "East":      [10, 25, 35, 55, 140, 300, 370, 350, 300, 150, 20, 5],
    "Central":   [12, 15, 20, 10, 15, 170, 300, 280, 180, 60, 15, 10],
    "Northeast": [10, 20, 60, 150, 280, 320, 330, 270, 190, 90, 15, 5],
}
# Approximate monthly mean temperature (deg C), Jan..Dec, per region.
MONTHLY_TEMP_C = {
    "North":     [14, 17, 23, 29, 33, 33, 31, 30, 29, 26, 20, 15],
    "West":      [24, 25, 27, 29, 30, 29, 28, 27, 28, 29, 28, 26],
    "South":     [24, 26, 28, 30, 30, 28, 27, 27, 27, 26, 25, 24],
    "East":      [20, 23, 28, 31, 31, 31, 30, 30, 30, 28, 24, 20],
    "Central":   [21, 24, 28, 33, 35, 31, 28, 27, 28, 27, 23, 20],
    "Northeast": [17, 19, 23, 26, 28, 29, 29, 29, 28, 26, 22, 18],
}

# ---------------------------------------------------------------------------
# Freight-disrupting festivals. The `holidays` package only covers gazetted
# holidays, not the multi-day slowdown around them.
# ---------------------------------------------------------------------------
FESTIVALS: dict[str, list[str]] = {
    "Diwali":           ["2023-11-12", "2024-10-31", "2025-10-20", "2026-11-08"],
    "Holi":             ["2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04"],
    "Dussehra":         ["2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20"],
    "Ganesh Chaturthi": ["2023-09-19", "2024-09-07", "2025-08-27", "2026-09-14"],
    "Eid al-Fitr":      ["2023-04-22", "2024-04-11", "2025-03-31", "2026-03-21"],
    "Pongal/Sankranti": ["2023-01-15", "2024-01-15", "2025-01-14", "2026-01-14"],
    "Christmas":        ["2023-12-25", "2024-12-25", "2025-12-25", "2026-12-25"],
}
# Disruption window (days either side of the festival)
FESTIVAL_WINDOW_DAYS = {"Diwali": 5, "Dussehra": 3, "Holi": 2, "Ganesh Chaturthi": 3,
                        "Eid al-Fitr": 2, "Pongal/Sankranti": 2, "Christmas": 2}

# National (gazetted) holidays used when the `holidays` package is not installed.
NATIONAL_HOLIDAYS_FALLBACK = {"01-26", "08-15", "10-02"}


def festival_dates() -> list[tuple[str, date]]:
    out = []
    for name, days in FESTIVALS.items():
        for d in days:
            out.append((name, date.fromisoformat(d)))
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle ('as the crow flies') distance in km. Works on scalars or numpy arrays."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def nearest_region(lat: float, lon: float) -> str:
    """Region of the closest known hub (used for coordinates that are not a known hub)."""
    best, best_d = "North", math.inf
    for _, (hlat, hlon, region, _, _) in HUBS.items():
        d = float(haversine_km(lat, lon, hlat, hlon))
        if d < best_d:
            best, best_d = region, d
    return best


def _stable_rng(*keys) -> np.random.Generator:
    """Deterministic RNG from arbitrary keys (same inputs -> same 'random' weather)."""
    h = hashlib.md5("|".join(map(str, keys)).encode()).hexdigest()
    return np.random.default_rng(int(h[:12], 16))


def climatology_weather(lat: float, lon: float, region: str, hilly: int, day: date) -> dict:
    """
    Offline weather estimate for a location and day, drawn from regional climate
    normals. Used when Open-Meteo is unreachable (and by the synthetic generator).
    It is deterministic: the same place and day always give the same values.
    """
    rng = _stable_rng(round(lat, 1), round(lon, 1), day.isoformat())
    m = day.month - 1
    monthly = MONTHLY_RAIN_MM[region][m] * (1.3 if hilly else 1.0)
    p_rain = monthly / (monthly + 60.0)                    # probability of a rainy day
    mean_amount = monthly / max(30 * p_rain, 1e-6)         # mm on a rainy day
    rain = float(rng.exponential(mean_amount)) if rng.random() < p_rain else 0.0
    temp = MONTHLY_TEMP_C[region][m] - (12 if hilly else 0) + float(rng.normal(0, 2.0))
    wind = float(abs(rng.normal(12 + 0.05 * rain, 5)))
    return {"rain_mm": round(min(rain, 250.0), 1), "temp_c": round(temp, 1),
            "wind_kmph": round(wind, 1)}
