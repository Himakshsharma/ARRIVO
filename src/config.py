"""
Central configuration for Arrivo.

Every path, constant and "magic number" lives here so that
training, the API and the dashboard all agree on the same settings.
Change a value here once, and the whole project follows.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# Weather / route API caches (so we never call an external API twice for the same thing)
WEATHER_CACHE_DIR = INTERIM_DIR / "weather_cache"
ROUTE_CACHE_FILE = INTERIM_DIR / "route_cache.csv"

# Raw dataset file names
KAGGLE_TRUCK_FILE = RAW_DIR / "Delivery_truck_trip_data.xlsx"   # real Kaggle file (if downloaded)
SYNTHETIC_FILE = RAW_DIR / "arrivo_trips_synthetic.csv"          # generated fallback

CLEAN_FILE = INTERIM_DIR / "trips_clean.csv"
FEATURES_FILE = PROCESSED_DIR / "trips_features.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Target definitions (explained in docs/data_dictionary.md)
# ---------------------------------------------------------------------------
# ETA target: actual transit time in hours (trip_end - trip_start)
ETA_TARGET = "actual_hours"
# Delay target: 1 if the truck arrives later than planned + a grace period
DELAY_TARGET = "is_delayed"
DELAY_GRACE_HOURS = 0.0   # 0 = same definition as the Kaggle dataset ("actual > planned")

# ---------------------------------------------------------------------------
# Time-based split (NO random split -> no peeking into the future)
# ---------------------------------------------------------------------------
TRAIN_END = "2024-12-31 23:59:59"   # train:      2023-01-01 -> 2024-12-31
VAL_END = "2025-06-30 23:59:59"     # validation: 2025-01-01 -> 2025-06-30
                                    # test:       2025-07-01 -> end of data

# ---------------------------------------------------------------------------
# Risk bands shown to operations managers
# ---------------------------------------------------------------------------
RISK_BANDS = {"Low": (0.0, 0.35), "Medium": (0.35, 0.65), "High": (0.65, 1.01)}

# ---------------------------------------------------------------------------
# Feature lists (used by training AND inference -> no train/serve skew)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    # route / distance
    "road_distance_km", "haversine_km", "route_detour_ratio", "osrm_duration_hrs",
    "planned_hours", "planned_speed_kmph",
    # time of dispatch
    "dispatch_hour", "dispatch_weekday", "dispatch_month",
    "is_weekend", "is_peak_hour", "is_night_dispatch",
    # calendar / season
    "is_national_holiday", "is_festival_window", "days_to_festival", "is_monsoon",
    # geography
    "origin_is_hilly", "dest_is_hilly",
    # weather on dispatch day
    "origin_rain_mm", "dest_rain_mm", "origin_temp_c", "dest_temp_c",
    "max_wind_kmph", "is_heavy_rain",
    # historical aggregates (computed from TRAINING data only)
    "route_hist_delay_rate", "route_hist_overrun_ratio", "route_hist_trips",
    "supplier_hist_delay_rate",
]
CATEGORICAL_FEATURES = ["vehicle_type", "material_type", "origin_region", "dest_region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that are only known AFTER delivery -> must NEVER be used as features
LEAKY_COLUMNS = [
    "actual_hours", "is_delayed", "trip_end_time", "delay_hours",
    "actual_eta", "trip_end_date", "ontime", "delay", "Current_Location",
    "Curr_lat", "Curr_lon", "Data_Ping_time",
]

# Heavy rain threshold (IMD classifies 64.5 mm/day as "heavy"; we flag >= 35 mm as
# "travel-disrupting" because trucks slow down well before IMD's heavy category)
HEAVY_RAIN_MM = 35.0

# Model artifact names
ETA_MODEL_FILE = MODELS_DIR / "eta_model.joblib"
DELAY_MODEL_FILE = MODELS_DIR / "delay_model.joblib"
ROUTE_STATS_FILE = MODELS_DIR / "route_stats.joblib"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

# MLflow
MLFLOW_EXPERIMENT = "arrivo"
MLFLOW_TRACKING_URI = f"file:{(ROOT / 'mlruns').as_posix()}"
