"""Project-wide paths, constants and feature lists shared by training, the API and the dashboard."""
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

# Caches for external API responses
WEATHER_CACHE_DIR = INTERIM_DIR / "weather_cache"
ROUTE_CACHE_FILE = INTERIM_DIR / "route_cache.csv"

# Raw dataset file names
KAGGLE_TRUCK_FILE = RAW_DIR / "Delivery_truck_trip_data.xlsx"   # Kaggle export, if present
SYNTHETIC_FILE = RAW_DIR / "arrivo_trips_synthetic.csv"          # generated fallback

CLEAN_FILE = INTERIM_DIR / "trips_clean.csv"
FEATURES_FILE = PROCESSED_DIR / "trips_features.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Targets (see docs/data_dictionary.md)
# ---------------------------------------------------------------------------
ETA_TARGET = "actual_hours"
DELAY_TARGET = "is_delayed"
DELAY_GRACE_HOURS = 0.0   # late when actual_hours > planned_hours + grace

# ---------------------------------------------------------------------------
# Chronological split boundaries
# ---------------------------------------------------------------------------
TRAIN_END = "2024-12-31 23:59:59"   # train:      2023-01-01 -> 2024-12-31
VAL_END = "2025-06-30 23:59:59"     # validation: 2025-01-01 -> 2025-06-30
                                    # test:       2025-07-01 -> end of data

# ---------------------------------------------------------------------------
# Delay-probability bands shown in the UI
# ---------------------------------------------------------------------------
RISK_BANDS = {"Low": (0.0, 0.35), "Medium": (0.35, 0.65), "High": (0.65, 1.01)}

# ---------------------------------------------------------------------------
# Model features (shared by training and inference)
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
    # history, fitted on the training split only
    "route_hist_delay_rate", "route_hist_overrun_ratio", "route_hist_trips",
    "supplier_hist_delay_rate",
]
CATEGORICAL_FEATURES = ["vehicle_type", "material_type", "origin_region", "dest_region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Post-delivery columns; never valid as features
LEAKY_COLUMNS = [
    "actual_hours", "is_delayed", "trip_end_time", "delay_hours",
    "actual_eta", "trip_end_date", "ontime", "delay", "Current_Location",
    "Curr_lat", "Curr_lon", "Data_Ping_time",
]

# Daily rainfall treated as travel-disrupting. Lower than IMD's 64.5 mm "heavy"
# category because road speeds drop well before that.
HEAVY_RAIN_MM = 35.0

# Model artifact names
ETA_MODEL_FILE = MODELS_DIR / "eta_model.joblib"
DELAY_MODEL_FILE = MODELS_DIR / "delay_model.joblib"
ROUTE_STATS_FILE = MODELS_DIR / "route_stats.joblib"
MODEL_CARD_FILE = MODELS_DIR / "model_card.json"

# MLflow
MLFLOW_EXPERIMENT = "arrivo"
MLFLOW_TRACKING_URI = f"file:{(ROOT / 'mlruns').as_posix()}"
