"""
Feature engineering.

The SAME functions are used for training and for live predictions in the API.
That single code path is how we avoid "train/serve skew" (the model seeing
features computed one way in training and another way in production).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import KFold

from src import config
from src.data.geo import haversine_km

PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}      # city rush hours -> slow exit from origin


def route_key(origin: pd.Series, dest: pd.Series) -> pd.Series:
    return origin.astype(str) + "→" + dest.astype(str)


# ---------------------------------------------------------------------------
# Row-level features (need nothing but the row itself)
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = df.dispatch_time
    df["dispatch_hour"] = t.dt.hour
    df["dispatch_weekday"] = t.dt.weekday                 # 0 = Monday
    df["dispatch_month"] = t.dt.month
    df["is_weekend"] = (t.dt.weekday >= 5).astype(int)
    df["is_peak_hour"] = t.dt.hour.isin(PEAK_HOURS).astype(int)
    df["is_night_dispatch"] = ((t.dt.hour >= 22) | (t.dt.hour <= 4)).astype(int)
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["haversine_km"] = haversine_km(df.origin_lat, df.origin_lon, df.dest_lat, df.dest_lon)
    df["route_detour_ratio"] = (df.road_distance_km / df.haversine_km.clip(lower=1)).clip(0.8, 3.0)
    df["planned_speed_kmph"] = df.road_distance_km / df.planned_hours.clip(lower=0.5)
    df["route"] = route_key(df.origin_city, df.destination_city)
    return df


# ---------------------------------------------------------------------------
# Historical aggregates (need the past -> fitted on TRAINING data only)
# ---------------------------------------------------------------------------
@dataclass
class RouteHistory:
    """
    Remembers how each route / supplier performed in the TRAINING period.

    * Smoothing (m-estimate): a route seen 3 times shouldn't get a 100% delay
      rate just because 3/3 were late. We blend its own rate with the global
      rate:  (sum + m * global) / (n + m).
    * Unknown routes at prediction time simply get the global average.
    """
    m: float = 20.0
    global_delay_rate: float = 0.0
    global_overrun: float = 1.0
    route_stats: dict = field(default_factory=dict)
    supplier_stats: dict = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "RouteHistory":
        overrun = df[config.ETA_TARGET] / df.planned_hours
        self.global_delay_rate = float(df[config.DELAY_TARGET].mean())
        self.global_overrun = float(overrun.mean())
        g = df.assign(_ov=overrun).groupby("route")
        n = g.size()
        rate = (g[config.DELAY_TARGET].sum() + self.m * self.global_delay_rate) / (n + self.m)
        ov = (g["_ov"].sum() + self.m * self.global_overrun) / (n + self.m)
        self.route_stats = {r: (float(rate[r]), float(ov[r]), int(n[r])) for r in n.index}
        gs = df.groupby("supplier_id")
        ns = gs.size()
        srate = (gs[config.DELAY_TARGET].sum() + self.m * self.global_delay_rate) / (ns + self.m)
        self.supplier_stats = {s: float(srate[s]) for s in ns.index}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        default = (self.global_delay_rate, self.global_overrun, 0)
        stats = [self.route_stats.get(r, default) for r in df.route]
        df["route_hist_delay_rate"] = [s[0] for s in stats]
        df["route_hist_overrun_ratio"] = [s[1] for s in stats]
        df["route_hist_trips"] = [s[2] for s in stats]
        df["supplier_hist_delay_rate"] = [self.supplier_stats.get(s, self.global_delay_rate)
                                          for s in df.supplier_id]
        return df


def add_history_out_of_fold(train: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """
    For TRAINING rows we compute route history *out-of-fold*: each row's stats
    come from the other 4/5 of the training data. Otherwise a row would "see"
    its own label inside its route average -> target leakage -> over-optimistic
    model. Validation/test rows use stats from the whole training set.
    """
    train = train.copy()
    cols = ["route_hist_delay_rate", "route_hist_overrun_ratio", "route_hist_trips",
            "supplier_hist_delay_rate"]
    parts = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    for fit_idx, apply_idx in kf.split(train):
        hist = RouteHistory().fit(train.iloc[fit_idx])
        parts.append(hist.transform(train.iloc[apply_idx])[cols])
    oof = pd.concat(parts).loc[train.index]
    oof["route_hist_trips"] = (oof["route_hist_trips"] * n_splits / (n_splits - 1)).round()
    return train.drop(columns=[c for c in cols if c in train], errors="ignore").join(oof)


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on the past, validate on the next period, test on the most recent period."""
    t = df.dispatch_time
    train = df[t <= config.TRAIN_END]
    val = df[(t > config.TRAIN_END) & (t <= config.VAL_END)]
    test = df[t > config.VAL_END]
    return train, val, test


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features shared by training and inference (history added separately)."""
    return add_route_features(add_time_features(df))


def build_model_frames(df_enriched: pd.DataFrame):
    """
    Full training-time pipeline:
      enriched data -> row features -> time split -> route history (train-only) .
    Returns train, val, test frames and the RouteHistory fitted on train
    (the one used to evaluate) plus one fitted on train+val (the one shipped).
    """
    df = base_features(df_enriched)
    train, val, test = time_split(df)
    hist_train = RouteHistory().fit(train)
    train = add_history_out_of_fold(train)
    val = hist_train.transform(val)
    test = hist_train.transform(test)
    return train, val, test, hist_train


def check_no_leakage(features: list[str]) -> None:
    """Fail loudly if a post-delivery column sneaks into the feature list."""
    leaked = set(features) & set(config.LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"Leaky features detected: {sorted(leaked)}")


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    check_no_leakage(config.ALL_FEATURES)
    X = df[config.ALL_FEATURES].copy()
    for c in config.CATEGORICAL_FEATURES:
        X[c] = X[c].astype(str)
    num = config.NUMERIC_FEATURES
    X[num] = X[num].astype(float)
    return X

