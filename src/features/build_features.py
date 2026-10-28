"""Feature engineering shared by training and inference."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import KFold

from src import config
from src.data.geo import haversine_km

PEAK_HOURS = {8, 9, 10, 17, 18, 19, 20}


def route_key(origin: pd.Series, dest: pd.Series) -> pd.Series:
    return origin.astype(str) + "→" + dest.astype(str)


# ---------------------------------------------------------------------------
# Row-level features
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
# Historical aggregates (fitted on the training split only)
# ---------------------------------------------------------------------------
@dataclass
class RouteHistory:
    """Route- and supplier-level delay statistics learned from training data.

    Rates are smoothed with an m-estimate, ``(sum + m * global) / (n + m)``, so
    rarely seen routes are pulled towards the global rate. Unseen routes and
    suppliers fall back to the global values.
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
    """Attach route history to training rows using K-fold out-of-fold estimates.

    Each row's statistics come from the other folds, so a row never contributes
    to its own route average. Validation and test rows are transformed with a
    ``RouteHistory`` fitted on the full training split instead.
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
    """Chronological train / validation / test split (boundaries in ``config``)."""
    t = df.dispatch_time
    train = df[t <= config.TRAIN_END]
    val = df[(t > config.TRAIN_END) & (t <= config.VAL_END)]
    test = df[t > config.VAL_END]
    return train, val, test


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features shared by training and inference (history added separately)."""
    return add_route_features(add_time_features(df))


def build_model_frames(df_enriched: pd.DataFrame):
    """Build train/validation/test frames from enriched data.

    Returns ``(train, val, test, history)`` where ``history`` is the
    ``RouteHistory`` fitted on the training split.
    """
    df = base_features(df_enriched)
    train, val, test = time_split(df)
    hist_train = RouteHistory().fit(train)
    train = add_history_out_of_fold(train)
    val = hist_train.transform(val)
    test = hist_train.transform(test)
    return train, val, test, hist_train


def check_no_leakage(features: list[str]) -> None:
    """Raise if any post-delivery column is in the feature list."""
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

