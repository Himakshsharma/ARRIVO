import pandas as pd
import pytest

from src import config
from src.data.enrich import calendar_features
from src.features.build_features import (RouteHistory, add_time_features, build_model_frames, check_no_leakage,
                                         feature_matrix)
from tests.helpers import small_enriched


def test_time_features():
    df = pd.DataFrame({"dispatch_time": pd.to_datetime(["2025-03-08 23:15", "2025-03-10 09:00"])})
    out = add_time_features(df)
    assert out.dispatch_hour.tolist() == [23, 9]
    assert out.is_weekend.tolist() == [1, 0]            # 8 Mar 2025 = Saturday
    assert out.is_night_dispatch.tolist() == [1, 0]
    assert out.is_peak_hour.tolist() == [0, 1]


def test_festival_and_monsoon_flags():
    diwali = calendar_features(pd.Timestamp("2025-10-19"))
    assert diwali["is_festival_window"] == 1 and diwali["days_to_festival"] == 1
    assert calendar_features(pd.Timestamp("2025-07-10"))["is_monsoon"] == 1
    assert calendar_features(pd.Timestamp("2025-01-26"))["is_national_holiday"] == 1


def test_no_leaky_features():
    check_no_leakage(config.ALL_FEATURES)
    with pytest.raises(ValueError):
        check_no_leakage(config.ALL_FEATURES + ["actual_hours"])


def test_time_split_has_no_overlap():
    train, val, test, _ = build_model_frames(small_enriched())
    assert train.dispatch_time.max() < val.dispatch_time.min()
    assert val.dispatch_time.max() < test.dispatch_time.min()
    assert len(train) > len(val) > 0 and len(test) > 0


def test_route_history_uses_training_only_and_handles_unknown_routes():
    train, val, _, hist = build_model_frames(small_enriched())
    assert set(hist.route_stats) <= set(train.route)          # nothing from val/test
    unseen = val.head(1).assign(route="Nowhere→Elsewhere", supplier_id="NEW")
    out = hist.transform(unseen)
    assert out.route_hist_delay_rate.iloc[0] == pytest.approx(hist.global_delay_rate)
    assert out.route_hist_trips.iloc[0] == 0


def test_route_history_smoothing():
    df = pd.DataFrame({"route": ["A→B"] * 2 + ["C→D"] * 98, "supplier_id": "S",
                       config.DELAY_TARGET: [1, 1] + [0] * 98,
                       config.ETA_TARGET: 10.0, "planned_hours": 10.0})
    h = RouteHistory(m=20).fit(df)
    rate = h.route_stats["A→B"][0]
    assert h.global_delay_rate < rate < 1.0                # pulled toward the global mean, not 100%


def test_feature_matrix_complete():
    train, *_ = build_model_frames(small_enriched())
    X = feature_matrix(train)
    assert list(X.columns) == config.ALL_FEATURES
    assert X[config.NUMERIC_FEATURES].isna().sum().sum() == 0
