import joblib
import numpy as np
import pytest

from src import config
from src.explain.shap_utils import REASON_GROUPS, explain, top_reasons
from src.features.build_features import build_model_frames, feature_matrix
from tests.helpers import small_enriched


def _X():
    _, _, test, _ = build_model_frames(small_enriched())
    return feature_matrix(test).head(50)


def test_reason_groups_cover_every_feature_once():
    grouped = [f for fs in REASON_GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(config.ALL_FEATURES)


def test_eta_contributions_add_up_to_prediction():
    art = joblib.load(config.ETA_MODEL_FILE)
    X = _X()
    contrib, base, _ = explain(art["pipeline"], X, "regression")
    np.testing.assert_allclose(base + contrib.sum(axis=1).values, art["pipeline"].predict(X), atol=1e-4)


def test_delay_contributions_add_up_to_raw_score():
    art = joblib.load(config.DELAY_MODEL_FILE)
    X = _X()
    contrib, base, method = explain(art["pipeline"], X, "classification")
    model = art["pipeline"]
    if method == "saabas" and "RandomForest" in art["model_name"]:
        raw = model.predict_proba(X)[:, 1]                   # RF explained in probability space
    else:
        raw = model.decision_function(X)                       # others in log-odds
    np.testing.assert_allclose(base + contrib.sum(axis=1).values, raw, atol=1e-3)


@pytest.mark.parametrize("prefer", ["raise", "lower"])
def test_top_reasons_shape(prefer):
    X = _X()
    eta_c, _, _ = explain(joblib.load(config.ETA_MODEL_FILE)["pipeline"], X, "regression")
    del_c, _, _ = explain(joblib.load(config.DELAY_MODEL_FILE)["pipeline"], X, "classification")
    i = X.index[0]
    reasons = top_reasons(eta_c.loc[i], del_c.loc[i], X.loc[i], k=3, prefer=prefer)
    assert len(reasons) == 3
    assert all(isinstance(r["eta_impact_hours"], float) and r["text"] for r in reasons)
    assert len({r["factor"] for r in reasons}) == 3
