"""SHAP-based explanations, reason text and explanation plots.

Contributions are additive: ``base + sum(contributions) == model output``.
The back-end is chosen per model type:

* linear models: exact closed form, ``coef * (x - mean)``;
* LightGBM / XGBoost: native TreeSHAP (``pred_contrib``);
* other tree ensembles: ``shap.TreeExplainer`` if installed, else Saabas
  path attribution.

One-hot columns are summed back to their source feature.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from src import config  # noqa: E402

try:
    import shap  # optional
except Exception:  # noqa: BLE001
    shap = None


# ---------------------------------------------------------------------------
# Raw contributions in the transformed (one-hot) space
# ---------------------------------------------------------------------------
def _saabas_tree(tree, X, value_fn):
    """Saabas path attribution for one fitted decision tree."""
    t = tree.tree_
    values = value_fn(t.value)                          # prediction at each node
    parent = np.full(t.node_count, -1)
    for p in range(t.node_count):
        for c in (t.children_left[p], t.children_right[p]):
            if c != -1:
                parent[c] = p
    nodes = np.where(parent >= 0)[0]
    delta = values[nodes] - values[parent[nodes]]       # change caused by the split above
    feat = t.feature[parent[nodes]]
    D = sparse.csr_matrix((delta, (nodes, feat)), shape=(t.node_count, X.shape[1]))
    path = tree.decision_path(X.astype(np.float32))
    return np.asarray((path @ D).todense()), float(values[0])


def _tree_contribs(model, X, kind):
    name = type(model).__name__
    if name.startswith("LGBM"):
        out = model.predict(X, pred_contrib=True)
        return out[:, :-1], float(out[0, -1]), "lightgbm-treeshap"
    if name.startswith("XGB"):
        import xgboost as xgb
        out = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return out[:, :-1], float(out[0, -1]), "xgboost-treeshap"
    if shap is not None:
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        base = ex.expected_value
        if isinstance(sv, list):                          # older shap: list per class
            sv, base = sv[1], base[1]
        elif np.ndim(sv) == 3:                            # newer shap: (n, features, classes)
            sv, base = sv[:, :, 1], np.atleast_1d(base)[1]
        return np.asarray(sv), float(np.atleast_1d(base)[0]), "shap-treeexplainer"
    # Saabas attribution for scikit-learn ensembles
    if name.startswith("RandomForest"):
        if kind == "classification":
            fn = lambda v: v[:, 0, 1] / v[:, 0, :].sum(axis=1)   # P(late) at each node
        else:
            fn = lambda v: v[:, 0, 0]
        parts = [_saabas_tree(est, X, fn) for est in model.estimators_]
        return np.mean([p[0] for p in parts], axis=0), float(np.mean([p[1] for p in parts])), "saabas"
    if name.startswith("GradientBoosting"):
        lr = model.learning_rate
        parts = [_saabas_tree(est, X, lambda v: v[:, 0, 0]) for est in model.estimators_[:, 0]]
        contrib = lr * np.sum([p[0] for p in parts], axis=0)
        # base value = model output for row 0 minus that row's contributions (public API only)
        raw0 = model.decision_function(X[:1])[0] if kind == "classification" else model.predict(X[:1])[0]
        return contrib, float(raw0 - contrib[0].sum()), "saabas"
    raise TypeError(f"No explainer for {name}")


def raw_contributions(pipeline, X_raw: pd.DataFrame, kind: str):
    """Return (contributions in transformed space, base value, transformed names, method)."""
    pre, model = pipeline.named_steps["pre"], pipeline.named_steps["model"]
    Xt = pre.transform(X_raw)
    names = list(pre.get_feature_names_out())
    if hasattr(model, "coef_"):                          # linear / logistic regression
        coef = np.ravel(model.coef_)
        mean = np.asarray(getattr(pipeline, "_train_mean_", np.zeros(Xt.shape[1])))
        contrib = (Xt - mean) * coef
        base = float(np.ravel(model.intercept_)[0] + mean @ coef)
        return contrib, base, names, "linear-exact"
    contrib, base, method = _tree_contribs(model, Xt, kind)
    return np.asarray(contrib), base, names, method


def explain(pipeline, X_raw: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, float, str]:
    """Per-row contributions aggregated back to ORIGINAL feature names."""
    contrib, base, names, method = raw_contributions(pipeline, X_raw, kind)
    groups = {}
    for i, n in enumerate(names):
        col = n.split("__", 1)[1]
        orig = next((c for c in config.CATEGORICAL_FEATURES if col.startswith(c + "_")), col)
        groups.setdefault(orig, []).append(i)
    agg = pd.DataFrame({f: contrib[:, idx].sum(axis=1) for f, idx in groups.items()}, index=X_raw.index)
    return agg[[c for c in config.ALL_FEATURES if c in agg]], base, method


def set_linear_background(pipeline, X_train_raw: pd.DataFrame):
    """Store the training mean of each transformed column, required for linear contributions."""
    if hasattr(pipeline.named_steps["model"], "coef_"):
        pipeline._train_mean_ = pipeline.named_steps["pre"].transform(X_train_raw).mean(axis=0)
    return pipeline


# ---------------------------------------------------------------------------
# Human-readable reasons
# ---------------------------------------------------------------------------
LABELS = {
    "road_distance_km": "Route length", "haversine_km": "Straight-line distance",
    "route_detour_ratio": "Winding / indirect road", "osrm_duration_hrs": "Free-flow drive time",
    "planned_hours": "Planned transit time", "planned_speed_kmph": "Tight planned schedule",
    "dispatch_hour": "Dispatch hour", "dispatch_weekday": "Day of week", "dispatch_month": "Month",
    "is_weekend": "Weekend dispatch", "is_peak_hour": "Rush-hour dispatch",
    "is_night_dispatch": "Night dispatch", "is_national_holiday": "National holiday",
    "is_festival_window": "Festival season traffic", "days_to_festival": "Proximity to a festival",
    "is_monsoon": "Monsoon season", "origin_is_hilly": "Hilly origin", "dest_is_hilly": "Hilly destination",
    "origin_rain_mm": "Rain at origin", "dest_rain_mm": "Rain at destination",
    "origin_temp_c": "Temperature at origin", "dest_temp_c": "Temperature at destination",
    "max_wind_kmph": "Strong wind", "is_heavy_rain": "Heavy rain on route",
    "route_hist_delay_rate": "Route's past delay record", "route_hist_overrun_ratio": "Route usually overruns plan",
    "route_hist_trips": "Route familiarity", "supplier_hist_delay_rate": "Supplier's past reliability",
    "vehicle_type": "Vehicle type", "material_type": "Cargo type", "origin_region": "Origin region",
    "dest_region": "Destination region",
}


def describe(feature: str, value) -> str:
    """Human-readable phrase for a single feature value."""
    label = LABELS.get(feature, feature)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{label}: {value}"
    if feature.endswith("rain_mm"):
        intensity = "Heavy rain" if v >= config.HEAVY_RAIN_MM else "Rain" if v >= 5 else "Little/no rain"
        where = "origin" if feature.startswith("origin") else "destination"
        return f"{intensity} at {where} ({v:.0f} mm)"
    if feature in ("route_hist_delay_rate", "supplier_hist_delay_rate"):
        return f"{label} ({v:.0%} late historically)"
    if feature == "route_hist_overrun_ratio":
        return f"{label} (avg {v:.2f}× planned time)"
    if feature == "road_distance_km":
        return f"{label} ({v:,.0f} km)"
    if feature == "planned_speed_kmph":
        return f"Planned average speed ({v:.0f} km/h)"
    if feature == "planned_hours":
        return f"{label} ({v:.0f} hrs)"
    if feature == "dispatch_hour":
        return f"Dispatched at {int(v):02d}:00"
    if feature == "days_to_festival":
        return f"{int(v)} days from a major festival" if v < 30 else "No festival nearby"
    if feature in ("origin_temp_c", "dest_temp_c"):
        return f"{label} ({v:.0f}°C)"
    if feature == "max_wind_kmph":
        return f"Wind up to {v:.0f} km/h"
    if feature.startswith("is_") or feature.endswith("_is_hilly"):
        return label if v >= 0.5 else f"Not {label[0].lower() + label[1:]}"
    return f"{label} ({v:,.2f})"


# Correlated features split SHAP credit, so reasons are reported per group
# (valid because contributions are additive).
REASON_GROUPS = {
    "route_length": ["road_distance_km", "haversine_km", "osrm_duration_hrs", "route_detour_ratio"],
    "schedule": ["planned_hours", "planned_speed_kmph"],
    "rain": ["origin_rain_mm", "dest_rain_mm", "is_heavy_rain"],
    "festival": ["is_festival_window", "days_to_festival", "is_national_holiday"],
    "season": ["is_monsoon", "dispatch_month"],
    "dispatch_time": ["dispatch_hour", "is_peak_hour", "is_night_dispatch"],
    "weekday": ["dispatch_weekday", "is_weekend"],
    "route_history": ["route_hist_delay_rate", "route_hist_overrun_ratio", "route_hist_trips"],
    "supplier": ["supplier_hist_delay_rate"],
    "terrain": ["origin_is_hilly", "dest_is_hilly"],
    "temperature": ["origin_temp_c", "dest_temp_c"],
    "wind": ["max_wind_kmph"],
    "vehicle": ["vehicle_type"], "cargo": ["material_type"],
    "regions": ["origin_region", "dest_region"],
}
SIZE_GROUPS = {"schedule", "route_length"}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def group_contributions(row: pd.Series) -> pd.Series:
    return pd.Series({g: float(sum(row.get(f, 0.0) for f in fs)) for g, fs in REASON_GROUPS.items()})


def describe_group(group: str, x: pd.Series, raises: bool) -> str:
    """Human-readable phrase for a reason group, using the shipment's values."""
    g = lambda f: x.get(f)  # noqa: E731
    if group == "route_length":
        km = float(g("road_distance_km"))
        return f"{'Long' if raises else 'Short'} route ({km:,.0f} km)"
    if group == "schedule":
        return (f"Tight planned schedule ({float(g('planned_hours')):.0f} hrs ≈ {float(g('planned_speed_kmph')):.0f} km/h)"
                if raises else f"Comfortable planned schedule ({float(g('planned_hours')):.0f} hrs)")
    if group == "rain":
        o, d = float(g("origin_rain_mm")), float(g("dest_rain_mm"))
        if max(o, d) < 1:
            return "Dry weather on dispatch day"
        if o >= 5 and d >= 5:
            return f"Rain at origin ({o:.0f} mm) and destination ({d:.0f} mm)"
        return describe("dest_rain_mm" if d >= o else "origin_rain_mm", max(o, d))
    if group == "festival":
        days = int(float(g("days_to_festival")))
        if float(g("is_festival_window")) >= 0.5:
            return ("Festival-day traffic" if days == 0 else
                    f"Festival-season traffic ({days} day{'s' if days > 1 else ''} from a major festival)")
        return "National holiday" if float(g("is_national_holiday")) >= 0.5 else "No festival rush"
    if group == "season":
        m = MONTHS[int(float(g("dispatch_month"))) - 1]
        return f"Monsoon season ({m})" if float(g("is_monsoon")) >= 0.5 else f"Dry season ({m})"
    if group == "dispatch_time":
        h = int(float(g("dispatch_hour")))
        tag = " – rush hour" if float(g("is_peak_hour")) >= 0.5 else " – night run" if float(g("is_night_dispatch")) >= 0.5 else ""
        return f"Dispatched at {h:02d}:00{tag}"
    if group == "weekday":
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return f"{days[int(float(g('dispatch_weekday')))]} dispatch"
    if group == "route_history":
        return f"Route's track record ({float(g('route_hist_delay_rate')):.0%} late, {int(float(g('route_hist_trips')))} past trips)"
    if group == "supplier":
        return describe("supplier_hist_delay_rate", g("supplier_hist_delay_rate"))
    if group == "terrain":
        hill = [s for s, f in (("origin", "origin_is_hilly"), ("destination", "dest_is_hilly")) if float(g(f)) >= 0.5]
        return f"Hilly terrain at {' and '.join(hill)}" if hill else "Plains route (no hill sections)"
    if group == "temperature":
        return f"Temperature {float(g('origin_temp_c')):.0f}°C → {float(g('dest_temp_c')):.0f}°C"
    if group == "wind":
        return describe("max_wind_kmph", g("max_wind_kmph"))
    if group == "vehicle":
        return f"Vehicle: {g('vehicle_type')}"
    if group == "cargo":
        return f"Cargo: {g('material_type')}"
    if group == "regions":
        return f"{g('origin_region')} → {g('dest_region')} corridor"
    return group


def top_reasons(eta_row: pd.Series, delay_row: pd.Series, raw_row: pd.Series, k: int = 3,
                prefer: str = "raise") -> list[dict]:
    """Return the top-k reason groups for one shipment.

    Groups are ranked by their contribution to delay risk. ``prefer="raise"``
    lists risk-increasing factors first; ``prefer="lower"`` lists risk-reducing
    ones first. Each reason also carries its ETA contribution in hours.
    """
    d, e = group_contributions(delay_row), group_contributions(eta_row)
    raising = d[d > 0].sort_values(ascending=False)
    lowering = d[d <= 0].sort_values()
    first, second = (raising, lowering) if prefer == "raise" else (lowering, raising)
    chosen = list(first.index[:k]) + list(second.index[: max(0, k - len(first))])
    out = []
    for grp in chosen:
        hrs = float(e[grp]); up = d[grp] > 0
        text = describe_group(grp, raw_row, up)
        # Size-related groups are relative to an average-length trip, so their hours
        # are left out of the text (still available in eta_impact_hours).
        if grp not in SIZE_GROUPS:
            text += f" ({hrs:+.1f} hrs)"
        out.append({"factor": grp, "text": text,
                    "eta_impact_hours": round(hrs, 2),
                    "risk_direction": "raises risk" if up else "lowers risk",
                    "risk_impact": round(float(d[grp]), 3)})
    return out


# ---------------------------------------------------------------------------
# Plots (matplotlib only, no dependency on the shap package)
# ---------------------------------------------------------------------------
def plot_importance(contrib: pd.DataFrame, path, title, unit):
    imp = contrib.abs().mean().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([LABELS.get(f, f) for f in imp.index], imp.values, color="#2b6cb0")
    ax.set(xlabel=f"Mean |SHAP value| ({unit})", title=title)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return imp.sort_values(ascending=False)


def plot_beeswarm(contrib: pd.DataFrame, X_raw: pd.DataFrame, path, title, unit, n=12, max_points=3000):
    """Beeswarm-style summary: one point per shipment, coloured by feature value."""
    feats = contrib.abs().mean().sort_values(ascending=False).head(n).index[::-1]
    sample = contrib.sample(min(max_points, len(contrib)), random_state=config.RANDOM_SEED).index
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, f in enumerate(feats):
        vals = pd.to_numeric(X_raw.loc[sample, f], errors="coerce")
        if vals.isna().all():                            # categorical feature
            colors = "#a0aec0"
        else:
            lo, hi = np.nanpercentile(vals, [5, 95])
            colors = plt.cm.coolwarm(np.clip((vals - lo) / max(hi - lo, 1e-9), 0, 1))
        ax.scatter(contrib.loc[sample, f], i + rng.normal(0, 0.12, len(sample)), s=5, c=colors, alpha=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(feats))); ax.set_yticklabels([LABELS.get(f, f) for f in feats])
    ax.set(xlabel=f"SHAP value ({unit})  ← lowers | raises →", title=title)
    sm = plt.cm.ScalarMappable(cmap="coolwarm"); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(["low", "high"]); cb.set_label("Feature value")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_waterfall(contrib_row: pd.Series, base: float, raw_row: pd.Series, path, title, unit, k=8):
    """Waterfall chart of grouped contributions for a single shipment."""
    grouped = group_contributions(contrib_row)
    top = grouped.reindex(grouped.abs().sort_values(ascending=False).index)
    shown, rest = top.head(k), top.iloc[k:].sum()
    items = [(describe_group(g, raw_row, v > 0), v) for g, v in shown.items()] + [("All other factors", rest)]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(items) + 1.8))
    running = base
    for i, (lab, v) in enumerate(items):
        ax.barh(i, v, left=running, color="#e53e3e" if v > 0 else "#3182ce")
        ax.text(running + v, i, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
        running += v
    ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items]); ax.invert_yaxis()
    ax.axvline(base, color="grey", ls="--", lw=1)
    ax.set(xlabel=unit, title=f"{title}\nbase {base:.2f} → prediction {running:.2f}")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
