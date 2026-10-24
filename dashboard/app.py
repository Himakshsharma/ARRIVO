"""
Arrivo operations dashboard (Streamlit).

Run:  streamlit run dashboard/app.py
It calls the FastAPI service at ARRIVO_API_URL (default http://localhost:8000);
if the API isn't running it works in embedded mode (same code, in-process).
"""
from __future__ import annotations

import io
import sys
from datetime import date, datetime, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # `streamlit run dashboard/app.py` only adds dashboard/ to the path
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import pydeck as pdk  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.client import ApiError, ArrivoClient  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
SAMPLE_BATCH = ROOT / "data" / "processed" / "sample_batch.csv"
RISK_COLORS = {"Low": "#2f855a", "Medium": "#d69e2e", "High": "#c53030"}

st.set_page_config(page_title="Arrivo – ETA & Delay Risk", page_icon="🚚", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  .risk-pill {display:inline-block;padding:2px 12px;border-radius:999px;color:white;font-weight:600}
  .reason {padding:8px 12px;border-radius:8px;margin-bottom:6px;background:rgba(127,127,127,0.08)}
  .muted {opacity:0.7;font-size:0.9rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_client() -> ArrivoClient:
    return ArrivoClient()


@st.cache_data(ttl=600)
def get_meta() -> dict:
    return get_client().meta()


@st.cache_data(ttl=600)
def get_metrics() -> dict:
    return get_client().metrics()


def risk_pill(level: str) -> str:
    return f'<span class="risk-pill" style="background:{RISK_COLORS[level]}">{level} risk</span>'


def gauge(prob: float, bands: dict) -> go.Figure:
    steps = [{"range": [lo * 100, min(hi, 1) * 100], "color": RISK_COLORS[name] + "33"}
             for name, (lo, hi) in bands.items()]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Probability of late arrival"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2d3748"}, "steps": steps}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def route_map(rows: list[dict], height: int = 380) -> pdk.Deck:
    """Arc from origin to destination per shipment, coloured by risk."""
    def rgb(level):
        h = RISK_COLORS[level].lstrip("#")
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    data = [{"from_lon": r["route"]["origin"][1], "from_lat": r["route"]["origin"][0],
             "to_lon": r["route"]["destination"][1], "to_lat": r["route"]["destination"][0],
             "color": rgb(r["risk_level"]), "label": f"{r['origin_city']} → {r['destination_city']}",
             "risk": f"{r['delay_probability']:.0%}"} for r in rows]
    points = ([{"lon": d["from_lon"], "lat": d["from_lat"], "c": [45, 55, 72]} for d in data]
              + [{"lon": d["to_lon"], "lat": d["to_lat"], "c": d["color"]} for d in data])
    layers = [
        pdk.Layer("ArcLayer", data, get_source_position=["from_lon", "from_lat"],
                  get_target_position=["to_lon", "to_lat"], get_source_color=[45, 55, 72],
                  get_target_color="color", get_width=3, pickable=True),
        pdk.Layer("ScatterplotLayer", points, get_position=["lon", "lat"], get_fill_color="c",
                  get_radius=25000, pickable=False),
    ]
    view = pdk.ViewState(latitude=22.5, longitude=80.0, zoom=3.6, pitch=30)
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto",
                    map_style=pdk.map_styles.CARTO_LIGHT, height=height,
                    tooltip={"text": "{label}\nDelay risk: {risk}"})


# ---------------------------------------------------------------------------
# Page 1 – single shipment
# ---------------------------------------------------------------------------
def page_predict():
    st.header("Check a shipment")
    meta = get_meta()
    cities = list(meta["cities"])
    with st.form("shipment"):
        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("Origin", cities, index=cities.index("Mumbai") if "Mumbai" in cities else 0)
        dest = c2.selectbox("Destination", cities, index=cities.index("Kolkata") if "Kolkata" in cities else 1)
        sid = c3.text_input("Shipment ID", "ARV-DEMO-1")
        c4, c5, c6 = st.columns(3)
        d = c4.date_input("Dispatch date", date.today())
        t = c5.time_input("Dispatch time", time(18, 30))
        vehicle = c6.selectbox("Vehicle", [v for v in meta["vehicle_types"] if v != "Unknown"])
        c7, c8, c9 = st.columns(3)
        cargo = c7.selectbox("Cargo", [m for m in meta["material_types"] if m != "Unknown"])
        supplier = c8.selectbox("Supplier", [s for s in meta["suppliers"] if s != "UNKNOWN"])
        planned = c9.number_input("Planner's ETA (hrs, 0 = estimate)", 0.0, 400.0, 0.0, 1.0)
        with st.expander("Override weather (what-if analysis)"):
            w1, w2, w3 = st.columns(3)
            use_w = w1.checkbox("Use these values instead of the weather lookup")
            o_rain = w2.slider("Rain at origin (mm)", 0, 200, 0)
            d_rain = w3.slider("Rain at destination (mm)", 0, 200, 0)
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        payload = {"shipment_id": sid, "origin_city": origin, "destination_city": dest,
                   "dispatch_time": datetime.combine(d, t).isoformat(), "vehicle_type": vehicle,
                   "material_type": cargo, "supplier_id": supplier}
        if planned > 0:
            payload["planned_hours"] = planned
        if use_w:
            payload.update(origin_rain_mm=o_rain, dest_rain_mm=d_rain)
        try:
            st.session_state["pred"] = get_client().predict(payload)
            st.session_state["payload"] = payload
        except ApiError as e:
            st.error(f"Could not predict: {e}")
            return

    pred = st.session_state.get("pred")
    if not pred:
        st.info("Fill in the shipment and press **Predict**.")
        return

    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown(f"### {pred['origin_city']} → {pred['destination_city']} &nbsp; {risk_pill(pred['risk_level'])}",
                    unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted transit", f"{pred['eta_hours']:.1f} hrs",
                  f"{pred['expected_delay_hours']:+.1f} hrs vs plan", delta_color="inverse")
        m2.metric("Expected arrival", pd.Timestamp(pred["expected_arrival"]).strftime("%d %b, %H:%M"))
        m3.metric("Planned arrival", pd.Timestamp(pred["planned_arrival"]).strftime("%d %b, %H:%M"))
        st.markdown("#### Top reasons")
        for r in pred["top_reasons"]:
            arrow = "▲" if r["risk_direction"] == "raises risk" else "▼"
            color = "#c53030" if arrow == "▲" else "#2f855a"
            st.markdown(f'<div class="reason"><b style="color:{color}">{arrow} {r["risk_direction"]}</b> · '
                        f'{r["text"]}</div>', unsafe_allow_html=True)
        ctx = pred["context"]
        st.markdown(f'<p class="muted">Route history: {ctx["route_hist_delay_rate"]:.0%} late over '
                    f'{ctx["route_hist_trips"]} past trips · weather source: {ctx["weather_source"]} · '
                    f'distance {pred["route"]["road_distance_km"]:,.0f} km</p>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(gauge(pred["delay_probability"], meta["risk_bands"]), width="stretch")
        st.pydeck_chart(route_map([pred], height=300))

    with st.expander("Draft a customer message (bonus: AI delay alert)"):
        a1, a2 = st.columns(2)
        name = a1.text_input("Customer name", "Priya")
        tone = a2.selectbox("Tone", ["friendly and professional", "formal", "brief and direct"])
        if st.button("Generate message"):
            try:
                out = get_client().alert(st.session_state["payload"], name, tone)
                st.success(out["message"])
                st.caption(f"Generated with: {out['mode']}")
            except ApiError as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Page 2 – operations view (batch)
# ---------------------------------------------------------------------------
def page_operations():
    st.header("Operations view")
    st.caption("Upload today's dispatch list; Arrivo ranks every shipment by delay risk.")
    up = st.file_uploader("CSV with origin_city, destination_city, dispatch_time (+ optional vehicle_type, "
                          "material_type, supplier_id, planned_hours, shipment_id)", type="csv")
    c1, c2 = st.columns([1, 3])
    use_sample = c1.button("Use sample batch") if SAMPLE_BATCH.exists() else False
    if SAMPLE_BATCH.exists():
        c2.download_button("Download sample CSV", SAMPLE_BATCH.read_bytes(), "sample_batch.csv", "text/csv")

    content = up.getvalue() if up is not None else (SAMPLE_BATCH.read_bytes() if use_sample else None)
    if content is not None:
        with st.spinner("Scoring shipments..."):
            try:
                st.session_state["batch"] = get_client().predict_batch(content)
            except ApiError as e:
                st.error(str(e))
                return
    res = st.session_state.get("batch")
    if not res:
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Shipments scored", res["count"])
    k2.metric("High risk", res["high_risk"]); k3.metric("Medium risk", res["medium_risk"])
    k4.metric("Low risk", res["low_risk"])
    if res["errors"]:
        with st.expander(f"{len(res['errors'])} rows could not be scored"):
            st.dataframe(pd.DataFrame(res["errors"]), width="stretch", hide_index=True)

    rows = res["results"]
    levels = st.multiselect("Show risk levels", ["High", "Medium", "Low"], default=["High", "Medium"])
    shown = [r for r in rows if r["risk_level"] in levels]
    table = pd.DataFrame([{
        "Shipment": r["shipment_id"], "Route": f"{r['origin_city']} → {r['destination_city']}",
        "Risk": r["risk_level"], "Delay prob.": r["delay_probability"],
        "ETA (hrs)": r["eta_hours"], "Late by (hrs)": max(r["expected_delay_hours"], 0),
        "Expected arrival": r["expected_arrival"].replace("T", " "),
        "Main reason": r["top_reasons"][0]["text"] if r["top_reasons"] else ""} for r in shown])
    if table.empty:
        st.info("No shipments at the selected risk levels.")
        return
    st.dataframe(table, width="stretch", hide_index=True, column_config={
        "Delay prob.": st.column_config.ProgressColumn("Delay prob.", min_value=0, max_value=1, format="percent")})
    buf = io.StringIO(); table.to_csv(buf, index=False)
    st.download_button("Download at-risk list", buf.getvalue(), "arrivo_at_risk.csv", "text/csv")
    st.pydeck_chart(route_map(shown[:200], height=420))


# ---------------------------------------------------------------------------
# Page 3 – model insights
# ---------------------------------------------------------------------------
def page_insights():
    st.header("Model insights")
    m = get_metrics()
    eta, dly = m["eta_model"], m["delay_model"]
    st.caption(f"Data: {m['data_source']} · trained {m['trained_at']} · delay = {m['delay_definition']}")
    c = st.columns(6)
    c[0].metric("ETA MAE", f"{eta['test_metrics']['MAE']:.2f} hrs")
    c[1].metric("ETA R²", f"{eta['test_metrics']['R2']:.3f}")
    c[2].metric("Within ±10%", f"{eta['test_metrics']['within_10pct_%']:.0f}%")
    c[3].metric("Delay ROC-AUC", f"{dly['test_metrics']['ROC_AUC']:.3f}")
    c[4].metric("Delay F1", f"{dly['test_metrics']['F1']:.3f}")
    c[5].metric("Recall (late caught)", f"{dly['test_metrics']['Recall']:.0%}")
    st.markdown(f"**Live models:** ETA = `{eta['name']}` · Delay = `{dly['name']}` "
                f"(threshold {dly['threshold']:.2f}, {dly['calibration']})")

    tab1, tab2, tab3, tab4 = st.tabs(["Model comparison", "Explainability (SHAP)", "Test performance", "EDA"])
    with tab1:
        cols_e = ["model", "stage", "val_MAE", "val_RMSE", "val_R2", "test_MAE", "test_RMSE", "test_R2"]
        cols_d = ["model", "stage", "val_ROC_AUC", "val_F1", "test_ROC_AUC", "test_F1", "test_Precision", "test_Recall"]
        st.subheader("ETA regression"); st.dataframe(pd.DataFrame(m["eta_model_comparison"])[cols_e].round(3),
                                                     width="stretch", hide_index=True)
        st.subheader("Delay classification"); st.dataframe(pd.DataFrame(m["delay_model_comparison"])[cols_d].round(3),
                                                           width="stretch", hide_index=True)
        if "risk_band_calibration" in m:
            st.subheader("Are the risk bands honest? (test set)")
            st.dataframe(pd.DataFrame(m["risk_band_calibration"]).round(3), width="stretch", hide_index=True)
    with tab2:
        for a, b in (("shap_eta_importance.png", "shap_delay_importance.png"),
                     ("shap_eta_summary.png", "shap_delay_summary.png"),
                     ("shap_eta_waterfall_example.png", "shap_delay_waterfall_example.png")):
            l, r = st.columns(2)
            for col, f in ((l, a), (r, b)):
                if (FIG_DIR / f).exists():
                    col.image(str(FIG_DIR / f))
    with tab3:
        for f in ("eta_test_performance.png", "delay_test_performance.png"):
            if (FIG_DIR / f).exists():
                st.image(str(FIG_DIR / f))
    with tab4:
        figs = sorted(FIG_DIR.glob("eda_*.png"))
        for i in range(0, len(figs), 2):
            cols = st.columns(2)
            for col, f in zip(cols, figs[i:i + 2]):
                col.image(str(f))


def page_about():
    st.header("About Arrivo")
    st.markdown("""
**Arrivo** predicts, for every truck shipment at dispatch time:
1. **ETA** – transit hours and expected arrival (regression)
2. **Delay risk** – calibrated probability of arriving later than planned (classification)
3. **Top reasons** – the factors behind each prediction, from SHAP values

Inputs used: route and distance, dispatch hour/day, vehicle and cargo type, weather at origin and destination
(Open-Meteo), Indian festivals and holidays, monsoon season, and each route's and supplier's track record
(learned from training data only). Nothing that is only known after delivery is ever used.
""")
    h = get_client().health()
    st.json({**h, "client_mode": get_client().mode})


PAGES = {"Check a shipment": page_predict, "Operations view": page_operations,
         "Model insights": page_insights, "About": page_about}

with st.sidebar:
    st.title("🚚 Arrivo")
    st.caption("Shipment ETA & delay-risk engine")
    page = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    try:
        client = get_client()
        st.caption(f"Backend: {'API' if client.mode == 'api' else 'embedded'} · "
                   f"{client.health().get('status', '?')}")
    except Exception as e:  # noqa: BLE001
        st.error(f"Backend unavailable: {e}")
        st.stop()

PAGES[page]()
