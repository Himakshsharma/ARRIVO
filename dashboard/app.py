"""
Arrivo operations dashboard (Streamlit).

Run:  streamlit run dashboard/app.py
It calls the FastAPI service at ARRIVO_API_URL (default http://localhost:8000);
if the API isn't running it works in embedded mode (same code, in-process).

Visual identity: Indian national-highway signage. The prediction is shown as a
kilometre stone (white stone, yellow cap); a "road strip" compares planned and
predicted arrival. Palette: asphalt, milestone yellow, highway-sign green.
"""
from __future__ import annotations

import io
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # `streamlit run dashboard/app.py` only adds dashboard/ to the path
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.client import ApiError, ArrivoClient  # noqa: E402

try:
    from src.explain.shap_utils import LABELS as FEATURE_LABELS  # noqa: E402
except Exception:  # noqa: BLE001
    FEATURE_LABELS = {}

FIG_DIR = ROOT / "reports" / "figures"
SAMPLE_BATCH = ROOT / "data" / "processed" / "sample_batch.csv"

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
ASPHALT = "#2A2F36"
ASPHALT_SOFT = "#4A525C"
YELLOW = "#F2C230"
SIGN_GREEN = "#1D6B47"
PAPER = "#F5F6F3"
LINE = "#DADDD6"
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#D9861C", "High": "#C8372D"}
FONT = "Barlow, system-ui, sans-serif"
DISPLAY = "'Barlow Condensed', Barlow, system-ui, sans-serif"

st.set_page_config(page_title="Arrivo – ETA & Delay Risk", page_icon="🚚", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, select, textarea {{
  font-family: {FONT};
}}
.stApp {{ background: {PAPER}; }}
[data-testid="stMain"], section.main {{ overflow-x: hidden; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ font-family: {DISPLAY}; letter-spacing: 0.01em; color: {ASPHALT}; }}
h2 {{ font-weight: 700; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* ---------- top band ---------- */
.arv-band {{
  background: {ASPHALT}; color: #fff; border-radius: 14px; padding: 22px 28px 26px;
  position: relative; overflow: hidden; margin-bottom: 18px;
}}
.arv-band::after {{             /* lane marking */
  content: ""; position: absolute; left: 0; right: 0; bottom: 10px; height: 4px;
  background: repeating-linear-gradient(90deg, {YELLOW} 0 38px, transparent 38px 66px);
  opacity: .9;
}}
.arv-brand {{ font-family: {DISPLAY}; font-size: 2.6rem; font-weight: 700; line-height: 1; }}
.arv-brand span {{ color: {YELLOW}; }}
.arv-tag {{ font-size: 1.05rem; opacity: .85; margin-top: 6px; max-width: 60ch; }}
.arv-band-stats {{ display: flex; gap: 34px; margin-top: 16px; flex-wrap: wrap; }}
.arv-band-stats div b {{ font-family: {DISPLAY}; font-size: 1.7rem; font-weight: 600; display: block; color: {YELLOW}; }}
.arv-band-stats div {{ font-size: .9rem; opacity: .9; }}

/* ---------- kilometre stone ---------- */
.stone-wrap {{ display: flex; justify-content: center; padding: 6px 0 2px; }}
.stone {{
  width: 230px; background: #fff; border: 2px solid {ASPHALT};
  border-radius: 115px 115px 10px 10px; overflow: hidden; text-align: center;
  box-shadow: 0 10px 0 -4px {LINE};
}}
.stone .cap {{
  background: {YELLOW}; color: {ASPHALT}; padding: 34px 12px 12px;
  font-family: {DISPLAY}; font-weight: 700; font-size: 1.55rem; line-height: 1.05;
  border-bottom: 2px solid {ASPHALT};
}}
.stone .num {{ font-family: {DISPLAY}; font-size: 4.2rem; font-weight: 700; color: {ASPHALT}; line-height: 1; margin-top: 14px; }}
.stone .unit {{ font-size: 1rem; color: {ASPHALT_SOFT}; }}
.stone .from {{ font-size: .9rem; color: {ASPHALT_SOFT}; padding: 10px 0 16px; }}

/* ---------- road strip ---------- */
.road {{ position: relative; height: 54px; background: {ASPHALT}; border-radius: 8px; margin: 34px 4px 46px; }}
.road::before {{ content: ""; position: absolute; left: 14px; right: 14px; top: 25px; height: 4px;
  background: repeating-linear-gradient(90deg, #fff 0 20px, transparent 20px 36px); opacity: .55; }}
.road .late {{ position: absolute; top: 0; bottom: 0; background: rgba(200,55,45,.55); }}
.road .early {{ position: absolute; top: 0; bottom: 0; background: rgba(46,139,87,.55); }}
.road .dot {{ position: absolute; top: 19px; width: 16px; height: 16px; border-radius: 50%;
  border: 3px solid #fff; transform: translateX(-50%); }}
.road .lbl {{ position: absolute; white-space: nowrap; font-size: .82rem; color: {ASPHALT}; }}
.road .lbl.top {{ top: -24px; }}
.road .lbl.bottom {{ bottom: -26px; }}

/* ---------- reasons ---------- */
.reason {{ display: grid; grid-template-columns: 26px 1fr auto; gap: 10px; align-items: center;
  background: #fff; border: 1px solid {LINE}; border-left: 6px solid var(--c); border-radius: 8px;
  padding: 10px 14px; margin-bottom: 8px; }}
.reason .ic {{ font-family: {DISPLAY}; font-weight: 700; color: var(--c); font-size: 1.3rem; }}
.reason .tx {{ color: {ASPHALT}; }}
.reason .hr {{ font-family: {DISPLAY}; font-weight: 600; font-size: 1.15rem; color: var(--c); }}

.pill {{ display:inline-block; padding: 3px 14px; border-radius: 999px; color: #fff; font-weight: 600;
  font-size: .95rem; vertical-align: middle; }}
.note {{ color: {ASPHALT_SOFT}; font-size: .9rem; }}

.tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 6px 0 14px; }}
.tile {{ flex: 1 1 150px; background: #fff; border: 1px solid {LINE}; border-left: 6px solid {SIGN_GREEN};
  border-radius: 10px; padding: 10px 14px; color: {ASPHALT_SOFT}; font-size: .92rem; }}
.tile b {{ display: block; font-family: {DISPLAY}; font-size: 1.9rem; font-weight: 600; color: {ASPHALT}; }}

/* ---------- native widgets ---------- */
div[data-testid="stMetric"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 16px; }}
div[data-testid="stMetricValue"] {{ font-family: {DISPLAY}; font-weight: 600; color: {ASPHALT}; }}
div[data-testid="stForm"] {{ background: #fff; border: 1px solid {LINE}; border-radius: 12px; }}
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button {{
  border-radius: 8px; font-weight: 600; }}
div[data-testid="stFormSubmitButton"] > button {{ background: {SIGN_GREEN}; color: #fff; border: 0; padding: .5rem 1.6rem; }}
div[data-testid="stFormSubmitButton"] > button:hover {{ background: #175a3b; color: #fff; }}
button:focus-visible {{ outline: 3px solid {YELLOW} !important; outline-offset: 2px; }}
div[data-testid="stTabs"] button p {{ font-family: {DISPLAY}; font-size: 1.1rem; font-weight: 600; }}
@media (max-width: 640px) {{
  .arv-brand {{ font-size: 2rem; }}
  .stone {{ width: 200px; }}
  .stone .num {{ font-size: 3.4rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the prediction engine…")
def get_client() -> ArrivoClient:
    client = ArrivoClient()
    client.health()          # loads (or, if needed, retrains) the models once
    return client


@st.cache_data(ttl=600)
def get_meta() -> dict:
    return get_client().meta()


@st.cache_data(ttl=600)
def get_metrics() -> dict:
    return get_client().metrics()


def style_fig(fig: go.Figure, height: int = 320, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fff", font=dict(family=FONT, color=ASPHALT, size=13),
        title_font=dict(family=DISPLAY, size=19, color=ASPHALT), showlegend=legend,
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="#fff", font_family=FONT),
    )
    fig.update_xaxes(gridcolor="#ECEEE9", zeroline=False)
    fig.update_yaxes(gridcolor="#ECEEE9", zeroline=False)
    return fig


def fmt_dt(iso: str) -> str:
    return pd.Timestamp(iso).strftime("%a %d %b, %H:%M")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def band(metrics: dict | None):
    stats = ""
    if metrics:
        e, d = metrics["eta_model"]["test_metrics"], metrics["delay_model"]["test_metrics"]
        stats = (f'<div class="arv-band-stats">'
                 f'<div><b>{e["MAE"]:.1f} hrs</b>average ETA error</div>'
                 f'<div><b>{d["ROC_AUC"]:.2f}</b>delay-risk ROC-AUC</div>'
                 f'<div><b>{d["Recall"]:.0%}</b>of late trucks caught</div></div>')
    st.markdown(f'<div class="arv-band"><div class="arv-brand">Arrivo<span>.</span></div>'
                f'<div class="arv-tag">Know which trucks will be late before they leave the yard: '
                f'predicted arrival, delay risk and the reasons behind it.</div>{stats}</div>',
                unsafe_allow_html=True)


def stone(pred: dict):
    st.markdown(f"""
<div class="stone-wrap"><div class="stone">
  <div class="cap">{pred['destination_city']}</div>
  <div class="num">{pred['eta_hours']:.0f}</div>
  <div class="unit">hours on the road</div>
  <div class="from">from {pred['origin_city']}, {pred['route']['road_distance_km']:,.0f} km</div>
</div></div>""", unsafe_allow_html=True)


def road_strip(pred: dict):
    """Dispatch → planned arrival → predicted arrival, to scale."""
    planned, eta = pred["planned_hours"], pred["eta_hours"]
    scale = max(planned, eta) * 1.08
    p, q = planned / scale * 100, eta / scale * 100
    gap_cls = "late" if eta > planned else "early"
    lo, hi = min(p, q), max(p, q)
    risk_c = RISK_COLORS[pred["risk_level"]]
    st.markdown(f"""
<div class="road">
  <div class="{gap_cls}" style="left:{lo}%;width:{hi - lo}%"></div>
  <div class="dot" style="left:0.8%;background:{ASPHALT_SOFT}"></div>
  <div class="dot" style="left:{p}%;background:{YELLOW}"></div>
  <div class="dot" style="left:{q}%;background:{risk_c}"></div>
  <div class="lbl top" style="left:0%">Dispatched {pd.Timestamp(pred['dispatch_time']):%d %b %H:%M}</div>
  <div class="lbl top" style="left:{p}%;transform:translateX(-{p}%)">Planned {pd.Timestamp(pred['planned_arrival']):%d %b %H:%M}</div>
  <div class="lbl bottom" style="left:{q}%;transform:translateX(-{q}%)"><b>Predicted {pd.Timestamp(pred['expected_arrival']):%d %b %H:%M}</b></div>
</div>""", unsafe_allow_html=True)


def gauge(prob: float, bands: dict) -> go.Figure:
    steps = [{"range": [lo * 100, min(hi, 1) * 100], "color": RISK_COLORS[n] + "40"} for n, (lo, hi) in bands.items()]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100,
        number={"suffix": "%", "font": {"size": 36, "family": DISPLAY, "color": ASPHALT}},
        gauge={"axis": {"range": [0, 100], "tickcolor": ASPHALT_SOFT}, "bar": {"color": ASPHALT, "thickness": 0.28},
               "steps": steps, "borderwidth": 0}))
    style_fig(fig, height=210)
    fig.update_layout(margin=dict(l=30, r=30, t=10, b=0), title=None)
    return fig


def route_map(rows: list[dict], height: int = 420) -> go.Figure:
    """Interactive map of routes (no API key needed: CARTO tiles)."""
    fig = go.Figure()
    new_api = hasattr(go, "Scattermap")            # plotly >= 5.24 (MapLibre, no token needed)
    Scatter = go.Scattermap if new_api else go.Scattermapbox
    for r in rows:
        (olat, olon), (dlat, dlon) = r["route"]["origin"], r["route"]["destination"]
        c = RISK_COLORS[r["risk_level"]]
        label = f"{r['origin_city']} → {r['destination_city']}"
        fig.add_trace(Scatter(lat=[olat, dlat], lon=[olon, dlon], mode="lines",
                              line=dict(width=3, color=c), hoverinfo="skip", showlegend=False))
        fig.add_trace(Scatter(lat=[olat], lon=[olon], mode="markers", marker=dict(size=9, color=ASPHALT),
                              hovertemplate=f"{r['origin_city']} (origin)<extra></extra>", showlegend=False))
        fig.add_trace(Scatter(lat=[dlat], lon=[dlon], mode="markers", marker=dict(size=13, color=c),
                              hovertemplate=(f"<b>{label}</b><br>Risk {r['delay_probability']:.0%} ({r['risk_level']})"
                                             f"<br>ETA {r['eta_hours']:.1f} hrs<extra></extra>"), showlegend=False))
    lats = [x for r in rows for x in (r["route"]["origin"][0], r["route"]["destination"][0])] or [22.5]
    lons = [x for r in rows for x in (r["route"]["origin"][1], r["route"]["destination"][1])] or [80]
    center = dict(lat=(min(lats) + max(lats)) / 2, lon=(min(lons) + max(lons)) / 2)
    span = max(max(lats) - min(lats), max(lons) - min(lons), 4)
    zoom = 5.6 if span < 6 else 4.6 if span < 12 else 3.7
    layout_map = dict(style="carto-positron", center=center, zoom=zoom)
    if new_api:
        fig.update_layout(map=layout_map)
    else:
        fig.update_layout(mapbox=layout_map)
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def reasons_block(pred: dict):
    for r in pred["top_reasons"]:
        up = r["risk_direction"] == "raises risk"
        c = RISK_COLORS["High"] if up else RISK_COLORS["Low"]
        text = r["text"].rsplit(" (", 1)[0] if r["text"].endswith("hrs)") else r["text"]
        hrs = f'{r["eta_impact_hours"]:+.1f} h' if r["factor"] not in ("schedule", "route_length") else ""
        st.markdown(f'<div class="reason" style="--c:{c}"><span class="ic">{"▲" if up else "▼"}</span>'
                    f'<span class="tx">{text}<br><span class="note">{"Raises" if up else "Lowers"} delay risk</span></span>'
                    f'<span class="hr">{hrs}</span></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 1 – single shipment
# ---------------------------------------------------------------------------
PRESETS = {
    "Monsoon long haul": dict(origin="Mumbai", dest="Kolkata", d=date(2026, 7, 21), t=time(18, 30),
                              vehicle="32 FT Multi-Axle", cargo="Cement & Steel", rain=(45, 60)),
    "Diwali rush": dict(origin="Delhi", dest="Jaipur", d=date(2026, 11, 6), t=time(9, 0),
                        vehicle="20 FT Container", cargo="FMCG", rain=None),
    "Hill run in the rains": dict(origin="Delhi", dest="Shimla", d=date(2026, 8, 10), t=time(9, 0),
                                  vehicle="17 FT Container", cargo="FMCG", rain=(20, 80)),
    "Quiet night, dry season": dict(origin="Pune", dest="Hyderabad", d=date(2026, 2, 11), t=time(23, 0),
                                    vehicle="20 FT Container", cargo="Pharma", rain=(0, 0)),
}


def apply_preset(name: str):
    p = PRESETS[name]
    ss = st.session_state
    ss.f_origin, ss.f_dest, ss.f_date, ss.f_time = p["origin"], p["dest"], p["d"], p["t"]
    ss.f_vehicle, ss.f_cargo = p["vehicle"], p["cargo"]
    ss.f_use_w = p["rain"] is not None
    if p["rain"] is not None:
        ss.f_orain, ss.f_drain = p["rain"]
    ss.auto_predict = True


def whatif(payload: dict, pred: dict):
    """Sweep rain and dispatch hour around this shipment (one batch call)."""
    key = str(sorted(payload.items()))
    if st.session_state.get("whatif_key") != key:
        base_o = pred["context"]["origin_rain_mm"]
        base_d = pred["context"]["dest_rain_mm"]
        rains = [0, 10, 20, 35, 50, 75, 100, 130]
        hours = list(range(0, 24, 2))
        rows = []
        for r in rains:
            rows.append({**payload, "shipment_id": f"rain-{r}", "origin_rain_mm": base_o, "dest_rain_mm": r})
        t0 = pd.Timestamp(payload["dispatch_time"])
        for h in hours:
            rows.append({**payload, "shipment_id": f"hour-{h}", "dispatch_time": t0.replace(hour=h, minute=0).isoformat(),
                         "origin_rain_mm": base_o, "dest_rain_mm": base_d})
        csv = pd.DataFrame(rows).to_csv(index=False).encode()
        res = {r["shipment_id"]: r for r in get_client().predict_batch(csv)["results"]}
        st.session_state.whatif = {
            "rain": pd.DataFrame([{"x": r, "risk": res[f"rain-{r}"]["delay_probability"],
                                   "eta": res[f"rain-{r}"]["eta_hours"]} for r in rains]),
            "hour": pd.DataFrame([{"x": h, "risk": res[f"hour-{h}"]["delay_probability"],
                                   "eta": res[f"hour-{h}"]["eta_hours"]} for h in hours]),
            "now": (base_d, t0.hour)}
        st.session_state.whatif_key = key
    return st.session_state.whatif


def whatif_chart(df: pd.DataFrame, title: str, xlab: str, marker_x: float, bands: dict) -> go.Figure:
    fig = go.Figure()
    for n, (lo, hi) in bands.items():
        fig.add_hrect(y0=lo * 100, y1=min(hi, 1) * 100, fillcolor=RISK_COLORS[n], opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(x=df.x, y=df.risk * 100, mode="lines+markers", line=dict(color=ASPHALT, width=3, shape="spline"),
                             marker=dict(size=8, color=ASPHALT), customdata=df.eta,
                             hovertemplate=f"{xlab} %{{x}}<br>Risk %{{y:.0f}}%<br>ETA %{{customdata:.1f}} hrs<extra></extra>"))
    fig.add_vline(x=marker_x, line=dict(color=YELLOW, width=3, dash="dot"))
    fig.update_yaxes(range=[0, 100], ticksuffix="%", title="Delay risk")
    fig.update_xaxes(title=xlab)
    fig.update_layout(title=title)
    return style_fig(fig, 300)


def page_predict():
    meta = get_meta()
    cities = list(meta["cities"])
    vehicles = [v for v in meta["vehicle_types"] if v != "Unknown"]
    cargos = [m for m in meta["material_types"] if m != "Unknown"]
    suppliers = [s for s in meta["suppliers"] if s != "UNKNOWN"]
    ss = st.session_state
    ss.setdefault("f_origin", "Mumbai" if "Mumbai" in cities else cities[0])
    ss.setdefault("f_dest", "Kolkata" if "Kolkata" in cities else cities[1])
    ss.setdefault("f_date", date.today() + timedelta(days=1))
    ss.setdefault("f_time", time(18, 30))
    ss.setdefault("f_vehicle", vehicles[0]); ss.setdefault("f_cargo", cargos[0]); ss.setdefault("f_supplier", suppliers[0])
    ss.setdefault("f_use_w", False); ss.setdefault("f_orain", 0); ss.setdefault("f_drain", 0)

    st.markdown("#### Try an example")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch")

    with st.form("shipment"):
        c1, c2, c3, c4 = st.columns(4)
        origin = c1.selectbox("From", cities, key="f_origin")
        dest = c2.selectbox("To", cities, key="f_dest")
        d = c3.date_input("Dispatch date", key="f_date")
        t = c4.time_input("Dispatch time", key="f_time", step=1800)
        c5, c6, c7, c8 = st.columns(4)
        vehicle = c5.selectbox("Vehicle", vehicles, key="f_vehicle")
        cargo = c6.selectbox("Cargo", cargos, key="f_cargo")
        supplier = c7.selectbox("Transport supplier", suppliers, key="f_supplier")
        planned = c8.number_input("Planner's ETA in hours (0 = estimate it)", 0.0, 400.0, 0.0, 1.0)
        with st.expander("Set the weather yourself"):
            use_w = st.checkbox("Use these rain values instead of looking up the weather", key="f_use_w")
            w1, w2 = st.columns(2)
            o_rain = w1.slider("Rain at origin (mm)", 0, 200, key="f_orain")
            d_rain = w2.slider("Rain at destination (mm)", 0, 200, key="f_drain")
        submitted = st.form_submit_button("Predict arrival")

    if submitted or ss.pop("auto_predict", False):
        if origin == dest:
            st.error("Pick a destination different from the origin.")
            return
        payload = {"shipment_id": "ARV-DEMO", "origin_city": origin, "destination_city": dest,
                   "dispatch_time": datetime.combine(d, t).isoformat(), "vehicle_type": vehicle,
                   "material_type": cargo, "supplier_id": supplier}
        if planned > 0:
            payload["planned_hours"] = planned
        if use_w:
            payload.update(origin_rain_mm=o_rain, dest_rain_mm=d_rain)
        try:
            with st.spinner("Predicting…"):
                ss.pred, ss.payload = get_client().predict(payload), payload
        except ApiError as e:
            st.error(f"Couldn't predict this shipment: {e}")
            return

    pred = ss.get("pred")
    if not pred:
        st.info("Choose an example above, or fill in a shipment and press **Predict arrival**.")
        return

    level = pred["risk_level"]
    late_by = pred["expected_delay_hours"]
    verdict = (f"likely to arrive about {late_by:.0f} hours late" if late_by >= 1 and level != "Low"
               else "on track to arrive on time" if level == "Low" else "at moderate risk of running late")
    st.markdown(f"## {pred['origin_city']} to {pred['destination_city']} "
                f"<span class='pill' style='background:{RISK_COLORS[level]}'>{level} risk</span>", unsafe_allow_html=True)
    st.markdown(f"<p class='note' style='font-size:1.05rem'>This shipment is {verdict}.</p>", unsafe_allow_html=True)

    left, mid, right = st.columns([1, 1.1, 1.1], gap="medium")
    with left:
        stone(pred)
    with mid:
        st.markdown("#### Chance of arriving late")
        st.plotly_chart(gauge(pred["delay_probability"], meta["risk_bands"]), width="stretch",
                        config={"displayModeBar": False})
    with right:
        st.metric("Predicted arrival", fmt_dt(pred["expected_arrival"]))
        st.metric("Planned arrival", fmt_dt(pred["planned_arrival"]),
                  f"{late_by:+.1f} hrs vs plan", delta_color="inverse")
    road_strip(pred)
    why, ctx_col = st.columns([1.7, 1], gap="large")
    with why:
        st.markdown("#### Why")
        reasons_block(pred)
    with ctx_col:
        ctx = pred["context"]
        fest = (f"{ctx['nearest_festival']} season traffic is expected. " if ctx.get("is_festival_window") else "")
        st.markdown("#### Conditions")
        st.markdown(f"<p class='note'>This route was late {ctx['route_hist_delay_rate']:.0%} of the time over "
                    f"{ctx['route_hist_trips']} past trips. Rain on dispatch day: {ctx['origin_rain_mm']:.0f} mm at "
                    f"origin and {ctx['dest_rain_mm']:.0f} mm at destination ({ctx['weather_source']}). "
                    f"{'Monsoon season. ' if ctx.get('is_monsoon') else ''}{fest}</p>", unsafe_allow_html=True)

    tab_map, tab_whatif, tab_alert = st.tabs(["Route map", "What if…", "Message the customer"])
    with tab_map:
        st.plotly_chart(route_map([pred], height=380), width="stretch", config={"scrollZoom": True})
    with tab_whatif:
        st.markdown("<p class='note'>The same shipment re-scored with different rain at the destination and different "
                    "dispatch hours. The dotted line marks the current choice.</p>", unsafe_allow_html=True)
        try:
            w = whatif(ss.payload, pred)
            a, b = st.columns(2)
            a.plotly_chart(whatif_chart(w["rain"], "More rain at the destination", "Rain (mm)", w["now"][0],
                                        meta["risk_bands"]), width="stretch", config={"displayModeBar": False})
            b.plotly_chart(whatif_chart(w["hour"], "A different dispatch hour", "Dispatch hour", w["now"][1],
                                        meta["risk_bands"]), width="stretch", config={"displayModeBar": False})
            best = w["hour"].loc[w["hour"].risk.idxmin()]
            if w["hour"].risk.max() - best.risk >= 0.05:
                st.markdown(f"<p class='note'>Lowest-risk dispatch slot for this route and day: <b>{int(best.x):02d}:00</b> "
                            f"({best.risk:.0%} risk).</p>", unsafe_allow_html=True)
        except ApiError as e:
            st.warning(f"What-if analysis isn't available right now: {e}")
    with tab_alert:
        a1, a2 = st.columns(2)
        name = a1.text_input("Customer name", "Priya")
        tone = a2.selectbox("Tone", ["friendly and professional", "formal", "brief and direct"])
        if st.button("Write the message"):
            try:
                out = get_client().alert(ss.payload, name, tone)
                st.success(out["message"])
                st.caption("Written by Claude" if out["mode"] == "llm" else "Written from a template (no LLM key configured)")
            except ApiError as e:
                st.error(str(e))


# ---------------------------------------------------------------------------
# Page 2 – operations view
# ---------------------------------------------------------------------------
def page_operations():
    st.markdown("## Today's dispatch board")
    st.markdown("<p class='note'>Upload the day's dispatch list and Arrivo ranks every truck by delay risk, "
                "so you know which ones to call about first.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.2, 1, 1])
    up = c1.file_uploader("Dispatch list (CSV with origin_city, destination_city, dispatch_time)", type="csv")
    use_sample = c2.button("Load sample day", type="primary", width="stretch") if SAMPLE_BATCH.exists() else False
    if SAMPLE_BATCH.exists():
        c3.download_button("Sample CSV", SAMPLE_BATCH.read_bytes(), "sample_batch.csv", "text/csv",
                           width="stretch")

    content = up.getvalue() if up is not None else (SAMPLE_BATCH.read_bytes() if use_sample else None)
    if content is not None:
        with st.spinner("Scoring every shipment…"):
            try:
                st.session_state.batch = get_client().predict_batch(content)
            except ApiError as e:
                st.error(f"Couldn't score this file: {e}")
                return
    res = st.session_state.get("batch")
    if not res:
        st.info("Load the sample day to see the board in action, or upload your own CSV.")
        return

    rows = res["results"]
    k = st.columns(4)
    k[0].metric("Shipments", res["count"])
    k[1].metric("High risk", res["high_risk"]); k[2].metric("Medium risk", res["medium_risk"])
    k[3].metric("Low risk", res["low_risk"])
    if res["errors"]:
        with st.expander(f"{len(res['errors'])} rows couldn't be scored"):
            st.dataframe(pd.DataFrame(res["errors"]), width="stretch", hide_index=True)

    df = pd.DataFrame([{
        "Shipment": r["shipment_id"], "Route": f"{r['origin_city']} → {r['destination_city']}",
        "Destination": r["destination_city"], "Risk": r["risk_level"], "Delay risk": r["delay_probability"],
        "ETA (hrs)": r["eta_hours"], "Late by (hrs)": max(r["expected_delay_hours"], 0.0),
        "Expected arrival": fmt_dt(r["expected_arrival"]),
        "Main reason": r["top_reasons"][0]["text"] if r["top_reasons"] else ""} for r in rows])

    a, b = st.columns([1, 1.4], gap="large")
    with a:
        counts = df.Risk.value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                               marker=dict(colors=[RISK_COLORS[x] for x in counts.index], line=dict(color="#fff", width=3)),
                               hovertemplate="%{label} risk: %{value} shipments<extra></extra>", textinfo="value"))
        fig.update_layout(title="Shipments by risk", annotations=[dict(text=f"<b>{len(df)}</b><br>trucks", showarrow=False,
                                                                       font=dict(family=DISPLAY, size=22))])
        st.plotly_chart(style_fig(fig, 300, legend=True), width="stretch", config={"displayModeBar": False})
    with b:
        top = df.sort_values("Delay risk", ascending=True).tail(12)
        fig = go.Figure(go.Bar(x=top["Delay risk"] * 100, y=top["Shipment"] + "  " + top["Route"], orientation="h",
                               marker_color=[RISK_COLORS[x] for x in top.Risk],
                               customdata=top[["ETA (hrs)", "Main reason"]],
                               hovertemplate="%{y}<br>Risk %{x:.0f}%<br>ETA %{customdata[0]:.1f} hrs<br>%{customdata[1]}<extra></extra>"))
        fig.update_xaxes(range=[0, 100], ticksuffix="%")
        fig.update_layout(title="The 12 riskiest shipments")
        st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Every shipment")
    f1, f2 = st.columns([1.4, 1])
    levels = f1.pills("Show", ["High", "Medium", "Low"], default=["High", "Medium"], selection_mode="multi")
    dests = f2.multiselect("Destination", sorted(df.Destination.unique()))
    shown = df[df.Risk.isin(levels or [])]
    if dests:
        shown = shown[shown.Destination.isin(dests)]
    if shown.empty:
        st.info("No shipments match these filters.")
    else:
        st.dataframe(shown.drop(columns="Destination"), width="stretch", hide_index=True, height=360,
                     column_config={"Delay risk": st.column_config.ProgressColumn("Delay risk", min_value=0, max_value=1,
                                                                                  format="percent"),
                                    "ETA (hrs)": st.column_config.NumberColumn(format="%.1f"),
                                    "Late by (hrs)": st.column_config.NumberColumn(format="%.1f")})
        buf = io.StringIO(); shown.to_csv(buf, index=False)
        st.download_button("Download this list", buf.getvalue(), "arrivo_at_risk.csv", "text/csv")
    ids = set(shown.Shipment)
    st.plotly_chart(route_map([r for r in rows if r["shipment_id"] in ids][:150], height=460), width="stretch",
                    config={"scrollZoom": True})


# ---------------------------------------------------------------------------
# Page 3 – model insights
# ---------------------------------------------------------------------------
def page_insights():
    m = get_metrics()
    eta, dly = m["eta_model"], m["delay_model"]
    st.markdown("## How good are the predictions?")
    st.markdown(f"<p class='note'>Measured on {m['split']['test'][2]:,} shipments from the most recent six months, "
                f"which the models never saw during training.</p>", unsafe_allow_html=True)
    tiles = [(f"{eta['test_metrics']['MAE']:.1f} hrs", "average ETA error"),
             (f"{eta['test_metrics']['within_10pct_%']:.0f}%", "of ETAs within ±10%"),
             (f"{dly['test_metrics']['ROC_AUC']:.3f}", "delay-risk ROC-AUC"),
             (f"{dly['test_metrics']['Recall']:.0%}", "of late trucks caught"),
             (f"{dly['test_metrics']['Precision']:.0%}", "of late flags were right")]
    st.markdown('<div class="tiles">' + "".join(f'<div class="tile"><b>{v}</b>{k}</div>' for v, k in tiles)
                + "</div>", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["Model comparison", "What drives delays", "Can you trust the risk levels?", "Data exploration"])
    with t1:
        e = pd.DataFrame(m["eta_model_comparison"]); d = pd.DataFrame(m["delay_model_comparison"])
        nice = lambda x: (x.replace(" (sklearn)", "").replace("GradientBoosting", "Gradient Boosting")  # noqa: E731
                          .replace("RandomForest", "Random Forest").replace("LinearRegression", "Linear Regression")
                          .replace("LogisticRegression", "Logistic Regression").replace(" + calibrated", ", calibrated")
                          .replace("Planner's ETA (no ML)", "Planner's ETA"))
        e["label"], d["label"] = e.model.map(nice), d.model.map(nice)
        a, b = st.container(), st.container()
        colors = [YELLOW if "Planner" in x else (SIGN_GREEN if x.startswith(eta["name"]) else "#9AA3AD") for x in e.model]
        fig = go.Figure(go.Bar(x=e.test_MAE, y=e.label, orientation="h", marker_color=colors,
                               text=[f"{v:.2f} h" for v in e.test_MAE], textposition="outside",
                               hovertemplate="%{y}<br>Test MAE %{x:.2f} hrs<extra></extra>"))
        fig.update_layout(title="ETA error by model (lower is better)")
        fig.update_xaxes(title="Mean absolute error, hours", range=[0, e.test_MAE.max() * 1.25])
        a.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})
        colors = [SIGN_GREEN if x.startswith(dly["name"]) else "#9AA3AD" for x in d.model]
        fig = go.Figure(go.Bar(x=d.test_ROC_AUC, y=d.label, orientation="h", marker_color=colors,
                               text=[f"{v:.3f}" for v in d.test_ROC_AUC], textposition="outside",
                               hovertemplate="%{y}<br>Test ROC-AUC %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Delay-risk ranking quality (higher is better)")
        fig.update_xaxes(title="ROC-AUC", range=[0.5, 1])
        b.plotly_chart(style_fig(fig, 280), width="stretch", config={"displayModeBar": False})
        st.markdown(f"<p class='note'>Yellow is the planner's own ETA with no machine learning. Green marks the "
                    f"models in use: <b>{eta['name']}</b> for ETA and <b>{dly['name']}</b> for delay risk "
                    f"(decision threshold {dly['threshold']:.2f}).</p>", unsafe_allow_html=True)
        with st.expander("Full comparison tables"):
            st.dataframe(e[["model", "stage", "val_MAE", "test_MAE", "test_RMSE", "test_R2"]].round(3), width="stretch", hide_index=True)
            st.dataframe(d[["model", "stage", "val_ROC_AUC", "test_ROC_AUC", "test_F1", "test_Precision", "test_Recall"]].round(3),
                         width="stretch", hide_index=True)
    with t2:
        a, b = st.columns(2, gap="large")
        for col, key, title, unit in ((a, "top_delay_drivers", "What moves delay risk", "mean |SHAP|"),
                                      (b, "top_eta_drivers", "What moves the ETA", "mean |SHAP|, hours")):
            s = pd.Series(m[key]).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=[FEATURE_LABELS.get(k, k) for k in s.index], orientation="h",
                                   marker_color=SIGN_GREEN, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(title=title); fig.update_xaxes(title=unit)
            col.plotly_chart(style_fig(fig, 380), width="stretch", config={"displayModeBar": False})
        img = [f for f in ("shap_delay_summary.png", "shap_delay_waterfall_example.png") if (FIG_DIR / f).exists()]
        if img:
            st.markdown("<p class='note'>Below, each dot is one shipment (red = high value of that factor), and one "
                        "high-risk shipment broken down factor by factor.</p>", unsafe_allow_html=True)
            cols = st.columns(len(img))
            for col, f in zip(cols, img):
                col.image(str(FIG_DIR / f))
    with t3:
        if "risk_band_calibration" in m:
            cal = pd.DataFrame(m["risk_band_calibration"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cal.band, y=cal.mean_predicted_risk * 100, name="Risk Arrivo predicted",
                                 marker_color="#B9C0C8"))
            fig.add_trace(go.Bar(x=cal.band, y=cal.actual_late_rate * 100, name="Share that actually arrived late",
                                 marker_color=[RISK_COLORS[b] for b in cal.band],
                                 text=[f"{v:.0%}" for v in cal.actual_late_rate], textposition="outside"))
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            fig.update_layout(title="Predicted risk vs what really happened", barmode="group")
            st.plotly_chart(style_fig(fig, 360, legend=True), width="stretch", config={"displayModeBar": False})
            st.markdown("<p class='note'>When Arrivo says a shipment is High risk, it really is late most of the time; "
                        "Low-risk shipments rarely are. That's what makes the levels safe to act on.</p>",
                        unsafe_allow_html=True)
        for f in ("delay_test_performance.png", "eta_test_performance.png"):
            if (FIG_DIR / f).exists():
                st.image(str(FIG_DIR / f))
    with t4:
        figs = sorted(FIG_DIR.glob("eda_*.png"))
        names = [f.stem.split("_", 2)[2].replace("_", " ").capitalize() for f in figs]
        if figs:
            pick = st.selectbox("Chart", names, index=min(3, len(names) - 1))
            st.image(str(figs[names.index(pick)]))


def page_about():
    st.markdown("## About Arrivo")
    st.markdown("""
Arrivo predicts three things for every truck at the moment it's dispatched:

1. **When it will arrive**: transit hours and the expected arrival time.
2. **How likely it is to be late**: a calibrated probability, shown as Low, Medium or High risk.
3. **Why**: the factors behind each prediction, worked out with SHAP and written in plain English.

It looks only at what's known at dispatch: the route and distance, the dispatch hour and day, the vehicle and cargo,
weather at both ends, Indian festivals and holidays, the monsoon, and each route's and supplier's past record.

Built with scikit-learn, SHAP, FastAPI and Streamlit. Source code:
[github.com/Himakshsharma/Arrivo](https://github.com/Himakshsharma/Arrivo)
""")
    with st.expander("System status"):
        st.json({**get_client().health(), "client_mode": get_client().mode})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PAGES = {"Check a shipment": page_predict, "Dispatch board": page_operations,
         "Model insights": page_insights, "About": page_about}

try:
    client = get_client()
    metrics = get_metrics()
except Exception as e:  # noqa: BLE001
    st.error(f"The prediction engine couldn't start: {e}")
    st.stop()

band(metrics)
page = st.segmented_control("Section", list(PAGES), default="Check a shipment", label_visibility="collapsed",
                            key="nav")
PAGES[page or "Check a shipment"]()
