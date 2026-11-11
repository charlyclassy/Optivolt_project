"""Public LSTM/grid-capacity landing page shown before sign-in."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from dashboard import charts, database, services
from dashboard import theme as ui

PUBLIC_CSS = """
<style>
/* The public navigation sidebar remains visible. */
/* High contrast on the dark hero: protect custom HTML from generic theme rules. */
.st-key-lp_hero .lp-lead,
.st-key-lp_hero .lp-trust,
.st-key-lp_hero [data-testid="stMarkdownContainer"] .lp-lead,
.st-key-lp_hero [data-testid="stMarkdownContainer"] .lp-trust {
  color: #f1f5ff !important;
  opacity: 1 !important;
  text-shadow: 0 1px 1px rgba(0,0,0,.16);
}
.st-key-lp_hero .lp-lead { font-size: 18px !important; line-height: 1.72 !important; }
/* Clear, visible Sign in label and clear hover/focus states. */
.st-key-lp_hero [data-testid="stButton"] button[kind="secondary"],
.st-key-lp_hero .stButton button[kind="secondary"] {
  background: rgba(255,255,255,.13) !important;
  border: 1.5px solid rgba(255,255,255,.72) !important;
  color: #ffffff !important;
  font-weight: 750 !important;
  opacity: 1 !important;
}
.st-key-lp_hero [data-testid="stButton"] button[kind="secondary"] *,
.st-key-lp_hero .stButton button[kind="secondary"] * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  opacity: 1 !important;
}
.st-key-lp_hero [data-testid="stButton"] button[kind="secondary"]:hover {
  background: rgba(255,255,255,.24) !important;
  border-color: #ffffff !important;
}
.st-key-lp_hero [data-testid="stButton"] button[kind="primary"],
.st-key-lp_hero .stButton button[kind="primary"] {
  background: #ffffff !important;
  color: #172554 !important;
  font-weight: 750 !important;
}
.st-key-lp_hero [data-testid="stButton"] button[kind="primary"] *,
.st-key-lp_hero .stButton button[kind="primary"] * {
  color: #172554 !important;
  -webkit-text-fill-color: #172554 !important;
}

.block-container { max-width: 1240px; padding-top: 1.4rem; }
.lp-brand { display:flex; align-items:center; gap:10px; font-size:20px; font-weight:700; color:#0f172a; letter-spacing:-0.02em; }
.st-key-lp_hero { background: radial-gradient(1200px 500px at 85% -10%, rgba(34,211,238,.28), transparent 60%),
  linear-gradient(135deg, #0b1020 0%, #1e1b4b 55%, #3730a3 100%); border-radius:28px; padding:54px 48px 48px 48px;
  margin-top:14px; box-shadow:0 30px 60px -30px rgba(30,27,75,.6); }
.lp-eyebrow { display:inline-flex; align-items:center; gap:8px; font-size:12.5px; font-weight:600; color:#c7d2fe;
  background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.16); border-radius:999px; padding:6px 12px; }
.lp-eyebrow i { width:7px; height:7px; border-radius:50%; background:#34d399; display:inline-block; }
.lp-h1 { font-size:50px; line-height:1.06; font-weight:800; color:#fff; letter-spacing:-0.04em; margin:18px 0 0 0; }
.lp-h1 span { background:linear-gradient(90deg,#a5b4fc,#67e8f9); -webkit-background-clip:text; background-clip:text; color:transparent; }
.lp-lead { font-size:17px; line-height:1.6; color:#cbd5e1; margin:18px 0 22px 0; max-width:520px; }
.lp-trust { font-size:12.5px; color:#94a3b8; margin-top:14px; }
.st-key-lp_hero .stButton button[kind="secondary"] { background:rgba(255,255,255,.08); color:#fff; border:1px solid rgba(255,255,255,.22); }
.st-key-lp_hero .stButton button[kind="primary"] { background:#fff; color:#312e81; box-shadow:0 12px 24px -12px rgba(255,255,255,.55); }
.st-key-lp_hero div[data-testid="stPlotlyChart"] { border:0; border-radius:18px; box-shadow:0 30px 60px -20px rgba(2,6,23,.7); padding:14px 12px 6px 12px; }
.lp-preview-head { display:flex; justify-content:space-between; align-items:center; color:#e2e8f0; font-size:13px; font-weight:600; margin-bottom:10px; }
.lp-preview-head span.live { font-size:11.5px; font-weight:600; color:#a7f3d0; background:rgba(16,185,129,.16); border-radius:999px; padding:3px 10px; }
.lp-stats { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:14px; margin:26px 0 8px 0; }
.lp-stat { background:#fff; border:1px solid #e6eaf2; border-radius:18px; padding:20px 22px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.lp-stat b { display:block; font-size:32px; font-weight:800; color:#0f172a; letter-spacing:-0.04em; line-height:1.1; }
.lp-stat span { font-size:13.5px; color:#475569; }
.lp-title { text-align:center; margin:54px 0 6px 0; font-size:32px; font-weight:800; letter-spacing:-0.03em; color:#0f172a; }
.lp-sub { text-align:center; font-size:16px; color:#475569; margin-bottom:26px; }
.lp-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; }
.lp-feature { background:#fff; border:1px solid #e6eaf2; border-radius:18px; padding:24px; box-shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -12px rgba(15,23,42,.10); }
.lp-feature .ic { width:42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#4338ca; background:#eef2ff; margin-bottom:14px; }
.lp-feature h3 { font-size:17px; font-weight:700; margin:0 0 6px 0; padding:0; color:#0f172a; letter-spacing:-0.01em; }
.lp-feature p { font-size:14px; line-height:1.6; color:#475569; margin:0; }
.lp-steps { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; }
.lp-step { border-radius:18px; padding:24px; background:linear-gradient(180deg,#eef2ff,#ffffff); border:1px solid #e0e7ff; }
.lp-step .n { width:32px; height:32px; border-radius:50%; background:#4f46e5; color:#fff; font-weight:700; display:flex; align-items:center; justify-content:center; font-size:14px; margin-bottom:12px; }
.lp-step h3 { font-size:16px; font-weight:700; margin:0 0 6px 0; padding:0; color:#0f172a; }
.lp-step p { font-size:14px; line-height:1.6; color:#475569; margin:0; }
.st-key-lp_cta { background:linear-gradient(135deg,#4f46e5,#0ea5e9); border-radius:24px; padding:36px 40px; margin-top:54px; }
.lp-cta-text { color:#fff; font-size:26px; font-weight:800; letter-spacing:-0.03em; line-height:1.2; }
.lp-cta-text small { display:block; font-size:15px; font-weight:500; color:#e0e7ff; margin-top:6px; letter-spacing:0; }
.st-key-lp_cta .stButton button { background:#fff; color:#312e81; border:0; }
.lp-foot { text-align:center; font-size:12.5px; color:#94a3b8; margin:34px 0 10px 0; line-height:1.7; }
.st-key-lp_hero [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 330px); }
.st-key-lp_hero [data-testid="stColumn"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 110px); }
.st-key-lp_cta [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width:min(100%, 220px); }
.lp-h1 { font-size:clamp(32px, 4.2vw, 50px) !important; }
.lp-title { font-size:clamp(24px, 3vw, 32px) !important; }
.lp-stats { grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)) !important; }
.lp-grid, .lp-steps { grid-template-columns:repeat(auto-fit, minmax(260px, 1fr)) !important; }
@media (max-width: 900px) { .lp-h1 { font-size:36px; } .lp-stats, .lp-grid, .lp-steps { grid-template-columns:1fr; } .st-key-lp_hero { padding:32px 22px; } }
</style>
"""

FEATURES = [
    ("trend", "LSTM-powered demand forecasts up to 72 hours ahead",
     "Saved Optivolt LSTM models generate 15-minute demand forecasts from historical depot readings."),
    ("alert", "Know your EV charging grid headroom",
     "Compare predicted charging demand with a configurable grid capacity limit and assess available grid headroom."),
    ("sliders", "Fleet growth scenarios",
     "Explore hypothetical fleet growth and capacity assumptions against the LSTM forecast."),
    ("layers", "Every depot in one view",
     "One status table for all sites, each with its own capacity and warning threshold."),
    ("upload", "Your own private workspace",
     "Each organisation has its own depots, users and history. Upload supported meter data for validation and Optivolt model training."),
    ("file", "Reports and audit trail",
     "Download Excel, PDF or CSV reports. Access reports and verified model evaluation where available."),
]
STEPS = [
    ("Create your workspace", "Sign up for your organisation, explore the sample depots, then upload your own meter data."),
    ("Understand your limit", "Set your depot grid capacity limit and compare it with historical LSTM predictions."),
    ("Plan smarter charging", "Explore charging schedules, growth scenarios and capacity risks with an accountable human review."),
]


def go(view: str) -> None:
    target = st.session_state.get("_public_nav_pages", {}).get(view)
    if target is not None:
        st.switch_page(target)
    else:
        st.session_state["public_view"] = view



def _model_facts() -> dict:
    """Verified summary of the saved held-out comparison (not live telemetry)."""
    return {"depots": "9", "interval": "15"}


def _load_heldout():
    """Read the evaluation data rather than generate synthetic predictions."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "heldout_predictions_all_models.csv"
    if not path.is_file():
        return None
    required = {"timestamp", "actual_kw", "baseline_kw", "lstm_kw"}
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if not required.issubset(frame.columns):
        return None
    for name in ("actual_kw", "baseline_kw", "lstm_kw"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame.dropna(subset=["timestamp", "actual_kw", "baseline_kw", "lstm_kw"]).sort_values("timestamp")


def _preview_chart() -> None:
    """Plot measured versus two forecast methods; never claim a new forecast is made."""
    import plotly.graph_objects as go
    data = _load_heldout()
    if data is None or data.empty:
        st.caption("The historical prediction comparison is not bundled with this deployment.")
        return
    day = pd.Timestamp("2022-07-07")
    block = data[data["timestamp"].dt.normalize() == day]
    if block.empty:
        block = data.iloc[:96]
    plot = go.Figure()
    plot.add_trace(go.Scatter(x=block.timestamp, y=block.actual_kw, name="Measured Demand", mode="lines", line=dict(color="#14213D", width=3.2)))
    plot.add_trace(go.Scatter(x=block.timestamp, y=block.baseline_kw, name="Previous-day Baseline", mode="lines", line=dict(color="#F59E0B", dash="dot", width=2.4)))
    plot.add_trace(go.Scatter(x=block.timestamp, y=block.lstm_kw, name="LSTM Historical Prediction", mode="lines", line=dict(color="#3B82F6", dash="dash", width=3)))
    plot.update_layout(
        height=370, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#334155", size=11),
        margin=dict(l=12, r=12, t=54, b=12),
        xaxis=dict(showgrid=False, title="Historical evaluation day", linecolor="#CBD5E1"),
        yaxis=dict(title="Demand (kW)", gridcolor="#E2E8F0", zerolinecolor="#CBD5E1"),
        legend=dict(orientation="h", y=1.16, x=0, font=dict(size=10, color="#1E293B")),
        hovermode="x unified",
    )
    st.markdown('<div class="lp-preview-head"><span>Measured vs predicted depot demand</span>'
                '<span class="live">Historical test data · 2022</span></div>', unsafe_allow_html=True)
    st.plotly_chart(plot, config={"displayModeBar":False}, key="landing_preview")
    st.caption("Saved held-out predictions; these are not current readings or a live model run.")


def _forecast_and_capacity_demo() -> None:
    """Interactive evidence-backed forecast, and illustrative capacity cap comparison."""
    import plotly.graph_objects as go
    data = _load_heldout()
    if data is None or data.empty:
        st.warning("Historical held-out prediction data is unavailable in this deployment.")
        return

    mae_base = float((data.actual_kw - data.baseline_kw).abs().mean())
    mae_lstm = float((data.actual_kw - data.lstm_kw).abs().mean())
    gain = 100.0 * (mae_base - mae_lstm) / mae_base if mae_base else 0.0
    c1,c2,c3 = st.columns(3)
    c1.metric("Previous-day forecast MAE", f"{mae_base:.2f} kW")
    c2.metric("LSTM forecast MAE", f"{mae_lstm:.2f} kW")
    c3.metric("Reduction in forecast MAE", f"{gain:.1f}%", help="Against the previous-day baseline on the bundled historical held-out dataset. This measures prediction accuracy, not energy savings.")

    days = data.timestamp.dt.strftime("%Y-%m-%d").unique().tolist()
    default = "2022-07-23" if "2022-07-23" in days else days[-1]
    day, limit = st.columns([1,1])
    selected = day.selectbox("Historical test day", days, index=days.index(default), key="lp_day")
    cap = limit.slider("Assumed Grid Capacity Limit (kW)", min_value=5, max_value=40, value=17, step=1, key="lp_capacity")
    block = data[data.timestamp.dt.strftime("%Y-%m-%d") == selected].copy()
    block["illustrative_cap_kw"] = block.lstm_kw.clip(upper=cap)
    block["above_limit_kw"] = (block.lstm_kw - cap).clip(lower=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=block.timestamp, y=block.actual_kw, mode="lines",name="Measured Demand",line=dict(color="#14213D",width=3)))
    fig.add_trace(go.Scatter(x=block.timestamp, y=block.baseline_kw,mode="lines",name="Previous-day Baseline",line=dict(color="#f59e0b",dash="dot",width=2)))
    fig.add_trace(go.Scatter(x=block.timestamp, y=block.lstm_kw,mode="lines",name="LSTM Historical Prediction",line=dict(color="#3B82F6",dash="dash",width=3)))
    fig.add_trace(go.Scatter(x=block.timestamp, y=block.illustrative_cap_kw,mode="lines",name="Capacity-Capped Profile",line=dict(color="#10b981",width=2,dash="dash")))
    fig.add_hline(y=cap, line_dash="dash", line_color="#ef4444", annotation_text=f"Grid Capacity Limit: {cap} kW")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             name="Grid Capacity Limit",
                             line=dict(color="#ef4444", width=2, dash="dash")))
    fig.update_layout(height=440,margin=dict(l=8,r=12,t=14,b=12),yaxis_title="Charging demand (kW)",
                      legend=dict(orientation="h",y=1.13),hovermode="x unified")
    st.plotly_chart(fig,config={"displayModeBar":False},key="lp_comparison")
    peak = float(block.lstm_kw.max())
    above_intervals = int((block.lstm_kw > cap).sum())
    unserved = float(block.above_limit_kw.sum()) * (config.SAMPLING_MINUTES / 60.0)
    a,b,c = st.columns(3)
    a.metric("Peak predicted demand", f"{peak:.1f} kW")
    b.metric("Intervals above limit", str(above_intervals),help="Each historical evaluation interval is 15 minutes.")
    c.metric("Energy requiring rescheduling", f"{unserved:.1f} kWh",help="Illustrative clipped load only; energy has not been shifted to another time.")
    st.info("**How to read this:** Blue shows saved LSTM predictions on historical test data; green shows a capacity-capped profile at the selected grid capacity limit. The capped profile is a scenario, not a verified optimised charging schedule: any excess charging energy would need to be rescheduled separately.")

def render() -> None:
    """Customer-facing public home. No model inference or test-data charts."""
    st.markdown(PUBLIC_CSS, unsafe_allow_html=True)
    st.markdown("""<style>
    .block-container {max-width:1160px!important; padding-top:1.1rem!important;}
    .lp-public-hero {background:linear-gradient(115deg,#10264c 0%,#174b72 58%,#078f9c 100%);
      border-radius:24px;padding:40px 42px 34px;box-shadow:0 14px 36px rgba(16,38,76,.16);}
    .lp-public-hero, .lp-public-hero * {color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;opacity:1!important;}
    .lp-public-hero .lp-kicker {color:#a7fff0!important;-webkit-text-fill-color:#a7fff0!important;
      font-weight:800;letter-spacing:.13em;font-size:12px;}
    .lp-public-hero h1 {font-size:clamp(29px,3.8vw,46px)!important;line-height:1.12!important;
      font-weight:850!important;margin:14px 0!important;max-width:800px;}
    .lp-public-hero p {font-size:17px!important;line-height:1.6!important;max-width:780px;margin:0!important;}
    .lp-public-hero .lp-note {font-size:13px!important;color:#e0f2fe!important;
      -webkit-text-fill-color:#e0f2fe!important;margin-top:16px!important;}
    .lp-section-title {color:#14294b!important;font-weight:800;font-size:22px;margin:28px 0 12px;}
    .lp-product-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;}
    .lp-product-card {border:1px solid #dce6f0;border-radius:17px;padding:20px;
      background:#ffffff;box-shadow:0 5px 20px rgba(20,41,75,.045);}
    .lp-product-card b {display:block;color:#14294b!important;font-size:17px;margin-bottom:8px;}
    .lp-product-card p {color:#40516b!important;font-size:14px;line-height:1.55;margin:0;}
    .lp-product-card .lp-icon {font-size:24px;margin-bottom:10px;}
    .lp-steps-strip {background:#eef6fa;border:1px solid #d4e8ee;border-radius:14px;
      padding:15px 18px;color:#17345c!important;font-size:14px;line-height:1.65;}
    .st-key-lp_public_actions [data-testid="stButton"] button {min-height:48px;font-weight:800!important;}
    .st-key-lp_public_actions [data-testid="stButton"] button[kind="primary"] {
      background:#514bf0!important;color:#ffffff!important;border:0!important;}
    .st-key-lp_public_actions [data-testid="stButton"] button[kind="primary"] * {
      color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;}
    @media(max-width:760px){.lp-product-grid{grid-template-columns:1fr;}
      .lp-public-hero{padding:26px 22px;}}
    </style>""", unsafe_allow_html=True)
    brand, _, register = st.columns([5, 1, 1.5], vertical_alignment="center")
    brand.markdown(f'<div class="lp-brand">{ui.LOGO_SVG.format(size=36)}{ui.esc(config.PRODUCT_NAME)}</div>',
                   unsafe_allow_html=True)
    register.button("Create account", key="lp_nav_signup", type="primary", on_click=go,
                    args=("signup",), width="stretch")
    st.markdown("""<section class="lp-public-hero">
      <div class="lp-kicker">EV DEPOT ENERGY INTELLIGENCE</div>
      <h1>Forecast depot demand. Plan charging capacity with confidence.</h1>
      <p>Use historical electricity readings and saved LSTM models to anticipate demand,
      understand available grid headroom, and assess the impact of fleet growth before making operational decisions.</p>
      <div class="lp-note">Your depot data · 15-minute demand forecasts · Practical capacity planning</div>
       </section>""", unsafe_allow_html=True)
    with st.container(key="lp_public_actions"):
        left, right, space = st.columns([1.3, 1, 2.2], vertical_alignment="center")
        left.button("Create your workspace", key="lp_hero_signup", type="primary", on_click=go,
                    args=("signup",), width="stretch")
        right.button("Sign in", key="lp_hero_signin", on_click=go,
                     args=("signin",), width="stretch")
    st.markdown("""<div class="lp-section-title">What Optivolt helps you do</div>
    <div class="lp-product-grid">
      <div class="lp-product-card"><div class="lp-icon">📈</div><b>Predict depot demand</b>
      <p>Generate time-stamped electricity demand forecasts with a saved depot LSTM model.</p></div>
      <div class="lp-product-card"><div class="lp-icon">⚡</div><b>Understand capacity risk</b>
      <p>Compare predicted peaks with configured grid capacity, warning thresholds and remaining headroom.</p></div>
      <div class="lp-product-card"><div class="lp-icon">🚚</div><b>Plan fleet growth</b>
      <p>Test charging and operational assumptions against the baseline demand forecast.</p></div>
    </div>
    <div class="lp-section-title">How it works</div>
    <div class="lp-steps-strip"><b>1. Set up your depot</b> &nbsp; → &nbsp;
    <b>2. Generate a demand forecast</b> &nbsp; → &nbsp;
    <b>3. Compare capacity and planning impacts</b> </div>""", unsafe_allow_html=True)
    st.caption("Historical UKPN Optimise Prime data supports the demonstration and model evaluation; it is not live customer telemetry.")
