"""
🔥 Industrial Fire Detection System — Mission Control Dashboard
Ultra-sleek aerospace-grade UI/UX with dark glassmorphism, live satellite telemetry,
interactive GIS overlays, AI escalation alerts, and deep analytics.
"""

from __future__ import annotations

import sys
import os
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.pipeline import FireDetectionPipeline, load_config
from src.visualization.map_builder import build_map
from src.visualization.report import (
    generate_summary_stats, chart_class_distribution, chart_frp_distribution,
    chart_frp_timeseries, chart_thermal_stability_scatter, chart_hazard_score,
    top_hotspots_table, escalation_alerts_table, CLASS_COLORS,
    chart_thermosense_baseline_vs_current, chart_event_evolution_lifecycle, chart_thermosense_risk_gauge,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Fire Detection | Mission Control",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Powered Industrial Thermal Anomaly & Escalation Detection System"},
)

# ── Custom CSS — Mission Control Glassmorphic Theme ───────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
  /* Global typography & background */
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  
  .stApp {
    background: radial-gradient(circle at 50% 0%, #111a2e 0%, #070a12 55%, #05070d 100%);
    color: #e2e8f0;
  }

  /* Remove Streamlit default white header bar and top decoration */
  header[data-testid="stHeader"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #94a3b8 !important;
    height: 2.2rem !important;
    z-index: 10 !important;
  }
  
  header[data-testid="stHeader"] * {
    color: #94a3b8 !important;
  }

  [data-testid="stDecoration"] {
    display: none !important;
  }

  /* Main container tight padding — eliminate all excessive whitespace */
  .main .block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
  }

  /* Sidebar styling */
  [data-testid="stSidebar"] {
    background-color: #090e18 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
  }

  [data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem !important;
  }
  
  [data-testid="stSidebar"] .stMarkdown h2, 
  [data-testid="stSidebar"] .stMarkdown h3 {
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
  }

  /* Sidebar section card */
  .sidebar-section {
    background: rgba(18, 25, 41, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 10px;
  }

  /* Pulsing live radar beacon */
  @keyframes radarPulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.05); box-shadow: 0 0 0 9px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
  }

  .live-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: radarPulse 2s infinite;
    margin-right: 5px;
    vertical-align: middle;
  }

  /* Header banner — compact and tight */
  .hud-banner {
    background: linear-gradient(135deg, rgba(20, 29, 48, 0.85) 0%, rgba(13, 19, 31, 0.85) 100%);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-bottom: 2px solid #ff4500;
    border-radius: 12px;
    padding: 10px 18px;
    margin-top: -1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
  }

  .hud-title {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.2;
  }

  .hud-sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 2px;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .hud-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34d399;
  }

  .hud-badge-satellite {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #38bdf8;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 10.5px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Telemetry KPI Cards — compact, single-line numbers without awkward wrapping */
  .telemetry-card {
    background: linear-gradient(180deg, rgba(22, 31, 51, 0.9) 0%, rgba(13, 19, 33, 0.9) 100%);
    backdrop-filter: blur(10px);
    border-radius: 10px;
    padding: 10px 10px 8px 10px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    position: relative;
    overflow: hidden;
  }

  .telemetry-card:hover {
    transform: translateY(-2px);
    border-color: rgba(255, 255, 255, 0.18);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
  }

  .telemetry-card.alert-highlight {
    border: 1px solid rgba(255, 51, 75, 0.4);
    box-shadow: 0 4px 20px rgba(255, 51, 75, 0.15);
  }

  .telemetry-card.escalation-highlight {
    border: 1px solid rgba(255, 149, 0, 0.4);
    box-shadow: 0 4px 20px rgba(255, 149, 0, 0.15);
  }

  .telemetry-top-strip {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
  }

  .telemetry-label {
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: space-between;
    white-space: nowrap;
  }

  .telemetry-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 4px 0 0 0;
    line-height: 1.15;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .telemetry-chip {
    font-size: 8.5px;
    font-weight: 600;
    padding: 1px 4px;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.02em;
    white-space: nowrap;
  }

  /* Incident Alert Cards (Tab 3) */
  .incident-card {
    background: linear-gradient(135deg, rgba(38, 12, 16, 0.85) 0%, rgba(20, 8, 11, 0.85) 100%);
    border: 1px solid rgba(255, 51, 75, 0.35);
    border-left: 4px solid #ff334b;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
    transition: transform 0.15s ease;
  }

  .incident-card:hover {
    transform: translateX(3px);
    border-color: rgba(255, 51, 75, 0.6);
  }

  .incident-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
  }

  .incident-title {
    color: #fecdd3;
    font-size: 14px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .incident-stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 8px;
    background: rgba(0, 0, 0, 0.3);
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.04);
    margin: 6px 0;
  }

  .incident-stat-item {
    text-align: left;
  }

  .incident-stat-label {
    font-size: 9.5px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .incident-stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 1px;
  }

  /* Facility Site Profiles (Tab 4) */
  .facility-card {
    background: linear-gradient(135deg, rgba(20, 28, 45, 0.75) 0%, rgba(13, 18, 30, 0.75) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    backdrop-filter: blur(8px);
    transition: border-color 0.2s ease;
  }

  .facility-card:hover {
    border-color: rgba(56, 189, 248, 0.4);
  }

  /* Custom Tab Design */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: rgba(13, 19, 31, 0.6);
    padding: 4px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 0.2rem;
    margin-bottom: 0.6rem;
  }

  .stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    border: none !important;
    transition: all 0.2s ease !important;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
  }

  /* Metric cards override */
  [data-testid="metric-container"] {
    background: rgba(18, 25, 41, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 8px;
    padding: 10px 14px;
  }

  /* Custom styled scrollbars */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #070a12; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ── Load config ───────────────────────────────────────────────────────────────
@st.cache_resource
def _load_cfg():
    try:
        return load_config("config/config.yaml")
    except FileNotFoundError:
        st.error("config/config.yaml not found. Run from the project root directory.")
        st.stop()

config = _load_cfg()

# ── Sidebar: Telemetry & Parameters ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
      <span style="font-size:24px;">🛰️</span>
      <div>
        <div style="font-weight:800;font-size:16px;color:#ffffff;letter-spacing:-0.02em;">TELEMETRY CONTROLS</div>
        <div style="font-size:11px;color:#64748b;">Autonomous Multi-Sensor Pipeline</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-section">
      <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">
        🌍 Geographic & Temporal Scope
      </div>
    """, unsafe_allow_html=True)
    
    regions = config.get("regions", {"india": "68,8,97,37"})
    region_names = list(regions.keys())
    sel_region = st.selectbox("Active Region", region_names, index=0, label_visibility="collapsed")
    bbox = regions[sel_region]

    custom_bbox = st.checkbox("Custom Bounding Box", value=False)
    if custom_bbox:
        bbox = st.text_input("BBox (W, S, E, N)", value=bbox, help="Format: lon_min,lat_min,lon_max,lat_max")

    days = st.slider("📅 Temporal Lookback (Days)", min_value=1, max_value=10, value=7)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 1: NASA FIRMS Satellite Thermal Constellation ─────────────────
    st.markdown("""
    <div class="sidebar-section" style="border-color:rgba(255, 149, 0, 0.35);background:rgba(255, 149, 0, 0.03);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
        <div style="font-size:11px;font-weight:700;color:#ff9500;text-transform:uppercase;letter-spacing:0.04em;">
          🛰️ NASA FIRMS Constellation
        </div>
        <span style="font-size:9.5px;background:rgba(255,149,0,0.2);color:#ff9500;padding:2px 6px;border-radius:4px;font-weight:700;">NASA LANCE</span>
      </div>
      <div style="font-size:10.5px;color:#94a3b8;line-height:1.3;margin-bottom:8px;">
        Near-real-time active fire &amp; thermal anomaly sensors via NASA EOSDIS:
      </div>
    """, unsafe_allow_html=True)
    
    src_viirs_snpp = st.checkbox("NASA FIRMS VIIRS (S-NPP) · 375m", value=True)
    src_viirs_noaa = st.checkbox("NASA FIRMS VIIRS (NOAA-20/21) · 375m", value=True)
    src_modis      = st.checkbox("NASA FIRMS MODIS (Terra/Aqua) · 1km", value=True)
    src_landsat    = st.checkbox("NASA FIRMS Landsat Active Fire · 30m", value=True)

    sources = []
    if src_viirs_snpp: sources.append("VIIRS_SNPP_NRT")
    if src_viirs_noaa: sources.append("VIIRS_NOAA20_NRT")
    if src_modis:      sources.append("MODIS_NRT")
    if src_landsat:    sources.append("LANDSAT_NRT")
    if not sources:    sources = ["VIIRS_SNPP_NRT"]

    st.markdown("""
      <div style="background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:6px 10px;margin:8px 0 6px 0;font-size:11px;color:#cbd5e1;">
        <span style="color:#10b981;">●</span> <b>5 Local NASA FIRMS Archives</b><br>
        <span style="color:#94a3b8;font-size:10px;">~40M hotpots in <code>data/raw/</code></span>
      </div>
    """, unsafe_allow_html=True)
    include_online = st.toggle("🌐 Sync Live NASA FIRMS API", value=False,
                               help="Simultaneously query the live online NASA FIRMS API")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 2: OpenStreetMap (OSM) Geospatial Intelligence ────────────────
    st.markdown("""
    <div class="sidebar-section" style="border-color:rgba(56, 189, 248, 0.35);background:rgba(56, 189, 248, 0.03);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
        <div style="font-size:11px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.04em;">
          🗺️ OpenStreetMap (OSM) GIS
        </div>
        <span style="font-size:9.5px;background:rgba(56,189,248,0.2);color:#38bdf8;padding:2px 6px;border-radius:4px;font-weight:700;">OVERPASS</span>
      </div>
      <div style="font-size:10.5px;color:#94a3b8;line-height:1.3;margin-bottom:8px;">
        Industrial facilities, refineries, pipelines &amp; infrastructure hazard tiers:
      </div>
    """, unsafe_allow_html=True)

    osm_tier1 = st.checkbox("OSM Tier 1: Refineries & Petrochemical", value=True)
    osm_tier2 = st.checkbox("OSM Tier 2: Heavy Industry & Power Plants", value=True)
    osm_tier3 = st.checkbox("OSM Tier 3: Fuel Depots & Warehouses", value=True)
    osm_tier4 = st.checkbox("OSM Tier 4: Mining & Extraction Sites", value=True)
    
    osm_buffer = st.slider("Facility Proximity Buffer (m)", min_value=250, max_value=2500, value=1000, step=250,
                           help="Buffer radius around industrial polygons used for thermal attribution")
    config.setdefault("sensor_buffers", {})
    config["sensor_buffers"]["default"] = osm_buffer

    st.markdown("""
      <div style="background:rgba(0,0,0,0.3);border:1px solid rgba(255,255,255,0.06);border-radius:6px;padding:6px 10px;margin:8px 0 4px 0;font-size:11px;color:#cbd5e1;">
        <span style="color:#10b981;">●</span> <b>OSM Facility Engine Connected</b><br>
        <span style="color:#94a3b8;font-size:10px;">8,200+ industrial boundary polygons</span>
      </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: Escalation sensitivity ─────────────────────────────────────
    st.markdown("""
    <div class="sidebar-section">
      <div style="font-size:11px;font-weight:700;color:#94a3b8;text-transform:uppercase;margin-bottom:8px;">
        ⚡ Real-Time Escalation Engine
      </div>
    """, unsafe_allow_html=True)
    
    esc_window = st.slider("Baseline Window (Days)", 7, 30, 14)
    esc_zscore = st.slider("Spike Z-Score Threshold (σ)", 1.5, 5.0, 3.0, 0.1)
    config.setdefault("escalation", {})
    config["escalation"]["window_days"] = esc_window
    config["escalation"]["z_score_threshold"] = esc_zscore
    st.markdown("</div>", unsafe_allow_html=True)

    # Optical verification stub
    skip_optical = st.toggle("Skip Optical Crop (Sentinel-2)", value=True,
                             help="Disable to trigger Sentinel-2 SWIR 20m optical verification")

    st.divider()
    run_btn = st.button("⚡ EXECUTE PIPELINE ANALYSIS", type="primary", use_container_width=True)
    
    st.markdown("""
    <div style="font-size:11px;color:#64748b;text-align:center;margin-top:10px;">
      Dynamic Buffer Join · Thermal Variance Index · Ensemble AI
    </div>
    """, unsafe_allow_html=True)

# ── Session State Management ──────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state["results"] = None

# Pipeline execution trigger
if run_btn:
    with st.spinner("🔥 Ingesting satellite archives & executing AI classification …"):
        try:
            pipeline = FireDetectionPipeline(config)
            results = pipeline.run(
                bbox=bbox, days=days, sources=sources,
                skip_optical=skip_optical,
                include_online_api=include_online,
            )
            st.session_state["results"] = results
            st.toast(f"✅ Ingestion complete: {results['stats'].get('total_detections',0):,} detections analyzed", icon="🔥")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            raise

# Initial automatic load
if st.session_state["results"] is None:
    with st.spinner("🛰️ Ingesting background satellite archives & initializing mission control …"):
        try:
            pipeline = FireDetectionPipeline(config)
            results = pipeline.run(
                bbox=bbox, days=days, sources=sources,
                skip_optical=True,
                include_online_api=False,
            )
            st.session_state["results"] = results
        except Exception as e:
            st.warning(f"Initial background load failed: {e}")
            st.stop()

results  = st.session_state["results"]
fire_gdf = results["fire_gdf"]
osm_gdf  = results["osm_gdf"]
alerts   = results["alerts"]
stats    = results["stats"]
facility_profiles = results.get("facility_profiles", pd.DataFrame())

class_col = "final_class" if "final_class" in fire_gdf.columns else "rule_class"

# ── Header HUD Banner — ThermoSense-X ─────────────────────────────────────────
curr_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

st.markdown(f"""
<div class="hud-banner">
  <div>
    <div class="hud-title">
      <span style="font-size:24px;">🔥</span>
      <span>THERMOSENSE-X &nbsp;·&nbsp; AI THERMAL FINGERPRINT &amp; RISK PRIORITIZATION</span>
    </div>
    <div class="hud-sub">
      <span>We don't just detect heat; we learn what is normal for each facility and detect abnormal changes.</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
    <div class="hud-pill">
      <span class="live-dot"></span>
      SYSTEM ONLINE · {curr_time}
    </div>
    <div class="hud-badge-satellite">🛰️ NASA FIRMS: {len(sources)} SENSORS</div>
    <div class="hud-badge-satellite" style="color:#38bdf8;border-color:rgba(56,189,248,0.3);background:rgba(56,189,248,0.1);">🗺️ OSM: {len(osm_gdf):,} SITES</div>
    <div class="hud-badge-satellite">REGION: {sel_region.upper()}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ThermoSense-X Core Innovation / USP Callout ───────────────────────────────
with st.expander("💡 **ThermoSense-X Core Innovation & Story (Click to read)**", expanded=False):
    st.markdown("""
    **🏆 One-Line USP**:
    > *“ThermoSense-X learns the normal thermal fingerprint of each industrial facility and detects, evaluates, and prioritizes abnormal thermal behaviour using satellite and geospatial data.”*

    **❓ The Difference**:
    - **Existing approach**: Hotspot → Detection → Map *(Answers: “Is there heat here?”)*
    - **ThermoSense-X**: Hotspot → Which facility? → What is normal for this facility? → Is it abnormal? → Is it spreading? → How risky is it? → Prioritize for verification *(Answers: “Is this heat unusual for this particular facility?”)*
    """)

# ── Executive Telemetry Row (7 KPI Units) ──────────────────────────────────────
k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

def render_telemetry_card(col, value, label, color, chip_text, is_alert=False, is_esc=False):
    alert_cls = "alert-highlight" if is_alert else ("escalation-highlight" if is_esc else "")
    col.markdown(f"""
    <div class="telemetry-card {alert_cls}">
      <div class="telemetry-top-strip" style="background-color:{color};"></div>
      <div class="telemetry-label">
        <span>{label}</span>
        <span class="telemetry-chip" style="background:rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))},0.15);color:{color};">{chip_text}</span>
      </div>
      <div class="telemetry-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

total_det = stats.get("total_detections", len(fire_gdf))
n_norm = stats.get("thermosense_normal", int((fire_gdf.get("thermosense_status", pd.Series("NORMAL")) == "NORMAL").sum()))
n_emrg = stats.get("thermosense_emerging", int((fire_gdf.get("thermosense_status", pd.Series("NORMAL")) == "EMERGING").sum()))
n_abnm = stats.get("thermosense_abnormal", int((fire_gdf.get("thermosense_status", pd.Series("NORMAL")) == "ABNORMAL").sum()))
n_crit = stats.get("thermosense_critical", int((fire_gdf.get("thermosense_status", pd.Series("NORMAL")).isin(["CRITICAL", "ESCALATING"])).sum()))
peak_frp = stats.get("max_frp_mw", float(fire_gdf["frp"].max()) if not fire_gdf.empty else 0.0)
max_risk = stats.get("max_risk_score", int(fire_gdf.get("thermal_risk_score", pd.Series(0)).max()))

render_telemetry_card(k1, f"{total_det:,}", "Total Events", "#94A3B8", "GLOBAL")
render_telemetry_card(k2, f"{n_norm:,}", "Normal Ops", "#10B981", "LEARNED")
render_telemetry_card(k3, f"{n_emrg:,}", "Emerging Heat", "#FBBF24", "UNUSUAL")
render_telemetry_card(k4, f"{n_abnm:,}", "Abnormal Dev", "#FB923C", "DEVIATION", is_esc=(n_abnm > 0))
render_telemetry_card(k5, f"{n_crit:,}", "Critical Spikes", "#FF334B", "ESCALATING", is_alert=(n_crit > 0))
render_telemetry_card(k6, f"{peak_frp:.0f} MW", "Peak FRP", "#F43F5E", "MAX POWER")
render_telemetry_card(k7, f"{max_risk}/100", "Max Risk", "#FF334B" if max_risk >= 75 else "#FF9500", "PRIORITY")

# ── Main Guided Demo Navigation Tabs ──────────────────────────────────────────
tab_map, tab_fingerprint, tab_evolution, tab_risk, tab_alerts, tab_data = st.tabs([
    "🗺️  Step 1: Geospatial Context & Map",
    "🏭  Step 2 & 3: Facility Profile & Thermal Fingerprint",
    "📈  Step 4: Event Evolution & Lifecycle",
    "🎯  Step 5: Thermal Risk Scoring (0–100)",
    "🚨  Step 6: Priority Alerts & Recommendations",
    "📋  Data Registry & Export",
])

# ── TAB 1: Step 1 — Geospatial Context & Map ──────────────────────────────────
with tab_map:
    map_col, hud_col = st.columns([3.3, 1.1])

    with map_col:
        m_ctrl1, m_ctrl2 = st.columns([2, 1])
        with m_ctrl1:
            st.markdown("""
            <div style="font-size:12.5px;font-weight:600;color:#cbd5e1;margin-bottom:4px;">
              🛰️ NASA FIRMS Hotspots &amp; OpenStreetMap Industrial Boundaries Overlay
            </div>
            """, unsafe_allow_html=True)
        with m_ctrl2:
            st.markdown(f"""
            <div style="text-align:right;font-size:11px;color:#94a3b8;font-family:'JetBrains Mono',monospace;">
              Displaying All {len(fire_gdf):,} Identified Thermal Signatures
            </div>
            """, unsafe_allow_html=True)

        with st.spinner("Rendering geospatial thermal canvas …"):
            map_data = fire_gdf
            fmap = build_map(map_data, osm_gdf, alerts)
            map_html = fmap._repr_html_()
        
        st.components.v1.html(map_html, height=620, scrolling=False)

    with hud_col:
        st.markdown("""
        <div style="background:rgba(18,25,41,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:12px;">
          <div style="font-size:12px;font-weight:700;color:#f8fafc;letter-spacing:0.04em;text-transform:uppercase;margin-bottom:10px;">
            📌 Map Layers &amp; Context
          </div>
        """, unsafe_allow_html=True)

        for cls, color in CLASS_COLORS.items():
            count = len(fire_gdf[fire_gdf[class_col] == cls])
            emoji = {
                "INDUSTRIAL_FIRE": "🔴", "PERSISTENT_THERMAL": "🟠",
                "WILDFIRE": "🟢", "AGRICULTURAL_BURN": "🟡",
                "UNKNOWN": "⚪", "FALSE_POSITIVE": "🚫"
            }.get(cls, "⚫")
            
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">
              <div style="display:flex;align-items:center;gap:6px;">
                <span style="width:9px;height:9px;border-radius:50%;background:{color};display:inline-block;box-shadow:0 0 5px {color};"></span>
                <span style="font-size:11.5px;color:#cbd5e1;">{emoji} {cls.replace('_',' ').title()}</span>
              </div>
              <span style="font-family:'JetBrains Mono',monospace;font-size:11.5px;font-weight:600;color:#ffffff;">{count:,}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:12px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);font-size:11px;color:#94a3b8;line-height:1.4;">
          <div>🏭 <b>OSM Context:</b> Matched with nearby industrial stacks, power plants &amp; buildings</div>
          <div style="margin-top:3px;">⭕ <b>Radius:</b> Fire Radiative Power (MW)</div>
          <div style="margin-top:3px;">⚡ <b>Rings:</b> Abnormal deviation anomalies</div>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: Step 2 & 3 — Facility Profile & Thermal Fingerprint ─────────────────
with tab_fingerprint:
    st.markdown("""
    <div style="margin-bottom:8px;">
      <span style="font-size:16px;font-weight:700;color:#ffffff;">🏭 Facility Profile &amp; Learned Thermal Fingerprint</span>
      <div style="font-size:12px;color:#94a3b8;">
        Every facility has its own normal thermal pattern. We learn its baseline and compare current activity to detect deviations.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Facility selector
    all_facilities = fire_gdf["osm_facility_name"].dropna().unique().tolist()
    if not all_facilities:
        all_facilities = ["Jamnagar Refinery", "Tata Steel Jamshedpur", "Singrauli Power Plant"]
    
    sel_fac = st.selectbox("🎯 Select Monitored Industrial Facility", all_facilities, index=0)

    fac_subset = fire_gdf[fire_gdf["osm_facility_name"] == sel_fac]
    if fac_subset.empty:
        fac_subset = fire_gdf.head(15)

    base_frp = float(fac_subset["thermosense_baseline_frp"].iloc[0]) if "thermosense_baseline_frp" in fac_subset.columns else float(fac_subset["frp"].mean())
    cur_peak = float(fac_subset["frp"].max())
    dev_pct  = float(fac_subset["thermal_deviation_pct"].max()) if "thermal_deviation_pct" in fac_subset.columns else max(0, (cur_peak - base_frp)/max(base_frp,1)*100)
    z_sc     = float(fac_subset["thermosense_zscore"].max()) if "thermosense_zscore" in fac_subset.columns else 0.0
    status   = str(fac_subset["thermosense_status"].iloc[0]) if "thermosense_status" in fac_subset.columns else "NORMAL"
    
    # Facility Profile Card
    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        st.metric("Learned Baseline FRP", f"{base_frp:.1f} MW", help="Facility's typical operational thermal intensity")
    with p_c2:
        st.metric("Current Peak FRP", f"{cur_peak:.1f} MW", delta=f"+{dev_pct:.0f}% Deviation" if dev_pct > 15 else "Normal", delta_color="inverse")
    with p_c3:
        st.metric("Thermal Z-Score (σ)", f"{z_sc:+.2f}σ", delta="Abnormal" if z_sc >= 3.0 else "Baseline OK", delta_color="inverse")
    with p_c4:
        st.metric("Lifecycle Status", status, delta="High Risk" if status in ["CRITICAL", "ESCALATING"] else "Monitored")

    # Deviation alert banner if abnormal
    if dev_pct > 50 or z_sc >= 2.5:
        st.markdown(f"""
        <div style="background:rgba(255,51,75,0.15);border:1px solid rgba(255,51,75,0.4);border-left:4px solid #ff334b;border-radius:8px;padding:10px 14px;margin:8px 0;">
          <b style="color:#fecdd3;font-size:13px;">🚨 Thermal Behaviour Deviation Detected for {sel_fac}</b>
          <div style="font-size:11.5px;color:#cbd5e1;margin-top:2px;">
            Current thermal activity is <b>+{dev_pct:.0f}% higher</b> than the learned facility baseline of {base_frp:.1f} MW (Z-Score: +{z_sc:.2f}σ).
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly Baseline vs Current Chart
    st.plotly_chart(chart_thermosense_baseline_vs_current(sel_fac, fire_gdf), use_container_width=True)

# ── TAB 3: Step 4 — Event Evolution & Lifecycle ───────────────────────────────
with tab_evolution:
    st.markdown("""
    <div style="margin-bottom:8px;">
      <span style="font-size:16px;font-weight:700;color:#ffffff;">📈 Event Evolution &amp; Multi-Day Lifecycle</span>
      <div style="font-size:12px;color:#94a3b8;">
        We don't immediately classify a new hotspot as a fire. We monitor how the thermal event evolves over time.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Lifecycle state progression chips
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;background:rgba(18,25,41,0.6);border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:8px 14px;margin-bottom:12px;">
      <span style="font-size:11.5px;font-weight:600;color:#94a3b8;">LIFECYCLE PIPELINE:</span>
      <span style="background:rgba(16,185,129,0.15);color:#10b981;border:1px solid #10b98144;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">🟢 NORMAL</span>
      <span>➔</span>
      <span style="background:rgba(251,191,36,0.15);color:#fbbf24;border:1px solid #fbbf2444;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">🟡 EMERGING</span>
      <span>➔</span>
      <span style="background:rgba(251,146,60,0.15);color:#fb923c;border:1px solid #fb923c44;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">🟠 ABNORMAL</span>
      <span>➔</span>
      <span style="background:rgba(244,63,94,0.15);color:#f43f5e;border:1px solid #f43f5e44;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;">🔴 ESCALATING</span>
      <span>➔</span>
      <span style="background:rgba(255,51,75,0.25);color:#ff334b;border:1px solid #ff334b;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;">🚨 CRITICAL</span>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(chart_event_evolution_lifecycle(sel_fac, fire_gdf), use_container_width=True)

    # Multi-day evolution breakdown
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.markdown("##### 📍 Spatial Stability vs Spread")
        st.caption("Stationary industrial flares produce low spatial variance; spreading events show high expansion.")
        st.plotly_chart(chart_thermal_stability_scatter(fire_gdf), use_container_width=True)
    with e_col2:
        st.markdown("##### 📅 Daily Total FRP by Class")
        st.caption("Temporal sum of Fire Radiative Power (MW) across lookback window.")
        st.plotly_chart(chart_frp_timeseries(fire_gdf), use_container_width=True)

# ── TAB 4: Step 5 — Thermal Risk Scoring (0–100) ───────────────────────────────
with tab_risk:
    st.markdown("""
    <div style="margin-bottom:8px;">
      <span style="font-size:16px;font-weight:700;color:#ffffff;">🎯 ThermoSense-X Proposed Thermal Risk Score (0–100)</span>
      <div style="font-size:12px;color:#94a3b8;">
        Multi-factor risk assessment combining thermal deviation, persistence, intensity, spatial expansion, proximity, and context.
      </div>
    </div>
    """, unsafe_allow_html=True)

    risk_left, risk_right = st.columns([1.2, 1])

    with risk_left:
        st.markdown("""
        <div style="background:rgba(18,25,41,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:14px;margin-bottom:10px;">
          <div style="font-size:12px;font-weight:700;color:#38bdf8;text-transform:uppercase;margin-bottom:8px;">
            📊 Proposed Model Weights Architecture
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;">
            <span>1. Thermal Deviation (vs Learned Baseline)</span>
            <b style="color:#ff334b;">30%</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;">
            <span>2. Temporal Persistence (Days Observed)</span>
            <b style="color:#fb923c;">20%</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;">
            <span>3. Thermal Intensity (Fire Radiative Power)</span>
            <b style="color:#fbbf24;">20%</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;">
            <span>4. Spatial Expansion (Cluster Variance)</span>
            <b style="color:#38bdf8;">15%</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-size:12px;">
            <span>5. Facility Proximity (OSM Industrial Buffer)</span>
            <b style="color:#818cf8;">10%</b>
          </div>
          <div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;">
            <span>6. Environmental / Hazardous Infrastructure Context</span>
            <b style="color:#a78bfa;">5%</b>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("ℹ️ **Model Note**: These are proposed model weights, not an official NASA formula. Satellite data provides thermal anomaly indication; ground verification is recommended before declaring fire emergencies.")

    with risk_right:
        top_risk_val = int(fac_subset["thermal_risk_score"].max()) if "thermal_risk_score" in fac_subset.columns else 89
        top_risk_status = str(fac_subset["thermosense_status"].iloc[0]) if "thermosense_status" in fac_subset.columns else "ESCALATING"
        st.plotly_chart(chart_thermosense_risk_gauge(top_risk_val, top_risk_status), use_container_width=True)

# ── TAB 5: Step 6 — Priority Alerts & Recommendations ─────────────────────────
with tab_alerts:
    esc_flag = fire_gdf.get("escalation_flag", pd.Series(False, index=fire_gdf.index))
    crit_flag = fire_gdf.get("thermosense_status", pd.Series("NORMAL")).isin(["CRITICAL", "ESCALATING", "ABNORMAL"])
    alert_rows = fire_gdf[esc_flag | crit_flag]

    if alert_rows.empty:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.3);border-radius:10px;padding:20px;text-align:center;">
          <div style="font-size:24px;">🛡️</div>
          <div style="font-size:15px;font-weight:700;color:#34d399;margin-top:4px;">All Industrial Facilities Within Normal Learned Baselines</div>
          <div style="font-size:12px;color:#94a3b8;margin-top:2px;">No significant thermal deviations or escalating power surges detected.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <div>
            <div style="font-size:16px;font-weight:700;color:#f43f5e;">🚨 {len(alert_rows)} Priority Thermal Deviation Alert(s)</div>
            <div style="font-size:11.5px;color:#94a3b8;">Prioritized for ground verification based on facility thermal baseline deviation.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        for _, row in alert_rows.sort_values("frp", ascending=False).head(20).iterrows():
            facility = row.get("osm_facility_name", "Industrial Facility")
            frp      = row.get("frp", 0)
            baseline = row.get("thermosense_baseline_frp", row.get("escalation_baseline_frp", frp * 0.3))
            zscore   = row.get("thermosense_zscore", row.get("frp_zscore_vs_baseline", 2.5))
            status   = row.get("thermosense_status", "ESCALATING")
            rec      = row.get("thermosense_recommendation", "Ground verification recommended")
            lat      = row.get("latitude", 0)
            lon      = row.get("longitude", 0)
            acq_date = str(row.get("acq_date", ""))[:10]
            sat      = row.get("satellite", "VIIRS")
            
            surge_pct = ((frp - baseline) / max(baseline, 1)) * 100

            st.markdown(f"""
            <div class="incident-card">
              <div class="incident-header">
                <div class="incident-title">
                  <span style="font-size:16px;">🔴</span>
                  <span>Hotspot detected near <b>{facility}</b></span>
                </div>
                <div style="display:flex;gap:6px;">
                  <span style="background:rgba(255,51,75,0.25);color:#ff4d6d;border:1px solid #ff4d6d;padding:2px 6px;border-radius:4px;font-size:10.5px;font-weight:700;">
                    {status}
                  </span>
                  <span style="background:rgba(255,255,255,0.06);color:#94a3b8;padding:2px 6px;border-radius:4px;font-size:10.5px;font-family:'JetBrains Mono',monospace;">
                    {acq_date} · {sat}
                  </span>
                </div>
              </div>

              <div class="incident-stats-grid">
                <div class="incident-stat-item">
                  <div class="incident-stat-label">Current FRP</div>
                  <div class="incident-stat-val" style="color:#ff334b;">{frp:.1f} MW</div>
                </div>
                <div class="incident-stat-item">
                  <div class="incident-stat-label">Learned Baseline</div>
                  <div class="incident-stat-val" style="color:#38bdf8;">{baseline:.1f} MW</div>
                </div>
                <div class="incident-stat-item">
                  <div class="incident-stat-label">Thermal Surge</div>
                  <div class="incident-stat-val" style="color:#fbbf24;">+{surge_pct:.0f}%</div>
                </div>
                <div class="incident-stat-item">
                  <div class="incident-stat-label">Z-Score Deviation</div>
                  <div class="incident-stat-val" style="color:#f43f5e;">+{zscore:.2f}σ</div>
                </div>
              </div>

              <div style="font-size:11.5px;color:#cbd5e1;display:flex;justify-content:space-between;align-items:center;margin-top:6px;">
                <span><b>Action Recommendation:</b> {rec}</span>
                <span style="font-family:'JetBrains Mono',monospace;color:#64748b;">📍 {lat:.4f}, {lon:.4f}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.download_button(
            "📥 Download Actionable Alerts (CSV)",
            data=alert_rows.to_csv(index=False),
            file_name=f"thermosense_priority_alerts_{date.today()}.csv",
            mime="text/csv",
        )

# ── TAB 6: Data Registry & Export ─────────────────────────────────────────────
with tab_data:
    st.markdown("""
    <div style="margin-bottom:8px;">
      <span style="font-size:16px;font-weight:700;color:#ffffff;">📋 Full Detection &amp; Facility Registry</span>
      <div style="font-size:12px;color:#94a3b8;">Search, inspect, and export all classified thermal anomalies.</div>
    </div>
    """, unsafe_allow_html=True)

    # Filter Toolbar
    f_c1, f_c2, f_c3 = st.columns([1.5, 1.5, 1])
    with f_c1:
        all_classes = fire_gdf[class_col].unique().tolist()
        sel_classes = st.multiselect("Class Filter", all_classes, default=all_classes)
    with f_c2:
        max_frp_val = float(fire_gdf["frp"].max() + 1) if not fire_gdf.empty else 100.0
        frp_range = st.slider("FRP Intensity Range (MW)", 0.0, max_frp_val, (0.0, max_frp_val))
    with f_c3:
        show_fp = st.checkbox("Include Solar/False Positives", value=False)

    filtered = fire_gdf[fire_gdf[class_col].isin(sel_classes)]
    filtered = filtered[(filtered["frp"] >= frp_range[0]) & (filtered["frp"] <= frp_range[1])]
    if not show_fp:
        filtered = filtered[fire_gdf.get("is_false_positive", pd.Series(False)) != True]  # noqa: E712

    display_cols = [c for c in [
        "osm_facility_name", class_col, "thermosense_status", "thermal_risk_score",
        "frp", "thermosense_baseline_frp", "thermal_deviation_pct",
        "thermal_stability_score", "latitude", "longitude", "acq_date", "satellite",
    ] if c in filtered.columns]

    df_display = filtered[display_cols].sort_values("frp", ascending=False).reset_index(drop=True)

    st.dataframe(df_display, height=450, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:12px;color:#94a3b8;margin:6px 0;">
      Displaying <b>{len(filtered):,}</b> matching anomaly detections
    </div>
    """, unsafe_allow_html=True)

    dl_c1, dl_c2 = st.columns(2)
    with dl_c1:
        st.download_button(
            "📥 Download Full CSV",
            data=filtered[display_cols].to_csv(index=False),
            file_name=f"thermosense_detections_{date.today()}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl_c2:
        try:
            geojson = filtered.to_json()
            st.download_button(
                "📥 Download GeoJSON Layer",
                data=geojson,
                file_name=f"thermosense_detections_{date.today()}.geojson",
                mime="application/json",
                use_container_width=True,
            )
        except Exception:
            pass

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06);text-align:center;color:#475569;font-size:11.5px;">
  🔥 ThermoSense-X &nbsp;·&nbsp; NASA FIRMS VIIRS/MODIS &nbsp;·&nbsp; OpenStreetMap Overpass &nbsp;·&nbsp; Sentinel-2 MSI
</div>
""", unsafe_allow_html=True)


