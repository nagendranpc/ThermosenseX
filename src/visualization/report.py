"""
Report Generator
Produces Plotly charts and summary statistics for the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

CLASS_COLORS = {
    "INDUSTRIAL_FIRE":    "#FF334B",  # Vivid crimson
    "PERSISTENT_THERMAL": "#FF9500",  # Vivid amber
    "WILDFIRE":           "#10B981",  # Vivid emerald
    "AGRICULTURAL_BURN":  "#FBBF24",  # Golden yellow
    "UNKNOWN":            "#94A3B8",  # Slate gray
    "FALSE_POSITIVE":     "#64748B",  # Muted slate
}

CHART_THEME = {
    "paper_bgcolor": "#0e131f",
    "plot_bgcolor": "#0e131f",
    "font": {"family": "Inter, -apple-system, system-ui, sans-serif", "color": "#E2E8F0"},
}


def generate_summary_stats(gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
    """Compute summary statistics dict for the dashboard KPI cards."""
    if gdf.empty:
        return {}

    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    vc        = gdf[class_col].value_counts()

    return {
        "total_detections":    int(len(gdf)),
        "industrial_fires":    int(vc.get("INDUSTRIAL_FIRE", 0)),
        "persistent_thermal":  int(vc.get("PERSISTENT_THERMAL", 0)),
        "wildfires":           int(vc.get("WILDFIRE", 0)),
        "agricultural_burns":  int(vc.get("AGRICULTURAL_BURN", 0)),
        "false_positives":     int(vc.get("FALSE_POSITIVE", 0)),
        "unknown":             int(vc.get("UNKNOWN", 0)),
        "escalation_alerts":   int(gdf.get("escalation_flag", pd.Series(False)).sum()),
        "mean_frp_mw":         round(float(gdf["frp"].mean()), 1),
        "max_frp_mw":          round(float(gdf["frp"].max()), 1),
        "sites_with_osm":      int(gdf.get("osm_overlap", pd.Series(False)).sum()),
    }


def chart_class_distribution(gdf: gpd.GeoDataFrame) -> go.Figure:
    """Horizontal bar chart: detection count by class."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    vc = gdf[class_col].value_counts().reset_index()
    vc.columns = ["Class", "Count"]
    vc["Color"] = vc["Class"].map(CLASS_COLORS).fillna("#888888")
    vc = vc.sort_values("Count", ascending=True)

    fig = go.Figure(go.Bar(
        x=vc["Count"], y=[c.replace("_", " ").title() for c in vc["Class"]],
        orientation="h",
        marker=dict(
            color=vc["Color"],
            line=dict(color="rgba(255,255,255,0.15)", width=1),
        ),
        text=[f"<b>{x:,}</b>" for x in vc["Count"]],
        textposition="outside",
        textfont=dict(size=12, color="#E2E8F0", family="JetBrains Mono, monospace"),
        hovertemplate="<b>%{y}</b><br>Detections: <b>%{x:,}</b><extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="<b>DETECTION CLASSIFICATION BREAKDOWN</b>", font=dict(size=13, color="#94A3B8")),
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=320,
        margin=dict(l=10, r=45, t=45, b=10),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False),
        yaxis=dict(tickfont=dict(size=12, color="#CBD5E1"), showgrid=False),
    )
    return fig


def chart_frp_distribution(gdf: gpd.GeoDataFrame) -> go.Figure:
    """Box plot of FRP distribution per class."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    fig = go.Figure()
    for cls, color in CLASS_COLORS.items():
        sub = gdf[gdf[class_col] == cls]["frp"].dropna()
        if sub.empty:
            continue
        fig.add_trace(go.Box(
            y=sub, name=cls.replace("_", " ").title(),
            marker_color=color, line_color=color,
            boxmean="sd",
            fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}",
            hovertemplate="<b>%{y:.1f} MW</b><extra>" + cls.replace("_", " ").title() + "</extra>",
        ))
    fig.update_layout(
        title=dict(text="<b>FRP INTENSITY SPREAD BY CLASS (MW)</b>", font=dict(size=13, color="#94A3B8")),
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=320,
        margin=dict(l=10, r=10, t=45, b=10),
        showlegend=False,
        yaxis=dict(title=dict(text="FRP (MW)", font=dict(size=11, color="#94A3B8")), showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        xaxis=dict(tickfont=dict(size=11, color="#CBD5E1"), showgrid=False),
    )
    return fig


def chart_frp_timeseries(gdf: gpd.GeoDataFrame) -> go.Figure:
    """Daily total FRP time series per class."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    if "acq_date" not in gdf.columns or gdf.empty:
        return go.Figure()

    df = gdf.copy()
    df["acq_date"] = pd.to_datetime(df["acq_date"])
    ts = (
        df.groupby([df["acq_date"].dt.date, class_col])["frp"]
        .sum()
        .reset_index()
    )
    ts.columns = ["date", "class", "total_frp"]

    fig = px.line(
        ts, x="date", y="total_frp", color="class",
        color_discrete_map=CLASS_COLORS,
        markers=True,
        labels={"total_frp": "Total FRP (MW)", "date": "Acquisition Date", "class": "Class"},
        title="<b>DAILY TOTAL FIRE RADIATIVE POWER EVOLUTION</b>",
    )
    fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        title_font=dict(size=13, color="#94A3B8"),
        height=330,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, color="#CBD5E1")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#CBD5E1")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig


def chart_thermal_stability_scatter(gdf: gpd.GeoDataFrame) -> go.Figure:
    """Scatter: thermal_stability_score vs FRP coloured by class."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    if "thermal_stability_score" not in gdf.columns or gdf.empty:
        return go.Figure()

    # Subsample top 2500 points by FRP for instant, smooth browser rendering
    plot_df = gdf.sort_values("frp", ascending=False).head(2500) if len(gdf) > 2500 else gdf

    fig = px.scatter(
        plot_df, x="thermal_stability_score", y="frp",
        color=class_col,
        color_discrete_map=CLASS_COLORS,
        size=plot_df["frp"].clip(1, 250),
        size_max=22,
        opacity=0.75,
        hover_data=["latitude", "longitude", "satellite", "acq_date"],
        labels={"thermal_stability_score": "Spatial-Temporal Stability Score", "frp": "FRP (MW)", class_col: "Class"},
        title="<b>THERMAL FINGERPRINT: STABILITY SCORE VS FRP INTENSITY</b>",
    )
    # Industrial persistence zone boundary
    fig.add_vline(
        x=0.70, line_dash="dash", line_color="#FF9500", line_width=1.5,
        annotation_text="Industrial Zone Threshold (≥ 0.70)",
        annotation_position="top right",
        annotation_font=dict(size=11, color="#FF9500"),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        title_font=dict(size=13, color="#94A3B8"),
        height=380,
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(range=[0, 1.05], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#CBD5E1")),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(size=11, color="#CBD5E1")),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=11, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
    )
    return fig


def chart_hazard_score(gdf: gpd.GeoDataFrame) -> go.Figure:
    """Gauge chart: overall industrial fire risk index."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    total = max(len(gdf), 1)
    n_ind = len(gdf[gdf[class_col] == "INDUSTRIAL_FIRE"])
    n_esc = int(gdf.get("escalation_flag", pd.Series(False)).sum())
    n_pt  = len(gdf[gdf[class_col] == "PERSISTENT_THERMAL"])

    risk_index = min(100, round(
        (n_ind / total * 60) + (n_esc * 8) + (n_pt / total * 20)
    ))

    # Gauge color calculation
    gauge_bar = "#10B981" if risk_index < 35 else "#FF9500" if risk_index < 70 else "#FF334B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk_index,
        title={"text": "<b>REGIONAL HAZARD INDEX</b>", "font": {"color": "#94A3B8", "size": 13, "family": "Inter, sans-serif"}},
        delta={"reference": 25, "increasing": {"color": "#FF334B"}, "decreasing": {"color": "#10B981"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickwidth": 1},
            "bar": {"color": gauge_bar, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0, 30],   "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [30, 65],  "color": "rgba(255, 149, 0, 0.15)"},
                {"range": [65, 85],  "color": "rgba(255, 51, 75, 0.2)"},
                {"range": [85, 100], "color": "rgba(255, 51, 75, 0.35)"},
            ],
            "threshold": {"line": {"color": "#FF334B", "width": 3}, "thickness": 0.8, "value": 75},
        },
        number={"suffix": "/100", "font": {"color": "#F8FAFC", "size": 32, "family": "JetBrains Mono, monospace"}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        height=330,
        margin=dict(l=25, r=25, t=50, b=20),
        font=CHART_THEME["font"],
    )
    return fig


def top_hotspots_table(gdf: gpd.GeoDataFrame, n: int = 15) -> pd.DataFrame:
    """Return the top N hotspots by FRP with key metadata."""
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"
    cols = [c for c in [
        class_col, "osm_facility_name", "frp", "latitude", "longitude",
        "acq_date", "satellite", "osm_hazard_weight",
        "thermal_stability_score", "escalation_flag",
    ] if c in gdf.columns]

    return (
        gdf[cols]
        .sort_values("frp", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def escalation_alerts_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return all escalated site detections for the alerts panel."""
    esc_col = "escalation_flag"
    if esc_col not in gdf.columns:
        return pd.DataFrame()
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"

    esc = gdf[gdf[esc_col] == True].copy()  # noqa: E712
    cols = [c for c in [
        class_col, "osm_facility_name", "frp", "escalation_baseline_frp",
        "frp_zscore_vs_baseline", "escalation_reason",
        "latitude", "longitude", "acq_date",
    ] if c in esc.columns]
    return esc[cols].sort_values("frp_zscore_vs_baseline", ascending=False).reset_index(drop=True)


def chart_thermosense_baseline_vs_current(facility_name: str, gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    ThermoSense-X Core Innovation Visualization:
    Compares the facility's learned historical baseline FRP vs current observed FRP readings.
    """
    if gdf.empty or "osm_facility_name" not in gdf.columns:
        return go.Figure()

    fac_df = gdf[gdf["osm_facility_name"] == facility_name].copy()
    if fac_df.empty:
        fac_df = gdf.head(10).copy()
        facility_name = fac_df["osm_facility_name"].iloc[0] if "osm_facility_name" in fac_df.columns else "Sample Facility"

    baseline_val = float(fac_df["thermosense_baseline_frp"].iloc[0]) if "thermosense_baseline_frp" in fac_df.columns else float(fac_df["frp"].mean())
    current_vals = fac_df["frp"].values
    dates = [str(d)[:10] for d in fac_df.get("acq_date", range(len(fac_df)))]

    fig = go.Figure()
    # Baseline benchmark line
    fig.add_trace(go.Scatter(
        x=dates,
        y=[baseline_val] * len(dates),
        mode="lines",
        name=f"Learned Baseline ({baseline_val:.1f} MW)",
        line=dict(color="#38BDF8", width=2, dash="dash"),
    ))
    # Observed detections
    fig.add_trace(go.Bar(
        x=dates,
        y=current_vals,
        name="Current Observed FRP",
        marker=dict(
            color=["#FF334B" if v > baseline_val * 2 else ("#FF9500" if v > baseline_val * 1.3 else "#10B981") for v in current_vals],
            line=dict(color="rgba(255,255,255,0.2)", width=1),
        ),
        hovertemplate="<b>%{x}</b><br>Observed FRP: <b>%{y:.1f} MW</b><extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"<b>THERMAL FINGERPRINT COMPARISON — {facility_name.upper()}</b>", font=dict(size=12, color="#94A3B8")),
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#CBD5E1")),
        yaxis=dict(title=dict(text="FRP (MW)", font=dict(size=11, color="#94A3B8")), showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#CBD5E1")),
    )
    return fig


def chart_event_evolution_lifecycle(facility_name: str, gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    Tracks how a thermal event progresses across lifecycle states:
    NORMAL -> EMERGING -> ABNORMAL -> ESCALATING -> CRITICAL
    """
    if gdf.empty:
        return go.Figure()

    fac_df = gdf[gdf.get("osm_facility_name", "") == facility_name].copy() if "osm_facility_name" in gdf.columns else gdf.copy()
    if fac_df.empty:
        fac_df = gdf.copy()

    if "acq_date" in fac_df.columns:
        fac_df["date_str"] = pd.to_datetime(fac_df["acq_date"]).dt.strftime("%b %d")
        daily = fac_df.groupby("date_str").agg(
            Daily_Max_FRP=("frp", "max"),
            Detections_Count=("frp", "count"),
            Max_Risk=("thermal_risk_score", "max") if "thermal_risk_score" in fac_df.columns else ("frp", lambda _: 50),
        ).reset_index()
    else:
        daily = pd.DataFrame({
            "date_str": [f"T+{i}d" for i in range(len(fac_df.head(7)))],
            "Daily_Max_FRP": fac_df["frp"].head(7).values,
            "Detections_Count": [1] * len(fac_df.head(7)),
            "Max_Risk": [40 + i * 8 for i in range(len(fac_df.head(7)))],
        })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["Daily_Max_FRP"],
        mode="lines+markers",
        name="Peak FRP (MW)",
        line=dict(color="#FF9500", width=3, shape="spline"),
        marker=dict(size=8, color="#FF9500"),
        yaxis="y1",
    ))
    fig.add_trace(go.Scatter(
        x=daily["date_str"], y=daily["Max_Risk"],
        mode="lines+markers",
        name="Thermal Risk Score (0–100)",
        line=dict(color="#FF334B", width=2.5, dash="dot"),
        marker=dict(size=7, color="#FF334B"),
        yaxis="y2",
    ))

    fig.update_layout(
        title=dict(text="<b>EVENT EVOLUTION & THERMAL INTENSITY LIFECYCLE</b>", font=dict(size=12, color="#94A3B8")),
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=300,
        margin=dict(l=10, r=20, t=40, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#CBD5E1")),
        yaxis=dict(title=dict(text="Peak FRP (MW)", font=dict(size=10, color="#FF9500")), showgrid=True, gridcolor="rgba(255,255,255,0.06)"),
        yaxis2=dict(
            title=dict(text="Risk Score (0-100)", font=dict(size=10, color="#FF334B")),
            overlaying="y", side="right", range=[0, 105], showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#CBD5E1")),
    )
    return fig


def chart_thermosense_risk_gauge(score: int, status: str = "NORMAL") -> go.Figure:
    """ThermoSense-X Multi-Factor Risk Gauge (0-100)."""
    gauge_color = "#10B981" if score < 30 else ("#FBBF24" if score < 55 else ("#FF9500" if score < 75 else "#FF334B"))
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"<b>THERMAL RISK: {status}</b>", "font": {"color": gauge_color, "size": 13, "family": "Inter, sans-serif"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#64748B", "tickwidth": 1},
            "bar": {"color": gauge_color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.08)",
            "steps": [
                {"range": [0, 30],   "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [30, 55],  "color": "rgba(251, 191, 36, 0.15)"},
                {"range": [55, 75],  "color": "rgba(255, 149, 0, 0.2)"},
                {"range": [75, 85],  "color": "rgba(255, 51, 75, 0.25)"},
                {"range": [85, 100], "color": "rgba(255, 51, 75, 0.4)"},
            ],
            "threshold": {"line": {"color": "#FF334B", "width": 3}, "thickness": 0.8, "value": 75},
        },
        number={"suffix": "/100", "font": {"color": "#F8FAFC", "size": 32, "family": "JetBrains Mono, monospace"}},
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        font=CHART_THEME["font"],
    )
    return fig


def chart_geostationary_rapid_cadence(facility_name: str, gdf: gpd.GeoDataFrame) -> go.Figure:
    """
    Plots high-frequency 10-minute cadence observations from Geostationary Satellites
    (Himawari-9, INSAT-3DR, GOES, Meteosat) for a monitored facility.
    Shows continuous real-time thermal tracking progression across the day.
    """
    if gdf.empty:
        return go.Figure()

    # Filter by facility if available
    fac_df = gdf[gdf.get("osm_facility_name", "") == facility_name].copy() if "osm_facility_name" in gdf.columns else gdf.copy()
    if fac_df.empty:
        fac_df = gdf.copy()

    # Filter for GEO detections if present, otherwise use all detections for the facility
    geo_df = fac_df[fac_df.get("orbit_type", "") == "GEOSTATIONARY"].copy()
    if geo_df.empty:
        geo_df = fac_df.copy()

    # Sort chronologically by acq_date and acq_time
    geo_df["time_sort"] = geo_df["acq_time"].astype(str).str.zfill(4)
    geo_df = geo_df.sort_values(["acq_date", "time_sort"])

    # Take the latest day (today)
    if "acq_date" in geo_df.columns:
        latest_date = geo_df["acq_date"].max()
        today_geo = geo_df[geo_df["acq_date"] == latest_date].copy()
    else:
        today_geo = geo_df.copy()

    if today_geo.empty:
        today_geo = geo_df.tail(72).copy()

    today_geo["time_formatted"] = today_geo["time_sort"].apply(lambda t: f"{t[:2]}:{t[2:4]} UTC" if len(t) >= 4 else str(t))

    # Calculate baseline and spike thresholds
    base_frp = float(today_geo["baseline_frp"].iloc[0]) if "baseline_frp" in today_geo.columns and not pd.isna(today_geo["baseline_frp"].iloc[0]) else float(today_geo["frp"].quantile(0.25))
    spike_thresh = base_frp + 2.5 * max(5.0, float(today_geo["frp"].std() if len(today_geo) > 2 else 10.0))

    # Point colors: cyan for normal observations, red/orange for spikes
    marker_colors = [
        "#FF334B" if val >= spike_thresh else ("#00E5FF" if val > base_frp * 1.3 else "#38BDF8")
        for val in today_geo["frp"]
    ]

    fig = go.Figure()

    # Baseline horizontal line
    fig.add_hline(
        y=base_frp,
        line_dash="dash",
        line_color="#10B981",
        line_width=1.5,
        annotation_text=f"Learned Facility Baseline ({base_frp:.1f} MW)",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#10B981"),
    )

    # Spike alert line
    fig.add_hline(
        y=spike_thresh,
        line_dash="dot",
        line_color="#FF334B",
        line_width=1.5,
        annotation_text=f"Abnormal Escalation Threshold ({spike_thresh:.1f} MW)",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#FF334B"),
    )

    # Main 10-min cadence continuous line trace
    satellite_label = today_geo["satellite"].iloc[0] if "satellite" in today_geo.columns and not today_geo.empty else "Himawari-9 / INSAT-3DR (GEO)"
    fig.add_trace(go.Scatter(
        x=today_geo["time_formatted"],
        y=today_geo["frp"],
        mode="lines+markers",
        name=f"GEO 10-Min FRP (MW) · {satellite_label}",
        line=dict(color="#00E5FF", width=2.5, shape="spline"),
        marker=dict(size=7, color=marker_colors, line=dict(width=1.5, color="#FFFFFF")),
        hovertemplate="<b>Time:</b> %{x}<br><b>FRP:</b> %{y:.1f} MW<br><b>Satellite:</b> " + str(satellite_label) + "<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=f"<b>🛰️ GEOSTATIONARY REAL-TIME 10-MINUTE CADENCE TRACKING — {facility_name.upper()}</b>",
            font=dict(size=12, color="#00E5FF")
        ),
        template="plotly_dark",
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        font=CHART_THEME["font"],
        height=320,
        margin=dict(l=10, r=20, t=45, b=20),
        xaxis=dict(
            title=dict(text="Observation Timestamp (UTC · 10-Minute Intervals)", font=dict(size=10, color="#94A3B8")),
            showgrid=False,
            tickfont=dict(size=9, color="#CBD5E1"),
            tickangle=-45,
            nticks=20,
        ),
        yaxis=dict(
            title=dict(text="Fire Radiative Power (MW)", font=dict(size=10, color="#00E5FF")),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.06)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#CBD5E1")),
    )
    return fig
