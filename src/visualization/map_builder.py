"""
Folium Interactive Map Builder
Generates a rich, layered HTML map with dark CartoDB basemap,
color-coded fire markers, OSM polygon overlays, and detailed popups.
"""

from __future__ import annotations

import logging
from typing import Optional

import folium
from folium.plugins import MarkerCluster, HeatMap, MiniMap
import geopandas as gpd
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Class → colour / icon mapping ─────────────────────────────────────────────
CLASS_STYLE = {
    "INDUSTRIAL_FIRE":    {"color": "#FF2222", "icon": "fire",          "icon_color": "white", "fill": "#FF2222"},
    "PERSISTENT_THERMAL": {"color": "#FF8C00", "icon": "thermometer",   "icon_color": "white", "fill": "#FF8C00"},
    "WILDFIRE":           {"color": "#22CC44", "icon": "tree",           "icon_color": "white", "fill": "#22CC44"},
    "AGRICULTURAL_BURN":  {"color": "#FFDD00", "icon": "leaf",           "icon_color": "black", "fill": "#FFDD00"},
    "UNKNOWN":            {"color": "#AAAAAA", "icon": "question-sign",  "icon_color": "white", "fill": "#AAAAAA"},
    "FALSE_POSITIVE":     {"color": "#666688", "icon": "ban-circle",     "icon_color": "white", "fill": "#666688"},
}

OSM_TIER_STYLE = {
    1: {"color": "#FF0000", "fill": "#FF444433", "weight": 2},
    2: {"color": "#FF8800", "fill": "#FF880033", "weight": 1.5},
    3: {"color": "#FFCC00", "fill": "#FFCC0022", "weight": 1},
    4: {"color": "#88AACC", "fill": "#88AACC11", "weight": 0.7},
}


def build_map(
    gdf: gpd.GeoDataFrame,
    osm_gdf: Optional[gpd.GeoDataFrame] = None,
    alerts: Optional[pd.DataFrame] = None,
    center: tuple[float, float] | None = None,
    zoom_start: int = 5,
) -> folium.Map:
    """
    Build a dark-themed Folium map with:
      - CartoDB dark_matter basemap
      - Per-class marker layers (toggle via LayerControl)
      - OSM industrial polygon overlays (colour-coded by hazard tier)
      - FRP heatmap overlay
      - Escalation alert markers (pulsing red)
      - Mini-map and fullscreen controls
      - Rich popups with all detection metadata

    Parameters
    ----------
    gdf        : Classified fire GeoDataFrame
    osm_gdf    : OSM industrial polygon GeoDataFrame (optional)
    alerts     : DataFrame of escalated/alerted detections (optional)
    center     : (lat, lon) for map centre; auto-computed if None
    zoom_start : Initial zoom level

    Returns
    -------
    folium.Map
    """
    if center is None:
        if not gdf.empty:
            center = (float(gdf["latitude"].mean()), float(gdf["longitude"].mean()))
        else:
            center = (22.0, 82.0)

    # ── Base map ───────────────────────────────────────────────────────────────
    m = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles=None,
        prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="CartoDB dark_matter",
        name="Dark Map",
        attr="&copy; CartoDB, OpenStreetMap contributors",
        control=False,
    ).add_to(m)

    # Secondary tile option
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        attr="&copy; OpenStreetMap contributors",
    ).add_to(m)

    # ── OSM industrial polygons ────────────────────────────────────────────────
    if osm_gdf is not None and not osm_gdf.empty:
        for tier in sorted(OSM_TIER_STYLE.keys()):
            style   = OSM_TIER_STYLE[tier]
            tier_gdf = osm_gdf[osm_gdf["osm_tier"] == tier]
            if tier_gdf.empty:
                continue
            tier_labels = {1: "Critical Hazard", 2: "High Hazard", 3: "Moderate Hazard", 4: "Low Hazard"}
            fg = folium.FeatureGroup(name=f"🏭 OSM {tier_labels[tier]}", show=(tier <= 2))
            for _, row in tier_gdf.iterrows():
                try:
                    folium.GeoJson(
                        data=row.geometry.__geo_interface__,
                        style_function=lambda f, s=style: {
                            "fillColor":   s["fill"],
                            "color":       s["color"],
                            "weight":      s["weight"],
                            "fillOpacity": 0.25,
                        },
                        tooltip=folium.Tooltip(
                            f"<b>{row.get('name','Unknown')}</b><br>"
                            f"Type: {row.get('osm_type','')}<br>"
                            f"Hazard: {row.get('hazard_label','')} (w={row.get('hazard_weight',0):.1f})"
                        ),
                    ).add_to(fg)
                except Exception:
                    pass
            fg.add_to(m)

    # ── FRP Heatmap ───────────────────────────────────────────────────────────
    if not gdf.empty and "frp" in gdf.columns:
        # Sample top 5000 detections by FRP for lightweight and instant heatmap rendering
        heat_gdf = gdf.sort_values("frp", ascending=False).head(5000) if len(gdf) > 5000 else gdf
        heat_data = [
            [float(row["latitude"]), float(row["longitude"]), min(float(row["frp"]), 200)]
            for _, row in heat_gdf.iterrows()
            if not pd.isna(row.get("frp"))
        ]
        HeatMap(
            heat_data,
            name="🔥 FRP Heatmap",
            min_opacity=0.3,
            radius=18,
            blur=15,
            gradient={"0.3": "blue", "0.55": "orange", "0.75": "red", "1.0": "white"},
            show=True,
        ).add_to(m)

    # ── Per-class marker layers ────────────────────────────────────────────────
    class_col = "final_class" if "final_class" in gdf.columns else "rule_class"

    for cls, style in CLASS_STYLE.items():
        # Only plot polar LEO archive detections in the classification cluster layers
        # Geostationary streams and Edge CCTV have their own dedicated high-resolution layers below
        cls_mask = (gdf.get(class_col, pd.Series()) == cls) & (~gdf.get("orbit_type", pd.Series("")).isin(["GEOSTATIONARY", "EDGE_GROUND_CAMERA"]))
        cls_gdf = gdf[cls_mask] if not gdf.empty else gdf.iloc[0:0]
        # Show all thermal detection layers by default so no thermal signatures are hidden
        show_layer = True
        fg = folium.FeatureGroup(name=f"{_cls_emoji(cls)} {cls} ({len(cls_gdf)})", show=show_layer)

        cluster = MarkerCluster(
            options={
                "maxClusterRadius": 45,
                "disableClusteringAtZoom": 16,
                "spiderfyOnMaxZoom": False,
                "zoomToBoundsOnClick": True,
            }
        )

        # For UNKNOWN background ambient points, sample top 150 by FRP to keep Leaflet canvas lightweight & instant
        # All points are still fully represented in the FRP Heatmap layer below
        if cls == "UNKNOWN" and len(cls_gdf) > 150:
            plot_gdf = cls_gdf.sort_values("frp", ascending=False).head(150)
        else:
            plot_gdf = cls_gdf.sort_values("frp", ascending=False).head(400) if len(cls_gdf) > 400 else cls_gdf

        for _, row in plot_gdf.iterrows():
            popup_html = _build_popup(row, class_col)
            radius = _frp_to_radius(row.get("frp", 5))
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=style["color"],
                fill=True,
                fill_color=style["fill"],
                fill_opacity=0.78,
                weight=1.5,
                popup=folium.Popup(popup_html, max_width=380),
                tooltip=folium.Tooltip(
                    f"<b>{cls}</b> | FRP: {row.get('frp', '?'):.1f} MW"
                    + (f" ⚠️ ESCALATED" if row.get("escalation_flag") else "")
                ),
            ).add_to(cluster)

        cluster.add_to(fg)
        fg.add_to(m)

    # ── Escalation alert markers ───────────────────────────────────────────────
    esc_mask = gdf.get("escalation_flag", pd.Series(False, index=gdf.index))
    esc_gdf  = gdf[esc_mask]
    if not esc_gdf.empty:
        fg_esc = folium.FeatureGroup(name=f"🚨 Escalation Alerts ({len(esc_gdf)})", show=True)
        for _, row in esc_gdf.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=22,
                color="#FF0000",
                fill=False,
                weight=3,
                opacity=0.9,
                tooltip="⚠️ ESCALATED SITE",
            ).add_to(fg_esc)
        fg_esc.add_to(m)

    # ── Geostationary Rapid Tracking Overlay Layer ────────────────────────────
    geo_mask = gdf.get("orbit_type", pd.Series("", index=gdf.index)) == "GEOSTATIONARY"
    geo_gdf = gdf[geo_mask]
    if not geo_gdf.empty:
        fg_geo = folium.FeatureGroup(name=f"🛰️ Geostationary Stream ({len(geo_gdf)} scans · 10m cadence)", show=True)
        
        # Group by facility location to provide clean site summaries and avoid marker clutter
        # Find unique facility locations
        site_groups = geo_gdf.groupby(["latitude", "longitude"])
        for (lat, lon), group in site_groups:
            peak_row = group.sort_values("frp", ascending=False).iloc[0]
            site_name = peak_row.get("osm_facility_name") or peak_row.get("_mock_site") or "Active Industrial Facility"
            scans_count = len(group)
            max_frp = group["frp"].max()
            sat_name = peak_row.get("satellite", "INSAT-3DR / Himawari-9")
            
            popup_html = _build_popup(peak_row, class_col)
            
            # Outer radar pulse ring
            folium.CircleMarker(
                location=[lat, lon],
                radius=26,
                color="#00E5FF",
                fill=True,
                fill_color="#00E5FF",
                fill_opacity=0.10,
                weight=2,
                dash_array="6, 6",
            ).add_to(fg_geo)
            
            # Inner distinct core marker with label
            folium.CircleMarker(
                location=[lat, lon],
                radius=14,
                color="#00B4D8",
                fill=True,
                fill_color="#00E5FF",
                fill_opacity=0.85,
                weight=2,
                popup=folium.Popup(popup_html, max_width=380),
                tooltip=folium.Tooltip(
                    f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#0f172a;font-weight:700;padding:2px 4px;'>"
                    f"🛰️ <b>GEO 10-MIN RAPID TRACKING</b><br/>"
                    f"🏭 <b>{site_name}</b><br/>"
                    f"📡 Sat: {sat_name}<br/>"
                    f"🔥 Peak FRP: <b>{max_frp:.1f} MW</b> ({scans_count} scans today)"
                    f"</div>"
                ),
            ).add_to(fg_geo)
            
        fg_geo.add_to(m)

    # ── Edge Real-Time Zero-Delay Camera Overlay Layer ────────────────────────
    edge_mask = gdf.get("orbit_type", pd.Series("", index=gdf.index)) == "EDGE_GROUND_CAMERA"
    edge_gdf = gdf[edge_mask]
    if not edge_gdf.empty:
        fg_edge = folium.FeatureGroup(name=f"📹 On-Site Edge Cameras ({len(edge_gdf)} streams · 0-Sec Instant)", show=True)
        for _, row in edge_gdf.iterrows():
            popup_html = _build_popup(row, class_col)
            site_name = row.get("osm_facility_name") or row.get("_mock_site") or "Monitored Industrial Complex"
            
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(color="red" if row.get("frp", 0) > 50 else "blue", icon="video-camera", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=380),
                tooltip=folium.Tooltip(
                    f"<div style='font-family:Inter,sans-serif;font-size:12px;color:#0f172a;font-weight:700;padding:2px 4px;'>"
                    f"📹 <b>ZERO-DELAY EDGE CAMERA (FLIR / YOLOv8)</b><br/>"
                    f"🏭 <b>{site_name}</b><br/>"
                    f"⚡ Latency: <b>&lt; 0.2 seconds</b> (Instant Flame Trigger)<br/>"
                    f"🔥 Heat Output: <b>{row.get('frp', 0):.1f} MW</b>"
                    f"</div>"
                ),
            ).add_to(fg_edge)
        fg_edge.add_to(m)

    # ── Map Controls & Dark Styling ───────────────────────────────────────────
    custom_map_css = """
    <style>
    .leaflet-control-layers {
        background: rgba(15, 23, 42, 0.92) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6) !important;
        font-family: 'Inter', Segoe UI, Arial, sans-serif !important;
        font-size: 11px !important;
        max-height: 260px !important;
        overflow-y: auto !important;
        padding: 8px 12px !important;
    }
    .leaflet-control-layers-toggle {
        background-color: rgba(15, 23, 42, 0.88) !important;
        border-radius: 6px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    .leaflet-control-layers label {
        color: #cbd5e1 !important;
        margin-bottom: 2px !important;
        cursor: pointer;
    }
    .leaflet-control-layers label:hover {
        color: #38bdf8 !important;
    }
    .leaflet-popup-content-wrapper {
        background: rgba(15, 23, 42, 0.96) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.7) !important;
    }
    .pop-card { font-family: 'Inter', Segoe UI, Arial, sans-serif; width: 300px; color: #f8fafc; }
    .pop-hdr { padding: 6px 10px; border-radius: 6px 6px 0 0; margin: -4px -4px 6px -4px; font-weight: 700; font-size: 12.5px; color: #fff; }
    .pop-table { width: 100%; font-size: 11px; border-collapse: collapse; }
    .pop-table td { padding: 2px 3px; }
    .pop-table td:first-child { color: #94a3b8; width: 38%; }
    .pop-table td:last-child { color: #f8fafc; font-weight: 600; }
    </style>
    """
    m.get_root().html.add_child(folium.Element(custom_map_css))
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    logger.info(f"Map built with {len(gdf)} detections.")
    return m


# ── Helpers ────────────────────────────────────────────────────────────────────

def _frp_to_radius(frp: float) -> float:
    return float(np.clip(4 + np.sqrt(max(0, frp)) * 0.55, 5, 30))


def _cls_emoji(cls: str) -> str:
    return {
        "INDUSTRIAL_FIRE":    "🔴",
        "PERSISTENT_THERMAL": "🟠",
        "WILDFIRE":           "🟢",
        "AGRICULTURAL_BURN":  "🟡",
        "UNKNOWN":            "⚪",
        "FALSE_POSITIVE":     "🚫",
    }.get(cls, "⚫")


def _build_popup(row: pd.Series, class_col: str) -> str:
    cls         = row.get(class_col, "UNKNOWN")
    style       = CLASS_STYLE.get(cls, CLASS_STYLE["UNKNOWN"])
    frp         = row.get("frp", 0)
    confidence  = row.get("rule_confidence", row.get("ml_confidence", 0))
    escalated   = row.get("escalation_flag", False)
    esc_reason  = row.get("escalation_reason", "")
    facility    = row.get("osm_facility_name", "—")
    osm_type    = row.get("osm_type", "—")
    hazard_wt   = row.get("osm_hazard_weight", 0.0)
    stability   = row.get("thermal_stability_score", 0.5)
    drift       = row.get("centroid_drift_km", 0.0)
    acq_date    = str(row.get("acq_date", ""))[:10]
    acq_time    = str(row.get("acq_time", ""))
    time_fmt    = f"{acq_time[:2]}:{acq_time[2:4]} UTC" if len(acq_time) >= 4 else ""
    satellite   = row.get("satellite", "—")
    orbit_type  = row.get("orbit_type", "POLAR_LEO")
    cadence     = row.get("cadence", "3-6 hours")
    footprint   = row.get("sensor_footprint", "375m-1km")
    daynight    = "☀️ Day" if row.get("daynight") == "D" else "🌙 Night"

    esc_badge = (
        f'<span style="background:#FF2222;color:white;padding:1px 5px;border-radius:3px;font-size:10px;">⚠️ ESCALATED</span>'
        if escalated else ""
    )
    opt_badge = (
        '<span style="background:#00AA44;color:white;padding:1px 5px;border-radius:3px;font-size:10px;">✅ Opt Verified</span>'
        if row.get("optical_verified") else ""
    )
    geo_badge = (
        f'<span style="background:rgba(0,229,255,0.2);color:#00E5FF;border:1px solid #00E5FF;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700;">🛰️ GEO 10-MIN</span>'
        if orbit_type == "GEOSTATIONARY" else ""
    )

    return f"""
    <div class="pop-card">
      <div class="pop-hdr" style="background:{style['color']};">
        {_cls_emoji(cls)} {cls.replace('_',' ')}
        <span style="float:right;font-size:10.5px;">Conf: {confidence:.0%}</span>
      </div>
      <div style="margin-bottom:4px;display:flex;gap:4px;flex-wrap:wrap;">{esc_badge} {opt_badge} {geo_badge}</div>
      <table class="pop-table">
        <tr><td>📅 Time</td><td>{acq_date} {time_fmt}</td></tr>
        <tr><td>🔥 FRP</td><td>{frp:.1f} MW</td></tr>
        <tr><td>🛰 Sat</td><td>{satellite} ({daynight})</td></tr>
        <tr><td>🌐 Orbit</td><td style="color:{'#00E5FF' if orbit_type == 'GEOSTATIONARY' else '#ffffff'};">{orbit_type} ({cadence})</td></tr>
        <tr><td>📏 Footprint</td><td>{footprint}</td></tr>
        <tr><td>🏭 Facility</td><td>{facility}</td></tr>
        <tr><td>⚠️ Hazard</td><td>{hazard_wt:.1f} ({osm_type})</td></tr>
        <tr><td>🌡 Stability</td><td>{stability:.2f} · {drift:.3f} km/d</td></tr>
        {"<tr><td colspan='2' style='color:#FF4444;'>⚡ " + esc_reason + "</td></tr>" if escalated else ""}
      </table>
    </div>
    """


def _build_legend() -> str:
    items = "".join(
        f'<div style="margin:3px 0;display:flex;align-items:center;gap:6px;">'
        f'<span style="background:{s["color"]};display:inline-block;width:10px;height:10px;'
        f'border-radius:50%;flex-shrink:0;box-shadow:0 0 4px {s["color"]};"></span>'
        f'<span style="font-size:11px;color:#e2e8f0;white-space:nowrap;">{_cls_emoji(cls)} {cls.replace("_"," ").title()}</span>'
        f'</div>'
        for cls, s in CLASS_STYLE.items()
    )
    return f"""
    <div style="position:absolute;bottom:25px;left:15px;z-index:999;
                background:rgba(15,23,42,0.92);backdrop-filter:blur(8px);
                border:1px solid rgba(255,255,255,0.12);border-radius:8px;
                padding:10px 14px;color:#f8fafc;font-family:Inter,Segoe UI,Arial,sans-serif;
                box-shadow:0 4px 16px rgba(0,0,0,0.6);pointer-events:auto;">
      <div style="font-size:11.5px;font-weight:700;color:#38bdf8;text-transform:uppercase;letter-spacing:0.04em;margin-bottom:6px;">
        🔥 Detection Classes
      </div>
      {items}
      <div style="font-size:9px;color:#94a3b8;margin-top:6px;border-top:1px solid rgba(255,255,255,0.08);padding-top:4px;">
        Circle size ∝ FRP intensity
      </div>
    </div>
    """
