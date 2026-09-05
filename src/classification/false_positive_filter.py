"""
Solar Reflectance False-Positive Filter  (Stage 0)
Identifies and tags thermal detections that are almost certainly caused by
solar glare off metallic rooftops, desert sand, or solar farms — not real fires.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# OSM types that strongly indicate solar/reflective surfaces
SOLAR_OSM_TYPES = {
    "power=generator;generator:source=solar",
    "landuse=solar_farm",
    "solar",
}

FP_CLASS = "FALSE_POSITIVE"


def filter_false_positives(
    gdf: gpd.GeoDataFrame,
    swir_threshold: float = 0.30,
    nir_threshold:  float = 0.25,
    frp_ceiling_mw: float = 5.0,
) -> gpd.GeoDataFrame:
    """
    Tag rows as FALSE_POSITIVE when they meet daytime solar-glare criteria.

    Rules applied (in priority order):
      1. OSM tag indicates a solar farm or solar generator → FP
      2. Daytime + low FRP (< frp_ceiling_mw) + low confidence → FP
      3. If SWIR/NIR reflectance data present:
             SWIR > swir_threshold AND NIR > nir_threshold AND FRP < ceiling → FP

    A new boolean column ``is_false_positive`` is added.
    Rows already marked FALSE_POSITIVE are passed through unchanged.

    Parameters
    ----------
    gdf             : GeoDataFrame with fire detections
    swir_threshold  : SWIR reflectance threshold (0–1)
    nir_threshold   : NIR reflectance threshold (0–1)
    frp_ceiling_mw  : Detections below this FRP value are eligible for FP filtering

    Returns
    -------
    GeoDataFrame with added column: is_false_positive (bool)
    """
    gdf = gdf.copy()
    gdf["is_false_positive"] = False
    gdf["fp_reason"]         = ""

    # ── Rule 1: OSM tag is solar farm / generator ─────────────────────────────
    if "osm_type" in gdf.columns:
        solar_mask = gdf["osm_type"].apply(
            lambda t: any(s in str(t).lower() for s in ["solar", "generator:source=solar"])
        )
        gdf.loc[solar_mask, "is_false_positive"] = True
        gdf.loc[solar_mask, "fp_reason"] = "osm_solar_facility"

    # ── Rule 2: Daytime + low FRP + low confidence ────────────────────────────
    night_ok = gdf.get("daynight", pd.Series("D", index=gdf.index)) != "N"
    low_frp  = gdf["frp"] < frp_ceiling_mw
    low_conf = gdf.get("confidence", pd.Series("nominal", index=gdf.index)) == "low"

    r2_mask  = night_ok & low_frp & low_conf
    newly_flagged = r2_mask & ~gdf["is_false_positive"]
    gdf.loc[newly_flagged, "is_false_positive"] = True
    gdf.loc[newly_flagged, "fp_reason"] = "daytime_low_frp_low_confidence"

    # ── Rule 3: Spectral reflectance test (when S2 data available) ────────────
    if "swir_reflectance" in gdf.columns and "nir_reflectance" in gdf.columns:
        high_swir   = gdf["swir_reflectance"].fillna(0) > swir_threshold
        high_nir    = gdf["nir_reflectance"].fillna(0)  > nir_threshold
        r3_mask     = high_swir & high_nir & low_frp & night_ok
        newly_r3    = r3_mask & ~gdf["is_false_positive"]
        gdf.loc[newly_r3, "is_false_positive"] = True
        gdf.loc[newly_r3, "fp_reason"] = "spectral_reflectance_solar_glare"

    # ── Mock class hint injection (demo mode) ─────────────────────────────────
    if "_mock_class_hint" in gdf.columns:
        mock_fp = gdf["_mock_class_hint"] == "FALSE_POSITIVE"
        gdf.loc[mock_fp, "is_false_positive"] = True
        gdf.loc[mock_fp & (gdf["fp_reason"] == ""), "fp_reason"] = "mock_solar_glare"

    n_fp = gdf["is_false_positive"].sum()
    logger.info(f"False-positive filter: {n_fp}/{len(gdf)} detections flagged as FP")
    return gdf
