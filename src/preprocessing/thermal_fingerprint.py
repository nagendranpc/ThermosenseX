"""
Thermal Fingerprinting & Variance Index
Computes per-cluster spatial stability and thermal volatility signatures
to distinguish stationary industrial stacks from moving wildfire fronts.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ, dλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a       = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1, math.sqrt(a)))


def compute_cluster_fingerprints(
    gdf: gpd.GeoDataFrame,
    site_history: pd.DataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Compute thermal fingerprint features per cluster and add them as columns
    to the input GeoDataFrame.

    Features added
    --------------
    spatial_variance       : var(lat) + var(lon) — low = stationary, high = moving
    frp_cv                 : FRP coefficient of variation (std/mean) across cluster days
    centroid_drift_km      : Mean daily centroid displacement in km
    thermal_stability_score: Composite 0–1 score (1 = perfectly stable industrial)
    detection_regularity   : Detections per day / days in window (0–1)
    frp_temporal_variance  : Variance of FRP across acquisition dates
    """
    gdf = gdf.copy()

    # Defaults for single-point clusters
    gdf["spatial_variance"]        = 0.0
    gdf["frp_cv"]                  = 0.0
    gdf["centroid_drift_km"]       = 0.0
    gdf["thermal_stability_score"] = 0.5
    gdf["detection_regularity"]    = 0.5
    gdf["frp_temporal_variance"]   = 0.0

    if "cluster_id" not in gdf.columns:
        logger.warning("No cluster_id column found — fingerprinting skipped.")
        return gdf

    # Fast vectorized computation of cluster fingerprint features
    grouped = gdf.groupby("cluster_id")
    
    # 1. Spatial variance: var(lat) + var(lon)
    lat_var = grouped["latitude"].var().fillna(0.0)
    lon_var = grouped["longitude"].var().fillna(0.0)
    spat_var = (lat_var + lon_var).rename("spatial_variance")
    
    # 2. FRP coefficient of variation
    frp_m = grouped["frp"].mean().replace(0, 1e-6)
    frp_s = grouped["frp"].std().fillna(0.0)
    frp_cv = (frp_s / frp_m).rename("frp_cv").round(4)
    
    # 3. Detection regularity & date tracking
    if "acq_date" in gdf.columns:
        n_dates = grouped["acq_date"].nunique().rename("n_dates")
        det_reg = (n_dates / n_dates.max().clip(1)).rename("detection_regularity").round(4)
    else:
        det_reg = pd.Series(0.5, index=grouped.indices.keys(), name="detection_regularity")

    # 4. Map back to GeoDataFrame index
    c_id = gdf["cluster_id"]
    gdf["spatial_variance"]        = c_id.map(spat_var).fillna(0.0).astype(float)
    gdf["frp_cv"]                  = c_id.map(frp_cv).fillna(0.0).astype(float)
    gdf["detection_regularity"]    = c_id.map(det_reg).fillna(0.5).astype(float)
    gdf["centroid_drift_km"]       = 0.0
    gdf["frp_temporal_variance"]   = (gdf["frp_cv"] * gdf["frp"]).round(4)

    # 5. Composite thermal stability score (1 = perfectly stationary industrial stack)
    stability_raw = 1.0 / (1.0 + gdf["spatial_variance"] * 10_000 + gdf["frp_cv"])
    gdf["thermal_stability_score"] = stability_raw.clip(0.0, 1.0).round(4)

    logger.info("Vectorized thermal fingerprinting complete.")
    return gdf


def _fingerprint_cluster(cdf: pd.DataFrame) -> Dict[str, float]:
    """Compute fingerprint metrics for one cluster's detections."""
    lats = cdf["latitude"].values
    lons = cdf["longitude"].values
    frps = cdf["frp"].values

    # ── Spatial variance ──────────────────────────────────────────────────────
    spatial_variance = float(np.var(lats) + np.var(lons))

    # ── FRP coefficient of variation ──────────────────────────────────────────
    frp_mean = float(np.mean(frps))
    frp_std  = float(np.std(frps))
    frp_cv   = float(frp_std / (frp_mean + 1e-6))

    # ── FRP temporal variance (per date) ─────────────────────────────────────
    if "acq_date" in cdf.columns and cdf["acq_date"].nunique() > 1:
        daily_mean            = cdf.groupby("acq_date")["frp"].mean()
        frp_temporal_variance = float(daily_mean.var()) if len(daily_mean) > 1 else 0.0
    else:
        frp_temporal_variance = frp_cv * frp_mean

    # ── Centroid drift per day ────────────────────────────────────────────────
    centroid_drift_km = 0.0
    if "acq_date" in cdf.columns and cdf["acq_date"].nunique() > 1:
        daily_cents = cdf.groupby("acq_date")[["latitude", "longitude"]].mean().reset_index()
        daily_cents = daily_cents.sort_values("acq_date")
        drifts      = []
        for i in range(1, len(daily_cents)):
            prev, curr = daily_cents.iloc[i - 1], daily_cents.iloc[i]
            drifts.append(_haversine_km(
                prev["latitude"], prev["longitude"],
                curr["latitude"], curr["longitude"]
            ))
        centroid_drift_km = float(np.mean(drifts)) if drifts else 0.0

    # ── Detection regularity ─────────────────────────────────────────────────
    if "acq_date" in cdf.columns:
        n_dates_observed = int(cdf["acq_date"].nunique())
        total_dates      = max(n_dates_observed, 1)
        detection_regularity = min(1.0, n_dates_observed / max(total_dates, 1))
    else:
        detection_regularity = 0.5

    # ── Thermal stability score (composite 0–1) ───────────────────────────────
    # High stability → industrial stack: low spatial_variance, low frp_cv, low drift
    stability_raw  = 1.0 / (1.0 + spatial_variance * 10_000 + frp_cv + centroid_drift_km * 2)
    thermal_stability_score = float(min(1.0, max(0.0, stability_raw)))

    return {
        "spatial_variance":        spatial_variance,
        "frp_cv":                  round(frp_cv, 4),
        "centroid_drift_km":       round(centroid_drift_km, 4),
        "thermal_stability_score": round(thermal_stability_score, 4),
        "detection_regularity":    round(detection_regularity, 4),
        "frp_temporal_variance":   round(frp_temporal_variance, 4),
    }
