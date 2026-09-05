"""
Feature Engineering
Assembles the complete ML feature matrix from all preprocessing outputs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)

# All features used by the ML model — must be present (or imputed) before training/prediction
ML_FEATURES = [
    "frp",
    "bright_ti4",
    "bright_ti5",
    "confidence_encoded",
    "daynight_encoded",
    "cluster_size",
    "cluster_mean_frp",
    "cluster_max_frp",
    "spatial_variance",
    "frp_cv",
    "centroid_drift_km",
    "thermal_stability_score",
    "detection_regularity",
    "frp_temporal_variance",
    "frp_zscore_vs_baseline",
    "osm_overlap_int",
    "osm_hazard_weight",
    "osm_tier",
    "dist_to_industrial_km",
]


def build_feature_matrix(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Extract and clean the ML feature matrix from the enriched GeoDataFrame.
    Missing columns are filled with sensible defaults.

    Returns
    -------
    DataFrame with columns = ML_FEATURES, index aligned with gdf.
    """
    df = pd.DataFrame(index=gdf.index)

    defaults = {
        "frp":                      0.0,
        "bright_ti4":               300.0,
        "bright_ti5":               270.0,
        "confidence_encoded":       1,
        "daynight_encoded":         0,
        "cluster_size":             1,
        "cluster_mean_frp":         0.0,
        "cluster_max_frp":          0.0,
        "spatial_variance":         0.0,
        "frp_cv":                   0.0,
        "centroid_drift_km":        0.0,
        "thermal_stability_score":  0.5,
        "detection_regularity":     0.5,
        "frp_temporal_variance":    0.0,
        "frp_zscore_vs_baseline":   0.0,
        "osm_overlap_int":          0,
        "osm_hazard_weight":        0.0,
        "osm_tier":                 0,
        "dist_to_industrial_km":    999.0,
    }

    for feat, default in defaults.items():
        if feat == "osm_overlap_int":
            # Derive from boolean osm_overlap
            if "osm_overlap" in gdf.columns:
                df[feat] = gdf["osm_overlap"].astype(int)
            else:
                df[feat] = 0
        elif feat in gdf.columns:
            df[feat] = pd.to_numeric(gdf[feat], errors="coerce").fillna(default)
        else:
            df[feat] = default

    # Clip FRP to avoid extreme outliers skewing the model
    df["frp"]             = df["frp"].clip(0, 2000)
    df["cluster_max_frp"] = df["cluster_max_frp"].clip(0, 2000)
    df["dist_to_industrial_km"] = df["dist_to_industrial_km"].clip(0, 500)

    logger.debug(f"Feature matrix built: {df.shape[0]} rows × {df.shape[1]} features")
    return df[ML_FEATURES]
