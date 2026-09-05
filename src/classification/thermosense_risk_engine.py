"""
🔥 ThermoSense-X — Thermal Risk Engine & Anomaly Classifier
Implements the core ThermoSense-X philosophy:
Learning the normal thermal behaviour of each industrial facility,
detecting deviations, tracking event evolution, and scoring thermal risk (0–100).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


class ThermoSenseRiskEngine:
    """
    Calculates facility-specific thermal fingerprints, deviation metrics,
    event evolution lifecycle states, and the proposed ThermoSense-X Thermal Risk Score.

    Model Weights:
      - Thermal Deviation : 30%
      - Persistence       : 20%
      - Thermal Intensity : 20%
      - Spatial Expansion : 15%
      - Facility Proximity: 10%
      - Environmental/Ctx :  5%
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def compute_facility_fingerprints(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Builds a facility-specific baseline (Thermal Fingerprint) for each monitored site.
        Learns: typical intensity, frequency, spatial stability, baseline FRP.
        """
        if gdf.empty or "osm_facility_name" not in gdf.columns:
            return pd.DataFrame()

        # Group by facility
        fac_group = gdf.groupby("osm_facility_name")
        profiles = fac_group.agg(
            baseline_frp=("frp", "mean"),
            typical_max_frp=("frp", "max"),
            typical_min_frp=("frp", "min"),
            std_frp=("frp", lambda s: float(s.std()) if len(s) > 1 else 1.0),
            total_observations=("frp", "count"),
            mean_stability=("thermal_stability_score", "mean") if "thermal_stability_score" in gdf.columns else ("frp", lambda _: 0.5),
            hazard_tier=("osm_hazard_weight", "first") if "osm_hazard_weight" in gdf.columns else ("frp", lambda _: 0.5),
            osm_type=("osm_type", "first") if "osm_type" in gdf.columns else ("frp", lambda _: "industrial"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        ).reset_index()

        profiles["baseline_frp"] = profiles["baseline_frp"].round(1)
        profiles["std_frp"] = profiles["std_frp"].fillna(1.0).round(2)
        return profiles

    def evaluate_detections(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Enriches detections with:
          - facility baseline comparison
          - thermal_deviation_pct
          - thermal_risk_score (0–100)
          - event_evolution_status (NORMAL, EMERGING, ABNORMAL, ESCALATING, CRITICAL)
          - context_description (e.g. 'Hotspot detected 280 m from Jamnagar Refinery')
          - recommendation ('Ground verification recommended')
        """
        if gdf.empty:
            return gdf

        df = gdf.copy()

        # Ensure baseline columns exist
        if "osm_facility_name" not in df.columns:
            df["osm_facility_name"] = "Unlabeled Industrial Site"

        # 1. Facility Baseline & Deviation (30% weight component)
        fac_mean = df.groupby("osm_facility_name")["frp"].transform("mean")
        fac_std = df.groupby("osm_facility_name")["frp"].transform("std").fillna(2.0)
        df["thermosense_baseline_frp"] = fac_mean.round(1)
        
        # Z-score and percentage deviation
        zscore = (df["frp"] - fac_mean) / (fac_std + 1e-5)
        deviation_pct = ((df["frp"] - fac_mean) / fac_mean.clip(lower=1.0)) * 100.0
        df["thermosense_zscore"] = zscore.round(2)
        df["thermal_deviation_pct"] = deviation_pct.clip(lower=0.0).round(1)
        score_deviation = np.clip(zscore * 20.0 + (deviation_pct / 5.0), 0.0, 100.0)

        # 2. Persistence Score (20% weight component)
        if "persistence_days" in df.columns:
            p_days = df["persistence_days"].fillna(1)
        else:
            p_days = df.groupby("osm_facility_name")["frp"].transform("count")
        score_persistence = np.clip((p_days / 7.0) * 100.0, 10.0, 100.0)

        # 3. Thermal Intensity Score (20% weight component)
        score_intensity = np.clip((df["frp"] / 150.0) * 100.0, 0.0, 100.0)

        # 4. Spatial Expansion Score (15% weight component)
        if "spatial_variance" in df.columns:
            score_expansion = np.clip(df["spatial_variance"] * 50000.0, 10.0, 100.0)
        else:
            score_expansion = np.full(len(df), 25.0)

        # 5. Facility Proximity Score (10% weight component)
        if "osm_overlap" in df.columns:
            score_proximity = np.where(df["osm_overlap"] == True, 100.0, 40.0)
        else:
            score_proximity = np.full(len(df), 60.0)

        # 6. Environmental / Hazard Context (5% weight component)
        if "osm_hazard_weight" in df.columns:
            score_env = np.clip(df["osm_hazard_weight"].fillna(0.5) * 100.0, 0.0, 100.0)
        else:
            score_env = np.full(len(df), 50.0)

        # Proposed ThermoSense-X Formula:
        # Risk Score = (Dev * 0.30) + (Pers * 0.20) + (Int * 0.20) + (Exp * 0.15) + (Prox * 0.10) + (Env * 0.05)
        raw_risk = (
            score_deviation * 0.30 +
            score_persistence * 0.20 +
            score_intensity * 0.20 +
            score_expansion * 0.15 +
            score_proximity * 0.10 +
            score_env * 0.05
        )
        df["thermal_risk_score"] = np.clip(np.round(raw_risk), 0, 100).astype(int)

        # 7. Event Evolution Lifecycle State: NORMAL -> EMERGING -> ABNORMAL -> ESCALATING -> CRITICAL
        def assign_lifecycle(r):
            score = r["thermal_risk_score"]
            dev = r["thermal_deviation_pct"]
            z = r["thermosense_zscore"]
            if score >= 85 or (z >= 4.0 and dev > 200):
                return "CRITICAL"
            elif score >= 70 or (z >= 2.8 and dev > 100):
                return "ESCALATING"
            elif score >= 50 or (z >= 1.8 and dev > 50):
                return "ABNORMAL"
            elif score >= 30:
                return "EMERGING"
            else:
                return "NORMAL"

        df["thermosense_status"] = df.apply(assign_lifecycle, axis=1)

        # 8. Human-Readable Context Description & Action Recommendation
        def make_context_desc(r):
            fac = r.get("osm_facility_name", "Industrial Site")
            frp_val = r.get("frp", 0)
            base_val = r.get("thermosense_baseline_frp", 0)
            status = r.get("thermosense_status", "NORMAL")
            dist_str = "within facility boundary" if r.get("osm_overlap", False) else "in facility buffer zone"
            return f"Thermal hotspot ({frp_val:.1f} MW) detected {dist_str} of {fac} (baseline: {base_val:.1f} MW) — Status: {status}"

        df["thermosense_context_desc"] = df.apply(make_context_desc, axis=1)
        df["thermosense_recommendation"] = df["thermosense_status"].map({
            "CRITICAL": "🚨 CRITICAL ALERT — Immediate ground verification & facility alert dispatched",
            "ESCALATING": "⚠️ ESCALATING EVENT — Priority ground inspection recommended; thermal spread active",
            "ABNORMAL": "⚡ ABNORMAL BEHAVIOUR — Continued satellite monitoring & verification advised",
            "EMERGING": "🟡 EMERGING HEAT SIGNATURE — Routine surveillance active",
            "NORMAL": "🟢 NORMAL OPERATION — Within learned facility baseline parameters",
        })

        return df
