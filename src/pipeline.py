"""
End-to-End Fire Detection Pipeline
Orchestrates all stages from data ingestion to classified output.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from datetime import date
from typing import List, Optional

import geopandas as gpd
import pandas as pd
import yaml

from src.ingestion.firms_client    import FIRMSClient
from src.ingestion.osm_client      import OSMClient
from src.ingestion.sentinel_client import SentinelClient
from src.preprocessing.clustering         import cluster_detections
from src.preprocessing.dynamic_buffer     import spatial_join_with_dynamic_buffer
from src.preprocessing.thermal_fingerprint import compute_cluster_fingerprints
from src.preprocessing.feature_engineering import build_feature_matrix
from src.classification.false_positive_filter import filter_false_positives
from src.classification.escalation_engine     import EscalationEngine
from src.classification.rule_based            import classify_rule_based
from src.classification.ml_classifier         import FireMLClassifier
from src.classification.optical_verifier      import run_optical_verification
from src.store.site_history_store             import SiteHistoryStore

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class FireDetectionPipeline:
    """
    Master pipeline that chains all stages sequentially:

    Stage 0 → Ingest FIRMS + OSM + (optionally) Sentinel data
    Stage 1 → Preprocess: cluster, dynamic-buffer join, fingerprint, feature matrix
    Stage 2 → Filter false positives
    Stage 3 → Escalation engine (FRP spike detection)
    Stage 4 → Rule-based classification
    Stage 5 → ML classification
    Stage 6 → Optical verification (INDUSTRIAL_FIRE detections only)
    Stage 7 → Final class resolution + output
    """

    def __init__(self, config: dict):
        self.cfg         = config
        self.firms       = FIRMSClient(config)
        self.osm         = OSMClient(config)
        self.sentinel    = SentinelClient(config)
        self.history     = SiteHistoryStore(config.get("paths", {}).get("site_history_db",
                           "data/processed/site_history/history.db"))
        self.escalation  = EscalationEngine(config, self.history)
        self.ml_clf      = FireMLClassifier(config)

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        bbox:    str         = "68,8,97,37",
        days:    int         = 7,
        sources: List[str]   = None,
        skip_optical: bool   = False,
        local_csv: Optional[str] = None,
        include_online_api: bool = False,
    ) -> dict:
        """
        Execute the full pipeline and return a results dict:
        {
          "fire_gdf"  : classified GeoDataFrame,
          "osm_gdf"   : OSM industrial polygons GeoDataFrame,
          "alerts"    : escalation alerts DataFrame,
          "stats"     : summary statistics dict,
        }
        """
        logger.info("=" * 65)
        logger.info("  Industrial Fire Detection Pipeline — START")
        logger.info(f"  bbox={bbox}  days={days}")
        logger.info("=" * 65)

        # ── S0: Ingest ─────────────────────────────────────────────────────────
        logger.info("[S0] Fetching FIRMS data …")
        fire_gdf = self.firms.fetch(
            bbox=bbox,
            days=days,
            sources=sources,
            local_csv_path=local_csv,
            include_online_api=include_online_api,
        )
        logger.info(f"     {len(fire_gdf)} raw detections loaded.")

        logger.info("[S0] Fetching OSM industrial facilities …")
        osm_gdf = self.osm.fetch_facilities(bbox=bbox)
        logger.info(f"     {len(osm_gdf)} industrial polygons loaded.")

        # ── S1a: Clustering ────────────────────────────────────────────────────
        logger.info("[S1] Clustering detections …")
        clust_cfg = self.cfg.get("clustering", {})
        fire_gdf  = cluster_detections(
            fire_gdf,
            eps_meters  = float(clust_cfg.get("eps_meters",  750)),
            min_samples = int(clust_cfg.get("min_samples", 1)),
        )

        # ── S1b: Dynamic buffer spatial join ──────────────────────────────────
        logger.info("[S1] Dynamic-buffer spatial join with OSM …")
        fire_gdf = spatial_join_with_dynamic_buffer(fire_gdf, osm_gdf, self.cfg)

        # ── S1c: Thermal fingerprinting ───────────────────────────────────────
        logger.info("[S1] Computing thermal fingerprints …")
        fire_gdf = compute_cluster_fingerprints(fire_gdf)

        # ── S1d: Assign geohash for history store ─────────────────────────────
        fire_gdf = self._assign_geohash(fire_gdf)

        # ── S1e: Persistence days from history store ──────────────────────────
        fire_gdf = self._add_persistence_days(fire_gdf)

        # ── S2: False-positive filter ─────────────────────────────────────────
        logger.info("[S2] Applying false-positive filter …")
        fp_cfg   = self.cfg.get("classification", {})
        fire_gdf = filter_false_positives(
            fire_gdf,
            frp_ceiling_mw = float(fp_cfg.get("fp_frp_ceiling_mw", 5)),
        )

        # ── S3: Escalation engine ─────────────────────────────────────────────
        logger.info("[S3] Running escalation engine …")
        fire_gdf = self.escalation.process(fire_gdf)

        # ── S4: Rule-based classification ─────────────────────────────────────
        logger.info("[S4] Rule-based classification …")
        fire_gdf = classify_rule_based(fire_gdf, self.cfg)

        # ── S5: ML classification ─────────────────────────────────────────────
        logger.info("[S5] ML classification …")
        X        = build_feature_matrix(fire_gdf)

        # Train on rule labels if no pre-trained model
        if not Path(self.ml_clf.model_path).exists():
            logger.info("[S5] Training ML model on rule-based labels …")
            self.ml_clf.train(X, fire_gdf["rule_class"])

        ml_preds = self.ml_clf.predict(X)
        fire_gdf = pd.concat([fire_gdf.reset_index(drop=True),
                               ml_preds.reset_index(drop=True)], axis=1)

        # ── S6: Final class resolution ────────────────────────────────────────
        fire_gdf["final_class"]      = self._resolve_class(fire_gdf)
        fire_gdf["final_confidence"] = self._resolve_confidence(fire_gdf)

        # ── S7: Optical verification ──────────────────────────────────────────
        if not skip_optical:
            logger.info("[S7] Optical verification (INDUSTRIAL_FIRE only) …")
            fire_gdf = run_optical_verification(fire_gdf, self.sentinel)

        # ── S8: ThermoSense-X Facility Fingerprinting & Risk Engine ───────────
        logger.info("[S8] ThermoSense-X facility baseline comparison & 6-factor risk scoring …")
        from src.classification.thermosense_risk_engine import ThermoSenseRiskEngine
        risk_engine = ThermoSenseRiskEngine(self.cfg)
        fire_gdf = risk_engine.evaluate_detections(fire_gdf)
        facility_profiles = risk_engine.compute_facility_fingerprints(fire_gdf)

        # ── Persist history ───────────────────────────────────────────────────
        self.history.upsert_detections(fire_gdf)

        # ── Extract alerts ────────────────────────────────────────────────────
        from src.visualization.report import escalation_alerts_table, generate_summary_stats
        alerts = escalation_alerts_table(fire_gdf)
        stats  = generate_summary_stats(fire_gdf)
        
        # Add ThermoSense-X specific summary metrics
        stats["thermosense_critical"]   = int((fire_gdf["thermosense_status"] == "CRITICAL").sum()) if "thermosense_status" in fire_gdf.columns else 0
        stats["thermosense_escalating"] = int((fire_gdf["thermosense_status"] == "ESCALATING").sum()) if "thermosense_status" in fire_gdf.columns else 0
        stats["thermosense_abnormal"]   = int((fire_gdf["thermosense_status"] == "ABNORMAL").sum()) if "thermosense_status" in fire_gdf.columns else 0
        stats["thermosense_emerging"]   = int((fire_gdf["thermosense_status"] == "EMERGING").sum()) if "thermosense_status" in fire_gdf.columns else 0
        stats["thermosense_normal"]     = int((fire_gdf["thermosense_status"] == "NORMAL").sum()) if "thermosense_status" in fire_gdf.columns else 0
        stats["mean_risk_score"]        = round(float(fire_gdf["thermal_risk_score"].mean()), 1) if "thermal_risk_score" in fire_gdf.columns and not fire_gdf.empty else 0.0
        stats["max_risk_score"]         = int(fire_gdf["thermal_risk_score"].max()) if "thermal_risk_score" in fire_gdf.columns and not fire_gdf.empty else 0

        logger.info("=" * 65)
        logger.info(f"  ThermoSense-X Pipeline complete — {len(fire_gdf)} classified detections")
        logger.info(f"  Critical / Escalating Events: {stats['thermosense_critical']} / {stats['thermosense_escalating']}")
        logger.info(f"  Facility Profiles Evaluated: {len(facility_profiles)}")
        logger.info("=" * 65)

        return {
            "fire_gdf":          fire_gdf,
            "osm_gdf":           osm_gdf,
            "alerts":            alerts,
            "stats":             stats,
            "facility_profiles": facility_profiles,
        }

    # ── Helper methods ─────────────────────────────────────────────────────────

    def _assign_geohash(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if "geohash" in gdf.columns:
            return gdf
        try:
            import geohash as gh_lib
            gdf["geohash"] = gdf.apply(
                lambda r: gh_lib.encode(r["latitude"], r["longitude"], precision=7), axis=1
            )
        except ImportError:
            gdf["geohash"] = (
                gdf["latitude"].round(2).astype(str) + "_" +
                gdf["longitude"].round(2).astype(str)
            )
        return gdf

    def _add_persistence_days(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add persistence_days by counting history store records per geohash."""
        gdf["persistence_days"] = 1
        if "geohash" not in gdf.columns:
            return gdf
        all_sites = self.history.get_all_sites()
        if all_sites.empty:
            return gdf
        gh_n = all_sites.set_index("geohash")["n_obs"].to_dict()
        gdf["persistence_days"] = gdf["geohash"].map(gh_n).fillna(1).astype(int)
        return gdf

    def _resolve_class(self, gdf: pd.DataFrame) -> pd.Series:
        """
        Final class: ML result when confidence > 0.70,
        otherwise fall back to rule-based result.
        Escalation always takes priority.
        """
        final = gdf.get("rule_class", pd.Series("UNKNOWN", index=gdf.index)).copy()
        if "ml_class" in gdf.columns and "ml_confidence" in gdf.columns:
            ml_confident = gdf["ml_confidence"] > 0.70
            final = final.where(~ml_confident, other=gdf["ml_class"])

        # Escalation always overrides
        if "escalation_flag" in gdf.columns:
            esc_mask   = gdf["escalation_flag"] == True  # noqa: E712
            osm_mask   = gdf.get("osm_overlap", pd.Series(False)).astype(bool)
            final_mask = esc_mask & osm_mask
            final[final_mask] = "INDUSTRIAL_FIRE"

        # False positives always override
        if "is_false_positive" in gdf.columns:
            final[gdf["is_false_positive"] == True] = "FALSE_POSITIVE"  # noqa: E712

        return final

    def _resolve_confidence(self, gdf: pd.DataFrame) -> pd.Series:
        if "ml_confidence" in gdf.columns:
            ml_conf   = gdf["ml_confidence"].fillna(0)
            rule_conf = gdf.get("rule_confidence", pd.Series(0.5, index=gdf.index)).fillna(0.5)
            return ml_conf.where(ml_conf > 0.70, other=rule_conf)
        return gdf.get("rule_confidence", pd.Series(0.5, index=gdf.index))
