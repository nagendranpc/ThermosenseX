"""
Rule-Based Classifier  (Stage 2)
Multi-stage heuristic engine assigning fire classes based on domain knowledge.
Runs before the ML model and provides labels used for training the RF classifier.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

# ── Class labels ───────────────────────────────────────────────────────────────
CLASS_INDUSTRIAL_FIRE    = "INDUSTRIAL_FIRE"
CLASS_PERSISTENT_THERMAL = "PERSISTENT_THERMAL"
CLASS_WILDFIRE           = "WILDFIRE"
CLASS_AGRICULTURAL_BURN  = "AGRICULTURAL_BURN"
CLASS_UNKNOWN            = "UNKNOWN"
CLASS_FALSE_POSITIVE     = "FALSE_POSITIVE"

ALL_CLASSES = [
    CLASS_INDUSTRIAL_FIRE,
    CLASS_PERSISTENT_THERMAL,
    CLASS_WILDFIRE,
    CLASS_AGRICULTURAL_BURN,
    CLASS_UNKNOWN,
    CLASS_FALSE_POSITIVE,
]


def classify_rule_based(
    gdf: gpd.GeoDataFrame,
    cfg: dict | None = None,
) -> gpd.GeoDataFrame:
    """
    Apply the rule-based classification cascade to each detection.

    Rules are applied in priority order (highest to lowest):
      0. FALSE_POSITIVE  — already flagged by the FP filter
      1. INDUSTRIAL_FIRE — escalation flag OR high-FRP industrial overlap < 5 days
      2. PERSISTENT_THERMAL — OSM overlap + recurring ≥ persistence threshold
      3. WILDFIRE        — no OSM overlap, transient, high FRP
      4. AGRICULTURAL_BURN — daytime, moderate FRP, no OSM overlap
      5. UNKNOWN         — everything else

    Adds columns: rule_class (str), rule_confidence (float 0–1)
    """
    if cfg is None:
        cfg = {}
    c = cfg.get("classification", {})

    frp_ind_min     = float(c.get("frp_industrial_fire_min_mw",        50))
    frp_wild_min    = float(c.get("frp_wildfire_min_mw",               15))
    frp_agri_max    = float(c.get("frp_agricultural_max_mw",           60))
    persist_pt_days = int(c.get("persistence_persistent_thermal_days",  4))
    persist_wf_max  = int(c.get("persistence_wildfire_max_days",        3))

    gdf = gdf.copy()
    gdf["rule_class"]      = CLASS_UNKNOWN
    gdf["rule_confidence"] = 0.5

    def _row_class(row) -> tuple[str, float]:
        # ── Stage 0: False positive ────────────────────────────────────────
        if row.get("is_false_positive", False):
            return CLASS_FALSE_POSITIVE, 0.95

        frp        = float(row.get("frp", 0))
        persist    = int(row.get("persistence_days", row.get("detection_regularity", 0) * 7))
        osm_ov     = bool(row.get("osm_overlap", False))
        osm_wt     = float(row.get("osm_hazard_weight", 0.0))
        daynight   = str(row.get("daynight", "D"))
        conf       = str(row.get("confidence", "nominal"))
        stability  = float(row.get("thermal_stability_score", 0.5))
        drift      = float(row.get("centroid_drift_km", 0.0))
        escalated  = bool(row.get("escalation_flag", False))

        # Proxy persistence from detection_regularity if needed
        if persist == 0 and "detection_regularity" in row:
            persist = round(row["detection_regularity"] * 7)

        # ── Stage 1: Escalated industrial fire ────────────────────────────
        if escalated and osm_ov:
            conf_val = min(0.99, 0.80 + osm_wt * 0.19)
            return CLASS_INDUSTRIAL_FIRE, conf_val

        # ── Stage 1b: Acute industrial fire (fresh, no escalation needed) ─
        if osm_ov and frp >= frp_ind_min and persist < 5:
            conf_val = min(0.95, 0.70 + osm_wt * 0.25)
            return CLASS_INDUSTRIAL_FIRE, conf_val

        # ── Stage 2: Persistent thermal (stable recurring source) ─────────
        if osm_ov and persist >= persist_pt_days and stability >= 0.55:
            conf_val = min(0.95, 0.65 + osm_wt * 0.30 + stability * 0.05)
            return CLASS_PERSISTENT_THERMAL, conf_val

        # ── Stage 2b: High-stability OSM overlap (even without long history)
        if osm_ov and stability >= 0.75 and frp < frp_ind_min:
            return CLASS_PERSISTENT_THERMAL, 0.72

        # ── Stage 3: Wildfire ─────────────────────────────────────────────
        if not osm_ov and frp >= frp_wild_min and persist <= persist_wf_max:
            high_drift = drift > 0.3
            conf_val   = 0.60 + (0.15 if high_drift else 0.0) + (0.10 if frp > 50 else 0.0)
            return CLASS_WILDFIRE, min(0.90, conf_val)

        # ── Stage 4: Agricultural burn ────────────────────────────────────
        if not osm_ov and frp <= frp_agri_max and daynight == "D" and conf in ("nominal", "high"):
            agri_conf = 0.55 + (0.15 if frp < 25 else 0.0) + (0.10 if conf == "high" else 0.0)
            return CLASS_AGRICULTURAL_BURN, min(0.85, agri_conf)

        # ── Stage 5: Unknown ──────────────────────────────────────────────
    # ── Fast Vectorized Rule Classification ────────────────────────────────────
    rule_class = pd.Series(CLASS_UNKNOWN, index=gdf.index)
    rule_conf  = pd.Series(0.30, index=gdf.index)

    frp       = pd.to_numeric(gdf["frp"], errors="coerce").fillna(0.0)
    osm_ov    = gdf["osm_overlap"].astype(bool) if "osm_overlap" in gdf.columns else pd.Series(False, index=gdf.index)
    osm_wt    = pd.to_numeric(gdf["osm_hazard_weight"], errors="coerce").fillna(0.0) if "osm_hazard_weight" in gdf.columns else pd.Series(0.0, index=gdf.index)
    stability = pd.to_numeric(gdf["thermal_stability_score"], errors="coerce").fillna(0.5) if "thermal_stability_score" in gdf.columns else pd.Series(0.5, index=gdf.index)
    escalated = gdf["escalation_flag"].astype(bool) if "escalation_flag" in gdf.columns else pd.Series(False, index=gdf.index)
    daynight  = gdf["daynight"].astype(str) if "daynight" in gdf.columns else pd.Series("D", index=gdf.index)
    conf      = gdf["confidence"].astype(str).str.lower() if "confidence" in gdf.columns else pd.Series("nominal", index=gdf.index)

    # 4. Agricultural Burn
    m_agri = (~osm_ov) & (frp <= frp_agri_max) & (daynight == "D") & (conf.isin(["nominal", "high"]))
    rule_class[m_agri] = CLASS_AGRICULTURAL_BURN
    rule_conf[m_agri]  = 0.80

    # 3. Wildfire
    m_wild = (~osm_ov) & (frp >= frp_wild_min)
    rule_class[m_wild] = CLASS_WILDFIRE
    rule_conf[m_wild]  = 0.85

    # 2. Persistent Thermal
    m_pt = (osm_ov) & (stability >= 0.50) & (frp < frp_ind_min)
    rule_class[m_pt] = CLASS_PERSISTENT_THERMAL
    rule_conf[m_pt]  = 0.88

    # 1. Industrial Fire
    m_ind = (osm_ov) & ((escalated) | (frp >= frp_ind_min))
    rule_class[m_ind] = CLASS_INDUSTRIAL_FIRE
    rule_conf[m_ind]  = 0.95

    # 0. False Positive
    if "is_false_positive" in gdf.columns:
        m_fp = gdf["is_false_positive"].astype(bool)
        rule_class[m_fp] = CLASS_FALSE_POSITIVE
        rule_conf[m_fp]  = 0.95

    gdf["rule_class"]      = rule_class
    gdf["rule_confidence"] = rule_conf

    dist = gdf["rule_class"].value_counts()
    logger.info(f"Vectorized rule-based classification:\n{dist.to_string()}")
    return gdf
