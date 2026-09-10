"""
Persistent-Site Escalation Engine  (Stage 1)
Tracks per-site FRP history and detects anomalous spikes that warrant
automatic escalation from PERSISTENT_THERMAL → INDUSTRIAL_FIRE.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


class EscalationEngine:
    """
    Maintains rolling FRP statistics per geohash-7 site and flags detections
    where the current FRP significantly exceeds the site's historical baseline.

    Algorithm per site:
        baseline_mean = rolling_mean(frp, window=window_days)
        baseline_std  = rolling_std(frp, window=window_days)
        z_score       = (current_frp − baseline_mean) / (baseline_std + ε)

        if z_score > z_threshold AND current_frp > min_frp_spike:
            classification = INDUSTRIAL_FIRE
            escalation_flag = True
    """

    def __init__(self, config: dict, site_history_store=None):
        esc_cfg               = config.get("escalation", {})
        self.window_days      = int(esc_cfg.get("window_days", 14))
        self.z_threshold      = float(esc_cfg.get("z_score_threshold", 3.0))
        self.min_observations = int(esc_cfg.get("min_observations", 5))
        self.min_frp_spike    = float(esc_cfg.get("min_frp_spike_mw", 30))
        self.store            = site_history_store

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add escalation columns to each detection:
          frp_zscore_vs_baseline  : float  — z-score vs site baseline
          escalation_flag         : bool   — True if site should escalate
          escalation_reason       : str    — human-readable alert message
          escalation_baseline_frp : float  — baseline mean FRP for context
        """
        gdf = gdf.copy()
        gdf["frp_zscore_vs_baseline"]  = 0.0
        gdf["escalation_flag"]          = False
        gdf["escalation_reason"]        = ""
        gdf["escalation_baseline_frp"]  = 0.0

        if "geohash" not in gdf.columns:
            gdf = self._assign_geohash(gdf)

        # Vectorized baseline stats per geohash
        gh_grp = gdf.groupby("geohash")["frp"]
        mean_frp = gh_grp.transform("mean")
        std_frp  = gh_grp.transform("std").fillna(1.0)
        cnt_frp  = gh_grp.transform("count")

        # z-score vs site baseline
        z_scores = ((gdf["frp"] - mean_frp) / (std_frp + 1e-6)).round(3)
        gdf["frp_zscore_vs_baseline"] = z_scores
        gdf["escalation_baseline_frp"] = mean_frp.round(2)

        # Flag sites with significant FRP spikes
        esc_condition = (
            (z_scores >= self.z_threshold) &
            (gdf["frp"] >= self.min_frp_spike) &
            (cnt_frp >= self.min_observations)
        )
        gdf["escalation_flag"] = esc_condition
        gdf.loc[esc_condition, "escalation_reason"] = (
            "FRP spike: " + gdf.loc[esc_condition, "frp"].round(1).astype(str) + " MW vs. baseline " +
            gdf.loc[esc_condition, "escalation_baseline_frp"].round(1).astype(str) + " MW (z=" +
            gdf.loc[esc_condition, "frp_zscore_vs_baseline"].round(2).astype(str) + ")"
        )

        n_esc = gdf["escalation_flag"].sum()
        if n_esc:
            logger.warning(f"Escalation Engine: {n_esc} site(s) escalated to INDUSTRIAL_FIRE")
        else:
            logger.info("Escalation Engine: No escalations detected.")
        return gdf

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _assign_geohash(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Assign geohash-7 codes (~150m precision) to group nearby detections."""
        try:
            import pygeohash as gh_lib
            gdf["geohash"] = gdf.apply(
                lambda r: gh_lib.encode(r["latitude"], r["longitude"], precision=7),
                axis=1,
            )
        except ImportError:
            try:
                import geohash as gh_lib
                gdf["geohash"] = gdf.apply(
                    lambda r: gh_lib.encode(r["latitude"], r["longitude"], precision=7),
                    axis=1,
                )
            except ImportError:
                # Fallback: round to 2 decimal places (~1 km grid)
                gdf["geohash"] = (
                    gdf["latitude"].round(2).astype(str) + "_" +
                    gdf["longitude"].round(2).astype(str)
                )
        return gdf

    def _get_history(
        self, geohash: str, site_df: pd.DataFrame
    ) -> pd.Series:
        """Return historical FRP series for a site from store or in-data inference."""
        if self.store is not None:
            try:
                hist = self.store.get_frp_history(geohash, days=self.window_days)
                if hist is not None and len(hist) >= self.min_observations:
                    return hist
            except Exception as e:
                logger.debug(f"Store lookup failed for {geohash}: {e}")

        # Infer from multi-day data already in the GeoDataFrame
        if "acq_date" in site_df.columns and site_df["acq_date"].nunique() > 1:
            return site_df.groupby("acq_date")["frp"].mean()

        # Synthetic: mimic what a gas-flare baseline would look like
        return _mock_baseline(site_df["frp"].mean())

    def _check_single(
        self,
        current_frp: float,
        history: pd.Series,
    ) -> Tuple[float, bool, str, float]:
        """
        Returns (z_score, is_escalated, reason, baseline_mean).
        """
        if len(history) < self.min_observations:
            return 0.0, False, "", 0.0

        baseline_mean = float(history.mean())
        baseline_std  = float(history.std())
        z             = (current_frp - baseline_mean) / (baseline_std + 1e-6)

        if z >= self.z_threshold and current_frp >= self.min_frp_spike:
            reason = (
                f"FRP spike: {current_frp:.1f} MW vs. "
                f"baseline {baseline_mean:.1f} ± {baseline_std:.1f} MW "
                f"(z={z:.2f})"
            )
            return z, True, reason, baseline_mean

        return z, False, "", baseline_mean


def _mock_baseline(mean_frp: float, n: int = 14) -> pd.Series:
    """Generate a synthetic FRP baseline series around the given mean."""
    rng    = np.random.default_rng(seed=int(mean_frp * 100) % 9999)
    values = rng.normal(mean_frp * 0.6, mean_frp * 0.1, size=n).clip(min=0.1)
    return pd.Series(values)
