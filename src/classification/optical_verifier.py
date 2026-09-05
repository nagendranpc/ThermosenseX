"""
Optical Verifier  (Stage 4)
Enriches INDUSTRIAL_FIRE detections with Sentinel-2 SWIR optical verification.
"""

from __future__ import annotations

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def run_optical_verification(
    gdf: gpd.GeoDataFrame,
    sentinel_client,
) -> gpd.GeoDataFrame:
    """
    For each detection classified as INDUSTRIAL_FIRE, attempt Sentinel-2
    optical verification via the SentinelClient.

    Adds columns:
      optical_verified     (bool)   — True if NBR confirms active burn
      optical_source       (str)    — Sensor used (e.g. "Sentinel-2 B08/B11/B12")
      optical_scene_date   (str)
      optical_nbr          (float)
      optical_swir_lat     (float)  — Sub-metre peak SWIR location
      optical_swir_lon     (float)
      optical_cloud_pct    (float)
      optical_reason       (str)

    Parameters
    ----------
    gdf              : Classified GeoDataFrame (must have final_class column)
    sentinel_client  : SentinelClient instance
    """
    gdf = gdf.copy()
    col_defaults = {
        "optical_verified":   False,
        "optical_source":     None,
        "optical_scene_date": None,
        "optical_nbr":        None,
        "optical_swir_lat":   None,
        "optical_swir_lon":   None,
        "optical_cloud_pct":  None,
        "optical_reason":     "not_triggered",
    }
    for col, val in col_defaults.items():
        gdf[col] = val

    fire_mask = gdf["final_class"] == "INDUSTRIAL_FIRE"
    fire_rows = gdf[fire_mask]

    if fire_rows.empty:
        logger.info("Optical verifier: no INDUSTRIAL_FIRE detections to verify.")
        return gdf

    logger.info(f"Optical verifier: verifying {len(fire_rows)} INDUSTRIAL_FIRE detections …")

    for idx, row in fire_rows.iterrows():
        acq_date = (
            row["acq_date"].strftime("%Y-%m-%d")
            if hasattr(row.get("acq_date"), "strftime")
            else str(row.get("acq_date", ""))[:10]
        )
        result = sentinel_client.verify(
            lat=row["latitude"],
            lon=row["longitude"],
            detection_date=acq_date,
        )
        gdf.at[idx, "optical_verified"]   = result.get("verified",         False)
        gdf.at[idx, "optical_source"]     = result.get("source",           None)
        gdf.at[idx, "optical_scene_date"] = result.get("scene_date",       None)
        gdf.at[idx, "optical_nbr"]        = result.get("nbr_value",        None)
        gdf.at[idx, "optical_swir_lat"]   = result.get("swir_peak_lat",    None)
        gdf.at[idx, "optical_swir_lon"]   = result.get("swir_peak_lon",    None)
        gdf.at[idx, "optical_cloud_pct"]  = result.get("cloud_cover_pct",  None)
        gdf.at[idx, "optical_reason"]     = result.get("optical_reason",   "unknown")

    verified = gdf["optical_verified"].sum()
    logger.info(f"Optical verifier: {verified}/{len(fire_rows)} fire detections optically confirmed.")
    return gdf
