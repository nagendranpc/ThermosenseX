"""
Sensor-Adaptive Dynamic Pixel Buffer
Replaces fixed-radius spatial joins with buffers tuned to each sensor's
native pixel footprint, preventing spatial mis-classification at zone edges.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

# Sensor → ground sample distance (meters)
SENSOR_BUFFER_MAP = {
    "VIIRS":     375,   # I-Band, 375 m
    "MODIS":    1000,   # 1-km active fire product
    "LANDSAT":    30,   # 30-m active fire product
    "HIMAWARI": 2000,   # 2-km geostationary rapid tracking
    "METEOSAT": 2000,   # 2-km geostationary rapid tracking
    "GOES":     2000,   # 2-km geostationary rapid tracking
    "INSAT":    2000,   # 2-km geostationary rapid tracking
    "GEO":      2000,   # 2-km generic geostationary footprint
}
DEFAULT_BUFFER = 500  # fallback


def get_sensor_buffer_m(source_tag: str) -> float:
    """Return the appropriate buffer radius (metres) for a given source string."""
    for sensor, radius in SENSOR_BUFFER_MAP.items():
        if sensor.upper() in source_tag.upper():
            return float(radius)
    return float(DEFAULT_BUFFER)


def _meters_to_deg(meters: float, latitude: float = 0.0) -> float:
    """Approximate conversion of metres to decimal degrees at a given latitude."""
    lat_deg = meters / 111_320
    return lat_deg   # simplification; valid for small buffers


def spatial_join_with_dynamic_buffer(
    fire_gdf: gpd.GeoDataFrame,
    osm_gdf: gpd.GeoDataFrame,
    config: dict | None = None,
) -> gpd.GeoDataFrame:
    """
    Vectorized sensor-adaptive spatial join: buffer each fire point by its
    sensor's pixel footprint radius, then use geopandas sjoin for speed.

    VIIRS I-Band = 375m buffer | MODIS = 1km buffer

    Adds columns: osm_overlap, osm_facility_name, osm_hazard_weight,
                  osm_tier, osm_type, dist_to_industrial_km
    """
    if fire_gdf.empty:
        return fire_gdf

    fire_gdf = fire_gdf.copy()
    fire_gdf["osm_overlap"]           = False
    fire_gdf["osm_facility_name"]     = "None"
    fire_gdf["osm_hazard_weight"]     = 0.0
    fire_gdf["osm_tier"]              = 0
    fire_gdf["osm_type"]              = "none"
    fire_gdf["dist_to_industrial_km"] = 999.0

    if osm_gdf is None or osm_gdf.empty:
        logger.warning("OSM GeoDataFrame is empty — skipping spatial join.")
        return fire_gdf

    if fire_gdf.crs is None:
        fire_gdf = fire_gdf.set_crs("EPSG:4326")
    if osm_gdf.crs is None:
        osm_gdf = osm_gdf.set_crs("EPSG:4326")

    # Project both to metric CRS for accurate buffering
    fire_proj = fire_gdf.to_crs("EPSG:3857")
    osm_proj  = osm_gdf.to_crs("EPSG:3857")

    # Build buffered point GDF (per-source radius)
    def _buf_radius(src):
        return get_sensor_buffer_m(str(src))

    src_col = "_source" if "_source" in fire_proj.columns else None
    if src_col:
        radii = fire_proj[src_col].apply(_buf_radius)
    else:
        radii = pd.Series(DEFAULT_BUFFER, index=fire_proj.index)

    buffered_geoms   = fire_proj.geometry.buffer(radii)
    fire_buf         = fire_proj.copy()
    fire_buf.geometry = buffered_geoms

    # Vectorized sjoin
    joined = gpd.sjoin(
        fire_buf[["geometry"]],
        osm_proj[["geometry", "name", "hazard_weight", "osm_tier", "osm_type"]],
        how="left",
        predicate="intersects",
    )

    # For detections with multiple OSM matches, keep highest hazard_weight
    if "hazard_weight" in joined.columns:
        best = (
            joined.reset_index()
            .sort_values("hazard_weight", ascending=False)
            .drop_duplicates(subset=["index"], keep="first")
            .set_index("index")
        )
        overlap_idx = best.index[best["hazard_weight"].notna()]
        fire_gdf.loc[overlap_idx, "osm_overlap"]       = True
        fire_gdf.loc[overlap_idx, "osm_facility_name"] = best.loc[overlap_idx, "name"].fillna("Unknown").values
        fire_gdf.loc[overlap_idx, "osm_hazard_weight"] = best.loc[overlap_idx, "hazard_weight"].values
        fire_gdf.loc[overlap_idx, "osm_tier"]          = best.loc[overlap_idx, "osm_tier"].fillna(0).astype(int).values
        fire_gdf.loc[overlap_idx, "osm_type"]          = best.loc[overlap_idx, "osm_type"].fillna("unknown").values
        fire_gdf.loc[overlap_idx, "dist_to_industrial_km"] = 0.0

    # Compute distance to nearest OSM polygon for non-overlapping points safely using STRtree / sjoin_nearest
    no_overlap = fire_gdf[~fire_gdf["osm_overlap"]].index
    if len(no_overlap) > 0 and not osm_proj.empty:
        try:
            # Make valid and use sjoin_nearest in metric CRS
            fire_no_ov = fire_proj.loc[no_overlap, ["geometry"]]
            osm_valid = osm_proj[["geometry"]].copy()
            osm_valid.geometry = osm_valid.geometry.make_valid()

            nearest = gpd.sjoin_nearest(
                fire_no_ov,
                osm_valid,
                how="left",
                distance_col="dist_m",
            )
            if "dist_m" in nearest.columns:
                # Group by original index to get minimum distance if multiple matches
                min_dists = nearest.groupby(nearest.index)["dist_m"].min()
                fire_gdf.loc[min_dists.index, "dist_to_industrial_km"] = (min_dists / 1000).round(3).values
        except Exception as e:
            logger.warning(f"Could not compute nearest OSM distance: {e}")
            fire_gdf.loc[no_overlap, "dist_to_industrial_km"] = 999.0

    overlapping = fire_gdf["osm_overlap"].sum()
    logger.info(
        f"Dynamic buffer join: {overlapping}/{len(fire_gdf)} detections overlap industrial polygons"
    )
    return fire_gdf
