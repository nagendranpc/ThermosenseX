"""
DBSCAN Spatial Clustering of Fire Detections
Groups nearby thermal anomaly pixels into discrete fire events.
Uses haversine distance to correctly handle geographic coordinates.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


def cluster_detections(
    gdf: gpd.GeoDataFrame,
    eps_meters: float = 750,
    min_samples: int = 1,
) -> gpd.GeoDataFrame:
    """
    Apply DBSCAN with haversine metric to group nearby fire detections
    into spatial clusters (representing single fire events).

    Parameters
    ----------
    gdf         : Input GeoDataFrame with latitude / longitude columns
    eps_meters  : Maximum distance between two points in the same cluster (meters)
    min_samples : Minimum detections to form a core cluster

    Returns
    -------
    GeoDataFrame with additional columns:
      cluster_id, cluster_size, cluster_centroid_lat, cluster_centroid_lon,
      cluster_mean_frp, cluster_max_frp
    """
    if gdf.empty:
        gdf["cluster_id"] = pd.Series(dtype=int)
        return gdf

    coords_rad = np.radians(gdf[["latitude", "longitude"]].values)
    eps_rad    = eps_meters / 1000 / EARTH_RADIUS_KM

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    ).fit(coords_rad)

    gdf = gdf.copy()
    gdf["cluster_id"] = db.labels_

    # Promote noise points (label=-1) to individual singleton clusters
    noise_mask = gdf["cluster_id"] == -1
    n_noise    = noise_mask.sum()
    if n_noise > 0:
        max_cid = gdf["cluster_id"].max()
        gdf.loc[noise_mask, "cluster_id"] = range(max_cid + 1, max_cid + 1 + n_noise)

    # Compute cluster-level statistics
    cluster_stats = (
        gdf.groupby("cluster_id")
        .agg(
            cluster_size      = ("frp", "count"),
            cluster_mean_frp  = ("frp", "mean"),
            cluster_max_frp   = ("frp", "max"),
            cluster_centroid_lat = ("latitude",  "mean"),
            cluster_centroid_lon = ("longitude", "mean"),
        )
        .reset_index()
    )

    gdf = gdf.merge(cluster_stats, on="cluster_id", how="left")
    n_clusters = gdf["cluster_id"].nunique()
    logger.info(
        f"Clustering: {len(gdf)} detections → {n_clusters} clusters "
        f"(eps={eps_meters}m, min_samples={min_samples})"
    )
    return gdf


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance in km between two (lat, lon) points."""
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ      = math.radians(lat2 - lat1)
    dλ      = math.radians(lon2 - lon1)
    a       = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
