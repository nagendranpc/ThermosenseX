"""
OpenStreetMap / Overpass API Client
Fetches industrial facility polygons and assigns a multi-tiered hazard weight.
Results are cached as GeoJSON to avoid repeated API calls.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import shape, Polygon, MultiPolygon, Point

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# ── Hazard tier query tags ─────────────────────────────────────────────────────
HAZARD_TIERS: List[Dict] = [
    {
        "tier": 1, "label": "CRITICAL", "weight": 1.0,
        "tags": [
            ("industrial", "refinery"),
            ("industrial", "oil"),
            ("man_made",   "petroleum_well"),
            ("industrial", "chemical_works"),
            ("man_made",   "pipeline"),
        ],
    },
    {
        "tier": 2, "label": "HIGH", "weight": 0.7,
        "tags": [
            ("power",    "plant"),
            ("man_made", "works"),
            ("industrial", "steel"),
            ("industrial", "foundry"),
            ("industrial", "smelting"),
        ],
    },
    {
        "tier": 3, "label": "MODERATE", "weight": 0.4,
        "tags": [
            ("landuse",  "industrial"),
            ("man_made", "storage_tank"),
            ("industrial", "warehouse"),
        ],
    },
    {
        "tier": 4, "label": "LOW", "weight": 0.1,
        "tags": [
            ("landuse",  "quarry"),
            ("man_made", "chimney"),
            ("industrial", "port"),
        ],
    },
]

# ── Mock OSM polygons for offline dev ─────────────────────────────────────────
_MOCK_OSM_SITES = [
    # (name, center_lat, center_lon, radius_deg, tier, weight, osm_type)
    ("Jamnagar Refinery",       22.470, 70.060, 0.06,  1, 1.0, "industrial=refinery"),
    ("Jharia Coalfield",        23.752, 86.415, 0.05,  3, 0.4, "landuse=industrial"),
    ("Tata Steel Jamshedpur",   22.802, 86.185, 0.04,  2, 0.7, "industrial=steel"),
    ("Ankleshwar GIDC",         21.630, 73.002, 0.04,  1, 1.0, "industrial=chemical_works"),
    ("Singrauli Power",         24.199, 82.685, 0.05,  2, 0.7, "power=plant"),
    ("GAIL Vaghodia",           22.314, 73.186, 0.03,  1, 1.0, "man_made=petroleum_well"),
    ("Numaligarh Refinery",     26.664, 93.697, 0.04,  1, 1.0, "industrial=refinery"),
    ("Panipat Refinery",        29.388, 76.970, 0.04,  1, 1.0, "industrial=refinery"),
    ("Rajasthan Solar Farm",    26.500, 72.500, 0.07,  4, 0.1, "power=generator;generator:source=solar"),
    ("Bhilai Steel Plant",      21.209, 81.378, 0.05,  2, 0.7, "industrial=steel"),
]


def _make_box(lat: float, lon: float, r: float) -> Polygon:
    """Create a square polygon centered at (lat, lon) with half-side r degrees."""
    return Polygon([
        (lon - r, lat - r), (lon + r, lat - r),
        (lon + r, lat + r), (lon - r, lat + r),
    ])


def _generate_mock_osm(bbox: str) -> gpd.GeoDataFrame:
    west, south, east, north = [float(x) for x in bbox.split(",")]
    rows = []
    for name, lat, lon, r, tier, weight, osm_type in _MOCK_OSM_SITES:
        if south <= lat <= north and west <= lon <= east:
            rows.append({
                "name": name, "osm_tier": tier,
                "hazard_label": ["", "CRITICAL", "HIGH", "MODERATE", "LOW"][tier],
                "hazard_weight": weight, "osm_type": osm_type,
                "geometry": _make_box(lat, lon, r),
                "center_lat": lat, "center_lon": lon,
            })
    return gpd.GeoDataFrame(rows, crs="EPSG:4326") if rows else gpd.GeoDataFrame()


class OSMClient:
    """
    Queries OpenStreetMap via the Overpass API for industrial facilities.
    Assigns a 4-tier hazard weight to each polygon.
    Results are disk-cached as GeoJSON files.
    """

    def __init__(self, config: dict):
        self.cfg         = config.get("osm", {})
        self.overpass    = self.cfg.get("overpass_url", OVERPASS_URL)
        self.cache_dir   = Path(self.cfg.get("cache_dir", "data/osm"))
        self.cache_ttl   = int(self.cfg.get("cache_ttl_hours", 24)) * 3600
        self.use_mock    = __import__("os").getenv("USE_MOCK_DATA", "true").lower() == "true"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hazard_tiers = HAZARD_TIERS

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_facilities(self, bbox: str) -> gpd.GeoDataFrame:
        """
        Return a GeoDataFrame of industrial facility polygons for the given bbox,
        each annotated with osm_tier (1–4) and hazard_weight (0.1–1.0).
        """
        cache_key  = bbox.replace(",", "_")
        cache_file = self.cache_dir / f"osm_{cache_key}.geojson"

        # Return cached version if still fresh
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_ttl:
                logger.info(f"OSM: Loading from cache ({age/3600:.1f}h old)")
                return gpd.read_file(cache_file)

        if self.use_mock:
            gdf = _generate_mock_osm(bbox)
            if gdf.empty:
                logger.info(f"OSM: No mock polygons for bbox={bbox}; querying live Overpass API...")
                gdf = self._query_overpass(bbox)
            else:
                logger.info(f"OSM running in MOCK mode — loaded {len(gdf)} synthetic polygons.")
        else:
            gdf = self._query_overpass(bbox)

        if not gdf.empty:
            gdf.to_file(cache_file, driver="GeoJSON")
        return gdf

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_query(self, bbox: str) -> str:
        """Build an Overpass QL query covering all hazard tiers."""
        west, south, east, north = bbox.split(",")
        ov_bbox = f"{south},{west},{north},{east}"  # Overpass uses S,W,N,E order

        tag_filters = []
        for tier in self.hazard_tiers:
            for key, val in tier["tags"]:
                tag_filters.append(f'  way["{key}"="{val}"]({ov_bbox});')
                tag_filters.append(f'  relation["{key}"="{val}"]({ov_bbox});')

        tags_block = "\n".join(tag_filters)
        return f"""
[out:json][timeout:60];
(
{tags_block}
);
out geom;
"""

    def _query_overpass(self, bbox: str) -> gpd.GeoDataFrame:
        query = self._build_query(bbox)
        try:
            resp = requests.post(
                self.overpass,
                data={"data": query},
                timeout=90,
                headers={"User-Agent": "IndustrialFireDetector/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Overpass API error: {e}")
            return gpd.GeoDataFrame()

        return self._parse_overpass(data)

    def _parse_overpass(self, data: dict) -> gpd.GeoDataFrame:
        rows = []
        for elem in data.get("elements", []):
            try:
                geom = self._elem_to_shape(elem)
                if geom is None:
                    continue
                tags       = elem.get("tags", {})
                tier, weight, label = self._classify_tags(tags)
                osm_type   = self._describe_tags(tags)
                center     = geom.centroid
                rows.append({
                    "name":         tags.get("name", tags.get("operator", "Unknown")),
                    "osm_tier":     tier,
                    "hazard_label": label,
                    "hazard_weight": weight,
                    "osm_type":     osm_type,
                    "geometry":     geom,
                    "center_lat":   center.y,
                    "center_lon":   center.x,
                })
            except Exception:
                continue

        if not rows:
            return gpd.GeoDataFrame()
        return gpd.GeoDataFrame(rows, crs="EPSG:4326")

    def _elem_to_shape(self, elem: dict):
        if "geometry" not in elem:
            return None
        coords = [(n["lon"], n["lat"]) for n in elem["geometry"] if "lat" in n and "lon" in n]
        if len(coords) >= 3:
            try:
                return Polygon(coords)
            except Exception:
                return None
        return None

    def _classify_tags(self, tags: dict):
        for tier_def in self.hazard_tiers:
            for key, val in tier_def["tags"]:
                if tags.get(key) == val:
                    return tier_def["tier"], tier_def["weight"], tier_def["label"]
        return 3, 0.4, "MODERATE"

    def _describe_tags(self, tags: dict) -> str:
        relevant = ["industrial", "man_made", "power", "landuse"]
        parts = [f"{k}={tags[k]}" for k in relevant if k in tags]
        return "; ".join(parts) or "unknown"
