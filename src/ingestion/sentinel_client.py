"""
Sentinel-2 / Landsat Optical Verification Client
Uses the Copernicus Data Space Ecosystem (CDSE) to fetch SWIR/NIR bands
for optical verification of INDUSTRIAL_FIRE detections.
Gracefully degrades to stub results when credentials are absent.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try importing sentinelhub; mark as unavailable if not installed
try:
    from sentinelhub import (
        SHConfig, SentinelHubCatalog, DataCollection,
        CatalogSearchIterator, BBox, CRS, MimeType,
        SentinelHubRequest, bbox_to_dimensions,
    )
    _SH_AVAILABLE = True
except ImportError:
    _SH_AVAILABLE = False
    logger.warning(
        "sentinelhub library not installed. "
        "Optical verification will return stub results. "
        "Run: pip install sentinelhub"
    )


def _stub_result(lat: float, lon: float, reason: str = "unavailable") -> dict:
    """Return a stub optical verification result."""
    return {
        "verified":         False,
        "source":           None,
        "scene_date":       None,
        "nbr_value":        None,
        "swir_peak_lat":    lat,
        "swir_peak_lon":    lon,
        "cloud_cover_pct":  None,
        "optical_reason":   reason,
    }


class SentinelClient:
    """
    Fetches Sentinel-2 L2A SWIR/NIR bands around a fire detection point
    and computes the Normalised Burn Ratio (NBR) to confirm active burning.

    Falls back to stub results if:
      - sentinelhub library not installed
      - CDSE credentials not configured
      - optical verification disabled in config
    """

    def __init__(self, config: dict):
        self.cfg         = config.get("optical", {})
        self.enabled     = self.cfg.get("enabled", True)
        self.cloud_max   = float(self.cfg.get("cloud_cover_max_pct", 30))
        self.window_days = int(self.cfg.get("scene_window_days", 3))
        self.nbr_thresh  = float(self.cfg.get("nbr_fire_threshold", -0.20))
        self.cache_dir   = Path(config.get("paths", {}).get("sentinel_cache", "data/sentinel"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._sh_config: Optional[object] = None
        if _SH_AVAILABLE and self.enabled:
            self._sh_config = self._init_sh_config()

    # ── Public API ─────────────────────────────────────────────────────────────

    def verify(
        self,
        lat: float,
        lon: float,
        detection_date: str,
        radius_m: float = 500,
    ) -> dict:
        """
        Attempt optical verification for a point (lat, lon) on a given date.

        Returns a dict with keys:
          verified, source, scene_date, nbr_value, swir_peak_lat, swir_peak_lon,
          cloud_cover_pct, optical_reason
        """
        if not self.enabled:
            return _stub_result(lat, lon, reason="disabled_in_config")
        if not _SH_AVAILABLE or self._sh_config is None:
            return _stub_result(lat, lon, reason="sentinelhub_unavailable")

        try:
            return self._fetch_and_analyze(lat, lon, detection_date, radius_m)
        except Exception as e:
            logger.warning(f"Sentinel verification failed for ({lat},{lon}): {e}")
            return _stub_result(lat, lon, reason=f"error: {e}")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _init_sh_config(self) -> Optional[object]:
        username = os.getenv("CDSE_USERNAME", "")
        password = os.getenv("CDSE_PASSWORD", "")
        if not username or not password:
            logger.info("CDSE credentials not set — optical verification stubbed.")
            return None
        try:
            config            = SHConfig()
            config.sh_client_id       = username
            config.sh_client_secret   = password
            config.sh_base_url        = "https://sh.dataspace.copernicus.eu"
            config.sh_token_url       = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            return config
        except Exception as e:
            logger.warning(f"SHConfig init failed: {e}")
            return None

    def _fetch_and_analyze(
        self, lat: float, lon: float,
        detection_date: str, radius_m: float
    ) -> dict:
        det_dt    = datetime.strptime(detection_date, "%Y-%m-%d")
        date_from = det_dt - timedelta(days=self.window_days)
        date_to   = det_dt + timedelta(days=self.window_days)

        # Bounding box ~radius_m metres around the point
        deg_offset = radius_m / 111_320
        bbox = BBox(
            bbox=[lon - deg_offset, lat - deg_offset,
                  lon + deg_offset, lat + deg_offset],
            crs=CRS.WGS84,
        )

        # Search catalog
        catalog = SentinelHubCatalog(config=self._sh_config)
        results = list(catalog.search(
            DataCollection.SENTINEL2_L2A,
            bbox=bbox,
            time=(date_from, date_to),
            filter=f"eo:cloud_cover < {self.cloud_max}",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]},
        ))

        if not results:
            return _stub_result(lat, lon, reason="no_scene_available")

        # Pick closest scene
        best = min(
            results,
            key=lambda r: abs((datetime.fromisoformat(r["properties"]["datetime"][:10]) - det_dt).days)
        )
        scene_date   = best["properties"]["datetime"][:10]
        cloud_cover  = best["properties"].get("eo:cloud_cover", 99)

        # Download B08 (NIR, 10m), B11 (SWIR, 20m) — request at 20m
        size  = bbox_to_dimensions(bbox, resolution=20)
        evalscript = """
//VERSION=3
function setup() {
  return { input: ["B08","B11","B12","SCL"], output: {bands:3, sampleType:"FLOAT32"} };
}
function evaluatePixel(s) {
  return [s.B08, s.B11, s.B12];
}
"""
        req = SentinelHubRequest(
            evalscript    = evalscript,
            input_data    = [SentinelHubRequest.input_data(
                data_collection = DataCollection.SENTINEL2_L2A,
                time_interval   = (scene_date, scene_date),
            )],
            responses     = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox          = bbox,
            size          = size,
            config        = self._sh_config,
        )
        data = req.get_data()[0]   # shape: (H, W, 3)  bands: B08, B11, B12

        nir  = data[:, :, 0]
        swir = data[:, :, 1]
        nbr  = np.where(
            (nir + swir) > 0,
            (nir - swir) / (nir + swir),
            0.0
        )

        # Locate hottest SWIR pixel
        peak_idx    = np.unravel_index(np.argmax(swir), swir.shape)
        h, w        = swir.shape
        deg_per_px  = deg_offset * 2 / max(h, w)
        peak_lat    = (lat + deg_offset) - peak_idx[0] * deg_per_px
        peak_lon    = (lon - deg_offset) + peak_idx[1] * deg_per_px

        nbr_val     = float(np.mean(nbr))
        verified    = nbr_val < self.nbr_thresh

        return {
            "verified":        verified,
            "source":          "Sentinel-2 B08/B11/B12",
            "scene_date":      scene_date,
            "nbr_value":       round(nbr_val, 4),
            "swir_peak_lat":   round(peak_lat, 6),
            "swir_peak_lon":   round(peak_lon, 6),
            "cloud_cover_pct": round(cloud_cover, 1),
            "optical_reason":  "confirmed" if verified else "nbr_above_threshold",
        }
