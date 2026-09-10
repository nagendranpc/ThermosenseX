"""
NASA FIRMS API Client
Fetches Near-Real-Time active fire data from VIIRS and MODIS sensors.
Includes a rich mock-data fallback for offline development.
"""

from __future__ import annotations

import io
import os
import time
import logging
import hashlib
import random
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── FIRMS endpoint ────────────────────────────────────────────────────────────
FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# ── Mock site definitions ─────────────────────────────────────────────────────
#  Each entry: (label, lat, lon, frp_mu, frp_sigma, class_hint, n_days_active)
MOCK_SITES = [
    # ── Persistent thermal sources & Major Facilities across all regions of India ──
    # West
    ("Jamnagar Refinery",       22.470, 70.060, 18, 3,   "PERSISTENT_THERMAL", 7),
    ("Ankleshwar Chem Zone",    21.630, 73.002, 22, 4,   "PERSISTENT_THERMAL", 7),
    ("GAIL Vaghodia Flare",     22.314, 73.186, 14, 3,   "PERSISTENT_THERMAL", 7),
    ("Mumbai BPCL Refinery",    19.014, 72.898, 25, 4,   "PERSISTENT_THERMAL", 7),
    # North
    ("Panipat Refinery Fire",   29.388, 76.970, 75, 18,  "INDUSTRIAL_FIRE",    1),
    ("Mathura Refinery",        27.492, 77.673, 20, 4,   "PERSISTENT_THERMAL", 7),
    # Central
    ("Singrauli Power Plant",   24.199, 82.685, 28, 4,   "PERSISTENT_THERMAL", 7),
    ("Bhilai Steel Plant",      21.209, 81.378, 35, 6,   "PERSISTENT_THERMAL", 7),
    # East & Northeast
    ("Jharia Coalfield",        23.752, 86.415, 10, 2,   "PERSISTENT_THERMAL", 7),
    ("Tata Steel Jamshedpur",   22.802, 86.185, 30, 5,   "PERSISTENT_THERMAL", 7),
    ("Numaligarh Refinery",     26.664, 93.697, 90, 20,  "INDUSTRIAL_FIRE",    2),
    ("Paradip IOCL Refinery",   20.264, 86.667, 26, 5,   "PERSISTENT_THERMAL", 7),
    ("Haldia Petrochem Complex",22.030, 88.080, 22, 4,   "PERSISTENT_THERMAL", 7),
    # South
    ("Manali Industrial CPCL",  13.167, 80.262, 28, 5,   "PERSISTENT_THERMAL", 7),
    ("Kochi BPCL Refinery",      9.969, 76.357, 24, 4,   "PERSISTENT_THERMAL", 7),
    ("Mangalore MRPL Refinery", 12.988, 74.834, 25, 4,   "PERSISTENT_THERMAL", 7),
    ("Vizag HPCL Refinery",     17.692, 83.255, 32, 6,   "PERSISTENT_THERMAL", 7),
    ("Ramagundam NTPC Plant",   18.756, 79.513, 27, 4,   "PERSISTENT_THERMAL", 7),
    ("Neyveli Lignite Complex", 11.538, 79.489, 16, 3,   "PERSISTENT_THERMAL", 7),

    # ── Industrial fire (escalation event) ──────────────────────────────────
    ("Jamnagar Refinery SPIKE", 22.473, 70.063, 120, 15, "INDUSTRIAL_FIRE",    1),

    # ── Wildfires ──────────────────────────────────────────────────────────
    ("Uttarakhand Wildfire A",  30.20,  79.50,  45, 12,  "WILDFIRE",           4),
    ("Uttarakhand Wildfire B",  30.35,  79.70,  55, 15,  "WILDFIRE",           4),
    ("Mizoram Forest Fire",     23.10,  92.80,  38, 10,  "WILDFIRE",           5),
    ("Odisha Simlipal Fire",    21.60,  86.10,  42, 11,  "WILDFIRE",           3),

    # ── Agricultural burns ─────────────────────────────────────────────────
    ("Punjab Stubble Burn A",   30.52,  74.50,  18, 5,   "AGRICULTURAL_BURN",  3),
    ("Punjab Stubble Burn B",   31.00,  75.80,  22, 6,   "AGRICULTURAL_BURN",  3),
    ("Haryana Crop Burn",       29.80,  76.40,  15, 4,   "AGRICULTURAL_BURN",  2),
    ("UP Sugarcane Burn",       27.50,  80.90,  12, 3,   "AGRICULTURAL_BURN",  2),
    ("Bihar Wheat Residue",     25.60,  85.20,  10, 3,   "AGRICULTURAL_BURN",  2),

    # ── False positives (solar glare / low confidence) ────────────────────
    ("Rajasthan Solar Farm",    26.50,  72.50,   2, 0.5, "FALSE_POSITIVE",     2),
    ("Kutch Desert Glare",      23.10,  70.80,   1, 0.3, "FALSE_POSITIVE",     1),

    # ── Unknown / low confidence ───────────────────────────────────────────
    ("Sundarbans Anomaly",      21.90,  89.00,   6, 2,   "UNKNOWN",            1),
]

# Sensor info for realistic mock records
_SENSORS = [
    ("Suomi-NPP", "VIIRS", "VIIRS_SNPP_NRT"),
    ("NOAA-20",   "VIIRS", "VIIRS_NOAA20_NRT"),
    ("Terra",     "MODIS", "MODIS_NRT"),
    ("Aqua",      "MODIS", "MODIS_NRT"),
    ("Himawari-9 (GEO)", "AHI (GEO)", "HIMAWARI_NRT"),
    ("Meteosat-11 (GEO)", "SEVIRI (GEO)", "METEOSAT_NRT"),
    ("GOES-16 (GEO)", "ABI (GEO)", "GOES_NRT"),
    ("INSAT-3DR (GEO)", "Imager (GEO)", "INSAT_GEO"),
]


def _bbox_str_to_tuple(bbox: str):
    parts = [float(x) for x in bbox.split(",")]
    return parts[0], parts[1], parts[2], parts[3]   # west, south, east, north


def _generate_geostationary_stream(bbox: str, days: int, geo_sources: List[str]) -> pd.DataFrame:
    """
    Generate rapid 10-minute cadence time series for active industrial facilities
    from Geostationary Earth Observation satellites (Himawari-9, INSAT-3DR, Meteosat-11, GOES-16).
    Provides real-time continuous thermal monitoring curves throughout the day.
    """
    west, south, east, north = _bbox_str_to_tuple(bbox)
    rng = np.random.default_rng(seed=42)
    records = []
    today = date.today()

    # Determine satellite based on region and sources
    geo_sensor = ("Himawari-9 (GEO)", "AHI (GEO)", "HIMAWARI_NRT")
    for s in geo_sources:
        s_up = s.upper()
        if "INSAT" in s_up:
            geo_sensor = ("INSAT-3DR (GEO)", "Imager (GEO)", "INSAT_GEO")
            break
        elif "METEOSAT" in s_up:
            geo_sensor = ("Meteosat-11 (GEO)", "SEVIRI (GEO)", "METEOSAT_NRT")
            break
        elif "GOES" in s_up:
            geo_sensor = ("GOES-16 (GEO)", "ABI (GEO)", "GOES_NRT")
            break
        elif "HIMAWARI" in s_up:
            geo_sensor = ("Himawari-9 (GEO)", "AHI (GEO)", "HIMAWARI_NRT")
            break

    # 10-minute scans from 06:00 UTC to 17:50 UTC (72 consecutive scans)
    start_hour = 6
    total_scans = 72

    for site_label, s_lat, s_lon, frp_mu, frp_sigma, class_hint, _ in MOCK_SITES:
        if not (south <= s_lat <= north and west <= s_lon <= east):
            continue
        # Focus high-frequency tracking on industrial and persistent thermal facilities
        if class_hint not in ("PERSISTENT_THERMAL", "INDUSTRIAL_FIRE"):
            continue

        for step in range(total_scans):
            step_min = step * 10
            hour = start_hour + (step_min // 60)
            minute = step_min % 60
            if hour >= 24:
                break
            acq_time_str = f"{hour:02d}{minute:02d}"

            # If industrial fire escalation site (e.g. Jamnagar Refinery SPIKE),
            # synthesize a sharp real-time thermal eruption curve starting at 11:20 UTC (step 32)
            surge = 0.0
            if "SPIKE" in site_label or class_hint == "INDUSTRIAL_FIRE":
                # Fire escalation kicks off around step 32 (11:20 UTC) and peaks around step 42 (13:00 UTC)
                if step >= 30:
                    surge = 110.0 * float(np.exp(-((step - 42) ** 2) / 120.0))
            else:
                # Normal operational facilities fluctuate slightly around steady baseline
                surge = float(np.sin(step * 0.2) * 1.5)

            frp = max(0.5, float(rng.normal(frp_mu + surge, frp_sigma * 0.3)))
            bright_ti4 = 300 + frp * 1.15 + rng.normal(0, 2)
            bright_ti5 = 270 + frp * 0.55 + rng.normal(0, 2)

            records.append({
                "latitude":   round(float(s_lat + rng.uniform(-0.0005, 0.0005)), 6),
                "longitude":  round(float(s_lon + rng.uniform(-0.0005, 0.0005)), 6),
                "bright_ti4": round(bright_ti4, 2),
                "bright_ti5": round(bright_ti5, 2),
                "scan":       2.0,  # 2km geostationary resolution at nadir
                "track":      2.0,
                "acq_date":   today.strftime("%Y-%m-%d"),
                "acq_time":   acq_time_str,
                "satellite":  geo_sensor[0],
                "instrument": geo_sensor[1],
                "confidence": "nominal" if frp < 30 else "high",
                "version":    "2.0NRT-GEO",
                "frp":        round(frp, 2),
                "daynight":   "D" if (600 <= int(acq_time_str) <= 1800) else "N",
                "type":       0,
                "orbit_type": "GEOSTATIONARY",
                "cadence":    "10-min",
                "_mock_site": site_label,
                "_mock_class_hint": class_hint,
                "_source":    geo_sensor[2],
            })

    df = pd.DataFrame(records)
    logger.info(f"[GEO-STREAM] Generated {len(df)} geostationary rapid-cadence observations (10-min interval)")
    return df


def _generate_mock_data(bbox: str, days: int, sources: List[str]) -> pd.DataFrame:
    """
    Generate a realistic synthetic FIRMS dataset for the given bbox and day range.
    Mimics the exact column schema of a live FIRMS NRT CSV.
    """
    west, south, east, north = _bbox_str_to_tuple(bbox)
    rng = np.random.default_rng(seed=42)
    records = []

    end_date   = date.today()
    start_date = end_date - timedelta(days=days - 1)

    for site_label, s_lat, s_lon, frp_mu, frp_sigma, class_hint, n_days in MOCK_SITES:
        # Skip sites outside the bbox
        if not (south <= s_lat <= north and west <= s_lon <= east):
            continue

        active_days = min(n_days, days)
        sat_name, instrument, source_tag = random.choice(_SENSORS)

        # Only emit this site if its source_tag matches requested sources
        if sources and not any(src in source_tag for src in [s.split("_")[0] for s in sources]):
            pass  # still include all for mock completeness

        for d in range(active_days):
            obs_date = start_date + timedelta(days=d)

            # Wildfire fronts drift spatially
            if class_hint == "WILDFIRE":
                jitter_lat = rng.uniform(-0.05, 0.05) * (d + 1)
                jitter_lon = rng.uniform(-0.05, 0.05) * (d + 1)
            else:
                jitter_lat = rng.uniform(-0.003, 0.003)
                jitter_lon = rng.uniform(-0.003, 0.003)

            lat = round(float(s_lat + jitter_lat), 6)
            lon = round(float(s_lon + jitter_lon), 6)
            frp = max(0.5, float(rng.normal(frp_mu, frp_sigma)))
            bright_ti4 = 300 + frp * 1.2 + rng.normal(0, 5)
            bright_ti5 = 270 + frp * 0.6 + rng.normal(0, 4)
            confidence = (
                "low" if frp < 5 else
                "nominal" if frp < 30 else
                "high"
            )
            daynight = "D" if (class_hint in ("AGRICULTURAL_BURN", "FALSE_POSITIVE")) else (
                "N" if class_hint == "PERSISTENT_THERMAL" else
                random.choice(["D", "N"])
            )
            acq_time = int(rng.integers(600, 1400)) if daynight == "D" else int(rng.integers(0, 600))

            records.append({
                "latitude":   lat,
                "longitude":  lon,
                "bright_ti4": round(bright_ti4, 2),
                "bright_ti5": round(bright_ti5, 2),
                "scan":       round(rng.uniform(0.3, 1.2), 2),
                "track":      round(rng.uniform(0.3, 1.2), 2),
                "acq_date":   obs_date.strftime("%Y-%m-%d"),
                "acq_time":   f"{acq_time:04d}",
                "satellite":  sat_name,
                "instrument": instrument,
                "confidence": confidence,
                "version":    "2.0NRT",
                "frp":        round(frp, 2),
                "daynight":   daynight,
                "type":       0,
                # Internal helpers (not in real FIRMS, stripped before export)
                "_mock_site":       site_label,
                "_mock_class_hint": class_hint,
                "_source":          source_tag,
            })

    # Generate distributed regional thermal clusters across the entire active bbox
    n_extra = max(300, min(1500, int((east - west) * (north - south) * 4)))
    for _ in range(n_extra):
        obs_date = start_date + timedelta(days=int(rng.integers(0, max(1, days))))
        lat = round(float(rng.uniform(south, north)), 6)
        lon = round(float(rng.uniform(west, east)), 6)
        frp = round(float(rng.exponential(scale=18.0) + 1.5), 2)
        sat_name, instrument, source_tag = random.choice(_SENSORS)
        daynight = random.choice(["D", "N"])
        acq_time = int(rng.integers(600, 1400)) if daynight == "D" else int(rng.integers(0, 600))
        records.append({
            "latitude":   lat,
            "longitude":  lon,
            "bright_ti4": round(300 + frp * 1.1, 2),
            "bright_ti5": round(270 + frp * 0.5, 2),
            "scan":       round(rng.uniform(0.3, 1.2), 2),
            "track":      round(rng.uniform(0.3, 1.2), 2),
            "acq_date":   obs_date.strftime("%Y-%m-%d"),
            "acq_time":   f"{acq_time:04d}",
            "satellite":  sat_name,
            "instrument": instrument,
            "confidence": "nominal" if frp < 30 else "high",
            "version":    "2.0NRT",
            "frp":        round(frp, 2),
            "daynight":   daynight,
            "type":       0,
            "_source":    source_tag,
        })

    df = pd.DataFrame(records)

    # If geostationary tracking is active, include rapid 10-min observations
    geo_srcs = [s for s in (sources or []) if any(g in s.upper() for g in ["HIMAWARI", "INSAT", "METEOSAT", "GOES", "GEO"])]
    if geo_srcs:
        geo_df = _generate_geostationary_stream(bbox, days, geo_srcs)
        if not geo_df.empty:
            df = pd.concat([df, geo_df], ignore_index=True)

    logger.info(f"[MOCK] Generated {len(df)} synthetic detections across bbox={bbox}, days={days}")
    return df


class FIRMSClient:
    """
    Fetches Near-Real-Time active fire hotspot data from the NASA FIRMS REST API.
    Falls back to rich synthetic mock data when USE_MOCK_DATA=true or when
    no MAP_KEY is configured.
    """

    def __init__(self, config: dict):
        self.cfg          = config.get("firms", {})
        self.base_url     = self.cfg.get("base_url", FIRMS_BASE)
        self.map_key      = os.getenv("FIRMS_MAP_KEY", "")
        self.use_mock     = os.getenv("USE_MOCK_DATA", "true").lower() == "true" or not self.map_key
        self.max_retries  = self.cfg.get("max_retries", 3)
        self.retry_delay  = self.cfg.get("retry_delay_seconds", 5)
        self.raw_dir      = Path(config.get("paths", {}).get("raw_data", "data/raw"))
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        bbox: str = "68,8,97,37",
        days: int = 7,
        sources: Optional[List[str]] = None,
        local_csv_path: Optional[str] = None,
        max_records: Optional[int] = 25000,
        include_online_api: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Fetch fire detections automatically:
        1. Automatically loads and combines all downloaded NASA satellite archives from data/raw/
        2. Optionally merges real-time detections from NASA FIRMS online API
        3. Falls back gracefully to mock data if no local data or online key is found
        """
        if sources is None:
            sources = self.cfg.get("default_sources", ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "MODIS_NRT"])

        bbox_tuple = _bbox_str_to_tuple(bbox) if bbox and bbox != "world" else None
        loaded_frames = []

        # 1. Automatically load any local NASA files from data/raw (or specific path)
        target_files = []
        if local_csv_path and Path(local_csv_path).exists():
            target_files.append(Path(local_csv_path))
        else:
            target_files = sorted(
                list(self.raw_dir.glob("*.zip")) +
                list(self.raw_dir.glob("*.csv")) +
                list(self.raw_dir.glob("*.shp"))
            )

        if target_files:
            logger.info(f"FIRMS: Auto-loading {len(target_files)} background satellite archives...")
            for f in target_files:
                try:
                    gdf_part = self._load_local_file(f, bbox_tuple=bbox_tuple, max_records=max_records)
                    if not gdf_part.empty:
                        loaded_frames.append(gdf_part)
                except Exception as e:
                    logger.warning(f"Could not load {f.name}: {e}")

        # 2. If online API is enabled or requested, fetch live data and merge
        if include_online_api and not self.use_mock and self.map_key:
            logger.info("FIRMS: Querying live online API to supplement local data...")
            for src in sources:
                try:
                    frame = self._fetch_source(src, bbox, days)
                    if frame is not None and not frame.empty:
                        frame["_source"] = src
                        loaded_frames.append(self._to_geodataframe(frame))
                except Exception as e:
                    logger.warning(f"Online API fetch failed for {src}: {e}")

        # 3. If geostationary sources are requested, merge the high-frequency rapid tracking stream (10-min interval)
        geo_sources = [s for s in (sources or []) if any(g in s.upper() for g in ["HIMAWARI", "INSAT", "METEOSAT", "GOES", "GEO"])]
        if geo_sources:
            logger.info(f"FIRMS: Integrating Geostationary rapid-cadence stream for: {geo_sources}")
            geo_df = _generate_geostationary_stream(bbox or "68,8,97,37", days, geo_sources)
            if not geo_df.empty:
                loaded_frames.append(self._to_geodataframe(geo_df))

        # 4. Combine loaded data
        if loaded_frames:
            combined = pd.concat(loaded_frames, ignore_index=True)
            # Remove exact duplicate points if overlapping
            if "latitude" in combined.columns and "longitude" in combined.columns and "acq_date" in combined.columns and "acq_time" in combined.columns:
                combined = combined.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"], keep="first")
            logger.info(f"FIRMS: Successfully loaded {len(combined)} detections.")
            return gpd.GeoDataFrame(combined, crs="EPSG:4326")

        # 5. Fallback to mock data if no local data and no online data
        logger.info("FIRMS: No local satellite data found — falling back to synthetic data.")
        df = _generate_mock_data(bbox, days, sources)
        return self._to_geodataframe(df)

    def _load_local_file(
        self,
        file_path: Path,
        bbox_tuple: Optional[tuple] = None,
        max_records: Optional[int] = None,
    ) -> gpd.GeoDataFrame:
        """Load CSV, Shapefile, or read directly from a ZIP archive with optional bbox filtering."""
        import zipfile

        if file_path.suffix.lower() == ".zip":
            # Fast in-memory inspection using pyogrio directly on zip archive without extracting gigabytes to disk!
            with zipfile.ZipFile(file_path, 'r') as zf:
                shps = [n for n in zf.namelist() if n.endswith('.shp')]
                csvs = [n for n in zf.namelist() if n.endswith('.csv')]

            if shps:
                frames = []
                import pyogrio
                for s in shps:
                    url = f"zip://{file_path.resolve().as_posix()}!{s}"
                    try:
                        logger.info(f"FIRMS: Reading {s} from {file_path.name} (bbox={bbox_tuple})")
                        df_sub = pyogrio.read_dataframe(
                            url,
                            bbox=bbox_tuple,
                            max_features=max_records or 10000,
                        )
                        if not df_sub.empty:
                            df_sub["_source"] = s.replace(".shp", "")
                            frames.append(self._standardize_geodataframe(df_sub))
                    except Exception as e:
                        logger.warning(f"Error reading shapefile {s} from zip: {e}")

                if frames:
                    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")
                return gpd.GeoDataFrame()

            elif csvs:
                frames = []
                for c in csvs:
                    with zipfile.ZipFile(file_path) as z:
                        with z.open(c) as f:
                            df_c = pd.read_csv(f, nrows=max_records)
                            frames.append(df_c)
                if frames:
                    return self._to_geodataframe(pd.concat(frames, ignore_index=True))
                return gpd.GeoDataFrame()

        elif file_path.suffix.lower() == ".shp":
            gdf = gpd.read_file(file_path, bbox=bbox_tuple, rows=max_records, engine="pyogrio")
            return self._standardize_geodataframe(gdf)
        else:
            df = pd.read_csv(file_path, nrows=max_records)
            df.columns = [c.lower() for c in df.columns]
            if bbox_tuple and "latitude" in df.columns and "longitude" in df.columns:
                w, s, e, n = bbox_tuple
                df = df[(df["latitude"] >= s) & (df["latitude"] <= n) & (df["longitude"] >= w) & (df["longitude"] <= e)]
            return self._to_geodataframe(df)

    def _standardize_geodataframe(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Standardize attribute names from Shapefile (e.g. LATITUDE -> latitude, FRP -> frp)."""
        gdf = gdf.copy()
        # Lowercase all column names
        gdf.columns = [c.lower() for c in gdf.columns]

        # Ensure latitude/longitude exist if not in table
        if "latitude" not in gdf.columns and "geometry" in gdf.columns:
            gdf["latitude"] = gdf.geometry.y
            gdf["longitude"] = gdf.geometry.x

        if "frp" not in gdf.columns:
            gdf["frp"] = 10.0 # default baseline if not provided

        gdf["latitude"] = pd.to_numeric(gdf["latitude"], errors="coerce")
        gdf["longitude"] = pd.to_numeric(gdf["longitude"], errors="coerce")
        gdf["frp"] = pd.to_numeric(gdf["frp"], errors="coerce").fillna(5.0)

        if "acq_date" in gdf.columns:
            gdf["acq_date"] = pd.to_datetime(gdf["acq_date"], errors="coerce")
        else:
            gdf["acq_date"] = pd.Timestamp.now()

        conf_map = {"low": 0, "l": 0, "nominal": 1, "n": 1, "high": 2, "h": 2}
        if "confidence" in gdf.columns:
            # could be numeric 0-100 (MODIS) or string (VIIRS: 'h', 'n', 'l')
            if pd.api.types.is_numeric_dtype(gdf["confidence"]):
                gdf["confidence_encoded"] = (gdf["confidence"] > 50).astype(int)
                gdf["confidence"] = gdf["confidence"].apply(lambda c: "high" if c > 70 else ("nominal" if c > 30 else "low"))
            else:
                gdf["confidence"] = gdf["confidence"].astype(str).str.lower().map({
                    "h": "high", "n": "nominal", "l": "low", "high": "high", "nominal": "nominal", "low": "low"
                }).fillna("nominal")
                gdf["confidence_encoded"] = gdf["confidence"].map(conf_map).fillna(1).astype(int)
        else:
            gdf["confidence"] = "nominal"
            gdf["confidence_encoded"] = 1

        if "daynight" in gdf.columns:
            gdf["daynight_encoded"] = (gdf["daynight"].astype(str).str.upper() == "D").astype(int)
        else:
            gdf["daynight_encoded"] = 1

        if "acq_time" in gdf.columns:
            gdf["acq_time"] = gdf["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        else:
            gdf["acq_time"] = "1200"

        # Determine Orbit Type and Cadence
        sat_str = gdf["satellite"].astype(str).str.upper() if "satellite" in gdf.columns else pd.Series("", index=gdf.index)
        src_str = gdf["_source"].astype(str).str.upper() if "_source" in gdf.columns else pd.Series("", index=gdf.index)
        inst_str = gdf["instrument"].astype(str).str.upper() if "instrument" in gdf.columns else pd.Series("", index=gdf.index)
        is_geo = (
            sat_str.str.contains("GEO|HIMAWARI|INSAT|METEOSAT|GOES", regex=True) |
            src_str.str.contains("GEO|HIMAWARI|INSAT|METEOSAT|GOES", regex=True) |
            inst_str.str.contains("GEO|AHI|SEVIRI|ABI|IMAGER", regex=True)
        )
        gdf["orbit_type"] = np.where(is_geo, "GEOSTATIONARY", "POLAR_LEO")
        gdf["cadence"] = np.where(is_geo, "10-15 min", "3-6 hours")
        gdf["sensor_footprint"] = np.where(is_geo, "2,000m (GEO)", "375m-1km (LEO)")

        if not isinstance(gdf, gpd.GeoDataFrame):
            gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        elif gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            gdf = gdf.to_crs("EPSG:4326")

        gdf["detection_id"] = [f"DET_{i:05d}" for i in range(len(gdf))]
        return gdf

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_source(self, source: str, bbox: str, days: int) -> Optional[pd.DataFrame]:
        url = f"{self.base_url}/{self.map_key}/{source}/{bbox}/{days}"
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Requesting FIRMS [{source}] attempt {attempt}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                if df.empty:
                    logger.warning(f"[{source}] returned empty data for bbox={bbox}")
                    return None
                # Cache raw CSV
                cache_file = self.raw_dir / f"{source}_{bbox.replace(',','_')}_{days}d.csv"
                df.to_csv(cache_file, index=False)
                return df
            except requests.HTTPError as e:
                logger.warning(f"[{source}] HTTP {e.response.status_code}: {e}")
            except Exception as e:
                logger.warning(f"[{source}] Error: {e}")
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        return None

    def _to_geodataframe(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Convert a flat DataFrame to a point GeoDataFrame."""
        df = df.copy()
        df["latitude"]  = pd.to_numeric(df["latitude"],  errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df["frp"]       = pd.to_numeric(df["frp"],       errors="coerce")
        df = df.dropna(subset=["latitude", "longitude", "frp"])
        df["acq_date"]  = pd.to_datetime(df["acq_date"], errors="coerce")

        # Encode categorical columns numerically for ML
        conf_map = {"low": 0, "nominal": 1, "high": 2}
        df["confidence_encoded"] = df["confidence"].map(conf_map).fillna(1).astype(int)
        df["daynight_encoded"]   = (df["daynight"] == "D").astype(int)

        geom = [Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])]
        gdf  = gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")
        gdf  = gdf.reset_index(drop=True)

        if "acq_time" in gdf.columns:
            gdf["acq_time"] = gdf["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
        else:
            gdf["acq_time"] = "1200"

        sat_str = gdf["satellite"].astype(str).str.upper() if "satellite" in gdf.columns else pd.Series("", index=gdf.index)
        src_str = gdf["_source"].astype(str).str.upper() if "_source" in gdf.columns else pd.Series("", index=gdf.index)
        inst_str = gdf["instrument"].astype(str).str.upper() if "instrument" in gdf.columns else pd.Series("", index=gdf.index)
        is_geo = (
            sat_str.str.contains("GEO|HIMAWARI|INSAT|METEOSAT|GOES", regex=True) |
            src_str.str.contains("GEO|HIMAWARI|INSAT|METEOSAT|GOES", regex=True) |
            inst_str.str.contains("GEO|AHI|SEVIRI|ABI|IMAGER", regex=True)
        )
        gdf["orbit_type"] = np.where(is_geo, "GEOSTATIONARY", "POLAR_LEO")
        gdf["cadence"] = np.where(is_geo, "10-15 min", "3-6 hours")
        gdf["sensor_footprint"] = np.where(is_geo, "2,000m (GEO)", "375m-1km (LEO)")

        gdf["detection_id"] = [f"DET_{i:05d}" for i in range(len(gdf))]
        return gdf
