# 🔥 ThermoSense-X — AI Industrial Thermal Anomaly & Risk Prioritization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NASA FIRMS](https://img.shields.io/badge/satellite-NASA%20FIRMS-orange.svg)](https://firms.modaps.eosdis.nasa.gov/)
[![OpenStreetMap](https://img.shields.io/badge/GIS-OpenStreetMap%20Overpass-green.svg)](https://www.openstreetmap.org/)

> **“ThermoSense-X learns the normal thermal fingerprint of each industrial facility and detects, evaluates, and prioritizes abnormal thermal behaviour using satellite and geospatial data.”**

---

### ❓ The Problem & Core Innovation
Satellite systems like **NASA FIRMS** detect thermal hotspots across the globe, but a hotspot does not always mean a fire disaster. It could be an authorized refinery flare, steel furnace, or normal manufacturing heat.

Instead of asking: **“Is there heat here?”**  
ThermoSense-X asks: **“Is this heat unusual for this particular facility?”**

---

## 🏗️ Proposed Thermal Risk Score (0–100) Architecture

| Dimension | Weight | Metric & Description |
|---|---|---|
| **1. Thermal Deviation** | **30%** | $Z$-score departure from learned facility baseline FRP |
| **2. Temporal Persistence** | **20%** | Consecutive days of sustained abnormal thermal emission |
| **3. Thermal Intensity** | **20%** | Fire Radiative Power (MW) absolute energy output |
| **4. Spatial Expansion** | **15%** | Cluster spatial variance growth (distinguishing stationary flares from expanding fronts) |
| **5. Facility Proximity** | **10%** | Sensor-adaptive dynamic buffer distance to OSM industrial boundary |
| **6. Environmental Context** | **5%** | Proximity to hazardous infrastructure, pipelines, and fuel storage |

---

## Features

| Feature | Description |
|---|---|
| **Persistent-Site Escalation** | FRP z-score spike detection auto-escalates gas flares to INDUSTRIAL_FIRE |
| **Thermal Fingerprinting** | 6-metric stability index separates stacks from moving wildfire fronts |
| **Sentinel-2 Verification** | SWIR B11/B12 optical confirmation of active fires at 20m resolution |
| **Solar False-Positive Filter** | Removes daytime solar glare from metallic roofs/solar farms |
| **OSM Hazard Matrix** | 4-tier industrial risk weights (refinery → quarry) |
| **Dynamic Pixel Buffers** | VIIRS=375m, MODIS=1km sensor-adaptive spatial joins |
| **ML Ensemble Classifier** | Random Forest + XGBoost trained on rule-based labels |
| **Interactive Dashboard** | Streamlit UI with Folium dark map + Plotly analytics |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for Windows:** If `python` is not on PATH, use the full path:
> `C:\Users\<user>\.local\bin\python3.14.exe -m pip install -r requirements.txt`

### 2. Configure API keys (optional — works with mock data)

```bash
copy .env.example .env
# Edit .env and add your FIRMS_MAP_KEY and CDSE credentials
```

> Without keys, the system runs in **mock demo mode** with realistic synthetic data for India.

### 3. Launch the dashboard

```bash
streamlit run app.py
```

### 4. Or use the CLI

```bash
# Analyse India, last 7 days
python main.py --region india --days 7

# Custom bounding box
python main.py --bbox "86,22,87,24" --days 3

# Middle East, tighter escalation threshold
python main.py --region middle_east --esc-zscore 2.5 --days 5
```

---

## Project Structure

```
industrial_fire_detection/
├── app.py                          # Streamlit dashboard (Perfect UI)
├── main.py                         # CLI entry point
├── config/config.yaml             # All tunable parameters
├── src/
│   ├── pipeline.py                # 7-stage orchestrator
│   ├── ingestion/
│   │   ├── firms_client.py        # NASA FIRMS API + mock data
│   │   ├── osm_client.py          # Overpass API + hazard matrix
│   │   └── sentinel_client.py    # Copernicus CDSE (optional)
│   ├── preprocessing/
│   │   ├── clustering.py          # DBSCAN haversine clustering
│   │   ├── dynamic_buffer.py      # Sensor-adaptive spatial join
│   │   ├── thermal_fingerprint.py # Variance index + stability
│   │   └── feature_engineering.py # 19-feature ML matrix
│   ├── classification/
│   │   ├── false_positive_filter.py  # Stage 0: Solar FP filter
│   │   ├── escalation_engine.py      # Stage 1: FRP spike detection
│   │   ├── rule_based.py             # Stage 2: Heuristic rules
│   │   ├── ml_classifier.py          # Stage 3: RF + XGBoost
│   │   └── optical_verifier.py      # Stage 4: S2 SWIR check
│   ├── store/
│   │   └── site_history_store.py  # SQLite rolling FRP history
│   └── visualization/
│       ├── map_builder.py          # Folium dark map
│       └── report.py              # Plotly charts + stats
└── data/
    ├── raw/                        # Cached FIRMS CSVs
    ├── osm/                        # Cached OSM polygons
    └── processed/
        └── site_history/          # SQLite history DB
```

---

## Classification Logic (7-Stage Pipeline)

```
┌── Stage 0 ──► Solar False-Positive Filter
│                (daytime + low FRP + high SWIR reflectance → FALSE_POSITIVE)
│
├── Stage 1 ──► Escalation Engine
│                (FRP z-score ≥ 3σ vs. 14-day baseline → INDUSTRIAL_FIRE ⚡)
│
├── Stage 2 ──► Rule-Based Classifier
│                (FRP thresholds + OSM overlap + persistence + stability)
│
├── Stage 3 ──► Random Forest + XGBoost
│                (19 features → per-class probabilities)
│
├── Stage 4 ──► [Optional] Sentinel-2 Optical Verification
│                (NBR < -0.20 confirms active burn at 20m resolution)
│
└── Final ───► Class resolution (ML when >70% conf, else rules; escalation always wins)
```

## Classification Classes

| Class | Colour | Criteria |
|---|---|---|
| `INDUSTRIAL_FIRE` | 🔴 | Acute fire at industrial site, OR escalated persistent source |
| `PERSISTENT_THERMAL` | 🟠 | Recurring heat at industrial site (gas flares, furnaces) |
| `WILDFIRE` | 🌿 | High FRP, transient, no industrial context |
| `AGRICULTURAL_BURN` | 🌾 | Daytime, moderate FRP, rural area |
| `FALSE_POSITIVE` | 🚫 | Solar glare, metallic roof, desert reflectance |
| `UNKNOWN` | ⚪ | Low confidence, insufficient context |

---

## API Keys

| Service | URL | Required for |
|---|---|---|
| NASA FIRMS MAP_KEY | https://firms.modaps.eosdis.nasa.gov/api/map_key/ | Live satellite data |
| Copernicus CDSE | https://dataspace.copernicus.eu/ | Sentinel-2 optical verification |

Both are **free**. Without them, the system uses synthetic demo data.

---

## Configuration

All thresholds are tunable in [`config/config.yaml`](config/config.yaml):

- Escalation z-score threshold (default: 3.0σ)
- Escalation rolling window (default: 14 days)
- Sensor buffer radii (VIIRS: 375m, MODIS: 1000m)
- DBSCAN eps (default: 750m)
- OSM hazard tier weights
- Solar false-positive FRP ceiling (default: 5 MW)
