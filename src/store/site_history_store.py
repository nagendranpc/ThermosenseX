"""
Site History Store
SQLite-backed persistent store for per-geohash FRP time-series data.
Used by the Escalation Engine to compute rolling baselines.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS site_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    geohash     TEXT    NOT NULL,
    obs_date    TEXT    NOT NULL,
    frp         REAL    NOT NULL,
    latitude    REAL,
    longitude   REAL,
    satellite   TEXT,
    inserted_at TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_gh_date ON site_history (geohash, obs_date);
"""


class SiteHistoryStore:
    """
    Lightweight SQLite wrapper that persists per-site (geohash) FRP observations
    for rolling-window escalation analysis.
    """

    def __init__(self, db_path: str | Path = "data/processed/site_history/history.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Context manager ────────────────────────────────────────────────────────
    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ── Init ───────────────────────────────────────────────────────────────────
    def _init_db(self) -> None:
        with self._conn() as con:
            con.executescript(CREATE_TABLE_SQL)
        logger.debug(f"SiteHistoryStore initialised at {self.db_path}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def upsert_detections(self, gdf) -> None:
        """
        Persist all detections from an enriched GeoDataFrame into the history store.
        Requires columns: geohash, acq_date, frp, latitude, longitude.
        """
        if gdf is None or gdf.empty:
            return

        required = {"geohash", "frp", "latitude", "longitude"}
        missing  = required - set(gdf.columns)
        if missing:
            logger.warning(f"SiteHistoryStore.upsert: missing columns {missing}")
            return

        rows = []
        for _, row in gdf.iterrows():
            obs_date = (
                row["acq_date"].strftime("%Y-%m-%d")
                if hasattr(row.get("acq_date"), "strftime")
                else str(row.get("acq_date", date.today()))[:10]
            )
            rows.append((
                str(row["geohash"]),
                obs_date,
                float(row["frp"]),
                float(row["latitude"]),
                float(row["longitude"]),
                str(row.get("satellite", "unknown")),
            ))

        with self._conn() as con:
            con.executemany(
                "INSERT OR IGNORE INTO site_history "
                "(geohash, obs_date, frp, latitude, longitude, satellite) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.info(f"SiteHistoryStore: upserted {len(rows)} records.")

    def get_frp_history(
        self, geohash: str, days: int = 14
    ) -> Optional[pd.Series]:
        """
        Return a Series of daily mean FRP values for the given geohash
        over the past `days` days.
        """
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        with self._conn() as con:
            rows = con.execute(
                "SELECT obs_date, AVG(frp) as mean_frp "
                "FROM site_history "
                "WHERE geohash=? AND obs_date>=? "
                "GROUP BY obs_date ORDER BY obs_date",
                (geohash, cutoff),
            ).fetchall()

        if not rows:
            return None
        s = pd.Series(
            {r["obs_date"]: r["mean_frp"] for r in rows},
            name="frp",
        )
        return s

    def get_all_sites(self) -> pd.DataFrame:
        """Return a summary of all tracked sites."""
        with self._conn() as con:
            rows = con.execute(
                "SELECT geohash, COUNT(*) as n_obs, AVG(frp) as mean_frp, "
                "MAX(frp) as max_frp, MIN(obs_date) as first_seen, MAX(obs_date) as last_seen "
                "FROM site_history GROUP BY geohash ORDER BY mean_frp DESC"
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def purge_old_records(self, keep_days: int = 90) -> int:
        """Remove records older than keep_days to bound database size."""
        cutoff = (date.today() - timedelta(days=keep_days)).isoformat()
        with self._conn() as con:
            cur = con.execute(
                "DELETE FROM site_history WHERE obs_date < ?", (cutoff,)
            )
            deleted = cur.rowcount
        logger.info(f"SiteHistoryStore: purged {deleted} records older than {cutoff}")
        return deleted
