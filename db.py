"""Schema, migrations and connections for garmin.db.

Two ways in, and only two: `connect_rw` for the importer and the Phase 2 sync,
`connect_ro` for the MCP server. The read-only connection uses SQLite's own
`mode=ro` URI, so the server cannot write even if its own guard rails are bypassed.

Units are SI throughout: metres, seconds, kilograms. Times are stored as UTC
ISO-8601 strings alongside a `local_date` column holding the calendar date in
the user's own timezone, so "Tuesday's run" means their Tuesday.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 2  # 2: vo2max, training_readiness, sync_state

DEFAULT_DB_PATH = Path.home() / "garmin.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS activities (
    activity_id       INTEGER PRIMARY KEY,
    start_time_utc    TEXT    NOT NULL,
    start_time_local  TEXT,
    local_date        TEXT    NOT NULL,
    type              TEXT,
    name              TEXT,
    duration_s        REAL,
    distance_m        REAL,
    avg_hr            INTEGER,
    max_hr            INTEGER,
    calories          REAL,
    elevation_gain_m  REAL,
    avg_speed_mps     REAL,
    avg_pace_s_per_km REAL,
    training_effect   REAL,
    raw               TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_activities_local_date ON activities(local_date);
CREATE INDEX IF NOT EXISTS idx_activities_type       ON activities(type);

CREATE TABLE IF NOT EXISTS activity_samples (
    activity_id  INTEGER NOT NULL,
    ts           TEXT    NOT NULL,
    hr           INTEGER,
    speed_mps    REAL,
    cadence      REAL,
    power_w      REAL,
    altitude_m   REAL,
    lat          REAL,
    lon          REAL,
    PRIMARY KEY (activity_id, ts)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_samples_activity ON activity_samples(activity_id);

CREATE TABLE IF NOT EXISTS daily_summary (
    date               TEXT PRIMARY KEY,
    steps              INTEGER,
    resting_hr         INTEGER,
    calories           REAL,
    intensity_minutes  INTEGER,
    stress_avg         INTEGER,
    body_battery_high  INTEGER,
    body_battery_low   INTEGER,
    raw                TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sleep (
    date         TEXT PRIMARY KEY,
    start_utc    TEXT,
    end_utc      TEXT,
    total_s      INTEGER,
    deep_s       INTEGER,
    light_s      INTEGER,
    rem_s        INTEGER,
    awake_s      INTEGER,
    sleep_score  INTEGER,
    raw          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hrv (
    date            TEXT PRIMARY KEY,
    nightly_avg_ms  REAL,
    status          TEXT,
    baseline_low    REAL,
    baseline_high   REAL,
    raw             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS body (
    date          TEXT PRIMARY KEY,
    weight_kg     REAL,
    body_fat_pct  REAL,
    bmi           REAL,
    raw           TEXT NOT NULL
);

-- One row per date Garmin published a new VO2 max estimate. Sparse: Garmin only
-- recalculates after a qualifying run or ride, so gaps between rows are normal.
CREATE TABLE IF NOT EXISTS vo2max (
    date            TEXT PRIMARY KEY,
    vo2max          REAL,        -- running / generic, ml/kg/min, Garmin's precise value
    vo2max_cycling  REAL,
    fitness_age     REAL,
    raw             TEXT NOT NULL
);

-- One row per day. Garmin recalculates readiness several times a day (on waking,
-- after workouts, during the day); `score` is the morning one - the number in
-- Garmin's Morning Report - and `latest_score` is the last of the day. `raw`
-- holds every entry for the day.
CREATE TABLE IF NOT EXISTS training_readiness (
    date                      TEXT PRIMARY KEY,
    score                     INTEGER,   -- 0-100, on waking
    level                     TEXT,      -- e.g. LOW, MODERATE, HIGH, PRIME
    feedback                  TEXT,      -- e.g. WELL_RECOVERED
    measured_at_local         TEXT,
    latest_score              INTEGER,
    latest_at_local           TEXT,
    recovery_time_min         INTEGER,   -- minutes, as Garmin reports it
    acute_load                INTEGER,   -- Garmin's acute training load (unitless)
    sleep_factor_pct          INTEGER,   -- each factor: 0-100, its contribution
    sleep_history_factor_pct  INTEGER,
    recovery_factor_pct       INTEGER,
    acwr_factor_pct           INTEGER,   -- acute:chronic workload ratio
    hrv_factor_pct            INTEGER,
    stress_factor_pct         INTEGER,
    raw                       TEXT NOT NULL
);

-- How far each ranged source has been *checked*, as opposed to how far it has
-- data. Sparse sources (weigh-ins, VO2 max) can go weeks without a row; without
-- this, their sync window would be anchored to the last row and stop reaching
-- today. Written by the Phase 2 sync only.
CREATE TABLE IF NOT EXISTS sync_state (
    source           TEXT PRIMARY KEY,
    checked_through  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS import_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name    TEXT NOT NULL,
    sha256       TEXT NOT NULL,
    rows_added   INTEGER NOT NULL DEFAULT 0,
    imported_at  TEXT NOT NULL,
    status       TEXT NOT NULL,
    note         TEXT
);
CREATE INDEX IF NOT EXISTS idx_import_log_sha ON import_log(sha256);
"""

# Tables the MCP server reports on, in the order data_status lists them.
DATA_TABLES = (
    "activities",
    "activity_samples",
    "daily_summary",
    "sleep",
    "hrv",
    "body",
    "vo2max",
    "training_readiness",
)

# Which column carries the calendar date, per table. Used by data_status.
DATE_COLUMN = {
    "activities": "local_date",
    "daily_summary": "date",
    "sleep": "date",
    "hrv": "date",
    "body": "date",
    "vo2max": "date",
    "training_readiness": "date",
}


def db_path() -> Path:
    """Where garmin.db lives: $GARMIN_DB if set, else ~/garmin.db."""
    env = os.environ.get("GARMIN_DB")
    return Path(env).expanduser() if env else DEFAULT_DB_PATH


def connect_rw(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open a writable connection and make sure the schema is current."""
    target = Path(path).expanduser() if path else db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def connect_ro(path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open a read-only connection.

    SQLite enforces this itself: any write raises `sqlite3.OperationalError`,
    whatever the caller intends. The server only ever uses this.
    """
    target = Path(path).expanduser() if path else db_path()
    if not target.exists():
        raise FileNotFoundError(
            f"No database at {target}. Run the importer first: "
            f"python -m garmin_claude.importer <garmin_export.zip>"
        )
    uri = f"file:{target.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    """Create or upgrade the schema. Safe to call on every open."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema is version {version}, newer than this code "
            f"(version {SCHEMA_VERSION}). Upgrade garmin-claude."
        )
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
