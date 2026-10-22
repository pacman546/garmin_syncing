"""Garmin full data export -> garmin.db.

Run it after every export:

    python -m garmin_claude.importer ~/Downloads/garmin_export.zip

Re-running the same export adds zero rows: every table upserts on its key, and
the export's SHA-256 is recorded in `import_log`, so a repeat is logged as a
duplicate and skipped.

Garmin reshuffles the export layout from time to time, so files are found by
pattern rather than by exact path, field names are looked up from a list of
candidates, and anything unrecognized is logged as a warning instead of
stopping the import. `--inspect` prints the raw values next to the converted
ones so the mapping can be checked against Garmin Connect by eye.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from . import db as dbmod

log = logging.getLogger("garmin_claude.importer")

# ---------------------------------------------------------------------------
# Unit conversion
#
# The export stores several fields in sub-SI units: durations in milliseconds,
# distances and elevation in centimetres. Each mapping below names the scale it
# applies, and `--inspect` prints raw and converted values side by side so the
# assumption can be confirmed against Garmin Connect (check R1 in the spec).
# ---------------------------------------------------------------------------

MS_TO_S = 1e-3
CM_TO_M = 1e-2
G_TO_KG = 1e-3

# Sanity bounds, used only to warn when a unit scale looks wrong.
MAX_SANE_DISTANCE_M = 1_000_000.0   # 1000 km in one activity
MAX_SANE_DURATION_S = 60 * 60 * 72  # 72 hours
MAX_SANE_SPEED_MPS = 35.0           # ~126 km/h


def pick(d: dict[str, Any], *keys: str) -> Any:
    """First present, non-null value among `keys`. Garmin renames fields."""
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def as_float(value: Any, scale: float = 1.0) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value) * scale
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    f = as_float(value)
    return None if f is None else int(round(f))


def epoch_ms_to_utc_iso(value: Any) -> str | None:
    """Epoch milliseconds -> '2026-10-17T18:00:00Z'."""
    f = as_float(value)
    if f is None:
        return None
    return (
        datetime.fromtimestamp(f / 1000.0, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def epoch_ms_to_wall_iso(value: Any) -> str | None:
    """Garmin's *Local* epoch fields are shifted so that reading them as UTC
    yields the local wall-clock time. Formatting as UTC therefore recovers the
    local clock without needing a timezone database."""
    return epoch_ms_to_utc_iso(value)


def parse_any_timestamp(value: Any) -> str | None:
    """Accept epoch ms, epoch s, or an ISO-8601 string; return UTC ISO."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Anything past ~1973 in seconds is still < 1e11; ms timestamps exceed it.
        return epoch_ms_to_utc_iso(value if value > 1e11 else value * 1000)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00").replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def date_of(iso: str | None) -> str | None:
    return iso[:10] if iso else None


def normalize_calendar_date(value: Any) -> str | None:
    """'2026-10-17', '2026-10-17T00:00:00.0', epoch ms -> '2026-10-17'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return date_of(parse_any_timestamp(value))
    text = str(value).strip()
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return date_of(parse_any_timestamp(text))


# ---------------------------------------------------------------------------
# Record extraction
#
# The export wraps its records in a few different envelopes depending on the
# file: a bare list, a list of one-key dicts, or an object with a single list
# value. `iter_records` flattens all of them.
# ---------------------------------------------------------------------------

def iter_records(payload: Any, hint_keys: Sequence[str] = ()) -> Iterator[dict]:
    """Yield the record dicts inside a parsed export file."""
    if isinstance(payload, dict):
        for key in hint_keys:
            if key in payload:
                yield from iter_records(payload[key], hint_keys)
                return
        list_values = [v for v in payload.values() if isinstance(v, list)]
        if len(list_values) == 1:
            yield from iter_records(list_values[0], hint_keys)
            return
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                # A wrapper like {"summarizedActivitiesExport": [...]}.
                inner = [v for v in item.values() if isinstance(v, list)]
                if len(item) == 1 and inner:
                    yield from iter_records(inner[0], hint_keys)
                else:
                    yield item
            elif isinstance(item, list):
                yield from iter_records(item, hint_keys)


# ---------------------------------------------------------------------------
# Row builders: one per table. Each takes a raw record and returns a dict of
# columns, or None when the record carries no usable key.
# ---------------------------------------------------------------------------

def build_activity(rec: dict) -> dict | None:
    activity_id = as_int(pick(rec, "activityId", "activity_id", "id"))
    if activity_id is None:
        return None

    start_utc = (
        epoch_ms_to_utc_iso(pick(rec, "beginTimestamp", "startTimeGmtEpoch"))
        or parse_any_timestamp(pick(rec, "startTimeGMT", "startTimeGmt", "startTimeUtc"))
    )
    start_local = (
        epoch_ms_to_wall_iso(pick(rec, "startTimeLocal"))
        if isinstance(pick(rec, "startTimeLocal"), (int, float))
        else parse_any_timestamp(pick(rec, "startTimeLocal"))
    )
    # The local clock decides the calendar date: an activity at 23:30 local
    # belongs to that local day, not to the UTC day it may fall into.
    local_date = date_of(start_local) or date_of(start_utc)
    if start_utc is None and start_local is None:
        return None

    duration_s = as_float(pick(rec, "duration", "elapsedDuration", "movingDuration"), MS_TO_S)
    distance_m = as_float(pick(rec, "distance"), CM_TO_M)
    elevation_m = as_float(pick(rec, "elevationGain"), CM_TO_M)
    avg_speed = as_float(pick(rec, "avgSpeed"), CM_TO_M)

    activity_type = pick(rec, "activityType", "sportType", "activityTypeDTO")
    if isinstance(activity_type, dict):
        activity_type = pick(activity_type, "typeKey", "key", "name")
    if activity_type is not None:
        activity_type = str(activity_type).lower()

    pace = None
    if distance_m and duration_s and distance_m > 0:
        pace = duration_s / (distance_m / 1000.0)

    return {
        "activity_id": activity_id,
        "start_time_utc": start_utc or start_local,
        "start_time_local": start_local,
        "local_date": local_date,
        "type": activity_type,
        "name": pick(rec, "name", "activityName"),
        "duration_s": duration_s,
        "distance_m": distance_m,
        "avg_hr": as_int(pick(rec, "avgHr", "averageHR", "avgHeartRate")),
        "max_hr": as_int(pick(rec, "maxHr", "maxHR", "maxHeartRate")),
        "calories": as_float(pick(rec, "calories", "activeKilocalories")),
        "elevation_gain_m": elevation_m,
        "avg_speed_mps": avg_speed,
        "avg_pace_s_per_km": pace,
        "training_effect": as_float(
            pick(rec, "aerobicTrainingEffect", "trainingEffect", "activityTrainingLoad")
        ),
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True),
    }


def build_sleep(rec: dict) -> dict | None:
    date = normalize_calendar_date(
        pick(rec, "calendarDate", "sleepStartTimestampLocal", "date")
    )
    if date is None:
        return None
    scores = pick(rec, "sleepScores", "sleepScore") or {}
    score = None
    if isinstance(scores, dict):
        overall = pick(scores, "overall", "overallScore", "value")
        score = as_int(pick(overall, "value", "score") if isinstance(overall, dict) else overall)
    else:
        score = as_int(scores)
    if score is None:
        score = as_int(pick(rec, "overallSleepScore", "sleepQualityTypePK"))

    deep = as_int(pick(rec, "deepSleepSeconds"))
    light = as_int(pick(rec, "lightSleepSeconds"))
    rem = as_int(pick(rec, "remSleepSeconds", "remSleepSecond"))
    awake = as_int(pick(rec, "awakeSleepSeconds", "awakeSleepSecond"))
    total = as_int(pick(rec, "sleepTimeSeconds", "totalSleepSeconds"))
    if total is None:
        parts = [p for p in (deep, light, rem) if p is not None]
        total = sum(parts) if parts else None

    return {
        "date": date,
        "start_utc": parse_any_timestamp(
            pick(rec, "sleepStartTimestampGMT", "sleepStartTimestampGmt")
        ),
        "end_utc": parse_any_timestamp(
            pick(rec, "sleepEndTimestampGMT", "sleepEndTimestampGmt")
        ),
        "total_s": total,
        "deep_s": deep,
        "light_s": light,
        "rem_s": rem,
        "awake_s": awake,
        "sleep_score": score,
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True),
    }


def intensity_minutes_total(rec: dict) -> int | None:
    """Moderate + 2 x vigorous minutes (Garmin counts vigorous time double).

    0 is a real answer - "no intense minutes today" - and must stay 0. None means
    Garmin gave no figure at all. Collapsing the two (as `... or None` did) turns
    every inactive day into a gap on a chart.
    """
    moderate = pick(rec, "moderateIntensityMinutes", "moderateIntensityDuration")
    vigorous = pick(rec, "vigorousIntensityMinutes", "vigorousIntensityDuration")
    if moderate is None and vigorous is None:
        return as_int(pick(rec, "intensityMinutes", "totalIntensityMinutes"))
    return (as_int(moderate) or 0) + 2 * (as_int(vigorous) or 0)


def stress_level(value: Any) -> int | None:
    """Garmin sends -1 (sometimes -2) when it had too little data to compute a
    stress level, e.g. a day the watch was not worn. That is "no data", not a
    stress level, and charting it would draw a dip below zero."""
    level = as_int(value)
    return None if level is None or level < 0 else level


def build_daily(rec: dict) -> dict | None:
    date = normalize_calendar_date(pick(rec, "calendarDate", "date", "statisticsStartDate"))
    if date is None:
        return None
    intensity = intensity_minutes_total(rec)

    return {
        "date": date,
        "steps": as_int(pick(rec, "totalSteps", "steps")),
        "resting_hr": as_int(
            pick(rec, "restingHeartRate", "minHeartRate", "restingHeartRateTimestamp")
        ),
        "calories": as_float(pick(rec, "totalKilocalories", "calories", "activeKilocalories")),
        "intensity_minutes": intensity,
        "stress_avg": stress_level(pick(rec, "averageStressLevel", "averageStress", "avgStressLevel")),
        "body_battery_high": as_int(
            pick(rec, "bodyBatteryHighestValue", "highestBodyBattery", "bodyBatteryHigh")
        ),
        "body_battery_low": as_int(
            pick(rec, "bodyBatteryLowestValue", "lowestBodyBattery", "bodyBatteryLow")
        ),
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True),
    }


def build_hrv(rec: dict) -> dict | None:
    date = normalize_calendar_date(pick(rec, "calendarDate", "date"))
    if date is None:
        return None
    baseline = pick(rec, "baseline") or {}
    if not isinstance(baseline, dict):
        baseline = {}
    status = pick(rec, "hrvStatus", "status")
    nightly = as_float(pick(rec, "lastNightAvg", "nightlyAvg", "weeklyAvg", "avgHrv"))
    if nightly is None and status is None:
        return None
    return {
        "date": date,
        "nightly_avg_ms": nightly,
        "status": str(status) if status is not None else None,
        "baseline_low": as_float(pick(baseline, "balancedLow", "lowUpper")),
        "baseline_high": as_float(pick(baseline, "balancedUpper", "markerValue")),
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True),
    }


def build_body(rec: dict) -> dict | None:
    date = normalize_calendar_date(
        pick(rec, "calendarDate", "date", "timestampGMT", "weighInDate")
    )
    if date is None:
        return None
    # Weight comes in grams in the export, kilograms from the Phase 2 API.
    weight_raw = pick(rec, "weight", "weightInGrams")
    weight = as_float(weight_raw, G_TO_KG)
    if weight is not None and weight < 1.0:  # already kilograms
        weight = as_float(weight_raw)
    fat = as_float(pick(rec, "bodyFat", "bodyFatPercentage", "bodyFatPct"))
    bmi = as_float(pick(rec, "bmi", "bodyMassIndex"))
    if weight is None and fat is None and bmi is None:
        return None
    return {
        "date": date,
        "weight_kg": weight,
        "body_fat_pct": fat,
        "bmi": bmi,
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True),
    }


# ---------------------------------------------------------------------------
# File recognition
# ---------------------------------------------------------------------------

class Handler:
    """One kind of export file: how to spot it, and how to turn it into rows."""

    def __init__(
        self,
        table: str,
        key: tuple[str, ...],
        patterns: Sequence[str],
        builder: Callable[[dict], dict | None],
        hint_keys: Sequence[str] = (),
    ) -> None:
        self.table = table
        self.key = key
        self.patterns = patterns
        self.builder = builder
        self.hint_keys = hint_keys

    def matches(self, name: str) -> bool:
        lowered = name.lower()
        return any(p in lowered for p in self.patterns)


HANDLERS: tuple[Handler, ...] = (
    Handler(
        "activities",
        ("activity_id",),
        ("summarizedactivities", "_activities.json", "activities.json"),
        build_activity,
        ("summarizedActivitiesExport", "activities"),
    ),
    Handler(
        "sleep",
        ("date",),
        ("sleepdata", "_sleep.json", "sleep_data"),
        build_sleep,
        ("sleepData", "dailySleepDTO"),
    ),
    Handler(
        "daily_summary",
        ("date",),
        ("udsfile", "aggregator", "dailysummary", "daily_summary"),
        build_daily,
        ("userDailySummary", "allDayStress"),
    ),
    Handler(
        "hrv",
        ("date",),
        ("hrvdata", "_hrv.json", "hrv_data"),
        build_hrv,
        ("hrvData", "hrvSummaries"),
    ),
    Handler(
        "body",
        ("date",),
        ("biometric", "weightdata", "weight_data", "bodycomposition"),
        build_body,
        ("weightData", "bodyCompositionData"),
    ),
)

# Files we knowingly skip, so they are not reported as unrecognized.
IGNORED_PATTERNS = (
    "__macosx", ".ds_store", "readme", ".txt", ".csv", ".pdf", ".html",
    "garmin-user-", "user_profile", "device", "gear", "goals", "course",
    "workout", "badge", "social", "segment", "personalrecord", "insight",
    "subscription", "request", "notification", "settings",
)


def is_ignored(name: str) -> bool:
    lowered = name.lower()
    return any(p in lowered for p in IGNORED_PATTERNS)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert(conn: sqlite3.Connection, table: str, key: tuple[str, ...], row: dict) -> int:
    """Insert or replace one row. Returns 1 if it was new, 0 if it already existed."""
    columns = list(row)
    placeholders = ", ".join("?" for _ in columns)
    updates = ", ".join(f"{c}=excluded.{c}" for c in columns if c not in key)
    conflict = ", ".join(key)
    where = " AND ".join(f"{k}=?" for k in key)

    existed = conn.execute(
        f"SELECT 1 FROM {table} WHERE {where}", [row[k] for k in key]
    ).fetchone() is not None

    conn.execute(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict}) DO UPDATE SET {updates}",
        [row[c] for c in columns],
    )
    return 0 if existed else 1


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

class ImportResult:
    def __init__(self) -> None:
        self.rows_added: dict[str, int] = {}
        self.rows_seen: dict[str, int] = {}
        self.unrecognized: list[str] = []
        self.warnings: list[str] = []
        self.duplicate = False
        self.files_read = 0

    @property
    def total_added(self) -> int:
        return sum(self.rows_added.values())

    def summary(self) -> str:
        if self.duplicate:
            return "duplicate export (already imported); 0 rows added"
        parts = [f"{t}+{n}" for t, n in sorted(self.rows_added.items()) if n]
        return (
            f"{self.files_read} files, {self.total_added} new rows"
            + (f" ({', '.join(parts)})" if parts else "")
            + (f", {len(self.unrecognized)} unrecognized" if self.unrecognized else "")
        )


def iter_export_files(root: Path) -> Iterator[Path]:
    """Every JSON file in an unpacked export, including nested zips already
    expanded by `unpack`."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def unpack(source: Path, workdir: Path) -> Path:
    """Expand the export into `workdir`, including the nested zips Garmin uses
    for uploaded files. A directory is used as-is."""
    if source.is_dir():
        return source
    if not zipfile.is_zipfile(source):
        raise ValueError(f"{source} is not a zip file or a directory")
    target = workdir / "export"
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as zf:
        zf.extractall(target)
    # Garmin nests UploadedFiles_*.zip inside the main export.
    for nested in sorted(target.rglob("*.zip")):
        try:
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(nested.with_suffix(""))
        except zipfile.BadZipFile:
            log.warning("skipping unreadable nested zip: %s", nested.name)
    return target


def import_export(
    source: Path,
    conn: sqlite3.Connection,
    *,
    with_samples: bool = False,
    force: bool = False,
) -> ImportResult:
    """Import one export (zip or unpacked directory) into an open connection."""
    result = ImportResult()
    source = source.expanduser()
    if not source.exists():
        raise FileNotFoundError(f"No export at {source}")

    digest = sha256_of(source) if source.is_file() else "directory"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if not force and digest != "directory":
        seen = conn.execute(
            "SELECT id FROM import_log WHERE sha256=? AND status='ok'", (digest,)
        ).fetchone()
        if seen is not None:
            result.duplicate = True
            conn.execute(
                "INSERT INTO import_log (file_name, sha256, rows_added, imported_at, status, note)"
                " VALUES (?, ?, 0, ?, 'duplicate', ?)",
                (source.name, digest, now, f"identical to import #{seen['id']}; skipped"),
            )
            conn.commit()
            log.info("%s already imported (sha256 %s...); nothing to do", source.name, digest[:12])
            return result

    with tempfile.TemporaryDirectory(prefix="garmin-import-") as tmp:
        root = unpack(source, Path(tmp))
        for path in iter_export_files(root):
            rel = str(path.relative_to(root))
            if path.suffix.lower() == ".fit":
                continue  # handled by fit_parser, below
            if path.suffix.lower() != ".json":
                if not is_ignored(rel) and path.suffix.lower() != ".zip":
                    result.unrecognized.append(rel)
                continue
            handler = next((h for h in HANDLERS if h.matches(rel)), None)
            if handler is None:
                if not is_ignored(rel):
                    result.unrecognized.append(rel)
                    log.warning("unrecognized export file: %s", rel)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                result.warnings.append(f"{rel}: unreadable JSON ({exc})")
                log.warning("unreadable JSON in %s: %s", rel, exc)
                continue

            result.files_read += 1
            added = 0
            seen = 0
            for rec in iter_records(payload, handler.hint_keys):
                row = handler.builder(rec)
                if row is None:
                    continue
                seen += 1
                added += upsert(conn, handler.table, handler.key, row)
            result.rows_added[handler.table] = result.rows_added.get(handler.table, 0) + added
            result.rows_seen[handler.table] = result.rows_seen.get(handler.table, 0) + seen
            log.info("%s -> %s: %d records, %d new", rel, handler.table, seen, added)

        if with_samples:
            from . import fit_parser

            fit_added, fit_warnings = fit_parser.import_fit_directory(root, conn)
            if fit_added:
                result.rows_added["activity_samples"] = fit_added
            result.warnings.extend(fit_warnings)

        result.warnings.extend(check_units(conn))

        conn.execute(
            "INSERT INTO import_log (file_name, sha256, rows_added, imported_at, status, note)"
            " VALUES (?, ?, ?, ?, 'ok', ?)",
            (
                source.name,
                digest,
                result.total_added,
                now,
                json.dumps(
                    {
                        "rows_added": result.rows_added,
                        "unrecognized": result.unrecognized[:50],
                        "warnings": result.warnings[:50],
                    },
                    separators=(",", ":"),
                ),
            ),
        )
        conn.commit()
    return result


def check_units(conn: sqlite3.Connection) -> list[str]:
    """Warn if converted values look like the wrong unit scale. Garmin has
    changed these before; this is how a silent 100x error gets caught."""
    warnings: list[str] = []
    row = conn.execute(
        "SELECT MAX(distance_m) AS d, MAX(duration_s) AS t, MAX(avg_speed_mps) AS s"
        " FROM activities"
    ).fetchone()
    if row is None:
        return warnings
    if row["d"] and row["d"] > MAX_SANE_DISTANCE_M:
        warnings.append(
            f"longest activity is {row['d'] / 1000:.0f} km - distance unit scale may be wrong"
        )
    if row["t"] and row["t"] > MAX_SANE_DURATION_S:
        warnings.append(
            f"longest activity is {row['t'] / 3600:.0f} h - duration unit scale may be wrong"
        )
    if row["s"] and row["s"] > MAX_SANE_SPEED_MPS:
        warnings.append(
            f"fastest average speed is {row['s']:.0f} m/s - speed unit scale may be wrong"
        )
    for w in warnings:
        log.warning("%s", w)
    return warnings


def inspect_export(source: Path, limit: int = 3) -> None:
    """Print what the importer finds, raw next to converted, without writing
    anything. Use this to confirm the mapping against Garmin Connect."""
    with tempfile.TemporaryDirectory(prefix="garmin-inspect-") as tmp:
        root = unpack(source.expanduser(), Path(tmp))
        matched: dict[str, list[str]] = {}
        unknown: list[str] = []
        for path in iter_export_files(root):
            rel = str(path.relative_to(root))
            if path.suffix.lower() == ".fit":
                matched.setdefault("activity_samples (.fit)", []).append(rel)
                continue
            if path.suffix.lower() != ".json":
                continue
            handler = next((h for h in HANDLERS if h.matches(rel)), None)
            if handler:
                matched.setdefault(handler.table, []).append(rel)
            elif not is_ignored(rel):
                unknown.append(rel)

        print("=== files matched ===")
        for table, files in sorted(matched.items()):
            print(f"\n{table}:")
            for f in files[:10]:
                print(f"  {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
        if unknown:
            print("\n=== unrecognized (will be logged, not imported) ===")
            for f in unknown[:20]:
                print(f"  {f}")

        print("\n=== sample conversions (check these against Garmin Connect) ===")
        for handler in HANDLERS:
            files = matched.get(handler.table, [])
            if not files:
                continue
            path = root / files[0]
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception as exc:  # noqa: BLE001 - inspection must not crash
                print(f"\n{handler.table}: unreadable ({exc})")
                continue
            print(f"\n--- {handler.table} (from {files[0]}) ---")
            shown = 0
            for rec in iter_records(payload, handler.hint_keys):
                row = handler.builder(rec)
                if row is None:
                    continue
                interesting = {k: v for k, v in row.items() if k != "raw"}
                print(f"  converted: {json.dumps(interesting, default=str)}")
                raw_keys = {
                    k: rec[k]
                    for k in ("duration", "distance", "elevationGain", "avgSpeed", "weight")
                    if k in rec
                }
                if raw_keys:
                    print(f"  raw units: {json.dumps(raw_keys, default=str)}")
                shown += 1
                if shown >= limit:
                    break


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m garmin_claude.importer",
        description="Import a Garmin full data export into garmin.db.",
    )
    parser.add_argument("export", type=Path, help="garmin_export.zip, or an unpacked directory")
    parser.add_argument("--db", type=Path, default=None, help="database path (default $GARMIN_DB or ~/garmin.db)")
    parser.add_argument("--with-samples", action="store_true", help="also parse FIT files into activity_samples")
    parser.add_argument("--force", action="store_true", help="re-import even if this export was already imported")
    parser.add_argument("--inspect", action="store_true", help="show what would be imported; write nothing")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.inspect:
        inspect_export(args.export)
        return 0

    conn = dbmod.connect_rw(args.db)
    try:
        result = import_export(
            args.export, conn, with_samples=args.with_samples, force=args.force
        )
    finally:
        conn.close()

    print(result.summary())
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    if result.unrecognized:
        print(
            f"note: {len(result.unrecognized)} unrecognized files logged "
            f"(run with --inspect to list them)",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
