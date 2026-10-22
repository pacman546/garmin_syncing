"""FIT files -> activity_samples.

Optional, and the slowest part of an import: a long history of FIT files holds
millions of sample rows. Enable it with `--with-samples`.

Samples are stored at full resolution and downsampled when the server reads
them, so the stored data stays faithful while what reaches Claude stays small.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger("garmin_claude.fit_parser")

# FIT stores coordinates in semicircles.
SEMICIRCLE_TO_DEGREES = 180.0 / (2**31)

LEADING_DIGITS = re.compile(r"(\d{6,})")


def _utc_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def _coord(value: Any) -> float | None:
    """Semicircles -> degrees. The SDK hands these over unconverted."""
    if value is None:
        return None
    try:
        degrees = float(value) * SEMICIRCLE_TO_DEGREES
    except (TypeError, ValueError):
        return None
    return degrees if -180.0 <= degrees <= 180.0 else None


def first_of(rec: dict, *keys: str) -> Any:
    """First key present with a non-null value. A FIT record may carry a key
    with no value, so `dict.get(k, default)` is not enough."""
    for key in keys:
        value = rec.get(key)
        if value is not None:
            return value
    return None


def parse_fit(path: Path) -> tuple[str | None, list[dict]]:
    """Return (session start as UTC ISO, sample rows) for one FIT file."""
    from garmin_fit_sdk import Decoder, Stream  # imported lazily: optional dependency

    stream = Stream.from_file(str(path))
    decoder = Decoder(stream)
    messages, errors = decoder.read(
        apply_scale_and_offset=True,
        convert_datetimes_to_dates=True,
        convert_types_to_strings=True,
        expand_sub_fields=True,
        expand_components=True,
    )
    for err in errors or []:
        log.debug("%s: %s", path.name, err)

    sessions = messages.get("session_mesgs") or []
    start = _utc_iso(sessions[0].get("start_time")) if sessions else None

    samples: list[dict] = []
    for rec in messages.get("record_mesgs") or []:
        ts = _utc_iso(rec.get("timestamp"))
        if ts is None:
            continue
        samples.append(
            {
                "ts": ts,
                "hr": first_of(rec, "heart_rate"),
                "speed_mps": first_of(rec, "enhanced_speed", "speed"),
                "cadence": first_of(rec, "cadence"),
                "power_w": first_of(rec, "power"),
                "altitude_m": first_of(rec, "enhanced_altitude", "altitude"),
                "lat": _coord(first_of(rec, "position_lat")),
                "lon": _coord(first_of(rec, "position_long")),
            }
        )
    if start is None and samples:
        start = samples[0]["ts"]
    return start, samples


def resolve_activity_id(
    conn: sqlite3.Connection, path: Path, session_start: str | None
) -> int | None:
    """Match a FIT file to a row in `activities`.

    First by the activity ID in the filename, which is how Garmin names them;
    failing that, by a session start time within a minute of a known activity.
    """
    for candidate in LEADING_DIGITS.findall(path.stem):
        row = conn.execute(
            "SELECT activity_id FROM activities WHERE activity_id=?", (int(candidate),)
        ).fetchone()
        if row:
            return int(row["activity_id"])

    if session_start:
        row = conn.execute(
            "SELECT activity_id FROM activities"
            " WHERE ABS(strftime('%s', start_time_utc) - strftime('%s', ?)) <= 60"
            " ORDER BY ABS(strftime('%s', start_time_utc) - strftime('%s', ?)) LIMIT 1",
            (session_start, session_start),
        ).fetchone()
        if row:
            return int(row["activity_id"])
    return None


def iter_fit_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".fit":
            yield path


def import_fit_directory(root: Path, conn: sqlite3.Connection) -> tuple[int, list[str]]:
    """Parse every FIT file under `root` into activity_samples.

    Returns (rows added, warnings). A file that cannot be parsed or matched is a
    warning, never a failure: the summary data is already in place without it.
    """
    try:
        import garmin_fit_sdk  # noqa: F401
    except ImportError:
        return 0, ["garmin-fit-sdk is not installed; skipped FIT samples"]

    added = 0
    warnings: list[str] = []
    for path in iter_fit_files(root):
        try:
            session_start, samples = parse_fit(path)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the import
            warnings.append(f"{path.name}: unreadable FIT ({exc})")
            log.warning("unreadable FIT %s: %s", path.name, exc)
            continue
        if not samples:
            continue
        activity_id = resolve_activity_id(conn, path, session_start)
        if activity_id is None:
            warnings.append(f"{path.name}: no matching activity, samples skipped")
            log.warning("no matching activity for %s", path.name)
            continue

        before = conn.execute(
            "SELECT COUNT(*) FROM activity_samples WHERE activity_id=?", (activity_id,)
        ).fetchone()[0]
        conn.executemany(
            "INSERT INTO activity_samples"
            " (activity_id, ts, hr, speed_mps, cadence, power_w, altitude_m, lat, lon)"
            " VALUES (:activity_id, :ts, :hr, :speed_mps, :cadence, :power_w,"
            "         :altitude_m, :lat, :lon)"
            " ON CONFLICT(activity_id, ts) DO UPDATE SET"
            "   hr=excluded.hr, speed_mps=excluded.speed_mps, cadence=excluded.cadence,"
            "   power_w=excluded.power_w, altitude_m=excluded.altitude_m,"
            "   lat=excluded.lat, lon=excluded.lon",
            [{"activity_id": activity_id, **s} for s in samples],
        )
        after = conn.execute(
            "SELECT COUNT(*) FROM activity_samples WHERE activity_id=?", (activity_id,)
        ).fetchone()[0]
        added += after - before
        log.info("%s -> activity %d: %d samples", path.name, activity_id, len(samples))

    conn.commit()
    return added, warnings
