"""MCP server exposing Garmin data to Claude Desktop, read-only.

Started by Claude Desktop as a child process and spoken to over stdio, so no
network port is ever opened. The database is opened with SQLite's `mode=ro`
URI: there are no write tools, and a write would fail at the SQLite layer even
if one were added by mistake.

Every tool returns compact JSON capped at 200 rows, and activity samples are
downsampled to 500 points, so answers stay small.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import sys
import warnings
from datetime import date as date_cls, datetime, timedelta, timezone
from typing import Any, Iterable

# Claude Desktop keeps this process's stderr in its logs, and anything written
# there is noise the user has to read past. The MCP SDK trips a
# pydantic_settings annotation warning on Python 3.14 as it is imported, so the
# filter has to be installed before the import, not in main().
warnings.filterwarnings("ignore", module="pydantic_settings.*")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from . import db as dbmod  # noqa: E402

log = logging.getLogger("garmin_claude.server")

ROW_CAP = 200
SAMPLE_CAP = 500

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

mcp = FastMCP(
    "garmin",
    instructions=(
        "Read-only access to the user's own Garmin history, held in a local "
        "SQLite file. Call data_status first when you are unsure whether a date "
        "is covered or where the data came from - it also tells you if the "
        "database holds test-fixture data rather than the user's real history. "
        "Units are SI: metres, seconds, kilograms, milliseconds for HRV; convert "
        "for display. Prefer get_trends over fetching every day for 'how has X "
        "trended' questions, and run_sql only for what the other tools do not "
        "cover. Nothing here can change Garmin data or the database."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ToolInput(ValueError):
    """Bad input from a tool call. Turned into a plain JSON error, never a traceback."""


def compact(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"), default=str)


def fail(message: str) -> str:
    return compact({"error": message})


def today_iso() -> str:
    return date_cls.today().isoformat()


def check_date(value: str | None, field: str, *, default: str | None = None) -> str:
    """Validate an ISO date. A missing end date means today."""
    if value is None or value == "":
        if default is None:
            raise ToolInput(f"{field} is required, as an ISO date like 2026-10-17")
        return default
    text = str(value).strip()
    if not DATE_RE.match(text):
        raise ToolInput(f"{field} must be an ISO date like 2026-10-17, got {value!r}")
    try:
        date_cls.fromisoformat(text)
    except ValueError:
        raise ToolInput(f"{field} is not a real date: {value!r}") from None
    return text


def date_window(start_date: str | None, end_date: str | None) -> tuple[str, str]:
    start = check_date(start_date, "start_date")
    end = check_date(end_date, "end_date", default=today_iso())
    if start > end:
        raise ToolInput(f"start_date {start} is after end_date {end}")
    return start, end


def check_limit(limit: Any, field: str = "limit") -> int:
    if limit is None:
        return ROW_CAP
    try:
        value = int(limit)
    except (TypeError, ValueError):
        raise ToolInput(f"{field} must be a whole number, got {limit!r}") from None
    if value < 1:
        raise ToolInput(f"{field} must be at least 1, got {value}")
    return min(value, ROW_CAP)


_connection: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    """One read-only connection, opened on first use and reused."""
    global _connection
    if _connection is None:
        _connection = dbmod.connect_ro(os.environ.get("GARMIN_DB"))
    return _connection


def existing_tables() -> set[str]:
    return {
        r[0] for r in conn().execute("SELECT name FROM sqlite_master WHERE type='table'")
    }


def rows(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    return [dict(r) for r in conn().execute(sql, tuple(params)).fetchall()]


def drop_nulls(record: dict) -> dict:
    """Leave out empty fields; they cost tokens and say nothing."""
    return {k: v for k, v in record.items() if v is not None}


def tool(fn):
    """Register a tool and turn bad input into a JSON error instead of a crash."""

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolInput as exc:
            return fail(str(exc))
        except FileNotFoundError as exc:
            return fail(str(exc))
        except sqlite3.Error as exc:
            log.warning("sqlite error in %s: %s", fn.__name__, exc)
            return fail(f"database error: {exc}")

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__annotations__ = fn.__annotations__
    wrapper.__signature__ = __import__("inspect").signature(fn)
    return mcp.tool()(wrapper)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def data_status() -> str:
    """What Garmin data is in the local database.

    Returns the date range covered, the row count for each table, and when the
    last import ran. Call this first when you are unsure whether a date is
    covered. Tables: activities, activity_samples, daily_summary, sleep, hrv,
    body, vo2max, training_readiness. Units are SI: metres, seconds, kilograms.
    """
    counts: dict[str, int] = {}
    ranges: dict[str, dict[str, str]] = {}
    present = existing_tables()
    for table in dbmod.DATA_TABLES:
        if table not in present:
            # An older database, not yet upgraded by a sync or import. Not an error.
            counts[table] = 0
            continue
        counts[table] = conn().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        column = dbmod.DATE_COLUMN.get(table)
        if column and counts[table]:
            row = conn().execute(
                f"SELECT MIN({column}) AS first, MAX({column}) AS last FROM {table}"
            ).fetchone()
            ranges[table] = {"first": row["first"], "last": row["last"]}

    overall = [r for r in ranges.values() if r.get("first")]
    last_import = conn().execute(
        "SELECT file_name, imported_at, rows_added, status FROM import_log"
        " ORDER BY id DESC LIMIT 1"
    ).fetchone()
    newest_ok = conn().execute(
        "SELECT file_name FROM import_log WHERE status='ok' ORDER BY id DESC LIMIT 1"
    ).fetchone()

    return compact(
        {
            "first_date": min(r["first"] for r in overall) if overall else None,
            "last_date": max(r["last"] for r in overall) if overall else None,
            "row_counts": counts,
            "per_table_range": ranges,
            "last_import": dict(last_import) if last_import else None,
            "data_source": describe_source(newest_ok["file_name"] if newest_ok else None),
            "units": {"distance": "m", "duration": "s", "weight": "kg", "hrv": "ms",
                      "vo2max": "ml/kg/min", "readiness": "0-100"},
        }
    )


FIXTURE_MARKER = "GARMIN_TEST_FIXTURE"


def describe_source(file_name: str | None) -> str:
    """Say plainly where these numbers came from.

    A database built from the test fixture holds invented data. Saying so here
    means Claude repeats the caveat instead of presenting made-up numbers as
    the user's own history.
    """
    if file_name is None:
        return "unknown: nothing has been imported yet"
    if FIXTURE_MARKER in file_name:
        return (
            "TEST FIXTURE - this is invented sample data, not the user's real "
            "Garmin history. Say so before quoting any number from it."
        )
    if file_name.startswith("garmin-connect-sync"):
        return "Garmin Connect live sync (Phase 2)"
    return f"Garmin data export (Phase 1): {file_name}"


@tool
def list_activities(
    start_date: str,
    end_date: str | None = None,
    type: str | None = None,
    limit: int | None = None,
) -> str:
    """List activities in a date range, one compact line each.

    Dates are ISO strings (2026-10-17) in local time; a missing end_date means
    today. `type` filters on Garmin's activity type and matches loosely, so
    "run" also finds "trail_running". At most 200 rows, newest first. Distances
    are metres and durations seconds. For one activity in full, use get_activity.
    """
    start, end = date_window(start_date, end_date)
    cap = check_limit(limit)

    sql = (
        "SELECT activity_id, local_date, type, name, duration_s, distance_m, avg_hr"
        " FROM activities WHERE local_date BETWEEN ? AND ?"
    )
    params: list[Any] = [start, end]
    if type:
        sql += " AND type LIKE ?"
        params.append(f"%{str(type).strip().lower()}%")
    sql += " ORDER BY local_date DESC, activity_id DESC LIMIT ?"
    params.append(cap)

    found = rows(sql, params)
    total = conn().execute(
        "SELECT COUNT(*) FROM activities WHERE local_date BETWEEN ? AND ?"
        + (" AND type LIKE ?" if type else ""),
        tuple([start, end] + ([f"%{str(type).strip().lower()}%"] if type else [])),
    ).fetchone()[0]

    return compact(
        {
            "range": {"start": start, "end": end},
            "type": type,
            "count_returned": len(found),
            "count_matching": total,
            "truncated": total > len(found),
            "activities": [drop_nulls(r) for r in found],
        }
    )


@tool
def get_activity(activity_id: int, include_samples: bool = False) -> str:
    """One activity in full: every summary field, plus per-kilometre splits.

    With include_samples=true it also returns the heart-rate and pace curve,
    downsampled to at most 500 points. GPS coordinates are left out unless
    include_samples is set, and even then only when the activity recorded them.
    """
    try:
        wanted = int(activity_id)
    except (TypeError, ValueError):
        raise ToolInput(f"activity_id must be a whole number, got {activity_id!r}") from None

    found = rows("SELECT * FROM activities WHERE activity_id=?", (wanted,))
    if not found:
        raise ToolInput(f"no activity with id {wanted}; use list_activities to find one")

    record = found[0]
    raw = record.pop("raw", None)
    payload: dict[str, Any] = {"activity": drop_nulls(record)}

    # Laps are not in the export's summary data, so splits are derived from the
    # samples when they were imported.
    sample_count = conn().execute(
        "SELECT COUNT(*) FROM activity_samples WHERE activity_id=?", (wanted,)
    ).fetchone()[0]
    payload["sample_count"] = sample_count

    if sample_count:
        payload["splits_per_km"] = _splits(wanted)

    if include_samples:
        if not sample_count:
            payload["samples"] = []
            payload["samples_note"] = (
                "no samples stored for this activity; re-run the importer with --with-samples"
            )
        else:
            step = max(1, sample_count // SAMPLE_CAP)
            payload["samples"] = [
                drop_nulls(r)
                for r in rows(
                    "SELECT ts, hr, speed_mps, cadence, power_w, altitude_m, lat, lon"
                    " FROM (SELECT *, ROW_NUMBER() OVER (ORDER BY ts) - 1 AS rn"
                    "       FROM activity_samples WHERE activity_id=?)"
                    " WHERE rn % ? = 0 ORDER BY ts LIMIT ?",
                    (wanted, step, SAMPLE_CAP),
                )
            ]
            payload["samples_downsampled_every"] = step
    elif raw:
        payload["extra_fields_available"] = True

    return compact(payload)


def _splits(activity_id: int) -> list[dict]:
    """Per-kilometre splits, integrated from the sample speeds."""
    samples = rows(
        "SELECT ts, speed_mps, hr FROM activity_samples"
        " WHERE activity_id=? ORDER BY ts",
        (activity_id,),
    )
    if len(samples) < 2:
        return []

    splits: list[dict] = []
    distance = 0.0
    km = 1
    start_ts = datetime.fromisoformat(samples[0]["ts"].replace("Z", "+00:00"))
    hrs: list[int] = []

    for prev, cur in zip(samples, samples[1:]):
        t0 = datetime.fromisoformat(prev["ts"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(cur["ts"].replace("Z", "+00:00"))
        dt = (t1 - t0).total_seconds()
        if dt <= 0 or dt > 60:  # a gap means the watch paused; do not integrate it
            continue
        speed = prev["speed_mps"] or 0.0
        distance += speed * dt
        if cur["hr"]:
            hrs.append(int(cur["hr"]))
        if distance >= km * 1000:
            seconds = (t1 - start_ts).total_seconds()
            splits.append(
                {
                    "km": km,
                    "elapsed_s": round(seconds, 1),
                    "avg_hr": round(sum(hrs) / len(hrs)) if hrs else None,
                }
            )
            km += 1
            start_ts = t1
            hrs = []
            if len(splits) >= ROW_CAP:
                break

    # Report each split's own duration, not the running total.
    out: list[dict] = []
    previous = 0.0
    for s in splits:
        out.append(
            drop_nulls(
                {
                    "km": s["km"],
                    "split_s": round(s["elapsed_s"], 1),
                    "avg_hr": s["avg_hr"],
                }
            )
        )
        previous = s["elapsed_s"]
    return out


DAILY_METRICS = {
    "steps": "steps",
    "resting_hr": "resting_hr",
    "calories": "calories",
    "intensity_minutes": "intensity_minutes",
    "stress": "stress_avg",
    "stress_avg": "stress_avg",
    "body_battery_high": "body_battery_high",
    "body_battery_low": "body_battery_low",
}


@tool
def get_daily_metrics(
    start_date: str,
    end_date: str | None = None,
    metrics: list[str] | None = None,
) -> str:
    """Per-day wellness numbers: steps, resting heart rate, stress, Body Battery.

    Dates are ISO strings; a missing end_date means today. `metrics` picks a
    subset from steps, resting_hr, calories, intensity_minutes, stress,
    body_battery_high, body_battery_low; omit it for all of them. At most 200
    days, newest first.
    """
    start, end = date_window(start_date, end_date)

    if metrics:
        if isinstance(metrics, str):
            metrics = [metrics]
        unknown = [m for m in metrics if m not in DAILY_METRICS]
        if unknown:
            raise ToolInput(
                f"unknown metric(s) {unknown}; choose from {sorted(set(DAILY_METRICS))}"
            )
        columns = list(dict.fromkeys(DAILY_METRICS[m] for m in metrics))
    else:
        columns = list(dict.fromkeys(DAILY_METRICS.values()))

    found = rows(
        f"SELECT date, {', '.join(columns)} FROM daily_summary"
        " WHERE date BETWEEN ? AND ? ORDER BY date DESC LIMIT ?",
        (start, end, ROW_CAP),
    )
    total = conn().execute(
        "SELECT COUNT(*) FROM daily_summary WHERE date BETWEEN ? AND ?", (start, end)
    ).fetchone()[0]

    return compact(
        {
            "range": {"start": start, "end": end},
            "count_returned": len(found),
            "count_matching": total,
            "truncated": total > len(found),
            "days": [drop_nulls(r) for r in found],
        }
    )


@tool
def get_sleep(start_date: str, end_date: str | None = None) -> str:
    """Per-night sleep: start and end, total, stage breakdown and score.

    Dates are ISO strings and name the night's calendar date as Garmin assigns
    it; a missing end_date means today. Durations are seconds. At most 200
    nights, newest first.
    """
    start, end = date_window(start_date, end_date)
    found = rows(
        "SELECT date, start_utc, end_utc, total_s, deep_s, light_s, rem_s, awake_s,"
        " sleep_score FROM sleep WHERE date BETWEEN ? AND ?"
        " ORDER BY date DESC LIMIT ?",
        (start, end, ROW_CAP),
    )
    total = conn().execute(
        "SELECT COUNT(*) FROM sleep WHERE date BETWEEN ? AND ?", (start, end)
    ).fetchone()[0]
    return compact(
        {
            "range": {"start": start, "end": end},
            "count_returned": len(found),
            "count_matching": total,
            "truncated": total > len(found),
            "nights": [drop_nulls(r) for r in found],
        }
    )


@tool
def get_hrv(start_date: str, end_date: str | None = None) -> str:
    """Nightly heart-rate variability in milliseconds, with status and baseline band.

    Dates are ISO strings; a missing end_date means today. At most 200 nights,
    newest first.
    """
    start, end = date_window(start_date, end_date)
    found = rows(
        "SELECT date, nightly_avg_ms, status, baseline_low, baseline_high FROM hrv"
        " WHERE date BETWEEN ? AND ? ORDER BY date DESC LIMIT ?",
        (start, end, ROW_CAP),
    )
    total = conn().execute(
        "SELECT COUNT(*) FROM hrv WHERE date BETWEEN ? AND ?", (start, end)
    ).fetchone()[0]
    return compact(
        {
            "range": {"start": start, "end": end},
            "count_returned": len(found),
            "count_matching": total,
            "truncated": total > len(found),
            "nights": [drop_nulls(r) for r in found],
        }
    )


# metric -> (table, date column, value expression)
TREND_METRICS: dict[str, tuple[str, str, str]] = {
    "steps": ("daily_summary", "date", "steps"),
    "resting_hr": ("daily_summary", "date", "resting_hr"),
    "calories": ("daily_summary", "date", "calories"),
    "intensity_minutes": ("daily_summary", "date", "intensity_minutes"),
    "stress": ("daily_summary", "date", "stress_avg"),
    "body_battery_high": ("daily_summary", "date", "body_battery_high"),
    "sleep_total": ("sleep", "date", "total_s"),
    "sleep_score": ("sleep", "date", "sleep_score"),
    "deep_sleep": ("sleep", "date", "deep_s"),
    "hrv": ("hrv", "date", "nightly_avg_ms"),
    "weight": ("body", "date", "weight_kg"),
    "vo2max": ("vo2max", "date", "vo2max"),
    "readiness": ("training_readiness", "date", "score"),
    "acute_load": ("training_readiness", "date", "acute_load"),
    "distance": ("activities", "local_date", "distance_m"),
    "duration": ("activities", "local_date", "duration_s"),
    "activity_count": ("activities", "local_date", "1"),
}

PERIOD_FORMAT = {"week": "%Y-W%W", "month": "%Y-%m", "year": "%Y", "day": "%Y-%m-%d"}


@tool
def get_trends(
    metric: str,
    period: str,
    start_date: str,
    end_date: str | None = None,
) -> str:
    """Aggregate one metric per week, month, year or day: sum, mean, min, max, count.

    `metric` is one of steps, resting_hr, calories, intensity_minutes, stress,
    body_battery_high, sleep_total, sleep_score, deep_sleep, hrv, weight,
    distance, duration, activity_count, vo2max, readiness (Garmin's morning
    training readiness, 0-100), acute_load. `period` is week, month, year or day.
    Use this for "how has X trended" questions instead of pulling every day.
    Distances are metres, durations seconds, weight kilograms, HRV milliseconds.
    """
    key = str(metric).strip().lower()
    if key not in TREND_METRICS:
        raise ToolInput(f"unknown metric {metric!r}; choose from {sorted(TREND_METRICS)}")
    bucket = str(period).strip().lower()
    if bucket not in PERIOD_FORMAT:
        raise ToolInput(f"period must be one of {sorted(PERIOD_FORMAT)}, got {period!r}")

    table, date_column, expression = TREND_METRICS[key]
    start, end = date_window(start_date, end_date)
    if table not in existing_tables():
        raise ToolInput(
            f"no {key} data in this database yet; the next sync adds the {table} table"
        )

    found = rows(
        f"SELECT strftime('{PERIOD_FORMAT[bucket]}', {date_column}) AS period,"
        f" COUNT({expression}) AS n,"
        f" SUM({expression}) AS sum,"
        f" AVG({expression}) AS mean,"
        f" MIN({expression}) AS min,"
        f" MAX({expression}) AS max"
        f" FROM {table} WHERE {date_column} BETWEEN ? AND ?"
        f" GROUP BY period ORDER BY period DESC LIMIT ?",
        (start, end, ROW_CAP),
    )
    for row in found:
        for field in ("sum", "mean", "min", "max"):
            if isinstance(row.get(field), float):
                row[field] = round(row[field], 3)

    return compact(
        {
            "metric": key,
            "period": bucket,
            "range": {"start": start, "end": end},
            "unit": _unit_for(key),
            "count_returned": len(found),
            "periods": found,
        }
    )


def _unit_for(metric: str) -> str | None:
    if metric in ("distance",):
        return "m"
    if metric in ("duration", "sleep_total", "deep_sleep"):
        return "s"
    if metric == "weight":
        return "kg"
    if metric == "vo2max":
        return "ml/kg/min"
    if metric == "readiness":
        return "0-100"
    if metric == "hrv":
        return "ms"
    if metric == "steps":
        return "steps"
    if metric in ("resting_hr",):
        return "bpm"
    return None


# ---------------------------------------------------------------------------
# run_sql
#
# Three layers hold: the connection is read-only at the SQLite level, the
# statement must be a single SELECT or WITH, and an authorizer callback denies
# every operation except reading.
# ---------------------------------------------------------------------------

COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)

ALLOWED_AUTH = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def strip_comments(sql: str) -> str:
    return COMMENT_RE.sub(" ", sql).strip()


def check_query(sql: str) -> str:
    """Accept exactly one SELECT or WITH statement. Reject everything else."""
    if not isinstance(sql, str) or not sql.strip():
        raise ToolInput("query is required")
    bare = strip_comments(sql)
    if not bare:
        raise ToolInput("query contains no statement")

    body = bare.rstrip().rstrip(";").strip()
    if not body:
        raise ToolInput("query contains no statement")
    if ";" in body:
        raise ToolInput("only one statement is allowed; remove the ';'")

    first = body.split(None, 1)[0].lower()
    if first not in ("select", "with"):
        raise ToolInput(
            f"only SELECT (or WITH ... SELECT) is allowed, got {first.upper()!r}. "
            "This database is read-only."
        )
    if not sqlite3.complete_statement(body + ";"):
        raise ToolInput("query is not a complete statement")
    return body


def _authorizer(action: int, *_args: Any) -> int:
    return sqlite3.SQLITE_OK if action in ALLOWED_AUTH else sqlite3.SQLITE_DENY


@tool
def run_sql(query: str) -> str:
    """Run one read-only SELECT against the Garmin database.

    For questions the other tools do not cover, such as joining sleep to
    activities. One statement only, starting with SELECT or WITH; anything else
    is rejected and the connection itself is read-only, so nothing can be
    changed. At most 200 rows.

    Tables and key columns, all units SI:
      activities(activity_id, start_time_utc, start_time_local, local_date, type,
                 name, duration_s, distance_m, avg_hr, max_hr, calories,
                 elevation_gain_m, avg_speed_mps, avg_pace_s_per_km,
                 training_effect, raw)
      activity_samples(activity_id, ts, hr, speed_mps, cadence, power_w,
                       altitude_m, lat, lon)
      daily_summary(date, steps, resting_hr, calories, intensity_minutes,
                    stress_avg, body_battery_high, body_battery_low, raw)
      sleep(date, start_utc, end_utc, total_s, deep_s, light_s, rem_s, awake_s,
            sleep_score, raw)
      hrv(date, nightly_avg_ms, status, baseline_low, baseline_high, raw)
      body(date, weight_kg, body_fat_pct, bmi, raw)
          weigh-ins; raw.sourceType USER_SETTING means a profile value, not a scale
      vo2max(date, vo2max, vo2max_cycling, fitness_age, raw)
          sparse: one row per date Garmin published a new estimate, ml/kg/min
      training_readiness(date, score, level, feedback, measured_at_local,
                         latest_score, latest_at_local, recovery_time_min,
                         acute_load, sleep_factor_pct, sleep_history_factor_pct,
                         recovery_factor_pct, acwr_factor_pct, hrv_factor_pct,
                         stress_factor_pct, raw)
          score is the on-waking value; raw holds every recalculation that day
      import_log(id, file_name, sha256, rows_added, imported_at, status, note)
    """
    body = check_query(query)
    connection = conn()
    connection.set_authorizer(_authorizer)
    try:
        cursor = connection.execute(body)
        columns = [d[0] for d in cursor.description or []]
        fetched = cursor.fetchmany(ROW_CAP + 1)
    except sqlite3.DatabaseError as exc:
        raise ToolInput(f"query rejected: {exc}") from None
    finally:
        connection.set_authorizer(None)

    truncated = len(fetched) > ROW_CAP
    data = [dict(zip(columns, r)) for r in fetched[:ROW_CAP]]
    return compact(
        {
            "columns": columns,
            "row_count": len(data),
            "truncated": truncated,
            "rows": data,
        }
    )


def main() -> None:
    level = os.environ.get("GARMIN_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s %(message)s",
        stream=sys.stderr,  # stdout is the MCP channel; never write to it
    )
    # The SDK narrates every request at INFO; that belongs behind GARMIN_LOG_LEVEL.
    logging.getLogger("mcp").setLevel(level)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
