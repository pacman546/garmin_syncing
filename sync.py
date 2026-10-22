"""Phase 2: pull new days straight from Garmin Connect into the same database.

Optional. Phase 1 keeps working whether or not this ever runs, and the MCP
server does not change: this writes to the same tables the importer writes to.

    python -m garmin_claude.sync login          # once, interactively
    python -m garmin_claude.sync                # daily, from cron

Three rules hold this module together:

1. Your password is typed by you, into `getpass()`, on the first run only. It is
   never stored, never passed as an argument, and never read from the
   environment - so it cannot end up in shell history, a config file, or a
   process list. Later runs resume from the saved tokens.
2. Only a fixed list of `get_*` methods is reachable. The underlying library can
   also upload, edit, delete and schedule data; `ReadOnlyGarmin` does not expose
   any of it, and the client object never leaves this module.
3. One sync per day, only dates newer than what is already stored, with a pause
   between requests. Repeated password logins are what trigger Garmin's
   rate-limit blocks.

A note on units: the Connect API returns seconds and metres, while the export
file returns milliseconds and centimetres. The builders here are separate from
the importer's for exactly that reason - sharing them would silently scale
every distance by 100.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from getpass import getpass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import db as dbmod
from .importer import (
    as_float,
    as_int,
    intensity_minutes_total,
    normalize_calendar_date,
    parse_any_timestamp,
    pick,
    stress_level,
    upsert,
)

log = logging.getLogger("garmin_claude.sync")

# The library's token store is a directory holding garmin_tokens.json.
TOKEN_DIR = Path.home() / ".garminconnect"
TOKEN_FILE_NAME = "garmin_tokens.json"
KEYRING_SERVICE = "garmin-claude"
KEYRING_KEY = "garmin_tokens"

# Be gentle: one request at a time, with a pause. Garmin rate-limits.
REQUEST_PAUSE_S = 1.5
# Never walk back further than this on a first sync; use a Phase 1 export for history.
MAX_BACKFILL_DAYS = 90
# Every run re-checks this many of the most recent stored days (the newest one
# included). A day is rarely final when first fetched: today is partial, and a
# watch that syncs late fills in yesterday afterwards. Writes are upserts, so
# re-fetching a settled day changes nothing.
REFRESH_DAYS = 3

# The only methods this module will ever call on the Garmin client.
ALLOWED_METHODS = frozenset(
    {
        "get_activities_by_date",   # ranged, paginates internally
        "get_user_summary",         # one day
        "get_sleep_data",           # one day
        "get_hrv_data",             # one day
        "get_hrv_data_range",       # ranged; preferred where available
        "get_body_composition",     # ranged
        "get_max_metrics_range",    # ranged; VO2 max (running, cycling)
        "get_training_readiness_range",  # ranged; several entries per day
    }
)


class SyncError(RuntimeError):
    """Something went wrong that the operator needs to read about."""


# ---------------------------------------------------------------------------
# The read-only wrapper
# ---------------------------------------------------------------------------

class ReadOnlyGarmin:
    """A keyhole onto the Garmin client: only `ALLOWED_METHODS` pass through.

    The library also has methods that upload, edit, delete and schedule data.
    None of them is reachable here: `__getattribute__` refuses every name that
    is not on the whitelist, including the private attribute holding the client
    and its name-mangled spelling, so neither a typo nor a later edit to this
    file can quietly gain write access.

    This is a guard rail, not a sandbox. Anything running inside this process
    could still reach the object through the garbage collector; what it stops is
    the realistic failure - code in this repository calling a write method,
    directly or by passing the client somewhere that does.
    """

    __slots__ = ("__holder",)

    # The only non-whitelisted names that still resolve, so the object stays
    # printable and introspectable.
    _PASSTHROUGH = frozenset({"__class__", "__dir__", "__repr__", "__slots__"})

    def __init__(self, client: Any) -> None:
        object.__setattr__(self, "_ReadOnlyGarmin__holder", (client,))

    def __getattribute__(self, name: str) -> Any:
        if name in ALLOWED_METHODS:
            client = object.__getattribute__(self, "_ReadOnlyGarmin__holder")[0]
            method = getattr(client, name, None)
            if method is None:
                raise SyncError(
                    f"the installed garminconnect has no {name!r}; its API has changed. "
                    "Pin a known version or update garmin_claude/sync.py."
                )

            def guarded(*args: Any, **kwargs: Any) -> Any:
                time.sleep(REQUEST_PAUSE_S)
                log.info("garmin: %s%s", name, args if args else "")
                return method(*args, **kwargs)

            guarded.__name__ = name
            return guarded

        if name in ReadOnlyGarmin._PASSTHROUGH:
            return object.__getattribute__(self, name)

        raise AttributeError(
            f"{name!r} is not on the read-only whitelist; "
            f"garmin-claude never writes to Garmin. "
            f"Allowed: {sorted(ALLOWED_METHODS)}"
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ReadOnlyGarmin is immutable")

    def __dir__(self) -> list[str]:
        return sorted(ALLOWED_METHODS)

    def __repr__(self) -> str:
        return f"<ReadOnlyGarmin: {len(ALLOWED_METHODS)} read methods>"


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

def _keyring():
    try:
        import keyring

        return keyring
    except ImportError:
        return None


def save_tokens_to_keychain() -> bool:
    """Move the library's token directory into the OS keychain and delete it.

    The refresh token grants lasting access to the account, so it is treated
    like a password. Returns False when `keyring` is not installed, in which
    case the files stay where the library put them, owner-readable only.
    """
    keyring = _keyring()
    if keyring is None:
        log.warning("keyring is not installed; tokens stay in %s", TOKEN_DIR)
        return False
    if not TOKEN_DIR.exists():
        return False

    bundle: dict[str, str] = {}
    for path in sorted(TOKEN_DIR.rglob("*")):
        if path.is_file():
            bundle[str(path.relative_to(TOKEN_DIR))] = path.read_text(encoding="utf-8")
    if not bundle:
        return False

    keyring.set_password(KEYRING_SERVICE, KEYRING_KEY, json.dumps(bundle))
    for path in sorted(TOKEN_DIR.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
    log.info("tokens moved into the OS keychain (%s); files deleted", KEYRING_SERVICE)
    return True


def restore_tokens_from_keychain() -> bool:
    """Write the keychain copy back out so the library can read it."""
    keyring = _keyring()
    if keyring is None:
        return False
    stored = keyring.get_password(KEYRING_SERVICE, KEYRING_KEY)
    if not stored:
        return False
    try:
        bundle = json.loads(stored)
    except json.JSONDecodeError:
        log.warning("keychain entry is not readable; ignoring it")
        return False

    TOKEN_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, content in bundle.items():
        target = TOKEN_DIR / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        target.chmod(0o600)
    return True


def forget_tokens() -> None:
    """Delete the local tokens, from disk and from the keychain.

    This does not revoke anything at Garmin: the library's `logout()` only
    removes the local copy. If a token may have leaked, revoke access in
    Garmin's account security settings as well.
    """
    keyring = _keyring()
    if keyring is not None:
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_KEY)
            print("removed the keychain entry")
        except Exception:  # noqa: BLE001 - absent entry is not an error
            pass
    removed = 0
    if TOKEN_DIR.exists():
        for path in sorted(TOKEN_DIR.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                removed += 1
    print(f"removed {removed} local token file(s)")
    print(
        "This did NOT revoke the token at Garmin. To do that, remove access in "
        "Garmin Connect > Account Settings > Security."
    )


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def _ask_mfa() -> str:
    """Prompt for the two-factor code. Passed to the library as a callback so it
    can ask at the moment it needs to, mid-login."""
    return input("Garmin MFA code (from your email or authenticator app): ").strip()


def _token_file() -> Path:
    return TOKEN_DIR / TOKEN_FILE_NAME


def _ensure_tokens_saved(raw_client: Any) -> bool:
    """Make sure the session tokens actually landed on disk.

    `login(tokenstore)` persists them itself, but only on some of its internal
    paths, and a silent miss here is expensive: every later run would fall back
    to a password login, and repeated password logins are exactly what trigger
    Garmin's rate-limit blocks. So verify, and dump explicitly if needed.
    """
    TOKEN_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not _token_file().exists():
        # `Garmin.client.dump(dir)` is the real API in 0.3.x. There is no
        # `.garth`: the library dropped garth in 0.3.0.
        dump = getattr(getattr(raw_client, "client", None), "dump", None)
        if dump is None:
            log.warning(
                "could not save tokens: this garminconnect release exposes no "
                "client.dump(). You will be asked to sign in again next run."
            )
            return False
        try:
            dump(str(TOKEN_DIR))
        except Exception as exc:  # noqa: BLE001
            log.warning("could not save tokens: %s", exc)
            return False

    if _token_file().exists():
        _token_file().chmod(0o600)
        return True
    return False


def connect(email: str | None = None, *, interactive: bool = False) -> ReadOnlyGarmin:
    """Return a read-only client, resuming saved tokens where possible.

    With `interactive=False` (the daily run) this never prompts: if the tokens
    are gone or expired it fails loudly, so cron does not hang forever waiting
    on a password nobody is there to type.
    """
    try:
        from garminconnect import Garmin
    except ImportError as exc:
        raise SyncError(
            "garminconnect is not installed. Phase 2 is optional:\n"
            "    pip install -r requirements-phase2.txt"
        ) from exc

    restore_tokens_from_keychain()
    tokenstore = str(TOKEN_DIR)

    # Every scheduled run takes this path. Constructed with no credentials, so
    # it can only resume - it cannot silently fall back to a password login.
    try:
        client = Garmin()
        client.login(tokenstore)
        log.info("resumed from saved tokens")
        return ReadOnlyGarmin(client)
    except Exception as exc:  # noqa: BLE001 - the library raises several types here
        if not interactive:
            raise SyncError(
                f"could not resume from saved tokens ({exc}). "
                "Run `python -m garmin_claude.sync login` once, by hand, to sign in again."
            ) from exc
        log.info("no usable tokens (%s); signing in", exc)

    # Interactive sign-in. The password is typed here and nowhere else.
    if not email:
        email = input("Garmin Connect email: ").strip()
    if os.environ.get("GARMIN_PASSWORD"):
        raise SyncError(
            "refusing to read your password from the environment. "
            "Run this command in a terminal and type it at the prompt instead."
        )
    password = getpass(f"Garmin Connect password for {email} (not stored): ")
    if not password:
        raise SyncError("no password entered")

    # prompt_mfa lets the library ask for the code itself, at the point it needs
    # it. The alternative, return_on_mfa, hands back a two-step dance whose
    # state argument the library then ignores.
    client = Garmin(email=email, password=password, prompt_mfa=_ask_mfa)
    try:
        client.login(tokenstore)
    except Exception as exc:  # noqa: BLE001
        raise SyncError(f"login failed: {exc}") from exc
    finally:
        del password

    if _ensure_tokens_saved(client):
        save_tokens_to_keychain()
    return ReadOnlyGarmin(client)


# ---------------------------------------------------------------------------
# Row builders for the Connect API
#
# The API speaks seconds and metres. The export file speaks milliseconds and
# centimetres. Keeping these separate from the importer's builders is what stops
# a 100x unit error from creeping in.
# ---------------------------------------------------------------------------

def build_activity_api(rec: dict) -> dict | None:
    activity_id = as_int(pick(rec, "activityId"))
    if activity_id is None:
        return None
    start_local = parse_any_timestamp(pick(rec, "startTimeLocal"))
    start_utc = parse_any_timestamp(pick(rec, "startTimeGMT", "startTimeGmt")) or start_local
    if start_utc is None:
        return None

    activity_type = pick(rec, "activityType")
    if isinstance(activity_type, dict):
        activity_type = pick(activity_type, "typeKey", "typeId")
    if activity_type is not None:
        activity_type = str(activity_type).lower()

    duration_s = as_float(pick(rec, "duration", "elapsedDuration"))
    distance_m = as_float(pick(rec, "distance"))
    pace = duration_s / (distance_m / 1000.0) if (duration_s and distance_m) else None

    return {
        "activity_id": activity_id,
        "start_time_utc": start_utc,
        "start_time_local": start_local,
        "local_date": (start_local or start_utc)[:10],
        "type": activity_type,
        "name": pick(rec, "activityName", "name"),
        "duration_s": duration_s,
        "distance_m": distance_m,
        "avg_hr": as_int(pick(rec, "averageHR", "avgHr")),
        "max_hr": as_int(pick(rec, "maxHR", "maxHr")),
        "calories": as_float(pick(rec, "calories")),
        "elevation_gain_m": as_float(pick(rec, "elevationGain")),
        "avg_speed_mps": as_float(pick(rec, "averageSpeed", "avgSpeed")),
        "avg_pace_s_per_km": pace,
        "training_effect": as_float(pick(rec, "aerobicTrainingEffect")),
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True, default=str),
    }


# Garmin answers for a date it has nothing for with a full response envelope
# whose every measurement is null (and includesWellnessData false). Storing that
# is worse than storing nothing: the row counts as "this date is done", so the
# incremental window never asks for it again once the watch does sync.
DAILY_SIGNAL_FIELDS = (
    "totalSteps", "restingHeartRate", "totalKilocalories", "activeKilocalories",
    "moderateIntensityMinutes", "vigorousIntensityMinutes", "averageStressLevel",
    "bodyBatteryHighestValue", "bodyBatteryLowestValue", "totalDistanceMeters",
    "maxHeartRate", "minHeartRate", "activeSeconds",
)

SLEEP_SIGNAL_FIELDS = (
    "sleepTimeSeconds", "deepSleepSeconds", "lightSleepSeconds", "remSleepSeconds",
    "awakeSleepSeconds", "napTimeSeconds", "sleepStartTimestampGMT",
    "sleepEndTimestampGMT", "sleepScores",
)


def has_any_measurement(rec: dict, fields: Sequence[str]) -> bool:
    """True when at least one real measurement is present and non-null."""
    return any(rec.get(f) is not None for f in fields)


def build_daily_api(rec: dict, date: str) -> dict | None:
    if not rec or not has_any_measurement(rec, DAILY_SIGNAL_FIELDS):
        return None
    return {
        "date": normalize_calendar_date(pick(rec, "calendarDate")) or date,
        "steps": as_int(pick(rec, "totalSteps")),
        "resting_hr": as_int(pick(rec, "restingHeartRate")),
        "calories": as_float(pick(rec, "totalKilocalories")),
        "intensity_minutes": intensity_minutes_total(rec),
        "stress_avg": stress_level(pick(rec, "averageStressLevel")),
        "body_battery_high": as_int(pick(rec, "bodyBatteryHighestValue")),
        "body_battery_low": as_int(pick(rec, "bodyBatteryLowestValue")),
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True, default=str),
    }


def build_sleep_api(payload: dict, date: str) -> dict | None:
    if not payload:
        return None
    dto = payload.get("dailySleepDTO") or payload
    if not isinstance(dto, dict) or not dto:
        return None
    if not has_any_measurement(dto, SLEEP_SIGNAL_FIELDS):
        return None  # an envelope of nulls: Garmin has no sleep for this date
    deep = as_int(pick(dto, "deepSleepSeconds"))
    light = as_int(pick(dto, "lightSleepSeconds"))
    rem = as_int(pick(dto, "remSleepSeconds"))
    total = as_int(pick(dto, "sleepTimeSeconds"))
    if total is None:
        parts = [p for p in (deep, light, rem) if p is not None]
        total = sum(parts) if parts else None
    scores = dto.get("sleepScores") or {}
    overall = scores.get("overall") if isinstance(scores, dict) else None
    score = as_int(pick(overall, "value") if isinstance(overall, dict) else overall)

    return {
        "date": normalize_calendar_date(pick(dto, "calendarDate")) or date,
        "start_utc": parse_any_timestamp(pick(dto, "sleepStartTimestampGMT")),
        "end_utc": parse_any_timestamp(pick(dto, "sleepEndTimestampGMT")),
        "total_s": total,
        "deep_s": deep,
        "light_s": light,
        "rem_s": rem,
        "awake_s": as_int(pick(dto, "awakeSleepSeconds")),
        "sleep_score": score,
        "raw": json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
    }


def build_hrv_api(payload: dict, date: str) -> dict | None:
    if not payload:
        return None
    summary = payload.get("hrvSummary") or payload
    if not isinstance(summary, dict) or not summary:
        return None
    baseline = summary.get("baseline") or {}
    if not isinstance(baseline, dict):
        baseline = {}
    nightly = as_float(pick(summary, "lastNightAvg", "weeklyAvg"))
    status = pick(summary, "status", "hrvStatus")
    if nightly is None and status is None:
        return None
    return {
        "date": normalize_calendar_date(pick(summary, "calendarDate")) or date,
        "nightly_avg_ms": nightly,
        "status": str(status) if status is not None else None,
        "baseline_low": as_float(pick(baseline, "balancedLow", "lowUpper")),
        "baseline_high": as_float(pick(baseline, "balancedUpper", "markerValue")),
        "raw": json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str),
    }


def build_body_api(rec: dict) -> dict | None:
    date = normalize_calendar_date(pick(rec, "calendarDate", "date"))
    if date is None:
        return None
    weight = as_float(pick(rec, "weight"))
    if weight is not None and weight > 500:  # grams, as the API sends them
        weight /= 1000.0
    fat = as_float(pick(rec, "bodyFat"))
    bmi = as_float(pick(rec, "bmi"))
    if weight is None and fat is None and bmi is None:
        return None
    return {
        "date": date,
        "weight_kg": weight,
        "body_fat_pct": fat,
        "bmi": bmi,
        "raw": json.dumps(rec, separators=(",", ":"), sort_keys=True, default=str),
    }


def build_vo2max_rows(entries: Any) -> list[dict]:
    """VO2 max entries -> one row per date.

    Garmin returns one entry per date it published a new estimate, with a
    `generic` (running) block and a `cycling` block, either of which may be
    null. Entries are merged per date so a running and a cycling estimate on the
    same day cannot overwrite each other.
    """
    merged: dict[str, dict] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        generic = entry.get("generic") if isinstance(entry.get("generic"), dict) else {}
        cycling = entry.get("cycling") if isinstance(entry.get("cycling"), dict) else {}
        running_value = as_float(pick(generic, "vo2MaxPreciseValue", "vo2MaxValue"))
        cycling_value = as_float(pick(cycling, "vo2MaxPreciseValue", "vo2MaxValue"))
        if running_value is None and cycling_value is None:
            continue
        date = normalize_calendar_date(
            pick(generic, "calendarDate") or pick(cycling, "calendarDate")
        )
        if date is None:
            continue
        row = merged.setdefault(
            date,
            {"date": date, "vo2max": None, "vo2max_cycling": None,
             "fitness_age": None, "_raw": []},
        )
        if running_value is not None:
            row["vo2max"] = running_value
        if cycling_value is not None:
            row["vo2max_cycling"] = cycling_value
        fitness_age = as_float(pick(generic, "fitnessAge"))
        if fitness_age is not None:
            row["fitness_age"] = fitness_age
        row["_raw"].append(entry)

    rows = []
    for row in merged.values():
        raw = row.pop("_raw")
        row["raw"] = json.dumps(raw if len(raw) > 1 else raw[0],
                                separators=(",", ":"), sort_keys=True, default=str)
        rows.append(row)
    return rows


READINESS_MORNING = "AFTER_WAKEUP_RESET"


def _local_ts(value: Any) -> str | None:
    """'2026-10-17T09:50:43.0' -> '2026-10-17T09:50:43' (local wall clock)."""
    if not value:
        return None
    text = str(value)
    return text.split(".", 1)[0] if "." in text else text


def build_readiness_rows(entries: Any) -> list[dict]:
    """Training readiness entries -> one row per day.

    Garmin recalculates readiness several times a day; each entry says why in
    `inputContext`. The row's `score` and factors come from the on-waking entry
    (`AFTER_WAKEUP_RESET`, the Morning Report number), falling back to the
    day's first entry on devices that do not tag them; `latest_score` is the
    day's last recalculation. Every entry is kept in `raw`.
    """
    by_day: dict[str, list[dict]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict) or entry.get("score") is None:
            continue
        day = normalize_calendar_date(entry.get("calendarDate"))
        if day:
            by_day.setdefault(day, []).append(entry)

    rows = []
    for day, items in by_day.items():
        items.sort(key=lambda e: str(e.get("timestamp") or e.get("timestampLocal") or ""))
        morning = next(
            (e for e in items if e.get("inputContext") == READINESS_MORNING), items[0]
        )
        latest = items[-1]
        rows.append(
            {
                "date": day,
                "score": as_int(morning.get("score")),
                "level": morning.get("level"),
                "feedback": morning.get("feedbackShort"),
                "measured_at_local": _local_ts(morning.get("timestampLocal")),
                "latest_score": as_int(latest.get("score")),
                "latest_at_local": _local_ts(latest.get("timestampLocal")),
                "recovery_time_min": as_int(morning.get("recoveryTime")),
                "acute_load": as_int(morning.get("acuteLoad")),
                "sleep_factor_pct": as_int(morning.get("sleepScoreFactorPercent")),
                "sleep_history_factor_pct": as_int(morning.get("sleepHistoryFactorPercent")),
                "recovery_factor_pct": as_int(morning.get("recoveryTimeFactorPercent")),
                "acwr_factor_pct": as_int(morning.get("acwrFactorPercent")),
                "hrv_factor_pct": as_int(morning.get("hrvFactorPercent")),
                "stress_factor_pct": as_int(morning.get("stressHistoryFactorPercent")),
                "raw": json.dumps(items, separators=(",", ":"), sort_keys=True, default=str),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Sync
#
# Two kinds of endpoint, two kinds of window:
#
# - Per-day endpoints (daily summary, sleep) cost one request per day, so their
#   window is bounded by --days and anchored to the newest stored row.
# - Ranged endpoints (activities, HRV, weigh-ins, VO2 max, readiness) cost one
#   request per chunk of up to RANGED_CHUNK_DAYS, so their window always runs
#   through today. It is anchored to how far that source has been *checked*
#   (sync_state), not to its newest row: weigh-ins and VO2 max go weeks without
#   a row, and a window anchored to the last row would stop reaching today.
# ---------------------------------------------------------------------------

RANGED_CHUNK_DAYS = 60


def latest_date(conn: sqlite3.Connection, table: str) -> str | None:
    """The newest calendar date stored in `table`, or None.

    Returns None for a table that has no date column at all - activity_samples
    is keyed by activity and timestamp, not by date - rather than guessing a
    column name that does not exist.
    """
    column = dbmod.DATE_COLUMN.get(table)
    if column is None:
        return None
    row = conn.execute(f"SELECT MAX({column}) AS last FROM {table}").fetchone()
    return row["last"] if row and row["last"] else None


def dates_to_fetch(
    conn: sqlite3.Connection, table: str, today: date_cls, max_days: int
) -> list[str]:
    """Per-day endpoints: the last `REFRESH_DAYS` stored days through today.

    Starting at "the day after the newest row" would be wrong: the newest row is
    often a partial day (fetched while the watch was mid-day, or before it had
    synced), and it would then never be corrected. So the window reaches back
    over the most recent days and re-fetches them.

    On an empty table this walks back `max_days` only: the full history belongs
    in a Phase 1 export, not in a few hundred API calls.
    """
    last = latest_date(conn, table)
    if last:
        start = date_cls.fromisoformat(last) - timedelta(days=REFRESH_DAYS - 1)
    else:
        start = today - timedelta(days=max_days - 1)
    floor = today - timedelta(days=MAX_BACKFILL_DAYS - 1)
    start = max(start, floor)
    if start > today:
        return []
    span = (today - start).days + 1
    return [(start + timedelta(days=i)).isoformat() for i in range(min(span, max_days))]


def checked_through(conn: sqlite3.Connection, source: str) -> str | None:
    row = conn.execute(
        "SELECT checked_through FROM sync_state WHERE source=?", (source,)
    ).fetchone()
    return row["checked_through"] if row else None


def mark_checked(conn: sqlite3.Connection, source: str, day: date_cls) -> None:
    conn.execute(
        "INSERT INTO sync_state (source, checked_through) VALUES (?, ?)"
        " ON CONFLICT(source) DO UPDATE SET checked_through=excluded.checked_through",
        (source, day.isoformat()),
    )


def ranged_window(
    conn: sqlite3.Connection, source: str, table: str, today: date_cls, max_days: int
) -> tuple[date_cls, date_cls]:
    """Ranged endpoints: from just before the last check (or the last row, or
    `max_days` back on a first run) through today. Always ends today."""
    anchor = checked_through(conn, source) or latest_date(conn, table)
    if anchor:
        start = date_cls.fromisoformat(anchor) - timedelta(days=REFRESH_DAYS - 1)
    else:
        start = today - timedelta(days=max_days - 1)
    return min(start, today), today


def chunk_range(start: date_cls, end: date_cls, size: int = RANGED_CHUNK_DAYS) -> list[tuple[str, str]]:
    """Split [start, end] into inclusive pieces of at most `size` days."""
    pieces = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=size - 1), end)
        pieces.append((cursor.isoformat(), stop.isoformat()))
        cursor = stop + timedelta(days=1)
    return pieces


# source -> (table, key, method, payload -> rows)
def _activity_rows(payload: Any) -> list[dict]:
    return [r for r in (build_activity_api(x) for x in (payload or [])) if r]


def _hrv_rows(payload: Any) -> list[dict]:
    entries = payload.get("hrvSummaries") if isinstance(payload, dict) else None
    rows = []
    for entry in entries or []:
        if not isinstance(entry, dict) or not entry.get("calendarDate"):
            continue  # SQLite would accept a NULL text key; never give it one
        row = build_hrv_api({"hrvSummary": entry}, entry["calendarDate"])
        if row and row.get("date"):
            rows.append(row)
    return rows


def _body_rows(payload: Any) -> list[dict]:
    entries = []
    if isinstance(payload, dict):
        entries = payload.get("dateWeightList") or payload.get("dailyWeightSummaries") or []
    return [r for r in (build_body_api(x) for x in entries) if r]


RANGED_SOURCES: tuple[tuple[str, str, tuple[str, ...], str, Callable[[Any], list[dict]]], ...] = (
    ("activities", "activities", ("activity_id",), "get_activities_by_date", _activity_rows),
    ("hrv", "hrv", ("date",), "get_hrv_data_range", _hrv_rows),
    ("body", "body", ("date",), "get_body_composition", _body_rows),
    ("vo2max", "vo2max", ("date",), "get_max_metrics_range", build_vo2max_rows),
    ("training_readiness", "training_readiness", ("date",),
     "get_training_readiness_range", build_readiness_rows),
)


def fetch_ranged(
    client: ReadOnlyGarmin,
    conn: sqlite3.Connection,
    source: str,
    table: str,
    key: tuple[str, ...],
    method: str,
    to_rows: Callable[[Any], list[dict]],
    start: date_cls,
    end: date_cls,
) -> int:
    """Fetch [start, end] in chunks and upsert. Raises on the first failed chunk,
    leaving sync_state unchanged so the next run asks for the same window."""
    count = 0
    pieces = chunk_range(start, end)
    for a, b in pieces:
        payload = getattr(client, method)(a, b)
        for row in to_rows(payload):
            count += upsert(conn, table, key, row)
    mark_checked(conn, source, end)
    log.info("%s %s..%s: %d new (%d request%s)", source, start, end, count,
             len(pieces), "" if len(pieces) == 1 else "s")
    return count


def _log_run(conn: sqlite3.Connection, name: str, added: dict[str, int], errors: list[str]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        "INSERT INTO import_log (file_name, sha256, rows_added, imported_at, status, note)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            name,
            "live-sync",
            sum(added.values()),
            now,
            "ok" if not errors else "partial",
            json.dumps({"rows_added": added, "errors": errors[:20]}, separators=(",", ":")),
        ),
    )
    conn.commit()
    for err in errors:
        log.warning("%s", err)


def sync(
    client: ReadOnlyGarmin,
    conn: sqlite3.Connection,
    *,
    today: date_cls | None = None,
    max_days: int = 7,
    errors_out: list[str] | None = None,
) -> dict[str, int]:
    """Pull everything new, and re-check the last few days. Returns rows added per table.

    Requests that failed are logged to import_log (status "partial") and, if
    `errors_out` is given, appended to it so the caller can tell the run was not
    clean.
    """
    today = today or date_cls.today()
    added: dict[str, int] = {}
    errors: list[str] = []
    hrv_fallback = False

    for source, table, key, method, to_rows in RANGED_SOURCES:
        start, end = ranged_window(conn, source, table, today, max_days)
        try:
            added[table] = fetch_ranged(
                client, conn, source, table, key, method, to_rows, start, end
            )
        except Exception as exc:  # noqa: BLE001
            if source == "hrv":
                # Older library releases have no ranged HRV; one call per day works.
                log.info("ranged HRV unavailable (%s); falling back to one call per day", exc)
                hrv_fallback = True
            else:
                errors.append(f"{source}: {exc}")
        conn.commit()

    per_day: list[tuple[str, tuple[str, ...], str, Callable[[Any, str], dict | None]]] = [
        ("daily_summary", ("date",), "get_user_summary", build_daily_api),
        ("sleep", ("date",), "get_sleep_data", build_sleep_api),
    ]
    if hrv_fallback:
        per_day.append(("hrv", ("date",), "get_hrv_data", build_hrv_api))

    for table, key, method, builder in per_day:
        days = dates_to_fetch(conn, table, today, max_days)
        count = 0
        for day in days:
            try:
                payload = getattr(client, method)(day)
            except Exception as exc:  # noqa: BLE001 - a missing day is normal
                errors.append(f"{table} {day}: {exc}")
                continue
            row = builder(payload, day)
            if row:
                count += upsert(conn, table, key, row)
        if days:
            added[table] = count
            log.info("%s %s..%s: %d new", table, days[0], days[-1], count)
        conn.commit()

    _log_run(conn, f"garmin-connect-sync {today.isoformat()}", added, errors)
    if errors_out is not None:
        errors_out.extend(errors)
    return added


def backfill(
    client: ReadOnlyGarmin,
    conn: sqlite3.Connection,
    since: date_cls,
    *,
    today: date_cls | None = None,
    errors_out: list[str] | None = None,
) -> dict[str, int]:
    """One-off: pull every ranged source from `since` through today.

    Cheap because these endpoints answer a whole chunk per request - about one
    request per source per two months. The per-day sources (daily summary,
    sleep) are deliberately left out: a year of them is several hundred
    requests, which is what the Garmin data export is for.
    """
    today = today or date_cls.today()
    if since > today:
        raise SyncError(f"--since {since} is in the future")
    added: dict[str, int] = {}
    errors: list[str] = []
    for source, table, key, method, to_rows in RANGED_SOURCES:
        try:
            added[table] = fetch_ranged(
                client, conn, source, table, key, method, to_rows, since, today
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")
        conn.commit()
    _log_run(conn, f"garmin-connect-backfill since {since.isoformat()}", added, errors)
    if errors_out is not None:
        errors_out.extend(errors)
    return added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _report_failures(failed: list[str]) -> int:
    """Exit non-zero when any request failed, so the scheduler's wrapper retries.

    Returning 0 here made a run that lost requests look like a success: the
    wrapper only retries on a non-zero exit. Retrying is safe, since every write
    is an upsert and the window re-checks recent days anyway.
    """
    if not failed:
        return 0
    print(f"{len(failed)} request(s) failed; the next attempt will retry them:", file=sys.stderr)
    for line in failed[:5]:
        print(f"  - {line}", file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m garmin_claude.sync",
        description="Phase 2: pull new Garmin days into garmin.db.",
        epilog="Your password is typed at the prompt and never stored.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="sync",
        choices=("sync", "login", "logout", "status", "backfill"),
        help="login signs in once interactively; sync is the daily run; "
             "backfill pulls the ranged sources (activities, HRV, weigh-ins, "
             "VO2 max, readiness) back to --since",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="backfill only: ISO date to start from (default: two years ago)",
    )
    parser.add_argument("--email", default=None, help="Garmin Connect email (login only)")
    parser.add_argument("--db", type=Path, default=None, help="database path")
    parser.add_argument("--days", type=int, default=7, help="most days to fetch in one run")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    if args.command == "logout":
        forget_tokens()
        return 0

    conn = dbmod.connect_rw(args.db)
    try:
        if args.command == "status":
            for table in dbmod.DATA_TABLES:
                rows = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                newest = latest_date(conn, table)
                when = newest if newest else ("n/a" if rows else "-")
                print(f"{table:<18} {rows:>8} rows   latest {when}")
            last = conn.execute(
                "SELECT * FROM import_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if last:
                print(f"\nlast import      {last['imported_at']} "
                      f"({last['status']}, {last['rows_added']} rows)")
            return 0

        try:
            client = connect(args.email, interactive=(args.command == "login"))
        except SyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if args.command == "login":
            print("signed in; tokens saved. Scheduled runs will not prompt again.")

        if args.command == "backfill":
            try:
                since = (
                    date_cls.fromisoformat(args.since)
                    if args.since
                    else date_cls.today() - timedelta(days=730)
                )
            except ValueError:
                print(f"error: --since must be an ISO date like 2025-01-01, got {args.since!r}",
                      file=sys.stderr)
                return 2
            failed: list[str] = []
            added = backfill(client, conn, since, errors_out=failed)
            total = sum(added.values())
            print(f"backfill since {since}: {total} new rows"
                  + (f" ({', '.join(f'{t}+{n}' for t, n in sorted(added.items()) if n)})" if total else ""))
            return _report_failures(failed)

        failed = []
        added = sync(client, conn, max_days=max(1, args.days), errors_out=failed)
        total = sum(added.values())
        print(
            f"{total} new rows"
            + (f" ({', '.join(f'{t}+{n}' for t, n in sorted(added.items()) if n)})" if total else "")
        )
        return _report_failures(failed)
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
