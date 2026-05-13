"""Shared date and timestamp helpers."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union


CHECKPOINT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
COMPACT_DAY_FORMAT = "%Y%m%d"

DayInput = Union[date, datetime, str]


def utc_today_date() -> date:
    """Return today's UTC date."""
    return datetime.now(timezone.utc).date()


def date_range_single_calendar_day(day: date) -> tuple[date, date]:
    """Return inclusive (day, day) for vector metadata day bounds."""
    return day, day


def date_range_last_n_calendar_days_inclusive(end: date, n: int) -> tuple[date, date]:
    """Return inclusive (start, end) spanning ``n`` calendar days ending on ``end``.

    ``n`` must be at least 1. For ``n == 1`` this matches :func:`date_range_single_calendar_day`.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    start = end - timedelta(days=n - 1)
    return start, end


def date_range_month_to_date(anchor: date) -> tuple[date, date]:
    """Return inclusive (first day of month, ``anchor``) for month-to-date bounds."""
    start = date(anchor.year, anchor.month, 1)
    return start, anchor


def utc_now_checkpoint_token() -> str:
    """Return the current UTC timestamp in checkpoint filename format."""
    return datetime.now(timezone.utc).strftime(CHECKPOINT_TIMESTAMP_FORMAT)


def coerce_day(value: DayInput) -> date:
    """Coerce supported day-like inputs into a native date object."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        try:
            if len(candidate) == 8 and candidate.isdigit():
                return datetime.strptime(candidate, COMPACT_DAY_FORMAT).date()
            return date.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(
                "day must be a date, datetime, YYYY-MM-DD string, or YYYYMMDD string"
            ) from exc
    raise ValueError("day must be a date, datetime, YYYY-MM-DD string, or YYYYMMDD string")


def format_day_iso(day: date) -> str:
    """Format a day as YYYY-MM-DD."""
    resolved_day = _require_date(day)
    return resolved_day.isoformat()


def format_day_compact(day: date) -> str:
    """Format a day as YYYYMMDD."""
    resolved_day = _require_date(day)
    return resolved_day.strftime(COMPACT_DAY_FORMAT)


def parse_utc_instant_iso_z(value: str) -> datetime:
    """Parse a UTC instant string that ends with ``Z`` (as from ``utc_now_iso_z``).

    Returns an aware datetime in UTC. Raises :class:`ValueError` when the string
    is empty, does not end with ``Z``, or is not a valid ISO-8601 timestamp.
    """
    candidate = value.strip()
    if not candidate:
        raise ValueError("instant string must be non-empty")
    if not candidate.endswith("Z"):
        raise ValueError("instant string must end with Z for UTC")
    normalized = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("instant string is not valid ISO-8601") from exc
    return parsed.astimezone(timezone.utc)


def parse_guardian_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse a Guardian API datetime string."""
    if value is None:
        return None
    candidate = value.strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        return None


def parse_checkpoint_timestamp(value: Union[str, Path]) -> Optional[datetime]:
    """Parse a checkpoint timestamp token from a filename or raw token."""
    stem = value.stem if isinstance(value, Path) else Path(value).stem
    token = stem.rsplit("_", 1)[-1]
    try:
        return datetime.strptime(token, CHECKPOINT_TIMESTAMP_FORMAT)
    except ValueError:
        return None


def _require_date(day: date) -> date:
    if isinstance(day, datetime) or not isinstance(day, date):
        raise TypeError("day must be a date instance")
    return day
