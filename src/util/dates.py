"""Shared date and timestamp helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Union


CHECKPOINT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
COMPACT_DAY_FORMAT = "%Y%m%d"

DayInput = Union[date, datetime, str]


def utc_today_date() -> date:
    """Return today's UTC date."""
    return datetime.now(timezone.utc).date()


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
