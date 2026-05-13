"""Briefing vector date window selectors (YAML)."""

from src.enums.base import BaseEnum


class BriefingDateFilter(BaseEnum):
    """How far back vector metadata ``source_day`` (or configured key) is restricted."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
