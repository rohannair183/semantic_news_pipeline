"""Relative ingestion timeframe enum."""

from src.enums.base import BaseEnum


class IngestionTimeframeRelative(BaseEnum):
    """Supported relative ingestion timeframe selectors."""

    PAST_DAY = "past_day"
    PAST_WEEK = "past_week"
    PAST_MONTH = "past_month"
