"""Ingestion timeframe mode enum."""

from src.enums.base import BaseEnum


class IngestionTimeframeMode(BaseEnum):
    """Supported ingestion timeframe mode values."""

    RELATIVE = "relative"
    EXPLICIT = "explicit"
