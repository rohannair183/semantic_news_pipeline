"""Enum describing supported Guardian API `use-date` values."""

from src.enums.base import BaseEnum


class GuardianUseDate(BaseEnum):
    """Supported `use-date` query values used by the ingestion client."""

    PUBLISHED = "published"
