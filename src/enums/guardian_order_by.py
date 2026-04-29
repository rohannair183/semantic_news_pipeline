"""Enum describing supported Guardian API ordering values."""

from src.enums.base import BaseEnum


class GuardianOrderBy(BaseEnum):
    """Supported `order-by` query values for Guardian content search."""

    NEWEST = "newest"
    OLDEST = "oldest"
    RELEVANCE = "relevance"
