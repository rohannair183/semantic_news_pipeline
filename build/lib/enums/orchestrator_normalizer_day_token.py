"""Normalizer day tokens allowed in orchestrator YAML task params."""

from src.enums.base import BaseEnum


class OrchestratorNormalizerDayToken(BaseEnum):
    """Special-case day values for ARTICLE_NORMALIZER orchestrator tasks."""

    UTC_TODAY = "utc_today"
