"""Enum describing YAML config groups under the configuration directory."""

from src.enums.base import BaseEnum


class YAMLConfigType(BaseEnum):
    """Top-level configuration groups available to YAML config parsing."""

    INGESTION = "ingestion"
    CHUNKING = "chunking"
