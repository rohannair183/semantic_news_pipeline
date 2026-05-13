"""Enum describing supported row-mapping transforms."""

from src.enums.base import BaseEnum


class ArticleRowTransform(BaseEnum):
    """Transforms supported by ArticleNormalizer row mappings."""

    PARSE_ISO = "parse_iso"
