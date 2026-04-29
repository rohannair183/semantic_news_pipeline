"""Enum describing supported row-mapping source selector kinds."""

from src.enums.base import BaseEnum


class ArticleRowSourceKind(BaseEnum):
    """Namespaces supported by ArticleNormalizer row-mapping sources."""

    PROFILE = "profile"
    PAYLOAD = "payload"
    FIELDS = "fields"
    ITEM = "item"
    DIRECT_KEY = "direct_key"
