"""Embedding provider selectors for YAML configuration."""

from src.enums.base import BaseEnum


class EmbeddingProvider(BaseEnum):
    """Supported embedding provider backends."""

    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OPENAI = "openai"
