"""Chunking strategy selectors for YAML configuration."""

from src.enums.base import BaseEnum


class ChunkingStrategy(BaseEnum):
    """Supported document chunking strategies."""

    SEMANTIC_SENTENCE = "semantic_sentence"
