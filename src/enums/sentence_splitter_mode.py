"""Sentence splitter modes for semantic chunking."""

from src.enums.base import BaseEnum


class SentenceSplitterMode(BaseEnum):
    """How to segment text into sentences before semantic grouping."""

    SIMPLE_REGEX = "simple_regex"
