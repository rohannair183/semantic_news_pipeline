"""Unit tests for chunk record helpers."""

import unittest

from src.chunking.chunk_records import build_chunk_row, semantic_params_fingerprint
from src.config.settings import SemanticChunkingParams
from src.enums.chunking_strategy import ChunkingStrategy
from src.enums.sentence_splitter_mode import SentenceSplitterMode


class TestSemanticParamsFingerprint(unittest.TestCase):
    """This class tests semantic_params_fingerprint."""

    def test_semantic_params_fingerprint_is_stable(self):
        """semantic_params_fingerprint: same params yield same digest."""
        params = SemanticChunkingParams(
            min_chars=10,
            max_chars=100,
            overlap_chars=2,
            similarity_threshold=0.3,
            sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
        )
        first = semantic_params_fingerprint(params)
        second = semantic_params_fingerprint(params)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)


class TestBuildChunkRow(unittest.TestCase):
    """This class tests build_chunk_row."""

    def test_build_chunk_row_includes_lineage_fields(self):
        """build_chunk_row: maps core schema fields."""
        semantic = SemanticChunkingParams(
            min_chars=1,
            max_chars=50,
            overlap_chars=0,
            similarity_threshold=0.5,
            sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
        )
        row = build_chunk_row(
            source_day="2026-04-29",
            source_row_index=3,
            chunk_index=0,
            chunk_text="Hello world.",
            chunk_start_char=0,
            chunk_end_char=12,
            source_text_column="body_text",
            strategy=ChunkingStrategy.SEMANTIC_SENTENCE,
            semantic=semantic,
            source_api_id="id-1",
            source_profile="main",
            passthrough={"headline": "H"},
        )
        self.assertEqual(row["source_day"], "2026-04-29")
        self.assertEqual(row["chunk_char_len"], 12)
        self.assertEqual(row["chunking_strategy"], "semantic_sentence")
        self.assertEqual(row["headline"], "H")
        self.assertEqual(row["source_api_id"], "id-1")
