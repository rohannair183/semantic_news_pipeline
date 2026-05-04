"""Unit tests for chunk record helpers."""

import unittest

from src.chunking.chunk_records import build_chunk_row, chunking_params_fingerprint
from src.enums.chunking_strategy import ChunkingStrategy


class TestChunkingParamsFingerprint(unittest.TestCase):
    """This class tests chunking_params_fingerprint."""

    def test_chunking_params_fingerprint_is_stable(self):
        """chunking_params_fingerprint: same strategy + params yield same digest."""
        params = {
            "min_chars": 10,
            "max_chars": 100,
            "overlap_chars": 2,
            "similarity_threshold": 0.3,
            "sentence_splitter": "simple_regex",
        }
        first = chunking_params_fingerprint(ChunkingStrategy.SEMANTIC_SENTENCE, params)
        second = chunking_params_fingerprint(ChunkingStrategy.SEMANTIC_SENTENCE, params)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)

    def test_chunking_params_fingerprint_differs_across_strategies(self):
        """chunking_params_fingerprint: different strategy values produce different hashes."""
        params = {"min_chars": 10, "max_chars": 100}
        hash_a = chunking_params_fingerprint(ChunkingStrategy.SEMANTIC_SENTENCE, params)
        hash_b = chunking_params_fingerprint("other_strategy", params)  # type: ignore[arg-type]
        self.assertNotEqual(hash_a, hash_b)


class TestBuildChunkRow(unittest.TestCase):
    """This class tests build_chunk_row."""

    def test_build_chunk_row_includes_lineage_fields(self):
        """build_chunk_row: maps core schema fields."""
        params = {
            "min_chars": 1,
            "max_chars": 50,
            "overlap_chars": 0,
            "similarity_threshold": 0.5,
            "sentence_splitter": "simple_regex",
        }
        row = build_chunk_row(
            source_day="2026-04-29",
            source_row_index=3,
            chunk_index=0,
            chunk_text="Hello world.",
            chunk_start_char=0,
            chunk_end_char=12,
            source_text_column="body_text",
            strategy=ChunkingStrategy.SEMANTIC_SENTENCE,
            params=params,
            source_api_id="id-1",
            source_profile="main",
            passthrough={"headline": "H"},
        )
        self.assertEqual(row["source_day"], "2026-04-29")
        self.assertEqual(row["chunk_char_len"], 12)
        self.assertEqual(row["chunking_strategy"], "semantic_sentence")
        self.assertEqual(row["headline"], "H")
        self.assertEqual(row["source_api_id"], "id-1")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
