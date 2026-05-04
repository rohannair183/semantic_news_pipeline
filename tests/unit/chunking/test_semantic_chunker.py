"""Unit tests for SemanticChunker strategy handler."""

import unittest
from typing import Any, Dict

from src.chunking.semantic_chunker import SemanticChunker


def _valid_params(**overrides: Any) -> Dict[str, Any]:
    """Build a valid params dict with optional overrides."""
    base: Dict[str, Any] = {
        "min_chars": 10,
        "max_chars": 200,
        "overlap_chars": 0,
        "similarity_threshold": 0.35,
        "sentence_splitter": "simple_regex",
    }
    base.update(overrides)
    return base


class TestSemanticChunkerChunk(unittest.TestCase):
    """This class tests chunk."""

    def test_chunk_returns_spans_for_simple_text(self) -> None:
        """chunk: produces spans from simple two-sentence input."""
        params = _valid_params(min_chars=1, max_chars=200)
        result = SemanticChunker().chunk("Hello. World.", params)
        self.assertGreater(len(result), 0)
        for text, start, end in result:
            self.assertEqual(text, "Hello. World."[start:end])

    def test_chunk_returns_empty_for_blank_text(self) -> None:
        """chunk: returns empty list for whitespace-only text."""
        params = _valid_params()
        result = SemanticChunker().chunk("   ", params)
        self.assertEqual(result, [])


class TestSemanticChunkerParseParams(unittest.TestCase):
    """This class tests _parse_params."""

    def test_parse_params_accepts_valid_params(self) -> None:
        """_parse_params: returns SemanticChunkingParams for a valid dict."""
        parsed = SemanticChunker._parse_params(  # pylint: disable=protected-access
            _valid_params()
        )
        self.assertEqual(parsed.min_chars, 10)
        self.assertEqual(parsed.max_chars, 200)

    def test_parse_params_raises_when_min_chars_not_int(self) -> None:
        """_parse_params: raises ValueError for non-integer min_chars."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(min_chars="bad")
            )
        self.assertIn("min_chars", str(ctx.exception))

    def test_parse_params_raises_when_min_chars_less_than_one(self) -> None:
        """_parse_params: raises ValueError for min_chars below 1."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(min_chars=0)
            )
        self.assertIn("min_chars", str(ctx.exception))

    def test_parse_params_raises_when_max_less_than_min(self) -> None:
        """_parse_params: raises ValueError when max_chars < min_chars."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(min_chars=200, max_chars=100)
            )
        self.assertIn("max_chars", str(ctx.exception))

    def test_parse_params_raises_when_overlap_not_int(self) -> None:
        """_parse_params: raises ValueError for non-integer overlap_chars."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(overlap_chars="bad")
            )
        self.assertIn("overlap_chars", str(ctx.exception))

    def test_parse_params_raises_when_overlap_negative(self) -> None:
        """_parse_params: raises ValueError for negative overlap_chars."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(overlap_chars=-1)
            )
        self.assertIn("overlap_chars", str(ctx.exception))

    def test_parse_params_raises_when_overlap_too_large(self) -> None:
        """_parse_params: raises ValueError when overlap_chars >= max_chars."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(max_chars=50, overlap_chars=50)
            )
        self.assertIn("overlap_chars", str(ctx.exception))

    def test_parse_params_raises_when_threshold_not_number(self) -> None:
        """_parse_params: raises ValueError for non-numeric similarity_threshold."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(similarity_threshold="bad")
            )
        self.assertIn("similarity_threshold", str(ctx.exception))

    def test_parse_params_raises_when_threshold_out_of_range(self) -> None:
        """_parse_params: raises ValueError when similarity_threshold > 1."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(similarity_threshold=2)
            )
        self.assertIn("similarity_threshold", str(ctx.exception))

    def test_parse_params_raises_when_splitter_invalid(self) -> None:
        """_parse_params: raises ValueError for unsupported sentence_splitter."""
        with self.assertRaises(ValueError) as ctx:
            SemanticChunker._parse_params(  # pylint: disable=protected-access
                _valid_params(sentence_splitter="bad")
            )
        self.assertIn("sentence_splitter", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
