"""Unit tests for semantic_sentence_chunks."""

import unittest
from types import SimpleNamespace

from src.chunking import semantic_split
from src.chunking.semantic_split import semantic_sentence_chunks
from src.config.settings import SemanticChunkingParams
from src.enums.sentence_splitter_mode import SentenceSplitterMode


class TestSemanticSentenceChunks(unittest.TestCase):
    """This class tests semantic_sentence_chunks."""

    def _params(
        self,
        *,
        min_chars: int = 1,
        max_chars: int = 500,
        overlap: int = 0,
        threshold: float = 0.2,
    ) -> SemanticChunkingParams:
        """_params: builds semantic params for tests."""
        return SemanticChunkingParams(
            min_chars=min_chars,
            max_chars=max_chars,
            overlap_chars=overlap,
            similarity_threshold=threshold,
            sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
        )

    def test_semantic_sentence_chunks_returns_empty_for_blank_text(self):
        """semantic_sentence_chunks: returns empty list for blank input."""
        self.assertEqual(semantic_sentence_chunks("  ", self._params()), [])

    def test_semantic_sentence_chunks_splits_on_low_similarity(self):
        """semantic_sentence_chunks: creates multiple chunks when similarity is low."""
        text = (
            "The stock market rose today on tech earnings. "
            "Volcanic ash disrupted flights in the southern region."
        )
        chunks = semantic_sentence_chunks(text, self._params(min_chars=5, threshold=0.8))
        self.assertGreaterEqual(len(chunks), 2)

    def test_semantic_sentence_chunks_merges_similar_short_sentences(self):
        """semantic_sentence_chunks: merges related short sentences."""
        text = "red car stops. red car goes."
        chunks = semantic_sentence_chunks(
            text,
            self._params(min_chars=1, max_chars=400, threshold=0.15),
        )
        self.assertEqual(len(chunks), 1)

    def test_semantic_sentence_chunks_applies_overlap(self):
        """semantic_sentence_chunks: second chunk includes overlap span."""
        text = "Onlyone. Onlytwo. Onlythree."
        chunks = semantic_sentence_chunks(
            text,
            self._params(min_chars=1, max_chars=200, overlap=4, threshold=0.9),
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertLess(chunks[1][1], chunks[1][2])

    def test_semantic_sentence_chunks_rejects_unknown_splitter(self):
        """semantic_sentence_chunks: raises for unsupported sentence splitter."""
        bad_params = SimpleNamespace(sentence_splitter=object())
        with self.assertRaises(ValueError) as ctx:
            semantic_sentence_chunks("Hello.", bad_params)
        self.assertIn("Unsupported sentence splitter", str(ctx.exception))


class TestSemanticSplitHelpers(unittest.TestCase):
    """This class tests helper functions in semantic_split."""

    def test_word_jaccard_handles_empty_inputs(self):
        """_word_jaccard: handles empty-token combinations."""
        self.assertEqual(semantic_split._word_jaccard("", ""), 1.0)  # pylint: disable=protected-access
        self.assertEqual(semantic_split._word_jaccard("a", ""), 0.0)  # pylint: disable=protected-access


    def test_split_sentences_returns_for_whitespace_only(self):
        """_split_sentences_simple_regex: returns quickly for whitespace text."""
        spans = semantic_split._split_sentences_simple_regex("   ")  # pylint: disable=protected-access
        self.assertEqual(spans, [])

    def test_split_sentences_handles_newline_and_terminal_state(self):
        """_split_sentences_simple_regex: supports newline sentence boundaries."""
        spans = semantic_split._split_sentences_simple_regex("A one.\nB two")  # pylint: disable=protected-access
        self.assertEqual(len(spans), 2)

    def test_split_long_sentence_spans_handles_max_and_blank(self):
        """_split_long_sentence_spans: handles invalid max and blank values."""
        self.assertEqual(
            semantic_split._split_long_sentence_spans("abc", 0, 0),  # pylint: disable=protected-access
            [],
        )
        self.assertEqual(
            semantic_split._split_long_sentence_spans("   ", 0, 10),  # pylint: disable=protected-access
            [],
        )

    def test_split_long_sentence_spans_breaks_long_sentence(self):
        """_split_long_sentence_spans: wraps words when sentence exceeds max_chars."""
        parts = semantic_split._split_long_sentence_spans(  # pylint: disable=protected-access
            "alpha beta gamma delta epsilon", 0, 10
        )
        self.assertGreaterEqual(len(parts), 2)

    def test_merge_sentence_spans_handles_empty_and_min_chars(self):
        """_merge_sentence_spans: handles empty input and min_chars continuation."""
        empty = semantic_split._merge_sentence_spans(  # pylint: disable=protected-access
            [],
            SemanticChunkingParams(
                min_chars=1,
                max_chars=500,
                overlap_chars=0,
                similarity_threshold=0.2,
                sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
            ),
        )
        self.assertEqual(empty, [])
        sentences = [("a", 0, 1), ("b", 2, 3), ("c", 4, 5)]
        spans = semantic_split._merge_sentence_spans(  # pylint: disable=protected-access
            sentences,
            SemanticChunkingParams(
                min_chars=10,
                max_chars=50,
                overlap_chars=0,
                similarity_threshold=0.9,
                sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
            ),
        )
        self.assertEqual(len(spans), 1)

    def test_merge_sentence_spans_splits_when_max_exceeded(self):
        """_merge_sentence_spans: flushes group when max_chars is exceeded."""
        sentences = [("abc", 0, 3), ("def", 4, 7), ("ghijklmnop", 8, 18)]
        spans = semantic_split._merge_sentence_spans(  # pylint: disable=protected-access
            sentences,
            SemanticChunkingParams(
                min_chars=1,
                max_chars=7,
                overlap_chars=0,
                similarity_threshold=0.0,
                sentence_splitter=SentenceSplitterMode.SIMPLE_REGEX,
            ),
        )
        self.assertGreaterEqual(len(spans), 2)
