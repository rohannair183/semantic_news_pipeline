"""Unit tests for the chunking strategy registry."""

import unittest
from unittest.mock import patch

from src.chunking.semantic_chunker import SemanticChunker
from src.chunking.strategies import (
    STRATEGY_HANDLERS,
    resolve_handler,
)
from src.enums.chunking_strategy import ChunkingStrategy
from tests.unit.chunking.test_chunker import _default_params


class TestStrategyRegistry(unittest.TestCase):
    """This class tests STRATEGY_HANDLERS."""

    def test_registry_contains_semantic_chunker(self) -> None:
        """STRATEGY_HANDLERS: registers a SemanticChunker for SEMANTIC_SENTENCE."""
        handler = STRATEGY_HANDLERS[ChunkingStrategy.SEMANTIC_SENTENCE]
        self.assertIsInstance(handler, SemanticChunker)


class TestSemanticChunkerHandler(unittest.TestCase):
    """This class tests SemanticChunker.chunk."""

    def test_chunk_delegates_to_semantic_sentence_chunks(self) -> None:
        """chunk: forwards arguments to semantic_sentence_chunks and returns its result."""
        params = _default_params()
        sentinel = [("chunk", 0, 5)]
        with patch(
            "src.chunking.semantic_chunker.semantic_sentence_chunks",
            return_value=sentinel,
        ) as mocked:
            result = SemanticChunker().chunk("Some text.", params)
        mocked.assert_called_once()
        self.assertEqual(result, sentinel)


class TestResolveHandler(unittest.TestCase):
    """This class tests resolve_handler."""

    def test_resolve_handler_returns_registered_handler(self) -> None:
        """resolve_handler: returns the handler registered for the strategy."""
        handler = resolve_handler(ChunkingStrategy.SEMANTIC_SENTENCE)
        self.assertIsInstance(handler, SemanticChunker)

    def test_resolve_handler_raises_for_unregistered_strategy(self) -> None:
        """resolve_handler: raises ValueError when no handler is registered."""
        with self.assertRaises(ValueError) as ctx:
            resolve_handler("totally_unknown_strategy")  # type: ignore[arg-type]
        self.assertIn("No chunking handler registered", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
