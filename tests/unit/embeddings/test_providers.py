"""Unit tests for the embedding provider registry."""

import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

from src.embeddings.providers import (
    SentenceTransformerHandler,
    resolve_provider,
)
from src.enums.embedding_provider import EmbeddingProvider
from src.utils.timer import Timer


class TestResolveProvider(unittest.TestCase):
    """This class tests resolve_provider."""

    def test_resolve_provider_returns_sentence_transformer_handler(self) -> None:
        """resolve_provider: returns SentenceTransformerHandler for SENTENCE_TRANSFORMERS."""
        handler = resolve_provider(
            EmbeddingProvider.SENTENCE_TRANSFORMERS,
            "all-MiniLM-L6-v2",
        )
        self.assertIsInstance(handler, SentenceTransformerHandler)

    def test_resolve_provider_raises_for_openai(self) -> None:
        """resolve_provider: raises ValueError for OPENAI (not yet implemented)."""
        with self.assertRaises(ValueError) as ctx:
            resolve_provider(EmbeddingProvider.OPENAI, "text-embedding-3-small")
        self.assertIn("No embedding handler registered", str(ctx.exception))

    def test_resolve_provider_passes_timer_to_handler(self) -> None:
        """resolve_provider: forwards timer to the constructed handler."""
        timer = Timer()
        handler = resolve_provider(
            EmbeddingProvider.SENTENCE_TRANSFORMERS,
            "model",
            timer=timer,
        )
        self.assertIs(handler._timer, timer)  # pylint: disable=protected-access


class TestSentenceTransformerHandler(unittest.TestCase):
    """This class tests SentenceTransformerHandler.embed."""

    def test_embed_delegates_to_sentence_transformer(self) -> None:
        """embed: forwards texts to the underlying SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_st_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock(SentenceTransformer=mock_st_cls)

        handler = SentenceTransformerHandler("test-model")
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = mock_st_module
        try:
            result = handler.embed(["hello", "world"], batch_size=32)
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original

        mock_st_cls.assert_called_once_with("test-model")
        mock_model.encode.assert_called_once_with(
            ["hello", "world"],
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0][0], 0.1)
        self.assertAlmostEqual(result[1][1], 0.4)

    def test_embed_loads_model_once(self) -> None:
        """embed: lazy-loads the model on first call and reuses on subsequent calls."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0]])
        mock_st_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock(SentenceTransformer=mock_st_cls)

        handler = SentenceTransformerHandler("test-model")
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = mock_st_module
        try:
            handler.embed(["a"])
            handler.embed(["b"])
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original

        self.assertEqual(mock_st_cls.call_count, 1)

    def test_embed_with_timer_records_sections(self) -> None:
        """embed: records provider.load_model and provider.encode timer sections."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5, 0.6]])
        mock_st_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock(SentenceTransformer=mock_st_cls)

        timer = Timer()
        handler = SentenceTransformerHandler("test-model", timer=timer)
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = mock_st_module
        try:
            result = handler.embed(["text"], batch_size=16)
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original

        labels = [r[0] for r in timer.records]
        self.assertIn("provider.load_model", labels)
        self.assertIn("provider.encode", labels)
        self.assertEqual(len(result), 1)

    def test_embed_without_timer_works(self) -> None:
        """embed: works correctly when no timer is provided."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0]])
        mock_st_cls = MagicMock(return_value=mock_model)
        mock_st_module = MagicMock(SentenceTransformer=mock_st_cls)

        handler = SentenceTransformerHandler("test-model")
        original = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = mock_st_module
        try:
            result = handler.embed(["hello"])
        finally:
            if original is None:
                sys.modules.pop("sentence_transformers", None)
            else:
                sys.modules["sentence_transformers"] = original

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0][0], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
