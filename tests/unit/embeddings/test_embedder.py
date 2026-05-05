"""Unit tests for Embedder."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config.settings import EmbeddingConfig, Settings
from src.embeddings.embedder import Embedder
from src.enums.embedding_provider import EmbeddingProvider


def _minimal_embedding_config(tmp_path: Path) -> EmbeddingConfig:
    """Build a minimal EmbeddingConfig pointing at temp directories."""
    return EmbeddingConfig(
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        text_column="chunk_text",
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name="test-model",
        batch_size=2,
    )


def _write_input_parquet(
    cfg: EmbeddingConfig,
    profile: str = "default",
    rows: list[dict] | None = None,
) -> Path:
    """Write an input parquet used by the embedder."""
    cfg.input_dir.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [
            {"chunk_text": "Hello world", "source_api_id": "a1"},
            {"chunk_text": "Foo bar baz", "source_api_id": "a2"},
            {"chunk_text": "Third chunk", "source_api_id": "a3"},
        ]
    path = cfg.input_dir / f"{profile}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings of length 3."""
    return [[float(i)] * 3 for i in range(len(texts))]


class _FakeHandler:  # pylint: disable=too-few-public-methods
    """Fake embedding handler that returns deterministic vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return fake embeddings for ``texts``."""
        return _fake_embed(texts)


class TestEmbedderEmbedToParquet(unittest.TestCase):
    """This class tests embed_to_parquet."""

    def test_embed_to_parquet_writes_output_with_embedding_column(self) -> None:
        """embed_to_parquet: writes parquet with embedding column appended."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg)

            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_FakeHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            self.assertIn("default", result)
            output_path = Path(result["default"])
            self.assertTrue(output_path.is_file())
            output_df = pd.read_parquet(output_path)
            self.assertIn("embedding", output_df.columns)
            self.assertEqual(len(output_df), 3)
            self.assertEqual(len(output_df["embedding"].iloc[0]), 3)

    def test_embed_to_parquet_returns_empty_for_empty_input(self) -> None:
        """embed_to_parquet: returns empty dict when input parquet has no rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[])

            with patch.object(Settings, "load_embedding_config", return_value=cfg):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            self.assertEqual(result, {})

    def test_embed_to_parquet_raises_for_missing_input(self) -> None:
        """embed_to_parquet: raises FileNotFoundError when input does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)

            with patch.object(Settings, "load_embedding_config", return_value=cfg):
                embedder = Embedder()
                with self.assertRaises(FileNotFoundError) as ctx:
                    embedder.embed_to_parquet(profile="missing")
                self.assertIn("missing", str(ctx.exception))

    def test_embed_to_parquet_batches_correctly(self) -> None:
        """embed_to_parquet: splits texts into batches of configured size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg)
            call_sizes: list[int] = []

            class _TrackingHandler:  # pylint: disable=too-few-public-methods
                def embed(self, texts: list[str]) -> list[list[float]]:
                    """Track call sizes and return fake embeddings."""
                    call_sizes.append(len(texts))
                    return _fake_embed(texts)

            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_TrackingHandler(),
                ),
            ):
                embedder = Embedder()
                embedder.embed_to_parquet(profile="default")

            self.assertEqual(call_sizes, [2, 1])

    def test_embed_to_parquet_preserves_original_columns(self) -> None:
        """embed_to_parquet: output contains all original columns plus embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(
                cfg,
                rows=[{"chunk_text": "text", "source_api_id": "x", "headline": "H"}],
            )
            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_FakeHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            output_df = pd.read_parquet(result["default"])
            self.assertIn("chunk_text", output_df.columns)
            self.assertIn("source_api_id", output_df.columns)
            self.assertIn("headline", output_df.columns)
            self.assertIn("embedding", output_df.columns)

    def test_embed_to_parquet_sanitizes_profile_slash(self) -> None:
        """embed_to_parquet: forward slashes in profile are replaced for filesystem safety."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(
                cfg,
                profile="team_profile",
                rows=[{"chunk_text": "text"}],
            )
            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_FakeHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="team/profile")

            self.assertIn("team/profile", result)
            output_path = Path(result["team/profile"])
            self.assertEqual(output_path.name, "team_profile.parquet")


class TestEmbedderProperties(unittest.TestCase):
    """This class tests Embedder property accessors."""

    def test_provider_name_and_model_name(self) -> None:
        """properties: expose provider and model name from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_embedding_config(Path(tmpdir))
            with patch.object(Settings, "load_embedding_config", return_value=cfg):
                embedder = Embedder()
            self.assertEqual(embedder.provider_name, "sentence_transformers")
            self.assertEqual(embedder.model_name, "test-model")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
