"""Unit tests for Embedder."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.config.settings import EmbeddingConfig, Settings
from src.embeddings.embedder import Embedder
from src.enums.embedding_provider import EmbeddingProvider
from src.utils.timer import Timer


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
            {"chunk_text": "Hello world", "source_api_id": "a1",
             "chunk_index": 0, "source_row_index": 0},
            {"chunk_text": "Foo bar baz", "source_api_id": "a2",
             "chunk_index": 0, "source_row_index": 1},
            {"chunk_text": "Third chunk", "source_api_id": "a3",
             "chunk_index": 0, "source_row_index": 2},
        ]
    path = cfg.input_dir / f"{profile}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_output_parquet(
    cfg: EmbeddingConfig,
    profile: str = "default",
    rows: list[dict] | None = None,
) -> Path:
    """Write a pre-existing output parquet to simulate a previous run."""
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / f"{profile}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Return deterministic fake embeddings of length 3."""
    return [[float(i)] * 3 for i in range(len(texts))]


class _FakeHandler:  # pylint: disable=too-few-public-methods
    """Fake embedding handler that returns deterministic vectors."""

    def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:  # pylint: disable=unused-argument
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

    def test_embed_to_parquet_passes_batch_size_to_handler(self) -> None:
        """embed_to_parquet: forwards configured batch_size to the handler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg)
            received_batch_sizes: list[int] = []

            class _TrackingHandler:  # pylint: disable=too-few-public-methods
                def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
                    """Track received batch_size and return fake embeddings."""
                    received_batch_sizes.append(batch_size)
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

            self.assertEqual(received_batch_sizes, [cfg.batch_size])

    def test_embed_to_parquet_preserves_original_columns(self) -> None:
        """embed_to_parquet: output contains all original columns plus embedding."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(
                cfg,
                rows=[{"chunk_text": "text", "source_api_id": "x",
                       "chunk_index": 0, "source_row_index": 0, "headline": "H"}],
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


class TestEmbedderIncremental(unittest.TestCase):
    """This class tests embed_to_parquet incremental caching."""

    def test_skips_already_embedded_chunks(self) -> None:
        """embed_to_parquet: reuses cached embeddings and does not re-embed them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[
                {"chunk_text": "Hello", "source_api_id": "a1",
                 "chunk_index": 0, "source_row_index": 0},
                {"chunk_text": "World", "source_api_id": "a2",
                 "chunk_index": 0, "source_row_index": 1},
            ])
            cached_embedding = [9.0, 9.0, 9.0]
            _write_output_parquet(cfg, rows=[
                {"chunk_text": "Hello", "source_api_id": "a1",
                 "chunk_index": 0, "source_row_index": 0,
                 "embedding": cached_embedding},
            ])

            embedded_texts: list[str] = []

            class _TrackHandler:  # pylint: disable=too-few-public-methods
                def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:  # pylint: disable=unused-argument
                    """Track which texts are embedded."""
                    embedded_texts.extend(texts)
                    return _fake_embed(texts)

            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_TrackHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            self.assertEqual(embedded_texts, ["World"])
            output_df = pd.read_parquet(result["default"])
            self.assertEqual(len(output_df), 2)
            a1_row = output_df[output_df["source_api_id"] == "a1"]
            self.assertEqual(
                list(a1_row["embedding"].iloc[0]), cached_embedding,
            )

    def test_all_cached_skips_embedding_entirely(self) -> None:
        """embed_to_parquet: skips provider call when all chunks are cached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            rows = [
                {"chunk_text": "Hello", "source_api_id": "a1",
                 "chunk_index": 0, "source_row_index": 0},
            ]
            _write_input_parquet(cfg, rows=rows)
            _write_output_parquet(cfg, rows=[
                {**rows[0], "embedding": [1.0, 2.0, 3.0]},
            ])

            handler_called = False

            class _FailHandler:  # pylint: disable=too-few-public-methods
                def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:  # pylint: disable=unused-argument
                    """Should never be called."""
                    nonlocal handler_called
                    handler_called = True
                    return _fake_embed(texts)

            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_FailHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            self.assertFalse(handler_called)
            output_df = pd.read_parquet(result["default"])
            self.assertEqual(len(output_df), 1)
            self.assertEqual(
                list(output_df["embedding"].iloc[0]), [1.0, 2.0, 3.0],
            )

    def test_no_output_embeds_everything(self) -> None:
        """embed_to_parquet: embeds all chunks when no output file exists yet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[
                {"chunk_text": "A", "source_api_id": "a1",
                 "chunk_index": 0, "source_row_index": 0},
                {"chunk_text": "B", "source_api_id": "a2",
                 "chunk_index": 0, "source_row_index": 1},
            ])

            embedded_texts: list[str] = []

            class _TrackHandler:  # pylint: disable=too-few-public-methods
                def embed(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:  # pylint: disable=unused-argument
                    """Track texts."""
                    embedded_texts.extend(texts)
                    return _fake_embed(texts)

            with (
                patch.object(Settings, "load_embedding_config", return_value=cfg),
                patch(
                    "src.embeddings.embedder.resolve_provider",
                    return_value=_TrackHandler(),
                ),
            ):
                embedder = Embedder()
                result = embedder.embed_to_parquet(profile="default")

            self.assertEqual(embedded_texts, ["A", "B"])
            output_df = pd.read_parquet(result["default"])
            self.assertEqual(len(output_df), 2)

    def test_stale_cached_rows_are_dropped(self) -> None:
        """embed_to_parquet: rows in output but not in input are excluded from result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[
                {"chunk_text": "New", "source_api_id": "a2",
                 "chunk_index": 0, "source_row_index": 0},
            ])
            _write_output_parquet(cfg, rows=[
                {"chunk_text": "Old", "source_api_id": "a1",
                 "chunk_index": 0, "source_row_index": 0,
                 "embedding": [1.0]},
            ])

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
            self.assertEqual(len(output_df), 1)
            self.assertEqual(output_df["source_api_id"].iloc[0], "a2")

    def test_input_without_id_columns_embeds_everything(self) -> None:
        """embed_to_parquet: embeds all rows when id columns are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[
                {"chunk_text": "no id"},
            ])
            _write_output_parquet(cfg, rows=[
                {"chunk_text": "no id", "embedding": [1.0]},
            ])

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
            self.assertEqual(len(output_df), 1)
            self.assertIn("embedding", output_df.columns)


class TestSplitNewAndCached(unittest.TestCase):
    """This class tests _split_new_and_cached."""

    def test_no_output_file_returns_all_new(self) -> None:
        """_split_new_and_cached: all rows are new when output file missing."""
        input_df = pd.DataFrame([
            {"source_api_id": "a1", "chunk_index": 0, "source_row_index": 0,
             "chunk_text": "x"},
        ])
        new_df, cached_df = Embedder._split_new_and_cached(  # pylint: disable=protected-access
            input_df, Path("/nonexistent/path.parquet"),
        )
        self.assertEqual(len(new_df), 1)
        self.assertTrue(cached_df.empty)

    def test_empty_output_returns_all_new(self) -> None:
        """_split_new_and_cached: all rows are new when output is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.parquet"
            pd.DataFrame().to_parquet(output_path, index=False)
            input_df = pd.DataFrame([
                {"source_api_id": "a1", "chunk_index": 0,
                 "source_row_index": 0, "chunk_text": "x"},
            ])
            new_df, cached_df = Embedder._split_new_and_cached(  # pylint: disable=protected-access
                input_df, output_path,
            )
            self.assertEqual(len(new_df), 1)
            self.assertTrue(cached_df.empty)

    def test_output_without_embedding_column_returns_all_new(self) -> None:
        """_split_new_and_cached: all rows new when output lacks embedding column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.parquet"
            pd.DataFrame([
                {"source_api_id": "a1", "chunk_index": 0,
                 "source_row_index": 0, "chunk_text": "x"},
            ]).to_parquet(output_path, index=False)
            input_df = pd.DataFrame([
                {"source_api_id": "a1", "chunk_index": 0,
                 "source_row_index": 0, "chunk_text": "x"},
            ])
            new_df, cached_df = Embedder._split_new_and_cached(  # pylint: disable=protected-access
                input_df, output_path,
            )
            self.assertEqual(len(new_df), 1)
            self.assertTrue(cached_df.empty)

    def test_partial_match_splits_correctly(self) -> None:
        """_split_new_and_cached: splits into matched and unmatched rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "out.parquet"
            pd.DataFrame([
                {"source_api_id": "a1", "chunk_index": 0,
                 "source_row_index": 0, "chunk_text": "x",
                 "embedding": [1.0, 2.0]},
            ]).to_parquet(output_path, index=False)
            input_df = pd.DataFrame([
                {"source_api_id": "a1", "chunk_index": 0,
                 "source_row_index": 0, "chunk_text": "x"},
                {"source_api_id": "a2", "chunk_index": 0,
                 "source_row_index": 1, "chunk_text": "y"},
            ])
            new_df, cached_df = Embedder._split_new_and_cached(  # pylint: disable=protected-access
                input_df, output_path,
            )
            self.assertEqual(len(cached_df), 1)
            self.assertEqual(len(new_df), 1)
            self.assertEqual(new_df["source_api_id"].iloc[0], "a2")
            self.assertIn("embedding", cached_df.columns)
            self.assertNotIn("embedding", new_df.columns)


class TestEmbedderTimer(unittest.TestCase):
    """This class tests timer integration in embed_to_parquet."""

    def test_embed_to_parquet_records_timer_sections(self) -> None:
        """embed_to_parquet: records read, embed, and write timer sections."""
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
                embedder.embed_to_parquet(profile="default")

            labels = [r[0] for r in embedder.timer.records]
            self.assertIn("embedder.read_parquet", labels)
            self.assertIn("embedder.embed_texts", labels)
            self.assertIn("embedder.write_parquet", labels)

    def test_shared_timer_receives_embedder_sections(self) -> None:
        """embed_to_parquet: an externally provided timer receives sections."""
        shared_timer = Timer()
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
                embedder = Embedder(timer=shared_timer)
                embedder.embed_to_parquet(profile="default")

            self.assertIs(embedder.timer, shared_timer)
            labels = [r[0] for r in shared_timer.records]
            self.assertIn("embedder.read_parquet", labels)

    def test_embed_to_parquet_empty_input_records_read_only(self) -> None:
        """embed_to_parquet: only read_parquet is recorded when input is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_embedding_config(base)
            _write_input_parquet(cfg, rows=[])

            with patch.object(Settings, "load_embedding_config", return_value=cfg):
                embedder = Embedder()
                embedder.embed_to_parquet(profile="default")

            labels = [r[0] for r in embedder.timer.records]
            self.assertEqual(labels, ["embedder.read_parquet"])


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

    def test_timer_returns_timer_instance(self) -> None:
        """properties: timer returns the Timer instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_embedding_config(Path(tmpdir))
            with patch.object(Settings, "load_embedding_config", return_value=cfg):
                embedder = Embedder()
            self.assertIsInstance(embedder.timer, Timer)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
