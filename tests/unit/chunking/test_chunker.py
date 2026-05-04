"""Unit tests for Chunker."""

import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional
from unittest.mock import patch

import pandas as pd

from src.chunking.chunker import Chunker
from src.config.settings import (
    ChunkingConfig,
    ChunkingProfileConfig,
    Settings,
)
from src.enums.chunking_strategy import ChunkingStrategy


def _default_params(**overrides: Any) -> Dict[str, Any]:
    """Build a raw params dict with sensible defaults overridable per test."""
    base: Dict[str, Any] = {
        "min_chars": 1,
        "max_chars": 200,
        "overlap_chars": 0,
        "similarity_threshold": 0.3,
        "sentence_splitter": "simple_regex",
    }
    base.update(overrides)
    return base


def _minimal_chunking_config(
    tmp_path: Path,
    *,
    profile_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_SENTENCE,
) -> ChunkingConfig:
    """Build a minimal ChunkingConfig with one default chunking profile."""
    return ChunkingConfig(
        profile_names=["p1"],
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
        text_columns=["body_text"],
        id_columns=["api_id"],
        profile_columns=["profile"],
        passthrough_columns=["headline"],
        chunking_profiles={
            "default": ChunkingProfileConfig(
                strategy=profile_strategy,
                params=_default_params(),
            ),
        },
    )


def _write_input_day_parquet(
    cfg: ChunkingConfig,
    *,
    day: str = "2026-05-01",
    rows: Optional[list[dict]] = None,
) -> Path:
    """Write an input parquet day file used by the chunker."""
    cfg.input_dir.mkdir(parents=True, exist_ok=True)
    if rows is None:
        rows = [
            {
                "body_text": "One. Two.",
                "api_id": "a1",
                "profile": "p",
                "headline": "H",
            }
        ]
    path = cfg.input_dir / f"{day}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


class TestChunkerChunkToParquet(unittest.TestCase):
    """This class tests chunk_to_parquet."""

    def test_chunk_to_parquet_writes_single_combined_parquet_per_profile(self) -> None:
        """chunk_to_parquet: writes one combined parquet covering every input day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_chunking_config(base)
            _write_input_day_parquet(
                cfg,
                day="2026-05-01",
                rows=[
                    {
                        "body_text": "One. Two.",
                        "api_id": "a1",
                        "profile": "p",
                        "headline": "H",
                    }
                ],
            )
            _write_input_day_parquet(
                cfg,
                day="2026-05-02",
                rows=[
                    {
                        "body_text": "Three. Four.",
                        "api_id": "a2",
                        "profile": "p",
                        "headline": "H2",
                    }
                ],
            )

            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
                written = chunker.chunk_to_parquet(profile="default")

            self.assertEqual(set(written.keys()), {"default"})
            combined_path = Path(written["default"])
            self.assertEqual(combined_path, cfg.output_dir / "default.parquet")
            self.assertTrue(combined_path.is_file())
            combined_df = pd.read_parquet(combined_path)
            self.assertEqual(set(combined_df["source_api_id"]), {"a1", "a2"})
            self.assertEqual(
                set(combined_df["source_day"]),
                {"2026-05-01", "2026-05-02"},
            )

    def test_chunk_to_parquet_returns_empty_when_no_records(self) -> None:
        """chunk_to_parquet: returns empty dict when no chunk rows are produced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_chunking_config(base)
            cfg.input_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["body_text", "api_id"]).to_parquet(
                cfg.input_dir / "2026-05-03.parquet", index=False
            )
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
                written = chunker.chunk_to_parquet(profile="default")

            self.assertEqual(written, {})
            self.assertFalse((cfg.output_dir / "default.parquet").exists())

    def test_chunk_to_parquet_skips_invalid_filename_and_empty_rows(self) -> None:
        """chunk_to_parquet: skips non-ISO stems and empty row payloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_chunking_config(base)
            cfg.input_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"body_text": "", "api_id": "x"}]).to_parquet(
                cfg.input_dir / "2026-05-02.parquet", index=False
            )
            pd.DataFrame([{"body_text": "text"}]).to_parquet(
                cfg.input_dir / "not-a-day.parquet", index=False
            )
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
                written = chunker.chunk_to_parquet(profile="default")

            self.assertEqual(written, {})

    def test_chunk_to_parquet_overwrites_existing_combined_parquet(self) -> None:
        """chunk_to_parquet: rebuilds the combined parquet on every call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            cfg = _minimal_chunking_config(base)
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            stale_path = cfg.output_dir / "default.parquet"
            pd.DataFrame(
                [
                    {
                        "source_api_id": "stale",
                        "chunk_index": 0,
                        "chunk_text": "Stale chunk",
                    }
                ]
            ).to_parquet(stale_path, index=False)
            _write_input_day_parquet(cfg)

            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
                chunker.chunk_to_parquet(profile="default")

            rebuilt_df = pd.read_parquet(stale_path)
            self.assertNotIn("stale", set(rebuilt_df["source_api_id"]))
            self.assertEqual(set(rebuilt_df["source_api_id"]), {"a1"})


class TestChunkerErrors(unittest.TestCase):
    """This class tests chunk_to_parquet error paths."""

    def test_chunk_to_parquet_raises_for_unknown_profile(self) -> None:
        """chunk_to_parquet: raises ValueError when profile is not configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_chunking_config(Path(tmpdir))
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
            with self.assertRaises(ValueError) as ctx:
                chunker.chunk_to_parquet(profile="missing")
            self.assertIn("Unknown chunking profile", str(ctx.exception))

    def test_chunk_to_parquet_raises_when_handler_missing(self) -> None:
        """chunk_to_parquet: raises ValueError for unregistered strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_chunking_config(Path(tmpdir))
            new_profiles = dict(cfg.chunking_profiles)
            new_profiles["default"] = replace(
                cfg.chunking_profiles["default"],
                strategy="unregistered_strategy",  # type: ignore[arg-type]
            )
            cfg = replace(cfg, chunking_profiles=new_profiles)
            _write_input_day_parquet(cfg)
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
                with self.assertRaises(ValueError) as ctx:
                    chunker.chunk_to_parquet(profile="default")
            self.assertIn("No chunking handler registered", str(ctx.exception))


class TestChunkerHelpers(unittest.TestCase):
    """This class tests helper methods on Chunker."""

    def test_profile_names_and_chunking_profile_names(self) -> None:
        """helpers: expose ingestion and chunking profile names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_chunking_config(Path(tmpdir))
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
            self.assertEqual(chunker.profile_names, ["p1"])
            self.assertEqual(chunker.chunking_profile_names, ["default"])

    def test_parse_day_helpers(self) -> None:
        """helpers: parse ISO day token from filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_chunking_config(Path(tmpdir))
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
            parsed = chunker._parse_day_from_filename(  # pylint: disable=protected-access
                Path("2026-01-01.parquet"),
            )
            self.assertEqual(parsed, "2026-01-01")
            bad = chunker._parse_day_from_filename(  # pylint: disable=protected-access
                Path("bad.parquet"),
            )
            self.assertIsNone(bad)

    def test_resolve_first_string_handles_missing_and_nan(self) -> None:
        """_resolve_first_string: skips missing and NaN values."""
        row = pd.Series({"a": float("nan"), "b": "  hi "})
        value, column = Chunker._resolve_first_string(  # pylint: disable=protected-access
            row,
            ["x", "a", "b"],
        )
        self.assertEqual(value, "hi")
        self.assertEqual(column, "b")

    def test_output_path_replaces_slashes_in_profile_name(self) -> None:
        """_output_path: forward slashes in profile name are replaced for filesystem safety."""
        path = Chunker._output_path(  # pylint: disable=protected-access
            Path("base"),
            "team/profile",
        )
        self.assertEqual(path, Path("base/team_profile.parquet"))


class TestChunkerChunkRow(unittest.TestCase):
    """This class tests _chunk_row."""

    def test_chunk_row_returns_empty_when_no_text_column(self) -> None:
        """_chunk_row: returns empty records when no configured text is present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = _minimal_chunking_config(Path(tmpdir))
            with patch.object(Settings, "load_chunking_config", return_value=cfg):
                chunker = Chunker()
            row = pd.Series({"headline": "h"})
            rows = chunker._chunk_row(  # pylint: disable=protected-access
                row,
                source_day="2026-05-01",
                source_row_index=0,
                profile=cfg.chunking_profiles["default"],
            )
        self.assertEqual(rows, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
