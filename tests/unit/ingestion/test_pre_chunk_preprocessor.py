"""Unit tests for YAML-driven pre-chunk parquet preprocessing."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from src.ingestion.prechunk_processing import PreChunkPreprocessor
from tests.unit.ingestion.test_config_helpers import (
    NORMALIZER_ROW_MAPPINGS,
    build_pre_chunk_preprocessor_config,
)


class TestPreChunkPreprocessorPreprocessDayToParquet(unittest.TestCase):
    """This class tests preprocess_all_to_parquet."""

    def test_preprocess_all_to_parquet_applies_operations_in_order(self):
        """preprocess_all_to_parquet: applies operations and writes chunk-ready parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "configuration"
            checkpoint_dir = root / "checkpoints" / "article_ingestor"
            parquet_dir = root / "checkpoints" / "parquet"
            output_dir = root / "checkpoints" / "pre_chunk"
            self._write_ingestion_config(
                config_root=config_root,
                checkpoint_dir=checkpoint_dir,
                parquet_dir=parquet_dir,
                output_dir=output_dir,
            )
            parquet_dir.mkdir(parents=True, exist_ok=True)
            source_path = parquet_dir / "2026-04-28.parquet"
            pd.DataFrame(
                [
                    {
                        "profile": "technology_daily",
                        "api_id": "tech-1",
                        "web_title": "  Title  ",
                        "headline": None,
                        "body_text": "Body\n  text",
                        "thumbnail": "https://example.com/image.png",
                    },
                    {
                        "profile": "technology_daily",
                        "api_id": "tech-2",
                        "web_title": "Next",
                        "headline": "Already set",
                        "body_text": "   ",
                        "thumbnail": "https://example.com/image-2.png",
                    },
                ]
            ).to_parquet(source_path, index=False)

            preprocessor = PreChunkPreprocessor(configuration_root=config_root)
            written = preprocessor.preprocess_to_parquet()

            written_path = Path(written["2026-04-28"])
            self.assertTrue(written_path.is_file())
            output_df = pd.read_parquet(written_path)
            self.assertEqual(list(output_df["api_id"]), ["tech-1"])
            self.assertEqual(output_df.iloc[0]["web_title"], "Title")
            self.assertEqual(output_df.iloc[0]["headline"], "Title")
            self.assertEqual(output_df.iloc[0]["body_text"], "Body text")
            self.assertNotIn("thumbnail", output_df.columns)

    def test_preprocess_all_to_parquet_skips_missing_inputs(self):
        """preprocess_all_to_parquet: skips profiles without source parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "configuration"
            checkpoint_dir = root / "checkpoints" / "article_ingestor"
            parquet_dir = root / "checkpoints" / "parquet"
            output_dir = root / "checkpoints" / "pre_chunk"
            self._write_ingestion_config(
                config_root=config_root,
                checkpoint_dir=checkpoint_dir,
                parquet_dir=parquet_dir,
                output_dir=output_dir,
            )
            preprocessor = PreChunkPreprocessor(configuration_root=config_root)
            written = preprocessor.preprocess_to_parquet()
            self.assertEqual(written, {})

    def test_preprocess_all_to_parquet_raises_for_missing_column_reference(self):
        """preprocess_all_to_parquet: raises when an operation references missing columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_root = root / "configuration"
            checkpoint_dir = root / "checkpoints" / "article_ingestor"
            parquet_dir = root / "checkpoints" / "parquet"
            output_dir = root / "checkpoints" / "pre_chunk"
            self._write_ingestion_config(
                config_root=config_root,
                checkpoint_dir=checkpoint_dir,
                parquet_dir=parquet_dir,
                output_dir=output_dir,
            )
            parquet_dir.mkdir(parents=True, exist_ok=True)
            source_path = parquet_dir / "2026-04-28.parquet"
            pd.DataFrame([{"api_id": "tech-1"}]).to_parquet(source_path, index=False)

            preprocessor = PreChunkPreprocessor(configuration_root=config_root)
            with self.assertRaises(ValueError):
                preprocessor.preprocess_to_parquet()

    @staticmethod
    def _write_ingestion_config(
        *,
        config_root: Path,
        checkpoint_dir: Path,
        parquet_dir: Path,
        output_dir: Path,
    ) -> None:
        config = build_pre_chunk_preprocessor_config(
            checkpoint_dir=str(checkpoint_dir),
            parquet_dir=str(parquet_dir),
            output_dir=str(output_dir),
        )
        config["pre_chunk_preprocessor"]["operations"] = [
            {"name": "drop_columns", "args": {"columns": ["thumbnail"]}},
            {"name": "trim_whitespace_columns", "args": {"columns": ["web_title"]}},
            {
                "name": "coalesce_columns",
                "args": {"target": "headline", "sources": ["headline", "web_title"]},
            },
            {"name": "drop_empty_rows", "args": {"required_columns": ["body_text"]}},
            {"name": "normalize_text_columns", "args": {"columns": ["body_text"]}},
        ]
        config["profiles"] = {"technology_daily": {"topic": "technology"}}
        config["article_normalizer"] = {"row_mappings": NORMALIZER_ROW_MAPPINGS}

        ingestion_config_path = config_root / "ingestion" / "ingestion_config.yaml"
        ingestion_config_path.parent.mkdir(parents=True, exist_ok=True)
        with ingestion_config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
