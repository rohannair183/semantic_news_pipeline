"""Unit tests for YAML-driven pre-chunk parquet preprocessing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from src.config.settings import PreChunkOperationConfig, PreChunkPreprocessorConfig
from src.enums.pre_chunk_operation import PreChunkOperation
from src.ingestion.prechunk_processing import PreChunkPreprocessor
from tests.unit.ingestion.test_config_helpers import (
    NORMALIZER_ROW_MAPPINGS,
    build_pre_chunk_preprocessor_config,
)


def _build_preprocessor_with_config(config: PreChunkPreprocessorConfig) -> PreChunkPreprocessor:
    with patch(
        "src.ingestion.prechunk_processing.Settings.load_pre_chunk_preprocessor_config"
    ) as mock_load_config:
        mock_load_config.return_value = config
        return PreChunkPreprocessor()


def _build_operation(
    name: PreChunkOperation,
    args: dict[str, object],
) -> PreChunkOperationConfig:
    return PreChunkOperationConfig(name=name, args=args)


class TestPreChunkPreprocessorHelpers(unittest.TestCase):
    """This class tests helper methods on PreChunkPreprocessor."""

    def test_profile_names_returns_copy(self):
        """profile_names: returns a copy of the configured profile list."""
        config = PreChunkPreprocessorConfig(
            profile_names=["technology_daily"],
            input_dir=Path("input"),
            output_dir=Path("output"),
            operations=[],
        )
        preprocessor = _build_preprocessor_with_config(config)

        profile_names = preprocessor.profile_names
        profile_names.append("mutated")

        self.assertEqual(preprocessor.profile_names, ["technology_daily"])

    def test_parse_day_from_filename_returns_none_for_invalid_name(self):
        """_parse_day_from_filename: returns None when the filename stem is not a date."""
        self.assertIsNone(
            PreChunkPreprocessor._parse_day_from_filename(  # pylint: disable=protected-access
                Path("not-a-day.parquet")
            )
        )

    def test_is_present_handles_none_and_non_string_values(self):
        """_is_present: returns False for None and True for non-string values."""
        self.assertFalse(
            PreChunkPreprocessor._is_present(None)  # pylint: disable=protected-access
        )
        self.assertTrue(
            PreChunkPreprocessor._is_present(0)  # pylint: disable=protected-access
        )

    def test_apply_operation_supports_rename_columns(self):
        """_apply_operation: renames columns when rename mapping is configured."""
        preprocessor = _build_preprocessor_with_config(
            PreChunkPreprocessorConfig(
                profile_names=["technology_daily"],
                input_dir=Path("input"),
                output_dir=Path("output"),
                operations=[],
            )
        )
        input_df = pd.DataFrame([{"old_name": "value"}])

        output_df = preprocessor._apply_operation(  # pylint: disable=protected-access
            input_df,
            _build_operation(
                PreChunkOperation.RENAME_COLUMNS,
                {"mapping": {"old_name": "new_name"}},
            ),
        )

        self.assertEqual(list(output_df.columns), ["new_name"])

    def test_apply_operation_returns_original_df_for_empty_rename_mapping(self):
        """_apply_operation: returns the original DataFrame for an empty rename mapping."""
        preprocessor = _build_preprocessor_with_config(
            PreChunkPreprocessorConfig(
                profile_names=["technology_daily"],
                input_dir=Path("input"),
                output_dir=Path("output"),
                operations=[],
            )
        )
        input_df = pd.DataFrame([{"headline": "value"}])

        output_df = preprocessor._apply_operation(  # pylint: disable=protected-access
            input_df,
            _build_operation(PreChunkOperation.RENAME_COLUMNS, {"mapping": {}}),
        )

        self.assertIs(output_df, input_df)

    def test_apply_operation_coalesces_into_new_target_column(self):
        """_apply_operation: creates the target column when coalescing into a new field."""
        preprocessor = _build_preprocessor_with_config(
            PreChunkPreprocessorConfig(
                profile_names=["technology_daily"],
                input_dir=Path("input"),
                output_dir=Path("output"),
                operations=[],
            )
        )
        input_df = pd.DataFrame([{"headline": None, "web_title": "Fallback"}])

        output_df = preprocessor._apply_operation(  # pylint: disable=protected-access
            input_df,
            _build_operation(
                PreChunkOperation.COALESCE_COLUMNS,
                {"target": "title", "sources": ["headline", "web_title"]},
            ),
        )

        self.assertEqual(output_df.iloc[0]["title"], "Fallback")

    def test_apply_operation_raises_for_unsupported_operation(self):
        """_apply_operation: raises when given an unsupported operation enum-like value."""
        preprocessor = _build_preprocessor_with_config(
            PreChunkPreprocessorConfig(
                profile_names=["technology_daily"],
                input_dir=Path("input"),
                output_dir=Path("output"),
                operations=[],
            )
        )
        unsupported_operation = PreChunkOperationConfig(
            name="unsupported",  # type: ignore[arg-type]
            args={},
        )

        with self.assertRaises(ValueError):
            preprocessor._apply_operation(  # pylint: disable=protected-access
                pd.DataFrame([{"headline": "value"}]),
                unsupported_operation,
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

    def test_preprocess_all_to_parquet_skips_invalid_empty_and_fully_filtered_inputs(
        self,
    ):
        """preprocess_all_to_parquet: skips invalid, empty, and fully filtered inputs."""
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
            pd.DataFrame([{"body_text": "text"}]).to_parquet(
                parquet_dir / "not-a-date.parquet",
                index=False,
            )
            pd.DataFrame(columns=["body_text"]).to_parquet(
                parquet_dir / "2026-04-28.parquet",
                index=False,
            )
            pd.DataFrame(
                [{"web_title": "Title", "headline": None, "body_text": "   "}],
            ).to_parquet(
                parquet_dir / "2026-04-29.parquet",
                index=False,
            )

            preprocessor = PreChunkPreprocessor(configuration_root=config_root)

            written = preprocessor.preprocess_to_parquet()

            self.assertEqual(written, {})

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
