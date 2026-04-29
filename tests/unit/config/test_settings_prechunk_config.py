# pylint: disable=duplicate-code
"""Unit tests for pre-chunk preprocessor config parsing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.pre_chunk_operation import PreChunkOperation


class TestSettingsLoadPreChunkPreprocessorConfig(unittest.TestCase):
    """This class tests load_pre_chunk_preprocessor_config."""

    def test_load_pre_chunk_preprocessor_config_returns_typed_config(self):
        """load_pre_chunk_preprocessor_config: returns typed config with validated operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"parquet_dir": "checkpoints/parquet"},
                "pre_chunk_preprocessor": {
                    "output_dir": "checkpoints/pre_chunk",
                    "operations": [
                        {"name": "drop_columns", "args": {"columns": ["thumbnail"]}},
                        {
                            "name": "coalesce_columns",
                            "args": {
                                "target": "headline",
                                "sources": ["headline", "web_title"],
                            },
                        },
                    ],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_pre_chunk_preprocessor_config(
                    configuration_root=root
                )

        self.assertEqual(typed_config.profile_names, ["technology_daily"])
        self.assertEqual(typed_config.input_dir, Path("checkpoints/parquet"))
        self.assertEqual(typed_config.output_dir, Path("checkpoints/pre_chunk"))
        self.assertEqual(typed_config.operations[0].name, PreChunkOperation.DROP_COLUMNS)
        self.assertEqual(typed_config.operations[1].name, PreChunkOperation.COALESCE_COLUMNS)

    def test_load_pre_chunk_preprocessor_config_raises_for_missing_operations(self):
        """load_pre_chunk_preprocessor_config: raises when operations are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_raises_for_invalid_operation_name(self):
        """load_pre_chunk_preprocessor_config: raises for unsupported operation names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [
                        {"name": "remove_fields", "args": {"columns": ["thumbnail"]}}
                    ]
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_raises_for_malformed_args(self):
        """load_pre_chunk_preprocessor_config: raises when operation args are malformed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [{"name": "drop_columns", "args": {"columns": []}}]
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_raises_for_non_mapping_operation(self):
        """load_pre_chunk_preprocessor_config: raises when an operation item is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {"operations": ["drop_columns"]},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_raises_for_empty_operation_name(self):
        """load_pre_chunk_preprocessor_config: raises when an operation name is blank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [{"name": "", "args": {}}],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_supports_rename_columns(self):
        """load_pre_chunk_preprocessor_config: parses rename_columns operation arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [
                        {
                            "name": "rename_columns",
                            "args": {"mapping": {"old_name": "new_name"}},
                        }
                    ],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_pre_chunk_preprocessor_config(
                    configuration_root=root
                )

        self.assertEqual(
            typed_config.operations[0].args["mapping"],
            {"old_name": "new_name"},
        )

    def test_load_pre_chunk_preprocessor_config_supports_filter_min_numeric(self):
        """load_pre_chunk_preprocessor_config: parses filter_min_numeric operation arguments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [
                        {
                            "name": "filter_min_numeric",
                            "args": {"column": "wordcount", "min_value": 500},
                        }
                    ],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_pre_chunk_preprocessor_config(
                    configuration_root=root
                )

        self.assertEqual(
            typed_config.operations[0].name,
            PreChunkOperation.FILTER_MIN_NUMERIC,
        )
        self.assertEqual(typed_config.operations[0].args["column"], "wordcount")
        self.assertEqual(typed_config.operations[0].args["min_value"], 500.0)

    def test_load_pre_chunk_preprocessor_config_rejects_non_numeric_min_value(self):
        """load_pre_chunk_preprocessor_config: raises when filter min_value is not numeric."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [
                        {
                            "name": "filter_min_numeric",
                            "args": {"column": "wordcount", "min_value": "not-a-number"},
                        }
                    ],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_raises_for_non_mapping_args(self):
        """load_pre_chunk_preprocessor_config: raises when operation args are not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {
                    "operations": [{"name": "drop_columns", "args": ["thumbnail"]}],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_pre_chunk_preprocessor_config_uses_empty_mapping_for_missing_args(self):
        """load_pre_chunk_preprocessor_config: treats missing args as an empty mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "pre_chunk_preprocessor": {"operations": [{"name": "drop_columns"}]},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
