# pylint: disable=duplicate-code
"""Unit tests for article normalizer config parsing."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.article_row_source_kind import ArticleRowSourceKind
from src.enums.article_row_transform import ArticleRowTransform


class TestSettingsLoadArticleNormalizerConfig(unittest.TestCase):
    """This class tests load_article_normalizer_config."""

    def test_load_article_normalizer_config_returns_typed_config(self):
        """load_article_normalizer_config: returns validated typed normalizer settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                },
                "article_ingestor": {
                    "checkpoint_dir": "checkpoints/article_ingestor",
                    "parquet_dir": "checkpoints/parquet",
                },
                "article_normalizer": {
                    "row_mappings": {"headline": {"sources": ["fields.headline"]}}
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_article_normalizer_config(configuration_root=root)

        self.assertEqual(typed_config.profile_names, ["technology_daily", "science_daily"])
        self.assertEqual(typed_config.checkpoint_dir, Path("checkpoints/article_ingestor"))
        self.assertEqual(typed_config.parquet_dir, Path("checkpoints/parquet"))
        self.assertEqual(
            typed_config.row_mappings["headline"].sources[0].kind,
            ArticleRowSourceKind.FIELDS,
        )
        self.assertEqual(typed_config.row_mappings["headline"].sources[0].path, "headline")

    def test_load_article_normalizer_config_parses_direct_key_and_transform(self):
        """load_article_normalizer_config: parses direct-key sources and enum transforms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {
                        "published_at": {
                            "sources": ["webPublicationDate"],
                            "transform": "parse_iso",
                        }
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_article_normalizer_config(configuration_root=root)

        row_mapping = typed_config.row_mappings["published_at"]
        self.assertEqual(row_mapping.sources[0].kind, ArticleRowSourceKind.DIRECT_KEY)
        self.assertEqual(row_mapping.sources[0].path, "webPublicationDate")
        self.assertEqual(row_mapping.transform, ArticleRowTransform.PARSE_ISO)

    def test_load_article_normalizer_config_parses_profile_source(self):
        """load_article_normalizer_config: parses the reserved profile source selector."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {"row_mappings": {"profile": {"sources": ["profile"]}}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_article_normalizer_config(configuration_root=root)

        self.assertEqual(
            typed_config.row_mappings["profile"].sources[0].kind,
            ArticleRowSourceKind.PROFILE,
        )

    def test_load_article_normalizer_config_raises_for_missing_row_mappings(self):
        """load_article_normalizer_config: raises when row_mappings are absent or empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_non_mapping_row_mapping(self):
        """load_article_normalizer_config: raises when one row mapping is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {"row_mappings": {"headline": "fields.headline"}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_invalid_sources_list(self):
        """load_article_normalizer_config: raises when sources is missing, empty, or malformed."""
        invalid_row_mappings = [
            {"headline": {"sources": []}},
            {"headline": {"sources": "fields.headline"}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for row_mappings in invalid_row_mappings:
                raw_config = {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {},
                    "article_normalizer": {"row_mappings": row_mappings},
                }
                with self.subTest(row_mappings=row_mappings), patch(
                    "src.config.settings.Settings.load_ingestion_config_from_root",
                    return_value=raw_config,
                ):
                    with self.assertRaises(ValueError):
                        Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_non_string_source(self):
        """load_article_normalizer_config: raises when a source entry is not a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {"row_mappings": {"headline": {"sources": [1]}}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_invalid_source_namespace(self):
        """load_article_normalizer_config: raises for unsupported dotted source namespaces."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {"headline": {"sources": ["feilds.headline"]}}
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_empty_dotted_path(self):
        """load_article_normalizer_config: raises when a dotted source path is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {"row_mappings": {"headline": {"sources": ["fields."]}}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_invalid_transform(self):
        """load_article_normalizer_config: raises when transform is unsupported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {
                        "headline": {
                            "sources": ["fields.headline"],
                            "transform": "uppercase",
                        }
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_non_string_transform(self):
        """load_article_normalizer_config: raises when transform is not a string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {
                        "headline": {
                            "sources": ["fields.headline"],
                            "transform": 1,
                        }
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)

    def test_load_article_normalizer_config_raises_for_invalid_profiles(self):
        """load_article_normalizer_config: raises when profiles is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": "not a dict",
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {"headline": {"sources": ["fields.headline"]}}
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_normalizer_config(configuration_root=root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
