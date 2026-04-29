"""Unit tests for the ArticleNormalizer class in the src.ingestion module."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.ingestion.article_normalizer import ArticleNormalizer


def _build_article_normalizer(config_root: Path) -> ArticleNormalizer:
    with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
        mock_parser = Mock()
        mock_parser.parse.return_value = {
            "profiles": {
                "technology_daily": {"topic": "technology"},
                "science_daily": {"topic": "science"},
            },
            "article_ingestor": {
                "checkpoint_dir": str(config_root / "checkpoints"),
                "parquet_dir": str(config_root / "parquet"),
            },
        }
        mock_parser_class.return_value = mock_parser
        return ArticleNormalizer(configuration_root=config_root)


class TestArticleNormalizerInit(unittest.TestCase):
    """This class tests __init__."""

    def test_init_loads_config(self):
        """__init__: loads ingestion config and resolves paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = _build_article_normalizer(Path(tmpdir))
            self.assertEqual(normalizer.profiles, ["technology_daily", "science_daily"])
            self.assertIsInstance(normalizer.checkpoint_dir, Path)
            self.assertIsInstance(normalizer.parquet_dir, Path)

    def test_init_raises_for_invalid_profiles(self):
        """__init__: raises ValueError if profiles is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {"profiles": "not a dict"}
                mock_parser_class.return_value = mock_parser
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))

    def test_init_defaults_parquet_dir(self):
        """__init__: uses default parquet_dir if not in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"test": {}},
                    "article_ingestor": {"checkpoint_dir": str(Path(tmpdir) / "check")},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=Path(tmpdir))
                self.assertIn("parquet", str(normalizer.parquet_dir))


class TestArticleNormalizerParseTS(unittest.TestCase):
    """This class tests _parse_ts_from_filename."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.normalizer = _build_article_normalizer(Path(tmpdir))

    def test_parse_ts_valid_filename(self):
        """_parse_ts_from_filename: parses valid timestamp from filename."""
        path = Path("technology_daily_20260428T221904Z.json")
        ts = self.normalizer._parse_ts_from_filename(path)  # pylint: disable=protected-access
        self.assertEqual(ts.year, 2026)
        self.assertEqual(ts.month, 4)
        self.assertEqual(ts.day, 28)

    def test_parse_ts_invalid_format(self):
        """_parse_ts_from_filename: returns None for invalid timestamp."""
        path = Path("technology_daily_invalid.json")
        ts = self.normalizer._parse_ts_from_filename(path)  # pylint: disable=protected-access
        self.assertIsNone(ts)

    def test_parse_ts_no_underscore(self):
        """_parse_ts_from_filename: returns None if no underscore in stem."""
        path = Path("invalidname.json")
        ts = self.normalizer._parse_ts_from_filename(path)  # pylint: disable=protected-access
        self.assertIsNone(ts)


class TestArticleNormalizerListProfileFiles(unittest.TestCase):
    """This class tests _list_profile_files."""

    def test_list_profile_files_finds_matching(self):
        """_list_profile_files: returns all matching JSON files for profile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            # Create test files
            (checkpoint_dir / "tech_daily_20260428T100000Z.json").touch()
            (checkpoint_dir / "tech_daily_20260427T100000Z.json").touch()
            (checkpoint_dir / "science_daily_20260428T100000Z.json").touch()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}, "science_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                # pylint: disable=protected-access
                files = normalizer._list_profile_files("tech_daily")
                # pylint: enable=protected-access
                self.assertEqual(len(files), 2)

    def test_list_profile_files_empty_when_no_match(self):
        """_list_profile_files: returns empty list if no matching files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                # pylint: disable=protected-access
                files = normalizer._list_profile_files("nonexistent")
                # pylint: enable=protected-access
                self.assertEqual(len(files), 0)


class TestArticleNormalizerParseISO(unittest.TestCase):
    """This class tests _parse_iso."""

    def test_parse_iso_with_z_suffix(self):
        """_parse_iso: parses ISO string with Z suffix."""
        # pylint: disable=protected-access
        dt = ArticleNormalizer._parse_iso("2026-04-28T22:19:04Z")
        # pylint: enable=protected-access
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 28)

    def test_parse_iso_with_timezone(self):
        """_parse_iso: parses ISO string with timezone offset."""
        # pylint: disable=protected-access
        dt = ArticleNormalizer._parse_iso("2026-04-28T22:19:04+00:00")
        # pylint: enable=protected-access
        self.assertEqual(dt.year, 2026)

    def test_parse_iso_returns_none_for_invalid(self):
        """_parse_iso: returns None for invalid ISO string."""
        dt = ArticleNormalizer._parse_iso("not-a-date")  # pylint: disable=protected-access
        self.assertIsNone(dt)

    def test_parse_iso_returns_none_for_empty(self):
        """_parse_iso: returns None for empty or None input."""
        self.assertIsNone(ArticleNormalizer._parse_iso(None))  # pylint: disable=protected-access
        self.assertIsNone(ArticleNormalizer._parse_iso(""))  # pylint: disable=protected-access


class TestArticleNormalizerFindLatest(unittest.TestCase):
    """This class tests find_latest_checkpoints_for_date."""

    def test_find_latest_returns_latest_per_profile(self):
        """find_latest_checkpoints_for_date: returns latest checkpoint per profile for day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            # Create checkpoints for tech_daily on same day
            (checkpoint_dir / "tech_daily_20260428T100000Z.json").touch()
            (checkpoint_dir / "tech_daily_20260428T200000Z.json").touch()
            (checkpoint_dir / "science_daily_20260428T150000Z.json").touch()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}, "science_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                latest = normalizer.find_latest_checkpoints_for_date(day)

                self.assertIn("tech_daily", latest)
                self.assertIn("science_daily", latest)
                self.assertIn("20260428T200000Z", str(latest["tech_daily"]))

    def test_find_latest_empty_for_missing_day(self):
        """find_latest_checkpoints_for_date: returns empty dict if no checkpoints for day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            (checkpoint_dir / "tech_daily_20260427T100000Z.json").touch()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                latest = normalizer.find_latest_checkpoints_for_date(day)
                self.assertEqual(len(latest), 0)

    def test_find_latest_skips_invalid_timestamps(self):
        """find_latest_checkpoints_for_date: skips files with invalid timestamp format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            (checkpoint_dir / "tech_daily_invalid.json").touch()
            (checkpoint_dir / "tech_daily_20260428T100000Z.json").touch()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                latest = normalizer.find_latest_checkpoints_for_date(day)

                self.assertEqual(len(latest), 1)
                self.assertIn("tech_daily", latest)


class TestArticleNormalizerNormalizeCheckpoint(unittest.TestCase):
    """This class tests normalize_checkpoint."""

    def test_normalize_checkpoint_extracts_fields(self):
        """normalize_checkpoint: extracts and normalizes article fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_file = tmppath / "test_checkpoint.json"

            checkpoint_data = {
                "profile": "test_profile",
                "items": [
                    {
                        "id": "article-1",
                        "webTitle": "Test Article",
                        "webPublicationDate": "2026-04-28T10:00:00Z",
                        "webUrl": "https://example.com/article-1",
                        "sectionName": "Technology",
                        "pillarName": "News",
                        "fields": {
                            "headline": "Test Headline",
                            "byline": "Test Author",
                            "bodyText": "Test content here",
                            "trailText": "Test trail",
                            "thumbnail": "https://example.com/thumb.jpg",
                            "wordcount": "500",
                            "lastModified": "2026-04-28T10:30:00Z",
                        },
                    }
                ],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            normalizer = _build_article_normalizer(tmppath)
            df = normalizer.normalize_checkpoint(checkpoint_file)

            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["api_id"], "article-1")
            self.assertEqual(df.iloc[0]["headline"], "Test Headline")
            self.assertEqual(df.iloc[0]["profile"], "test_profile")

    def test_normalize_checkpoint_override_profile(self):
        """normalize_checkpoint: allows profile override."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_file = tmppath / "test_checkpoint.json"

            checkpoint_data = {
                "profile": "original_profile",
                "items": [{"id": "article-1", "fields": {}}],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            normalizer = _build_article_normalizer(tmppath)
            df = normalizer.normalize_checkpoint(checkpoint_file, profile="override_profile")

            self.assertEqual(df.iloc[0]["profile"], "override_profile")

    def test_normalize_checkpoint_empty_items(self):
        """normalize_checkpoint: returns empty DataFrame if no items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_file = tmppath / "test_checkpoint.json"

            checkpoint_data = {"profile": "test", "items": []}

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            normalizer = _build_article_normalizer(tmppath)
            df = normalizer.normalize_checkpoint(checkpoint_file)

            self.assertEqual(len(df), 0)

    def test_normalize_checkpoint_handles_missing_fields(self):
        """normalize_checkpoint: handles missing optional fields gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_file = tmppath / "test_checkpoint.json"

            checkpoint_data = {
                "profile": "test",
                "items": [{"id": "article-1"}],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            normalizer = _build_article_normalizer(tmppath)
            df = normalizer.normalize_checkpoint(checkpoint_file)

            self.assertEqual(len(df), 1)
            self.assertIsNone(df.iloc[0]["headline"])


class TestArticleNormalizerWriteParquet(unittest.TestCase):
    """This class tests _write_parquet."""

    def test_write_parquet_creates_file(self):
        """_write_parquet: creates parquet file with correct path structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            parquet_dir = tmppath / "parquet"

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech": {}},
                    "article_ingestor": {"parquet_dir": str(parquet_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
                day = date(2026, 4, 28)

                with patch.object(df, "to_parquet"):
                    # pylint: disable=protected-access
                    out_path = normalizer._write_parquet(df, "tech", day)
                    # pylint: enable=protected-access
                    self.assertIn("tech", str(out_path))
                    self.assertIn("20260428", str(out_path))
                    self.assertIn("parquet", str(out_path))


class TestArticleNormalizerNormalizeDay(unittest.TestCase):
    """This class tests normalize_day_to_parquet."""

    def test_normalize_day_to_parquet_writes_profiles(self):
        """normalize_day_to_parquet: finds checkpoints and writes parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()
            parquet_dir = tmppath / "parquet"

            checkpoint_file = checkpoint_dir / "tech_daily_20260428T100000Z.json"
            checkpoint_data = {
                "profile": "tech_daily",
                "items": [
                    {
                        "id": "article-1",
                        "webTitle": "Test",
                        "fields": {"headline": "Test"},
                    }
                ],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {
                        "checkpoint_dir": str(checkpoint_dir),
                        "parquet_dir": str(parquet_dir),
                    },
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                with patch("src.ingestion.article_normalizer.pd.DataFrame.to_parquet"):
                    written = normalizer.normalize_day_to_parquet(day)
                    self.assertIn("tech_daily", written)

    def test_normalize_day_to_parquet_empty_when_no_checkpoints(self):
        """normalize_day_to_parquet: returns empty dict if no checkpoints for day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                written = normalizer.normalize_day_to_parquet(day)

                self.assertEqual(len(written), 0)

    def test_normalize_day_to_parquet_skips_empty_dataframes(self):
        """normalize_day_to_parquet: skips checkpoints that normalize to empty DataFrames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            checkpoint_file = checkpoint_dir / "tech_daily_20260428T100000Z.json"
            checkpoint_data = {
                "profile": "tech_daily",
                "items": [],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            with patch("src.ingestion.article_normalizer.YAMLConfigParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser.parse.return_value = {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {
                        "checkpoint_dir": str(checkpoint_dir),
                        "parquet_dir": str(tmppath / "parquet"),
                    },
                }
                mock_parser_class.return_value = mock_parser
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                written = normalizer.normalize_day_to_parquet(day)

                self.assertEqual(len(written), 0)


if __name__ == "__main__":
    unittest.main()
