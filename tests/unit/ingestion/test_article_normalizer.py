"""Unit tests for the ArticleNormalizer class in the src.ingestion module."""

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, cast
from unittest.mock import PropertyMock, patch

import pandas as pd

from src.config.settings import ArticleNormalizerConfig, ArticleRowMappingConfig
from src.ingestion.article_normalizer import ArticleNormalizer
from tests.unit.ingestion.test_config_helpers import (
    NORMALIZER_ROW_MAPPINGS,
    build_normalizer_config,
    build_row_mapping_configs,
    build_row_source_config,
)


def _build_article_normalizer_config(
    config_root: Path,
    config: Optional[Dict[str, Any]] = None,
) -> ArticleNormalizerConfig:
    if config is None:
        config = build_normalizer_config(
            checkpoint_dir=str(config_root / "checkpoints"),
            parquet_dir=str(config_root / "parquet"),
        )

    article_ingestor_config = config.get("article_ingestor", {}) or {}
    article_normalizer_config = config["article_normalizer"]

    return ArticleNormalizerConfig(
        profile_names=list(config["profiles"].keys()),
        checkpoint_dir=Path(
            str(article_ingestor_config.get("checkpoint_dir", "checkpoints/article_ingestor"))
        ),
        parquet_dir=Path(str(article_ingestor_config.get("parquet_dir", "checkpoints/parquet"))),
        row_mappings=build_row_mapping_configs(article_normalizer_config["row_mappings"]),
    )


def _build_article_normalizer(config_root: Path) -> ArticleNormalizer:
    with patch(
        "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
    ) as mock_load_config:
        mock_load_config.return_value = _build_article_normalizer_config(config_root)
        return ArticleNormalizer(configuration_root=config_root)


class TestArticleNormalizerInit(unittest.TestCase):
    """This class tests __init__."""

    def test_init_loads_config(self):
        """__init__: loads ingestion config and exposes config-backed properties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = _build_article_normalizer(Path(tmpdir))
            self.assertEqual(normalizer.profiles, ["technology_daily", "science_daily"])
            self.assertIsInstance(normalizer.checkpoint_dir, Path)
            self.assertIsInstance(normalizer.parquet_dir, Path)
            self.assertEqual(
                set(normalizer.row_mappings.keys()),
                set(NORMALIZER_ROW_MAPPINGS.keys()),
            )

    def test_init_raises_for_invalid_profiles(self):
        """__init__: raises ValueError if profiles is not a dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.side_effect = ValueError(
                    "Ingestion config must define a non-empty 'profiles' mapping"
                )
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))

    def test_init_defaults_parquet_dir(self):
        """__init__: uses default parquet_dir if not in config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    Path(tmpdir),
                    {
                        "profiles": {"test": {}},
                        "article_ingestor": {"checkpoint_dir": str(Path(tmpdir) / "check")},
                        "article_normalizer": {
                            "row_mappings": NORMALIZER_ROW_MAPPINGS.copy()
                        },
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=Path(tmpdir))
                self.assertIn("parquet", str(normalizer.parquet_dir))

    def test_init_rejects_invalid_article_normalizer_config(self):
        """__init__: raises ValueError if article_normalizer is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.side_effect = ValueError(
                    "Ingestion config must contain an 'article_normalizer' mapping"
                )
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))

    def test_init_rejects_null_article_normalizer_config(self):
        """__init__: raises ValueError if article_normalizer is null."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.side_effect = ValueError(
                    "Ingestion config must contain an 'article_normalizer' mapping"
                )
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))

    def test_init_rejects_missing_row_mappings(self):
        """__init__: raises ValueError if row_mappings are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.side_effect = ValueError(
                    "Ingestion config must contain a non-empty 'row_mappings' mapping"
                )
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))

    def test_init_rejects_empty_row_mappings(self):
        """__init__: raises ValueError if row_mappings is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.side_effect = ValueError(
                    "Ingestion config must contain a non-empty 'row_mappings' mapping"
                )
                with self.assertRaises(ValueError):
                    ArticleNormalizer(configuration_root=Path(tmpdir))


class TestArticleNormalizerParseTS(unittest.TestCase):
    """This class tests _parse_ts_from_filename."""

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.normalizer = _build_article_normalizer(Path(tmpdir))

    def test_parse_ts_valid_filename(self):
        """_parse_ts_from_filename: parses valid timestamp from filename."""
        path = Path("technology_daily_20260428T221904Z.json")
        ts = self.normalizer._parse_ts_from_filename(path)  # pylint: disable=protected-access
        self.assertIsNotNone(ts)
        ts = cast(datetime, ts)
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                        "profiles": {"tech_daily": {}, "science_daily": {}},
                        "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                        "article_normalizer": {
                            "row_mappings": NORMALIZER_ROW_MAPPINGS.copy()
                        },
                    },
                )
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                        "profiles": {"tech_daily": {}},
                        "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                        "article_normalizer": {
                            "row_mappings": NORMALIZER_ROW_MAPPINGS.copy()
                        },
                    },
                )
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
        self.assertIsNotNone(dt)
        dt = cast(datetime, dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 4)
        self.assertEqual(dt.day, 28)

    def test_parse_iso_with_timezone(self):
        """_parse_iso: parses ISO string with timezone offset."""
        # pylint: disable=protected-access
        dt = ArticleNormalizer._parse_iso("2026-04-28T22:19:04+00:00")
        # pylint: enable=protected-access
        self.assertIsNotNone(dt)
        dt = cast(datetime, dt)
        self.assertEqual(dt.year, 2026)

    def test_parse_iso_returns_none_for_invalid(self):
        """_parse_iso: returns None for invalid ISO string."""
        dt = ArticleNormalizer._parse_iso("not-a-date")  # pylint: disable=protected-access
        self.assertIsNone(dt)

    def test_parse_iso_returns_none_for_empty(self):
        """_parse_iso: returns None for empty or None input."""
        self.assertIsNone(ArticleNormalizer._parse_iso(None))  # pylint: disable=protected-access
        self.assertIsNone(ArticleNormalizer._parse_iso(""))  # pylint: disable=protected-access


class TestArticleNormalizerRowMappingHelpers(unittest.TestCase):
    """This class tests the row mapping helper methods."""

    def setUp(self):
        self.normalizer = _build_article_normalizer(Path(tempfile.mkdtemp()))

    def test_resolve_nested_value_returns_none_for_scalar_path(self):
        """_resolve_nested_value: returns None when a path walks into a scalar."""
        result = ArticleNormalizer._resolve_nested_value(  # pylint: disable=protected-access
            {"outer": "value"},
            "outer.inner",
        )
        self.assertIsNone(result)

    def test_resolve_source_value_supports_item_nested_path(self):
        """_resolve_source_value: resolves nested item paths."""
        context = {
            "item": {"fields": {"headline": "Nested Headline"}},
            "fields": {},
            "payload": {},
            "profile_value": "test_profile",
        }
        result = self.normalizer._resolve_source_value(  # pylint: disable=protected-access
            source_config=build_row_source_config("item.fields.headline"),
            context=context,
        )
        self.assertEqual(result, "Nested Headline")

    def test_first_non_empty_skips_blank_values(self):
        """_first_non_empty: skips None and blank strings before returning a value."""
        result = ArticleNormalizer._first_non_empty(  # pylint: disable=protected-access
            [None, "", "chosen", "later"]
        )
        self.assertEqual(result, "chosen")

    def test_apply_row_transform_rejects_unknown_transform(self):
        """_apply_row_transform: raises ValueError for unsupported transforms."""
        with self.assertRaises(ValueError):
            self.normalizer._apply_row_transform(  # pylint: disable=protected-access
                value="2026-04-28T10:00:00Z",
                transform="unknown",
            )

    def test_build_row_rejects_empty_sources(self):
        """_build_row: raises ValueError when a mapping has no sources."""
        with patch.object(
            ArticleNormalizer,
            "row_mappings",
            new_callable=PropertyMock,
            return_value={"headline": ArticleRowMappingConfig(sources=[])},
        ):
            with self.assertRaises(ValueError):
                self.normalizer._build_row(  # pylint: disable=protected-access
                    payload={"profile": "test_profile"},
                    item={"fields": {}},
                    profile=None,
                )


class TestArticleNormalizerResolveSourceValue(unittest.TestCase):
    """This class tests _resolve_source_value."""

    def setUp(self):
        self.normalizer = _build_article_normalizer(Path(tempfile.mkdtemp()))
        self.context = {
            "item": {"headline": "Item headline"},
            "fields": {"headline": "Fields headline"},
            "payload": {"headline": "Payload headline"},
            "profile_value": "test_profile",
        }

    def test_resolve_source_value_prefers_item_for_direct_key(self):
        """_resolve_source_value: prefers item values for direct-key sources."""
        result = self.normalizer._resolve_source_value(  # pylint: disable=protected-access
            source_config=build_row_source_config("headline"),
            context=self.context,
        )
        self.assertEqual(result, "Item headline")

    def test_resolve_source_value_falls_back_to_payload_for_direct_key(self):
        """_resolve_source_value: falls back to payload when item and fields are missing."""
        context = {
            "item": {},
            "fields": {},
            "payload": {"headline": "Payload headline"},
            "profile_value": "test_profile",
        }
        result = self.normalizer._resolve_source_value(  # pylint: disable=protected-access
            source_config=build_row_source_config("headline"),
            context=context,
        )
        self.assertEqual(result, "Payload headline")


class TestArticleNormalizerProperties(unittest.TestCase):
    """This class tests the config-backed properties."""

    def test_profiles_returns_new_list_each_time(self):
        """profiles: returns a copy so callers cannot mutate config state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = _build_article_normalizer(Path(tmpdir))
            profiles = normalizer.profiles
            profiles.append("mutated")

            self.assertEqual(normalizer.profiles, ["technology_daily", "science_daily"])

    def test_row_mappings_returns_configured_mapping(self):
        """row_mappings: returns the configured row mappings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            normalizer = _build_article_normalizer(Path(tmpdir))

            self.assertEqual(
                set(normalizer.row_mappings.keys()),
                set(NORMALIZER_ROW_MAPPINGS.keys()),
            )


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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}, "science_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}},
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                latest = normalizer.find_latest_checkpoints_for_date(day)

                self.assertEqual(len(latest), 1)
                self.assertIn("tech_daily", latest)


class TestArticleNormalizerEnsureDay(unittest.TestCase):
    """This class tests _ensure_day."""

    def test_ensure_day_raises_for_datetime_input(self):
        """_ensure_day: raises when the caller passes a datetime instead of a date."""
        with self.assertRaises(TypeError):
            ArticleNormalizer._ensure_day(  # pylint: disable=protected-access
                datetime(2026, 4, 28, 10, 0)
            )


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

    def test_normalize_checkpoint_uses_yaml_row_mapping_override(self):
        """normalize_checkpoint: respects YAML row mapping overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_file = tmppath / "test_checkpoint.json"

            checkpoint_data = {
                "profile": "test_profile",
                "items": [
                    {
                        "id": "article-1",
                        "webTitle": "Wrong Title",
                        "fields": {
                            "headline": "Preferred Title",
                            "firstPublicationDate": "2026-04-28T10:10:00Z",
                        },
                    }
                ],
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f)

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                config = build_normalizer_config(
                    checkpoint_dir=str(tmppath / "checkpoints"),
                    parquet_dir=str(tmppath / "parquet"),
                )
                config["article_normalizer"]["row_mappings"]["web_title"]["sources"] = [
                    "fields.headline"
                ]
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    config,
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                df = normalizer.normalize_checkpoint(checkpoint_file)

                self.assertEqual(df.iloc[0]["web_title"], "Preferred Title")
                self.assertEqual(df.iloc[0]["first_publication_date"].year, 2026)
                self.assertIsInstance(df.iloc[0]["first_publication_date"], datetime)

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
    """This class tests _write_combined_parquet."""

    def test_write_parquet_creates_file(self):
        """_write_combined_parquet: creates parquet file with correct path structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            parquet_dir = tmppath / "parquet"

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech": {}},
                    "article_ingestor": {"parquet_dir": str(parquet_dir)},
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
                day = date(2026, 4, 28)

                with patch.object(df, "to_parquet"):
                    # pylint: disable=protected-access
                    out_path = normalizer._write_combined_parquet(df, day)
                    # pylint: enable=protected-access
                    self.assertIn("2026-04-28.parquet", str(out_path))
                    self.assertIn("2026-04-28", str(out_path))
                    self.assertIn("parquet", str(out_path))


class TestArticleNormalizerNormalizeDay(unittest.TestCase):
    """This class tests normalize_day_to_parquet."""

    def test_normalize_day_to_parquet_writes_combined_output(self):
        """normalize_day_to_parquet: combines same-day checkpoints into one parquet."""
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {
                        "checkpoint_dir": str(checkpoint_dir),
                        "parquet_dir": str(parquet_dir),
                    },
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                with patch("src.ingestion.article_normalizer.pd.DataFrame.to_parquet"):
                    written = normalizer.normalize_day_to_parquet(day)
                    self.assertIn("2026-04-28", written)

    def test_normalize_day_to_parquet_empty_when_no_checkpoints(self):
        """normalize_day_to_parquet: returns empty dict if no checkpoints for day."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {"checkpoint_dir": str(checkpoint_dir)},
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
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

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                    "profiles": {"tech_daily": {}},
                    "article_ingestor": {
                        "checkpoint_dir": str(checkpoint_dir),
                        "parquet_dir": str(tmppath / "parquet"),
                    },
                    "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                day = date(2026, 4, 28)
                written = normalizer.normalize_day_to_parquet(day)

                self.assertEqual(len(written), 0)

    def test_normalize_day_to_parquet_uses_all_same_day_checkpoints(self):
        """normalize_day_to_parquet: includes rows from all same-day checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            checkpoint_dir = tmppath / "checkpoints"
            checkpoint_dir.mkdir()

            first_checkpoint = checkpoint_dir / "tech_daily_20260428T100000Z.json"
            second_checkpoint = checkpoint_dir / "tech_daily_20260428T120000Z.json"

            with first_checkpoint.open("w", encoding="utf-8") as checkpoint_file:
                json.dump(
                    {
                        "profile": "tech_daily",
                        "items": [{"id": "article-1", "fields": {"headline": "First"}}],
                    },
                    checkpoint_file,
                )
            with second_checkpoint.open("w", encoding="utf-8") as checkpoint_file:
                json.dump(
                    {
                        "profile": "tech_daily",
                        "items": [{"id": "article-2", "fields": {"headline": "Second"}}],
                    },
                    checkpoint_file,
                )

            with patch(
                "src.ingestion.article_normalizer.Settings.load_article_normalizer_config"
            ) as mock_load_config:
                mock_load_config.return_value = _build_article_normalizer_config(
                    tmppath,
                    {
                        "profiles": {"tech_daily": {}},
                        "article_ingestor": {
                            "checkpoint_dir": str(checkpoint_dir),
                            "parquet_dir": str(tmppath / "parquet"),
                        },
                        "article_normalizer": {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()},
                    },
                )
                normalizer = ArticleNormalizer(configuration_root=tmppath)

                written = normalizer.normalize_day_to_parquet(date(2026, 4, 28))
                combined_df = pd.read_parquet(Path(written["2026-04-28"]))
                self.assertEqual(list(combined_df["api_id"]), ["article-1", "article-2"])


if __name__ == "__main__":
    unittest.main()
