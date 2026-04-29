# pylint: disable=duplicate-code
"""Unit tests for Guardian profile config parsing."""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.guardian_order_by import GuardianOrderBy


class TestSettingsLoadGuardianProfileConfigs(unittest.TestCase):
    """This class tests load_guardian_profile_configs."""

    def test_load_guardian_profile_configs_returns_typed_profiles(self):
        """load_guardian_profile_configs: parses typed Guardian profile config objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "run_date": "2026-04-28",
                        "page_size": 3,
                        "query": "chips",
                        "section": "technology",
                        "order_by": "oldest",
                        "use_next_fallback": False,
                        "content_show_fields": "headline,bodyText",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_profiles = Settings.load_guardian_profile_configs(configuration_root=root)

        profile = typed_profiles["technology_daily"]
        self.assertEqual(profile.topic, "technology")
        self.assertEqual(profile.page_size, 3)
        self.assertEqual(profile.order_by, GuardianOrderBy.OLDEST)
        self.assertEqual(profile.section, "technology")
        self.assertFalse(profile.use_next_fallback)
        self.assertEqual(profile.content_show_fields, "headline,bodyText")

    def test_load_guardian_profile_configs_parses_profile_timeframe_override(self):
        """load_guardian_profile_configs: parses per-profile timeframe overrides."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "timeframe": {
                            "mode": "explicit",
                            "from_date": "2026-04-20",
                            "to_date": "2026-04-28",
                        },
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_profiles = Settings.load_guardian_profile_configs(configuration_root=root)
        profile = typed_profiles["technology_daily"]
        self.assertEqual(profile.from_date, date(2026, 4, 20))
        self.assertEqual(profile.to_date, date(2026, 4, 28))

    def test_load_guardian_profile_configs_rejects_mixed_timeframe_fields(self):
        """load_guardian_profile_configs: rejects mixed timeframe and date fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "run_date": "2026-04-28",
                        "timeframe": {"mode": "relative", "relative": "past_week"},
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_defaults_optional_fields(self):
        """load_guardian_profile_configs: defaults optional profile fields when they are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": {"topic": "technology"}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_profiles = Settings.load_guardian_profile_configs(configuration_root=root)

        self.assertIsNone(typed_profiles["technology_daily"].query)
        self.assertIsNone(typed_profiles["technology_daily"].section)
        self.assertEqual(typed_profiles["technology_daily"].content_show_fields, "all")

    def test_load_guardian_profile_configs_raises_for_missing_profiles(self):
        """load_guardian_profile_configs: raises when profiles is empty or missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_non_mapping_profile(self):
        """load_guardian_profile_configs: raises when a profile value is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": "invalid"},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_allows_missing_topic(self):
        """load_guardian_profile_configs: allows global ingestion profiles without topic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": {"page_size": 3}},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_profiles = Settings.load_guardian_profile_configs(configuration_root=root)
        self.assertEqual(typed_profiles["technology_daily"].topic, "")

    def test_load_guardian_profile_configs_raises_for_invalid_order_by(self):
        """load_guardian_profile_configs: raises when order_by is unsupported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "order_by": "sideways",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_invalid_page_size(self):
        """load_guardian_profile_configs: raises when page_size falls outside configured bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "page_size": 0,
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_unknown_keys(self):
        """load_guardian_profile_configs: raises when unsupported profile keys are present."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "extra_filters": {"section": "technology"},
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_invalid_string_fields(self):
        """load_guardian_profile_configs: raises for invalid typed string fields."""
        invalid_profiles = [
            {"topic": 123},
            {"topic": "technology", "query": 123},
            {"topic": "technology", "content_show_fields": 5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for raw_profile in invalid_profiles:
                raw_config = {
                    "base_url": "https://content.guardianapis.com",
                    "default_page_size": 10,
                    "max_page_size": 50,
                    "timeout_seconds": 30,
                    "profiles": {"technology_daily": raw_profile},
                }
                with self.subTest(raw_profile=raw_profile), patch(
                    "src.config.settings.Settings.load_ingestion_config_from_root",
                    return_value=raw_config,
                ):
                    with self.assertRaises(ValueError):
                        Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_allows_empty_section(self):
        """load_guardian_profile_configs: allows empty section strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "section": "",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_profiles = Settings.load_guardian_profile_configs(configuration_root=root)
        self.assertEqual(typed_profiles["technology_daily"].section, "")

    def test_load_guardian_profile_configs_raises_for_invalid_page_size_settings(self):
        """load_guardian_profile_configs: raises when global page-size settings are invalid."""
        invalid_configs = [
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 0,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": {"topic": "technology"}},
            },
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 0,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": {"topic": "technology"}},
            },
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 51,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {"technology_daily": {"topic": "technology"}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for raw_config in invalid_configs:
                with self.subTest(raw_config=raw_config), patch(
                    "src.config.settings.Settings.load_ingestion_config_from_root",
                    return_value=raw_config,
                ):
                    with self.assertRaises(ValueError):
                        Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_invalid_use_next_fallback(self):
        """load_guardian_profile_configs: raises when use_next_fallback is not boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "use_next_fallback": "yes",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_timeframe_type(self):
        """load_guardian_profile_configs: raises when timeframe is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "timeframe": "past_week",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_when_run_date_with_from_to(self):
        """load_guardian_profile_configs: rejects run_date mixed with from/to dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "run_date": "2026-04-28",
                        "from_date": "2026-04-20",
                        "to_date": "2026-04-28",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_partial_from_to_date(self):
        """load_guardian_profile_configs: requires both from_date and to_date together."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "from_date": "2026-04-20",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)

    def test_load_guardian_profile_configs_raises_for_invalid_from_to_order(self):
        """load_guardian_profile_configs: rejects from_date later than to_date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 10,
                "max_page_size": 50,
                "timeout_seconds": 30,
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                        "from_date": "2026-04-29",
                        "to_date": "2026-04-28",
                    }
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_guardian_profile_configs(configuration_root=root)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
