# pyright: reportPrivateUsage=false
# pylint: disable=too-many-lines
"""Unit tests for settings loading and validation."""

import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.article_row_source_kind import ArticleRowSourceKind
from src.enums.article_row_transform import ArticleRowTransform
from src.enums.guardian_order_by import GuardianOrderBy
from src.enums.pre_chunk_operation import PreChunkOperation
from src.enums.yaml_config_type import YAMLConfigType


class TestSettingsFromEnv(unittest.TestCase):
    """This class tests from_env."""

    def assert_raises_value_error(self, call):
        """Run a callable and assert it raises ValueError."""
        try:
            call()
        except ValueError:
            return
        self.fail("ValueError was not raised")

    def test_from_env_uses_environment_key_and_yaml_overrides(self):
        """from_env: environment key can be combined with YAML defaults."""
        with (
            patch.dict(os.environ, {"GUARDIAN_API_KEY": "from_env"}, clear=True),
            patch.object(
                Settings,
                "load_ingestion_config",
                return_value={
                    "base_url": "https://example.test",
                    "default_page_size": 10,
                    "max_page_size": 20,
                    "timeout_seconds": 15,
                },
            ),
        ):
            settings = Settings.load_settings(load_dotenv=False)

        self.assertEqual(settings.api_key, "from_env")
        self.assertEqual(settings.base_url, "https://example.test")
        self.assertEqual(settings.default_page_size, 10)
        self.assertEqual(settings.max_page_size, 20)
        self.assertEqual(settings.timeout_seconds, 15)

    def test_from_env_reads_key_from_environment(self):
        """from_env: falls back to environment variable when key arg missing."""
        with patch.dict(os.environ, {"GUARDIAN_API_KEY": "from_env"}, clear=True):
            settings = Settings.load_settings(load_dotenv=False)

        self.assertEqual(settings.api_key, "from_env")

    def test_from_env_loads_dotenv_when_enabled(self):
        """from_env: calls _load_env_file when load_dotenv is enabled."""
        with patch.dict(os.environ, {"GUARDIAN_API_KEY": "from_env"}, clear=True), patch.object(
            Settings,
            "_load_env_file",
        ) as mock_load_env_file:
            settings = Settings.load_settings(load_dotenv=True)

        self.assertEqual(settings.api_key, "from_env")
        mock_load_env_file.assert_called_once()

    def test_from_env_raises_when_key_missing(self):
        """from_env: raises when no explicit or environment key is present."""
        with patch.dict(os.environ, {}, clear=True):
            self.assert_raises_value_error(
                lambda: Settings.load_settings(load_dotenv=False)
            )

    def test_from_env_validates_page_and_timeout_constraints(self):
        """from_env: rejects invalid page and timeout configuration values."""
        invalid_configs = [
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 0,
                "max_page_size": 50,
                "timeout_seconds": 30,
            },
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 50,
                "max_page_size": 0,
                "timeout_seconds": 30,
            },
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 51,
                "max_page_size": 50,
                "timeout_seconds": 30,
            },
            {
                "base_url": "https://content.guardianapis.com",
                "default_page_size": 50,
                "max_page_size": 50,
                "timeout_seconds": 0,
            },
        ]

        with patch.dict(os.environ, {"GUARDIAN_API_KEY": "k"}, clear=True):
            for config_values in invalid_configs:
                with self.subTest(config_values=config_values), patch.object(
                    Settings, "load_ingestion_config", return_value=config_values
                ):
                    self.assert_raises_value_error(
                        lambda: Settings.load_settings(load_dotenv=False)
                    )


class TestSettingsLoadIngestionConfig(unittest.TestCase):
    """This class tests load_ingestion_config."""

    def test_load_ingestion_config_uses_default_path(self):
        """load_ingestion_config: loads the repository default YAML file."""
        values = Settings.load_ingestion_config()

        self.assertEqual(values["base_url"], "https://content.guardianapis.com")
        self.assertEqual(values["default_page_size"], 50)
        self.assertEqual(values["max_page_size"], 50)
        self.assertEqual(values["timeout_seconds"], 30)
        self.assertIn("profiles", values)

    def test_load_ingestion_config_calls_parser_with_defaults(self):
        """load_ingestion_config: loads split ingestion files and optional legacy overrides."""
        with patch("src.config.settings.YAMLConfigParser") as parser_cls:
            parser_instance = parser_cls.return_value
            parser_instance.parse.side_effect = [
                {"base_url": "https://mock"},
                {"profiles": {"daily": {"topic": "x"}}},
                {"article_ingestor": {"limit_per_profile": 5}},
                {
                    "article_normalizer": {
                        "row_mappings": {
                            "headline": {"sources": ["fields.headline"]}
                        }
                    }
                },
                {
                    "pre_chunk_preprocessor": {
                        "operations": [
                            {"name": "drop_columns", "args": {"columns": ["thumbnail"]}}
                        ]
                    }
                },
                {"article_ingestor": {"limit_per_profile": 7}},
            ]

            values = Settings.load_ingestion_config()

        parser_cls.assert_called_once_with()
        self.assertEqual(parser_instance.parse.call_count, 6)
        self.assertEqual(
            parser_instance.parse.call_args_list[0].kwargs,
            {"config_type": YAMLConfigType.INGESTION, "filename": "base.yaml"},
        )
        self.assertEqual(
            parser_instance.parse.call_args_list[-1].kwargs,
            {"config_type": YAMLConfigType.INGESTION, "filename": "ingestion_config.yaml"},
        )
        self.assertEqual(values["base_url"], "https://mock")
        self.assertEqual(values["article_ingestor"]["limit_per_profile"], 7)

    def test_load_ingestion_config_from_root_uses_custom_root(self):
        """load_ingestion_config_from_root: passes custom root and merges split config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("src.config.settings.YAMLConfigParser") as parser_cls:
                parser_instance = parser_cls.return_value
                parser_instance.parse.side_effect = [
                    {"base_url": "https://root.test"},
                    {},
                    {},
                    {},
                    {},
                    {},
                ]

                values = Settings.load_ingestion_config_from_root(configuration_root=root)

        parser_cls.assert_called_once_with(configuration_root=root)
        self.assertEqual(parser_instance.parse.call_count, 6)
        self.assertEqual(values, {"base_url": "https://root.test"})

    def test_load_article_ingestor_config_returns_typed_config(self):
        """load_article_ingestor_config: returns validated typed ingest settings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                },
                "article_ingestor": {
                    "profiles_to_run": ["science_daily"],
                    "limit_per_profile": 3,
                    "save_local_checkpoint": True,
                    "checkpoint_dir": "checkpoints/custom",
                    "enable_usage_logging": True,
                    "logs_dir": "custom_logs",
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_article_ingestor_config(configuration_root=root)

        self.assertEqual(typed_config.profile_names, ["technology_daily", "science_daily"])
        self.assertEqual(typed_config.profiles_to_run, ["science_daily"])
        self.assertEqual(typed_config.limit_per_profile, 3)
        self.assertTrue(typed_config.save_local_checkpoint)
        self.assertEqual(typed_config.checkpoint_dir, Path("checkpoints/custom"))
        self.assertTrue(typed_config.enable_usage_logging)
        self.assertEqual(typed_config.logs_dir, Path("custom_logs"))

    def test_load_article_ingestor_config_uses_default_usage_logging_values(self):
        """load_article_ingestor_config: uses usage logging defaults when fields are absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                typed_config = Settings.load_article_ingestor_config(configuration_root=root)
        self.assertFalse(typed_config.enable_usage_logging)
        self.assertEqual(typed_config.logs_dir, Path("logs"))

    def test_load_article_ingestor_config_raises_for_invalid_profiles_to_run(self):
        """load_article_ingestor_config: raises for malformed profiles_to_run values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"profiles_to_run": "technology_daily"},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_ingestor_config(configuration_root=root)

    def test_load_article_ingestor_config_raises_for_invalid_limit(self):
        """load_article_ingestor_config: raises when limit_per_profile is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"limit_per_profile": 0},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_ingestor_config(configuration_root=root)

    def test_load_article_ingestor_config_raises_for_invalid_save_local_checkpoint(self):
        """load_article_ingestor_config: raises when save_local_checkpoint is not boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"save_local_checkpoint": "yes"},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_ingestor_config(configuration_root=root)

    def test_load_article_ingestor_config_raises_for_invalid_enable_usage_logging(self):
        """load_article_ingestor_config: raises when enable_usage_logging is not boolean."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"enable_usage_logging": "yes"},
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_article_ingestor_config(configuration_root=root)


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
                "profiles": {
                    "technology_daily": {
                        "topic": "technology",
                    }
                },
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
                    "row_mappings": {
                        "headline": {"sources": ["fields.headline"]}
                    }
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
                "article_normalizer": {
                    "row_mappings": {
                        "profile": {"sources": ["profile"]}
                    }
                },
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
                    "operations": [
                        {"name": "drop_columns", "args": {"columns": []}}
                    ]
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
                "pre_chunk_preprocessor": {
                    "operations": ["drop_columns"],
                },
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
                    "operations": [
                        {"name": "drop_columns", "args": ["thumbnail"]},
                    ],
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
                "pre_chunk_preprocessor": {
                    "operations": [{"name": "drop_columns"}],
                },
            }
            with patch(
                "src.config.settings.Settings.load_ingestion_config_from_root",
                return_value=raw_config,
            ):
                with self.assertRaises(ValueError):
                    Settings.load_pre_chunk_preprocessor_config(configuration_root=root)

    def test_load_non_empty_string_rejects_blank_values(self):
        """_load_non_empty_string: raises for blank string values."""
        with self.assertRaises(ValueError):
            Settings._load_non_empty_string("   ", field_name="example")  # pylint: disable=protected-access

    def test_load_non_empty_string_mapping_returns_valid_mapping(self):
        """_load_non_empty_string_mapping: returns parsed string mapping values."""
        resolved = Settings._load_non_empty_string_mapping(  # pylint: disable=protected-access
            {"old_name": "new_name"},
            field_name="example",
        )
        self.assertEqual(resolved, {"old_name": "new_name"})

    def test_load_non_empty_string_mapping_rejects_empty_mapping(self):
        """_load_non_empty_string_mapping: raises for empty mapping values."""
        with self.assertRaises(ValueError):
            Settings._load_non_empty_string_mapping(  # pylint: disable=protected-access
                {},
                field_name="example",
            )

    def test_load_article_normalizer_config_raises_for_non_mapping_row_mapping(self):
        """load_article_normalizer_config: raises when one row mapping is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {},
                "article_normalizer": {
                    "row_mappings": {
                        "headline": "fields.headline"
                    }
                },
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
                "article_normalizer": {
                    "row_mappings": {
                        "headline": {"sources": [1]}
                    }
                },
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
                    "row_mappings": {
                        "headline": {"sources": ["feilds.headline"]}
                    }
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
                "article_normalizer": {
                    "row_mappings": {
                        "headline": {"sources": ["fields."]}
                    }
                },
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
