# pyright: reportPrivateUsage=false
"""Unit tests for settings loading and validation."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings
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


class TestSettingsLoadEnvFile(unittest.TestCase):
    """This class tests _load_env_file."""

    def test_load_env_file_reads_repo_root_env(self):
        """_load_env_file: loads key/value pairs from repository-root .env."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_dir = root / "src" / "config"
            src_dir.mkdir(parents=True)
            fake_settings_path = src_dir / "settings.py"
            env_path = root / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "IGNORED_LINE",
                        "GUARDIAN_API_KEY=from_file",
                        "EXTRA='value'",
                    ]
                ),
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("src.config.settings.__file__", str(fake_settings_path)),
            ):
                getattr(Settings, "_load_env_file")()
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "from_file")
                self.assertEqual(os.environ.get("EXTRA"), "value")

    def test_load_env_file_returns_early_if_key_already_set(self):
        """_load_env_file: does nothing when GUARDIAN_API_KEY already exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_dir = root / "src" / "config"
            src_dir.mkdir(parents=True)
            fake_settings_path = src_dir / "settings.py"
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("GUARDIAN_API_KEY=from_file\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {"GUARDIAN_API_KEY": "existing"}, clear=True),
                patch("src.config.settings.__file__", str(fake_settings_path)),
            ):
                getattr(Settings, "_load_env_file")()
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "existing")

    def test_load_env_file_returns_when_root_env_missing(self):
        """_load_env_file: leaves environment unchanged when root .env is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_dir = root / "src" / "config"
            src_dir.mkdir(parents=True)
            fake_settings_path = src_dir / "settings.py"

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("src.config.settings.__file__", str(fake_settings_path)),
            ):
                getattr(Settings, "_load_env_file")()
                self.assertIsNone(os.environ.get("GUARDIAN_API_KEY"))


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
        """load_ingestion_config: calls parser with ingestion defaults."""
        with patch("src.config.settings.YAMLConfigParser") as parser_cls:
            parser_instance = parser_cls.return_value
            parser_instance.parse.return_value = {"base_url": "https://mock"}

            values = Settings.load_ingestion_config()

        parser_cls.assert_called_once_with()
        parser_instance.parse.assert_called_once_with(
            config_type=YAMLConfigType.INGESTION,
            filename="guardian_client.yaml",
        )
        self.assertEqual(values, {"base_url": "https://mock"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
