# pyright: reportPrivateUsage=false
"""Unit tests for settings loading and validation."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.application.settings import Settings


class TestSettingsFromEnv(unittest.TestCase):
    """This class tests from_env."""

    def assert_raises_value_error(self, call):
        """Run a callable and assert it raises ValueError."""
        try:
            call()
        except ValueError:
            return
        self.fail("ValueError was not raised")

    def test_from_env_with_explicit_key_and_yaml_overrides(self):
        """from_env: explicit key can be combined with YAML defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "guardian_client.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        'base_url: "https://example.test"',
                        "default_page_size: 10",
                        "max_page_size: 20",
                        "timeout_seconds: 15",
                    ]
                ),
                encoding="utf-8",
            )
            settings = Settings.load_settings(
                api_key="explicit",
                load_dotenv=False,
                config_path=config_path,
            )

        self.assertEqual(settings.api_key, "explicit")
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
            self.assert_raises_value_error(lambda: Settings.load_settings(load_dotenv=False))

    def test_from_env_validates_page_and_timeout_constraints(self):
        """from_env: rejects invalid page and timeout configuration values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            bad_default = root / "bad_default.yaml"
            bad_default.write_text("default_page_size: 0\n", encoding="utf-8")

            bad_max = root / "bad_max.yaml"
            bad_max.write_text("max_page_size: 0\n", encoding="utf-8")

            bad_order = root / "bad_order.yaml"
            bad_order.write_text("default_page_size: 51\nmax_page_size: 50\n", encoding="utf-8")

            bad_timeout = root / "bad_timeout.yaml"
            bad_timeout.write_text("timeout_seconds: 0\n", encoding="utf-8")

            self.assert_raises_value_error(
                lambda: Settings.load_settings(api_key="k", load_dotenv=False, config_path=bad_default)
            )
            self.assert_raises_value_error(
                lambda: Settings.load_settings(api_key="k", load_dotenv=False, config_path=bad_max)
            )
            self.assert_raises_value_error(
                lambda: Settings.load_settings(api_key="k", load_dotenv=False, config_path=bad_order)
            )
            self.assert_raises_value_error(
                lambda: Settings.load_settings(api_key="k", load_dotenv=False, config_path=bad_timeout)
            )


class TestSettingsLoadEnvFile(unittest.TestCase):
    """This class tests _load_env_file."""

    def test_load_env_file_uses_first_existing_file(self):
        """_load_env_file: loads key/value pairs and ignores comments/invalid lines."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
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

            with patch.dict(os.environ, {}, clear=True):
                getattr(Settings, "_load_env_file")(search_paths=[Path(tmp_dir)])
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "from_file")
                self.assertEqual(os.environ.get("EXTRA"), "value")

    def test_load_env_file_returns_early_if_key_already_set(self):
        """_load_env_file: does nothing when GUARDIAN_API_KEY already exists."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("GUARDIAN_API_KEY=from_file\n", encoding="utf-8")

            with patch.dict(os.environ, {"GUARDIAN_API_KEY": "existing"}, clear=True):
                getattr(Settings, "_load_env_file")(search_paths=[Path(tmp_dir)])
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "existing")

    def test_load_env_file_skips_duplicate_search_paths(self):
        """_load_env_file: ignores duplicate roots after the first pass."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            env_path.write_text("GUARDIAN_API_KEY=from_file\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                getattr(Settings, "_load_env_file")(search_paths=[root, root])
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "from_file")

    def test_load_env_file_duplicate_and_missing_search_paths(self):
        """_load_env_file: gracefully handles duplicate and non-existent paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            missing = root / "missing"
            with patch.dict(os.environ, {}, clear=True):
                getattr(Settings, "_load_env_file")(search_paths=[missing, missing])
                self.assertIsNone(os.environ.get("GUARDIAN_API_KEY"))

    def test_load_env_file_uses_default_search_paths(self):
        """_load_env_file: discovers .env files using the default search path list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            env_path = root / ".env"
            env_path.write_text("GUARDIAN_API_KEY=from_default_paths\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True), patch("src.application.settings.Path.cwd", return_value=root):
                getattr(Settings, "_load_env_file")()
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "from_default_paths")


class TestSettingsLoadIngestionConfig(unittest.TestCase):
    """This class tests _load_ingestion_config."""

    def test_load_ingestion_config_uses_default_path(self):
        """_load_ingestion_config: loads the repository default YAML file."""
        values = getattr(Settings, "_load_ingestion_config")()

        self.assertEqual(values["base_url"], "https://content.guardianapis.com")
        self.assertEqual(values["default_page_size"], 50)
        self.assertEqual(values["max_page_size"], 50)
        self.assertEqual(values["timeout_seconds"], 30)

    def test_load_ingestion_config_parses_supported_values(self):
        """_load_ingestion_config: parses quoted strings, ints, and plain strings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "guardian_client.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "invalid",
                        "base_url: 'https://one.test'",
                        "label: plain",
                        "default_page_size: 12",
                        "empty_key: value",
                        ": ignore",
                    ]
                ),
                encoding="utf-8",
            )

            values = getattr(Settings, "_load_ingestion_config")(config_path=config_path)

        self.assertEqual(values["base_url"], "https://one.test")
        self.assertEqual(values["default_page_size"], 12)
        self.assertEqual(values["label"], "plain")
        self.assertEqual(values["empty_key"], "value")

    def test_load_ingestion_config_returns_empty_for_missing_file(self):
        """_load_ingestion_config: returns empty mapping for absent file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.yaml"
            values = getattr(Settings, "_load_ingestion_config")(config_path=missing)
        self.assertEqual(values, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()