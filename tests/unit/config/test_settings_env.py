# pyright: reportPrivateUsage=false
"""Unit tests for Settings env-file loading helpers."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.settings import Settings


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

    def test_load_env_file_does_not_override_existing_key(self):
        """_load_env_file: preserves existing env values while loading other keys."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_dir = root / "src" / "config"
            src_dir.mkdir(parents=True)
            fake_settings_path = src_dir / "settings.py"
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "GUARDIAN_API_KEY=from_file\nSUPABASE_URL=https://from-file.supabase.co\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {"GUARDIAN_API_KEY": "existing"}, clear=True),
                patch("src.config.settings.__file__", str(fake_settings_path)),
            ):
                getattr(Settings, "_load_env_file")()
                self.assertEqual(os.environ.get("GUARDIAN_API_KEY"), "existing")
                self.assertEqual(
                    os.environ.get("SUPABASE_URL"),
                    "https://from-file.supabase.co",
                )

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

    def test_load_repository_dotenv_delegates_to_load_env_file(self):
        """load_repository_dotenv: delegates to _load_env_file."""
        with patch.object(Settings, "_load_env_file") as mocked:
            Settings.load_repository_dotenv()
        mocked.assert_called_once()


class TestLoadSupabaseCredentials(unittest.TestCase):
    """This class tests load_supabase_credentials."""

    def test_load_supabase_credentials_returns_trimmed_pair(self):
        """load_supabase_credentials: returns stripped URL + service key."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": " https://x.supabase.co ",
                "SUPABASE_SERVICE_ROLE_KEY": " key ",
            },
            clear=True,
        ):
            url, key = Settings.load_supabase_credentials(load_dotenv=False)
        self.assertEqual(url, "https://x.supabase.co")
        self.assertEqual(key, "key")

    def test_load_supabase_credentials_raises_when_url_missing(self):
        """load_supabase_credentials: requires SUPABASE_URL."""
        with patch.dict(
            os.environ,
            {"SUPABASE_SERVICE_ROLE_KEY": "key"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as raised:
                Settings.load_supabase_credentials(load_dotenv=False)
        self.assertIn("SUPABASE_URL", str(raised.exception))

    def test_load_supabase_credentials_raises_when_service_key_missing(self):
        """load_supabase_credentials: requires SUPABASE_SERVICE_ROLE_KEY."""
        with patch.dict(
            os.environ,
            {"SUPABASE_URL": "https://x.supabase.co"},
            clear=True,
        ):
            with self.assertRaises(ValueError) as raised:
                Settings.load_supabase_credentials(load_dotenv=False)
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY", str(raised.exception))

    def test_load_supabase_credentials_raises_when_service_key_empty(self):
        """load_supabase_credentials: requires non-empty SUPABASE_SERVICE_ROLE_KEY."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://x.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "  ",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError) as raised:
                Settings.load_supabase_credentials(load_dotenv=False)
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY", str(raised.exception))

    def test_load_supabase_credentials_raises_when_url_empty(self):
        """load_supabase_credentials: requires non-empty SUPABASE_URL."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "  ",
                "SUPABASE_SERVICE_ROLE_KEY": "key",
            },
            clear=True,
        ):
            with self.assertRaises(ValueError) as raised:
                Settings.load_supabase_credentials(load_dotenv=False)
        self.assertIn("SUPABASE_URL", str(raised.exception))

    def test_load_supabase_credentials_with_dotenv_loading(self):
        """load_supabase_credentials: loads .env file when load_dotenv=True."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            src_dir = root / "src" / "config"
            src_dir.mkdir(parents=True)
            fake_settings_path = src_dir / "settings.py"
            env_path = root / ".env"
            env_path.write_text(
                "SUPABASE_URL=https://x.supabase.co\n"
                "SUPABASE_SERVICE_ROLE_KEY=key_from_file\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("src.config.settings.__file__", str(fake_settings_path)),
            ):
                url, key = Settings.load_supabase_credentials(load_dotenv=True)
            self.assertEqual(url, "https://x.supabase.co")
            self.assertEqual(key, "key_from_file")


class TestLoadOptionalPostgresConninfo(unittest.TestCase):
    """This class tests load_optional_postgres_conninfo."""

    def test_returns_supabase_postgres_url_first(self) -> None:
        """load_optional_postgres_conninfo: prefers SUPABASE_POSTGRES_URL over DATABASE_URL."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_POSTGRES_URL": "postgresql://first/db",
                "DATABASE_URL": "postgresql://second/db",
            },
            clear=True,
        ):
            uri = Settings.load_optional_postgres_conninfo(load_dotenv=False)
        self.assertEqual(uri, "postgresql://first/db")

    def test_returns_database_url_when_postgres_url_missing(self) -> None:
        """load_optional_postgres_conninfo: returns None when SUPABASE_POSTGRES_URL missing."""
        with patch.dict(os.environ, {"DATABASE_URL": " postgresql://pool/db "}, clear=True):
            uri = Settings.load_optional_postgres_conninfo(load_dotenv=False)
        self.assertIsNone(uri)

    def test_returns_none_when_unset(self) -> None:
        """load_optional_postgres_conninfo: returns None when neither URI is set."""
        with patch.dict(os.environ, {}, clear=True):
            uri = Settings.load_optional_postgres_conninfo(load_dotenv=False)
        self.assertIsNone(uri)

    def test_load_dotenv_merges_env_file(self) -> None:
        """load_optional_postgres_conninfo: merges repository .env when load_dotenv is True."""
        with patch.object(Settings, "_load_env_file") as mock_load:
            with patch.object(
                Settings,
                "_merge_dotenv_values_for_keys_always",
            ) as mock_merge:
                with patch.dict(
                    os.environ,
                    {"SUPABASE_POSTGRES_URL": "postgresql://x"},
                    clear=True,
                ):
                    Settings.load_optional_postgres_conninfo(load_dotenv=True)
        mock_load.assert_called_once()
        mock_merge.assert_called_once_with(("SUPABASE_POSTGRES_URL",))

    def test_dotenv_database_url_overrides_empty_shell_value(self) -> None:
        """load_optional_postgres_conninfo: DATABASE_URL is ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "DATABASE_URL=postgresql://from-dotenv/db\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {"DATABASE_URL": "", "SUPABASE_POSTGRES_URL": ""},
                    clear=True,
                ),
                patch.object(
                    Settings,
                    "_repository_root_dotenv_path",
                    return_value=env_path,
                ),
            ):
                uri = Settings.load_optional_postgres_conninfo(load_dotenv=True)
        self.assertIsNone(uri)

    def test_merge_dotenv_skips_comments_blank_and_irrelevant_keys(self) -> None:
        """_merge_dotenv_values_for_keys_always: ignores comments, blanks, and other keys."""
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "\n# comment\nIGNORE\nDATABASE_URL=postgresql://only/db\n  \nOTHER=x\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    Settings,
                    "_repository_root_dotenv_path",
                    return_value=env_path,
                ),
            ):
                Settings._merge_dotenv_values_for_keys_always(("DATABASE_URL",))
                self.assertEqual(os.environ.get("DATABASE_URL"), "postgresql://only/db")
                self.assertIsNone(os.environ.get("OTHER"))

    def test_merge_dotenv_returns_when_env_file_missing(self) -> None:
        """_merge_dotenv_values_for_keys_always: no-op when .env path does not exist."""
        # pylint: disable=protected-access
        missing = Path("/nonexistent/path/that/should/not/exist/.env")
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(
                Settings,
                "_repository_root_dotenv_path",
                return_value=missing,
            ):
                Settings._merge_dotenv_values_for_keys_always(("DATABASE_URL",))
                self.assertIsNone(os.environ.get("DATABASE_URL"))

    def test_merge_skips_empty_value_for_wanted_key(self) -> None:
        """_merge_dotenv_values_for_keys_always: skips wanted key with empty value."""
        # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("DATABASE_URL=\n", encoding="utf-8")
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(
                    Settings,
                    "_repository_root_dotenv_path",
                    return_value=env_path,
                ),
            ):
                Settings._merge_dotenv_values_for_keys_always(("DATABASE_URL",))
                self.assertIsNone(os.environ.get("DATABASE_URL"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
