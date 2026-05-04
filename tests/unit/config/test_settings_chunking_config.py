# pylint: disable=duplicate-code
"""Unit tests for chunking config parsing."""

import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.chunking_strategy import ChunkingStrategy


def _full_chunking_config(
    *,
    params_overrides: Dict[str, Any] | None = None,
    profile_overrides: Dict[str, Any] | None = None,
    section_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a minimal ingestion config dict including a valid chunking section."""
    params: Dict[str, Any] = {
        "min_chars": 1,
        "max_chars": 50,
        "overlap_chars": 0,
        "similarity_threshold": 0.5,
    }
    if params_overrides is not None:
        params.update(params_overrides)
    profile: Dict[str, Any] = {
        "strategy": "semantic_sentence",
        "params": params,
    }
    if profile_overrides is not None:
        profile.update(profile_overrides)
    chunking_section: Dict[str, Any] = {
        "input_dir": "in/parquet",
        "output_dir": "out/chunked",
        "text_columns": ["body_text"],
        "id_columns": ["api_id"],
        "profile_columns": ["profile"],
        "passthrough_columns": ["headline"],
        "profiles": {"default": profile},
    }
    if section_overrides is not None:
        chunking_section.update(section_overrides)
    return {
        "profiles": {"technology_daily": {"topic": "technology"}},
        "chunking": chunking_section,
    }


class _ChunkingConfigTestBase(unittest.TestCase):
    """Shared helpers for chunking config parsing tests."""

    def _patch_load(self, raw_config: Dict[str, Any]):
        """Return a patch context that returns ``raw_config`` from ingestion loaders."""
        return patch(
            "src.config.settings.Settings.load_ingestion_config_from_root",
            return_value=raw_config,
        )


class TestSettingsLoadChunkingConfig(_ChunkingConfigTestBase):
    """This class tests load_chunking_config."""

    def test_load_chunking_config_returns_typed_config(self) -> None:
        """load_chunking_config: returns typed paths, output_dir, and profile map."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config(
                params_overrides={
                    "min_chars": 50,
                    "max_chars": 400,
                    "overlap_chars": 10,
                    "similarity_threshold": 0.4,
                    "sentence_splitter": "simple_regex",
                },
            )
            with self._patch_load(raw_config):
                typed = Settings.load_chunking_config(configuration_root=root)

        self.assertEqual(typed.profile_names, ["technology_daily"])
        self.assertEqual(typed.input_dir, Path("in/parquet"))
        self.assertEqual(typed.output_dir, Path("out/chunked"))
        self.assertEqual(typed.text_columns, ["body_text"])
        self.assertEqual(typed.id_columns, ["api_id"])
        default_profile = typed.chunking_profiles["default"]
        self.assertEqual(default_profile.strategy, ChunkingStrategy.SEMANTIC_SENTENCE)
        self.assertEqual(default_profile.params["min_chars"], 50)
        self.assertEqual(default_profile.params["max_chars"], 400)
        self.assertEqual(default_profile.params["overlap_chars"], 10)
        self.assertEqual(default_profile.params["similarity_threshold"], 0.4)
        self.assertEqual(default_profile.params["sentence_splitter"], "simple_regex")

    def test_load_chunking_config_defaults_output_dir(self) -> None:
        """load_chunking_config: defaults output_dir to checkpoints/chunked_parquet."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            del raw_config["chunking"]["output_dir"]
            with self._patch_load(raw_config):
                typed = Settings.load_chunking_config(configuration_root=root)

        self.assertEqual(typed.output_dir, Path("checkpoints/chunked_parquet"))

    def test_load_chunking_config_allows_optional_id_and_profile_lists(self) -> None:
        """load_chunking_config: treats missing id/profile lists as empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            for key in ("id_columns", "profile_columns", "passthrough_columns"):
                del raw_config["chunking"][key]
            with self._patch_load(raw_config):
                typed = Settings.load_chunking_config(configuration_root=root)

        self.assertEqual(typed.id_columns, [])
        self.assertEqual(typed.profile_columns, [])
        self.assertEqual(typed.passthrough_columns, [])


class TestSettingsLoadChunkingConfigValidation(_ChunkingConfigTestBase):
    """This class tests load_chunking_config validation paths."""

    def test_load_chunking_config_raises_when_chunking_missing(self) -> None:
        """load_chunking_config: raises when chunking section is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = {"profiles": {"p": {"topic": "t"}}}
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking", str(ctx.exception))

    def test_load_chunking_config_raises_when_profiles_missing(self) -> None:
        """load_chunking_config: raises when chunking.profiles is missing or empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["profiles"] = {}
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profiles", str(ctx.exception))

    def test_load_chunking_config_raises_when_profile_not_mapping(self) -> None:
        """load_chunking_config: raises when a profile entry is not a mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["profiles"]["default"] = "bad"
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profiles.default", str(ctx.exception))

    def test_load_chunking_config_raises_when_strategy_invalid(self) -> None:
        """load_chunking_config: raises when a profile strategy is unsupported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config(profile_overrides={"strategy": "bad"})
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profiles.default.strategy", str(ctx.exception))

    def test_load_chunking_config_raises_when_params_missing(self) -> None:
        """load_chunking_config: raises when a profile's params mapping is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["profiles"]["default"].pop("params")
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profiles.default.params", str(ctx.exception))

    def test_load_chunking_config_raises_when_params_empty(self) -> None:
        """load_chunking_config: raises when a profile's params mapping is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["profiles"]["default"]["params"] = {}
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profiles.default.params", str(ctx.exception))

    def test_load_chunking_config_raises_when_optional_list_not_list(self) -> None:
        """load_chunking_config: raises when optional list fields are malformed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["id_columns"] = "api_id"
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.id_columns", str(ctx.exception))

    def test_load_chunking_config_raises_when_optional_list_item_blank(self) -> None:
        """load_chunking_config: raises when optional list contains blank item."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_config = _full_chunking_config()
            raw_config["chunking"]["profile_columns"] = [""]
            with self._patch_load(raw_config):
                with self.assertRaises(ValueError) as ctx:
                    Settings.load_chunking_config(configuration_root=root)
        self.assertIn("chunking.profile_columns", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
