"""Unit tests for Settings.load_briefing_persistence_config."""

import tempfile
import unittest
from pathlib import Path

from src.config.settings import BriefingPersistenceConfig, Settings


def _write_persistence_yaml(
    config_root: Path,
    content: str,
    filename: str = "briefing_persistence.yaml",
) -> Path:
    """Write briefing_persistence YAML under a temporary configuration root."""
    section_dir = config_root / "process"
    section_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = section_dir / filename
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


_VALID = """\
briefing_persistence:
  table_name: news_briefings
  schema_name: public
  ensure_table: true
"""


class TestLoadBriefingPersistenceConfig(unittest.TestCase):
    """This class tests load_briefing_persistence_config."""

    def test_load_returns_typed_config(self) -> None:
        """load_briefing_persistence_config: returns typed config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_persistence_yaml(root, _VALID)
            cfg = Settings.load_briefing_persistence_config(configuration_root=root)

        self.assertIsInstance(cfg, BriefingPersistenceConfig)
        self.assertEqual(cfg.table_name, "news_briefings")
        self.assertEqual(cfg.schema_name, "public")
        self.assertTrue(cfg.ensure_table)

    def test_defaults_schema_and_ensure(self) -> None:
        """load_briefing_persistence_config: defaults schema_name and ensure_table."""
        yaml_text = """\
briefing_persistence:
  table_name: t1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_persistence_yaml(root, yaml_text)
            cfg = Settings.load_briefing_persistence_config(configuration_root=root)

        self.assertEqual(cfg.schema_name, "public")
        self.assertTrue(cfg.ensure_table)

    def test_ensure_table_false(self) -> None:
        """load_briefing_persistence_config: parses ensure_table false."""
        yaml_text = """\
briefing_persistence:
  table_name: t1
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_persistence_yaml(root, yaml_text)
            cfg = Settings.load_briefing_persistence_config(configuration_root=root)

        self.assertFalse(cfg.ensure_table)

    def test_rejects_invalid_table_identifier(self) -> None:
        """load_briefing_persistence_config: rejects invalid table_name characters."""
        yaml_text = """\
briefing_persistence:
  table_name: bad-name
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_persistence_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_persistence_config(configuration_root=root)
        self.assertIn("table_name", str(ctx.exception))

    def test_rejects_non_boolean_ensure_table(self) -> None:
        """load_briefing_persistence_config: ensure_table must be boolean."""
        yaml_text = """\
briefing_persistence:
  table_name: ok
  ensure_table: "yes"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_persistence_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_persistence_config(configuration_root=root)
        self.assertIn("ensure_table", str(ctx.exception))
