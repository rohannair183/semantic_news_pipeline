"""This class tests `_parse_orchestrator_config_mapping` and related parsers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.settings import Settings
from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.orchestrator_task_kind import OrchestratorTaskKind
from src.enums.yaml_config_type import YAMLConfigType


class TestOrchestratorConfigParsing(unittest.TestCase):
    """This class tests `_parse_orchestrator_config_mapping`"""

    def test_parse_minimal_valid_config(self):
        """_parse_orchestrator_config_mapping: builds tasks with defaults."""
        raw = {
            "fail_fast": False,
            "tasks": [
                {"kind": "article_ingestor"},
            ],
        }
        config = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertFalse(config.fail_fast)
        self.assertEqual(len(config.tasks), 1)
        task = config.tasks[0]
        self.assertEqual(task.kind, OrchestratorTaskKind.ARTICLE_INGESTOR)
        self.assertEqual(task.task_id, "article_ingestor")
        self.assertTrue(task.enabled)
        self.assertIsNone(task.skip_when)

    def test_parse_rejects_empty_mapping(self):
        """_parse_orchestrator_config_mapping: raises on empty YAML."""
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping({})  # pylint: disable=protected-access
        self.assertIn("non-empty", str(raised.exception))

    def test_parse_rejects_non_bool_fail_fast(self):
        """_parse_orchestrator_config_mapping: fail_fast must be boolean."""
        raw = {"fail_fast": "yes", "tasks": [{"kind": "chunking"}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("fail_fast", str(raised.exception))

    def test_parse_rejects_empty_tasks(self):
        """_parse_orchestrator_config_mapping: tasks list must be non-empty."""
        raw = {"tasks": []}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("tasks", str(raised.exception))

    def test_parse_skip_when_missing_env_var(self):
        """_parse_orchestrator_task_spec: parses missing_env_var guard."""
        raw = {
            "tasks": [
                {
                    "id": "ingest",
                    "kind": "article_ingestor",
                    "skip_when": {"missing_env_var": "GUARDIAN_API_KEY"},
                }
            ]
        }
        config = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIsNotNone(config.tasks[0].skip_when)
        assert config.tasks[0].skip_when is not None
        self.assertEqual(config.tasks[0].skip_when.missing_env_var, "GUARDIAN_API_KEY")

    def test_parse_skip_when_missing_env_vars(self):
        """_parse_orchestrator_task_spec: parses missing_env_vars guard."""
        raw = {
            "tasks": [
                {
                    "id": "persist",
                    "kind": "briefing_persistence",
                    "skip_when": {
                        "missing_env_vars": [
                            "SUPABASE_URL",
                            "SUPABASE_SERVICE_ROLE_KEY",
                            "SUPABASE_POSTGRES_URL",
                        ]
                    },
                }
            ]
        }
        config = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIsNotNone(config.tasks[0].skip_when)
        assert config.tasks[0].skip_when is not None
        self.assertEqual(
            config.tasks[0].skip_when.missing_env_vars,
            (
                "SUPABASE_URL",
                "SUPABASE_SERVICE_ROLE_KEY",
                "SUPABASE_POSTGRES_URL",
            ),
        )

    def test_parse_normalizer_day_validates_iso(self):
        """_parse_orchestrator_task_params: validates article_normalizer day."""
        raw = {
            "tasks": [
                {
                    "kind": "article_normalizer",
                    "params": {"day": "not-a-date"},
                }
            ]
        }
        with self.assertRaises(ValueError):
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access

    def test_chunking_unknown_param_key(self):
        """_parse_orchestrator_task_params: rejects stray keys on chunking."""
        raw = {
            "tasks": [
                {"kind": "chunking", "params": {"profile": "p", "extra": True}},
            ]
        }
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("unknown keys", str(raised.exception))


class TestLoadOrchestratorConfigFromPath(unittest.TestCase):
    """This class tests `load_orchestrator_config_from_path`"""

    def test_load_from_path_reads_file(self):
        """load_orchestrator_config_from_path: parses file via YAML parser."""
        content = """\
fail_fast: true
tasks:
  - kind: chunking
    params:
      profile: custom
"""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(content)
            path = Path(tmp.name)
        self.addCleanup(path.unlink)
        config = Settings.load_orchestrator_config_from_path(path)
        self.assertEqual(config.tasks[0].params.profile, "custom")

    def test_load_orchestrator_config_orchestration_dir(self):
        """load_orchestrator_config: reads default orchestration YAML when present."""
        parser = YAMLConfigParser()
        path = parser.get_config_path(YAMLConfigType.ORCHESTRATION, "orchestrator.yaml")
        if not path.is_file():
            self.skipTest("repository orchestrator.yaml not present")
        config = Settings.load_orchestrator_config()
        self.assertGreaterEqual(len(config.tasks), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
