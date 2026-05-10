"""This class tests `_parse_orchestrator_task_spec` validation errors."""

from __future__ import annotations

import unittest

from src.config.settings import Settings
from src.enums.orchestrator_normalizer_day_token import OrchestratorNormalizerDayToken


class TestOrchestratorConfigValidationErrors(unittest.TestCase):
    """This class covers orchestrator YAML validation failure paths."""

    def test_tasks_must_be_list(self):
        """_parse_orchestrator_config_mapping: tasks rejects non-list types."""
        raw = {"tasks": {}}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("tasks", str(raised.exception))

    def test_task_entry_must_be_mapping(self):
        """_parse_orchestrator_task_spec: rejects non-mapping task entries."""
        raw = {"tasks": [[]]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("tasks[0]", str(raised.exception))

    def test_task_requires_kind(self):
        """_parse_orchestrator_task_spec: requires declarative kinds."""
        raw = {"tasks": [{}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("kind", str(raised.exception))

    def test_unknown_kind_reports_prefix(self):
        """_parse_orchestrator_task_spec: wraps orchestrator.kind errors."""
        raw = {"tasks": [{"kind": "unknown_runner"}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("OrchestratorTaskKind", str(raised.exception))

    def test_task_id_must_be_non_empty_when_present(self):
        """_parse_orchestrator_task_spec: rejects whitespace ids."""
        raw = {"tasks": [{"kind": "chunking", "id": "   "}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("id", str(raised.exception))

    def test_enabled_requires_boolean(self):
        """_parse_orchestrator_task_spec: enabled rejects non-bools."""
        raw = {"tasks": [{"kind": "chunking", "enabled": "no"}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("enabled", str(raised.exception))

    def test_skip_when_requires_mapping_when_present(self):
        """_parse_orchestrator_skip_when: rejects non-mapping guard blocks."""
        raw = {"tasks": [{"kind": "chunking", "skip_when": "bad"}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("skip_when", str(raised.exception))

    def test_skip_when_unknown_key(self):
        """_parse_orchestrator_skip_when: rejects unrecognized guard branches."""
        raw = {
            "tasks": [
                {
                    "kind": "chunking",
                    "skip_when": {"unexpected": True},
                },
            ],
        }
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("unknown keys", str(raised.exception))

    def test_skip_when_empty_returns_skip_object(self):
        """_parse_orchestrator_skip_when: empty mapping yields empty guard."""
        raw = {
            "tasks": [
                {
                    "kind": "chunking",
                    "skip_when": {},
                },
            ],
        }
        parsed = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        guard = parsed.tasks[0].skip_when
        self.assertIsNotNone(guard)
        assert guard is not None
        self.assertIsNone(guard.missing_env_var)

    def test_skip_when_bad_missing_env_shape(self):
        """_parse_orchestrator_skip_when: validates ``missing_env_var`` strings."""
        raw = {
            "tasks": [
                {
                    "kind": "chunking",
                    "skip_when": {"missing_env_var": 10},
                },
            ],
        }
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("missing_env_var", str(raised.exception))

    def test_params_must_map(self):
        """_parse_orchestrator_task_params: rejects stray param container types."""
        raw = {"tasks": [{"kind": "chunking", "params": "oops"}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("mapping", str(raised.exception))

    def test_chunk_profile_must_be_nonempty_string_when_present(self):
        """_parse_orchestrator_task_params: rejects empty chunk profiles."""
        raw = {"tasks": [{"kind": "chunking", "params": {"profile": ""}}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("profile", str(raised.exception))

    def test_normalizer_day_token_rejects_blank(self):
        """_parse_orchestrator_normalizer_day_value: rejects blank strings."""
        raw = {"tasks": [{"kind": "article_normalizer", "params": {"day": "   "}}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("day", str(raised.exception))

    def test_article_normalizer_rejects_unknown_keys(self):
        """_parse_orchestrator_task_params: normalizer restricts params keys."""
        raw = {"tasks": [{"kind": "article_normalizer", "params": {"oops": True}}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("unknown keys", str(raised.exception))

    def test_article_ingestor_rejects_unknown_keys(self):
        """_parse_orchestrator_task_params: ingest rejects arbitrary params."""
        raw = {"tasks": [{"kind": "article_ingestor", "params": {"oops": True}}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("unknown keys", str(raised.exception))

    def test_vector_sync_rejects_unknown_keys(self):
        """_parse_orchestrator_task_params: sync task rejects stray params."""
        raw = {"tasks": [{"kind": "vector_sync", "params": {"oops": True}}]}
        with self.assertRaises(ValueError) as raised:
            Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIn("unknown keys", str(raised.exception))

    def test_normalizer_params_accepts_utctoday_literal(self):
        """_parse_orchestrator_normalizer_day_value: parses utc_today tokens."""
        raw = {"tasks": [{"kind": "article_normalizer", "params": {"day": "utc_today"}}]}
        parsed = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertEqual(
            parsed.tasks[0].params.normalizer_day_raw,
            OrchestratorNormalizerDayToken.UTC_TODAY.value,
        )

    def test_normalizer_params_omit_day(self):
        """_parse_orchestrator_normalizer_day_value: absent day keys map to None."""
        raw = {"tasks": [{"kind": "article_normalizer", "params": {}}]}
        parsed = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertIsNone(parsed.tasks[0].params.normalizer_day_raw)

    def test_normalizer_params_store_explicit_iso_literals(self):
        """_parse_orchestrator_normalizer_day_value: persists validated ISO strings."""
        raw = {"tasks": [{"kind": "article_normalizer", "params": {"day": "1999-09-09"}}]}
        parsed = Settings._parse_orchestrator_config_mapping(raw)  # pylint: disable=protected-access
        self.assertEqual(parsed.tasks[0].params.normalizer_day_raw, "1999-09-09")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
