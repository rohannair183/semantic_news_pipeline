"""Integration wiring for declarative YAML orchestrator loading."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.application.orchestrator import Orchestrator
from src.config.settings import Settings
from src.enums.orchestrator_task_kind import OrchestratorTaskKind


def _noop_runner(_spec, _root, _timer):
    """Return sentinel output without touching external services."""
    return None


class TestOrchestratorRepoYaml(unittest.TestCase):
    """This class tests repository orchestrator presets."""

    def test_orchestrator_ci_yaml_loads_under_orchestration_dir(self):
        """load_orchestrator_config_from_path: accepts CI YAML on disk."""

        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "configuration" / "orchestration" / "orchestrator_ci.yaml"
        config = Settings.load_orchestrator_config_from_path(path)

        noop_map = {kind: _noop_runner for kind in OrchestratorTaskKind}

        orchestrator = Orchestrator(config, runners=noop_map)
        summary = orchestrator.run()

        self.assertFalse(summary.has_failure)
        disabled = sum(1 for r in summary.task_results if r.outcome == "skipped_disabled")
        self.assertGreaterEqual(disabled, 2)

    def test_orchestrator_ci_yaml_runs_with_default_runners(self):
        """Orchestrator.run: executes CI YAML stages with real registered runners."""
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "configuration" / "orchestration" / "orchestrator_ci.yaml"
        config = Settings.load_orchestrator_config_from_path(path)
        summary = Orchestrator(config).run()

        self.assertFalse(summary.has_failure)
        outcomes = {result.outcome for result in summary.task_results}
        self.assertIn("success", outcomes)
        self.assertIn("skipped_disabled", outcomes)

    def test_orchestrator_yaml_includes_briefing_persistence_stage(self):
        """load_orchestrator_config_from_path: default YAML wires briefing persistence."""
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "configuration" / "orchestration" / "orchestrator.yaml"
        config = Settings.load_orchestrator_config_from_path(path)

        kinds = [task.kind for task in config.tasks]
        self.assertIn(OrchestratorTaskKind.BRIEFING_PERSISTENCE, kinds)

        noop_map = {kind: _noop_runner for kind in OrchestratorTaskKind}
        with mock.patch(
            "src.application.orchestrator.evaluate_briefing_persistence_skip",
            return_value=(True, "briefing persistence already ran on 2026-05-13"),
        ):
            summary = Orchestrator(config, runners=noop_map).run()

        briefing_result = next(
            result
            for result in summary.task_results
            if result.kind == OrchestratorTaskKind.BRIEFING_PERSISTENCE
        )
        self.assertEqual(briefing_result.outcome, "skipped_predicate")

    def test_load_via_custom_configuration_root(self):
        """load_orchestrator_config: honors alternate configuration directories."""

        with TemporaryDirectory() as tmp:
            cfg_root = Path(tmp)
            orchestration_dir = cfg_root / "orchestration"
            orchestration_dir.mkdir(parents=True)
            orch_path = orchestration_dir / "orchestrator.yaml"
            orch_path.write_text(
                "fail_fast: true\n"
                "tasks:\n"
                "  - kind: chunking\n"
                "    params:\n"
                "      profile: probe\n",
                encoding="utf-8",
            )
            parsed = Settings.load_orchestrator_config(configuration_root=cfg_root)
            self.assertEqual(parsed.tasks[0].params.profile, "probe")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
