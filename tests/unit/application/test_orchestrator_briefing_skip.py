"""Tests for orchestrator behavior around briefing persistence skip/exception paths."""

from __future__ import annotations

import unittest

from src.application.orchestrator import Orchestrator
from src.config.settings import OrchestratorConfig, OrchestratorTaskSpec, OrchestratorTaskParams, OrchestratorTaskKind


class TestOrchestratorBriefingSkipAndError(unittest.TestCase):
    """This class tests orchestrator handling of briefing persistence skip/error."""

    def _make_config(self) -> OrchestratorConfig:
        spec = OrchestratorTaskSpec(
            task_id="briefing",
            kind=OrchestratorTaskKind.BRIEFING_PERSISTENCE,
            enabled=True,
            skip_when=None,
            params=OrchestratorTaskParams(),
        )
        return OrchestratorConfig(fail_fast=True, tasks=(spec,))

    def test_evaluate_briefing_persistence_raises_is_handled(self) -> None:
        config = self._make_config()
        orch = Orchestrator(config, runners={})
        def _raise(*args, **kwargs):
            raise RuntimeError("boom")
        with unittest.mock.patch(
            "src.application.orchestrator.evaluate_briefing_persistence_skip",
            side_effect=_raise,
        ):
            summary = orch.run()
        self.assertEqual(len(summary.task_results), 1)
        res = summary.task_results[0]
        self.assertEqual(res.outcome, "failed")
        self.assertIn("boom", res.detail)

    def test_evaluate_briefing_persistence_skips_when_true(self) -> None:
        config = self._make_config()
        orch = Orchestrator(config, runners={})
        with unittest.mock.patch(
            "src.application.orchestrator.evaluate_briefing_persistence_skip",
            return_value=(True, "missing stuff"),
        ):
            summary = orch.run()
        self.assertEqual(len(summary.task_results), 1)
        res = summary.task_results[0]
        self.assertEqual(res.outcome, "skipped_predicate")
        self.assertIn("missing stuff", res.detail)


if __name__ == "__main__":
    unittest.main()
