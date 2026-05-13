"""This module tests orchestrator sequencing, skips, and ``fail_fast``."""

from __future__ import annotations

import os
import unittest
from typing import Dict
from unittest import mock

from src.application.orchestrator import Orchestrator
from src.application.orchestrator import OrchestratorRunSummary, OrchestratorTaskResult
from src.application.orchestrator import evaluate_orchestrator_skip_when
from src.config.settings import (
    OrchestratorConfig,
    OrchestratorSkipWhen,
    OrchestratorTaskParams,
    OrchestratorTaskSpec,
)
from src.enums.orchestrator_task_kind import OrchestratorTaskKind
from src.utils.timer import Timer


def _stub_spec(
    task_id: str,
    kind: OrchestratorTaskKind,
    *,
    enabled: bool = True,
    skip_when: OrchestratorSkipWhen | None = None,
    params: OrchestratorTaskParams | None = None,
) -> OrchestratorTaskSpec:
    params_value = OrchestratorTaskParams() if params is None else params
    return OrchestratorTaskSpec(
        enabled=enabled,
        params=params_value,
        skip_when=skip_when,
        kind=kind,
        task_id=task_id,
    )


class TestEvaluateOrchestratorSkipWhen(unittest.TestCase):
    """This class tests evaluate_orchestrator_skip_when."""

    def test_no_skip_predicate(self):
        """evaluate_orchestrator_skip_when: returns false when guards absent."""
        self.assertEqual(evaluate_orchestrator_skip_when(None), (False, ""))
        skip = OrchestratorSkipWhen(missing_env_var=None)
        self.assertEqual(evaluate_orchestrator_skip_when(skip), (False, ""))

    def test_missing_env_skips_when_unset(self):
        """evaluate_orchestrator_skip_when: skips unless env appears."""
        unique_name = "ORCH_GUARD_TEST_VAR_NO_ONE_SETS_THIS"
        skip = OrchestratorSkipWhen(missing_env_var=unique_name)
        with mock.patch.dict(os.environ):
            os.environ.pop(unique_name, None)
            should_skip, reason = evaluate_orchestrator_skip_when(skip)
        self.assertTrue(should_skip)
        self.assertIn(unique_name, reason)

        with mock.patch.dict(os.environ):
            os.environ[unique_name] = "x"
            self.assertEqual(evaluate_orchestrator_skip_when(skip), (False, ""))


class TestOrchestratorRun(unittest.TestCase):
    """This class tests Orchestrator."""

    def test_run_loads_repository_dotenv_before_tasks(self):
        """Orchestrator.run: merges repository .env before evaluating tasks."""

        def ok(_spec, _root, _timer):
            return None

        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(_stub_spec("only", OrchestratorTaskKind.CHUNKING),),
        )
        with mock.patch("src.application.orchestrator.Settings.load_repository_dotenv") as mocked:
            Orchestrator(
                config,
                runners={OrchestratorTaskKind.CHUNKING: ok},
            ).run()
        mocked.assert_called_once()

    def test_fail_fast_halts_remaining_tasks(self):
        """Orchestrator.run: fail_fast skips later tasks after a failure."""

        calls: list[str] = []

        def ok_runner(_spec, _root, _timer):
            calls.append("ok")

        def boom(_spec, _root, _timer):
            raise RuntimeError("expected failure")

        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(
                _stub_spec("a", OrchestratorTaskKind.ARTICLE_INGESTOR),
                _stub_spec("b", OrchestratorTaskKind.ARTICLE_NORMALIZER),
                _stub_spec("c", OrchestratorTaskKind.CHUNKING),
            ),
        )
        runners = {
            OrchestratorTaskKind.ARTICLE_INGESTOR: ok_runner,
            OrchestratorTaskKind.ARTICLE_NORMALIZER: boom,
            OrchestratorTaskKind.CHUNKING: ok_runner,
        }
        summary = Orchestrator(config, runners=runners).run()
        self.assertEqual(calls, ["ok"])
        self.assertEqual(
            tuple(r.outcome for r in summary.task_results),
            ("success", "failed", "skipped_fail_fast"),
        )

    def test_continue_after_failure_when_fail_fast_disabled(self):
        """Orchestrator.run: executes later tasks after failure when relaxed."""

        counts: Dict[str, int] = {"n": 0}

        def fail_once(_spec, _root, _timer):
            counts["n"] += 1
            if counts["n"] == 1:
                raise RuntimeError("first-only")

        def ok(_spec, _root, _timer):
            counts["n"] += 10

        config = OrchestratorConfig(
            fail_fast=False,
            tasks=(
                _stub_spec("a", OrchestratorTaskKind.CHUNKING),
                _stub_spec("b", OrchestratorTaskKind.EMBEDDINGS),
            ),
        )
        runners = {
            OrchestratorTaskKind.CHUNKING: fail_once,
            OrchestratorTaskKind.EMBEDDINGS: ok,
        }
        summary = Orchestrator(config, runners=runners).run()
        self.assertEqual(counts["n"], 11)
        outcomes = tuple(r.outcome for r in summary.task_results)
        self.assertEqual(outcomes, ("failed", "success"))
        self.assertTrue(summary.has_failure)

    def test_disabled_tasks_record_skip(self):
        """Orchestrator.run: skips disabled declarative tasks."""
        invoked = mock.Mock()

        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(
                _stub_spec("a", OrchestratorTaskKind.CHUNKING, enabled=False),
            ),
        )
        summary = Orchestrator(
            config,
            runners={
                OrchestratorTaskKind.CHUNKING: invoked,
            },
        ).run()
        invoked.assert_not_called()
        self.assertEqual(summary.task_results[0].outcome, "skipped_disabled")

    def test_skip_predicate_records_without_running(self):
        """Orchestrator.run: skips when ``skip_when`` predicate matches."""
        unique_name = "ORCH_PREDICATE_TEST_VAR_ABSENT_ABSENT_ABSENT"

        invoked = mock.Mock()

        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(
                _stub_spec(
                    "guard",
                    OrchestratorTaskKind.ARTICLE_INGESTOR,
                    skip_when=OrchestratorSkipWhen(missing_env_var=unique_name),
                ),
            ),
        )
        with mock.patch.dict(os.environ):
            os.environ.pop(unique_name, None)
            summary = Orchestrator(
                config,
                runners={OrchestratorTaskKind.ARTICLE_INGESTOR: invoked},
            ).run()

        invoked.assert_not_called()
        self.assertEqual(summary.task_results[0].outcome, "skipped_predicate")

    def test_missing_runner_is_failure(self):
        """Orchestrator.run: reports failure when no runner is registered."""
        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(_stub_spec("x", OrchestratorTaskKind.CHUNKING),),
        )
        summary = Orchestrator(config, runners={}).run()
        result = summary.task_results[0]
        self.assertEqual(result.outcome, "failed")
        self.assertIn("runner", result.detail.lower())
        self.assertTrue(summary.has_failure)

    def test_success_records_timer_sections(self):
        """Orchestrator.run: timers collect task sections."""

        def ok(_spec, _root, _timer):
            return {"ok": True}

        timer = Timer()
        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(
                _stub_spec("only", OrchestratorTaskKind.CHUNKING),
                _stub_spec(
                    "off",
                    OrchestratorTaskKind.CHUNKING,
                    enabled=False,
                ),
            ),
        )
        summary = Orchestrator(
            config,
            timer=timer,
            runners={OrchestratorTaskKind.CHUNKING: ok},
        ).run()
        self.assertFalse(summary.has_failure)
        labels = {label for label, _elapsed in timer.records}
        self.assertTrue(any(label.startswith("orchestrator.task.") for label in labels))

    def test_briefing_persistence_skip_check_records_without_running(self):
        """Orchestrator.run: skips briefing persistence when freshness gate matches."""

        invoked = mock.Mock()

        config = OrchestratorConfig(
            fail_fast=True,
            tasks=(
                _stub_spec("briefing", OrchestratorTaskKind.BRIEFING_PERSISTENCE),
            ),
        )
        with mock.patch(
            "src.application.orchestrator.evaluate_briefing_persistence_skip",
            return_value=(True, "briefing persistence already ran on 2026-05-13"),
        ) as mocked_skip:
            summary = Orchestrator(
                config,
                runners={OrchestratorTaskKind.BRIEFING_PERSISTENCE: invoked},
            ).run()

        mocked_skip.assert_called_once()
        invoked.assert_not_called()
        self.assertEqual(summary.task_results[0].outcome, "skipped_predicate")


class TestOrchestratorSummary(unittest.TestCase):
    """This class tests OrchestratorRunSummary helpers."""

    def test_has_failure_ignores_skips(self):
        """OrchestratorRunSummary.has_failure: skips are not failures."""
        summary = OrchestratorRunSummary(
            task_results=(
                OrchestratorTaskResult(
                    task_id="a",
                    kind=OrchestratorTaskKind.CHUNKING,
                    outcome="skipped_fail_fast",
                    elapsed_seconds=None,
                    detail="x",
                ),
            )
        )
        self.assertFalse(summary.has_failure)

    def test_has_failure_detects_failed(self):
        """OrchestratorRunSummary.has_failure: flags failed outcomes."""
        summary = OrchestratorRunSummary(
            task_results=(
                OrchestratorTaskResult(
                    task_id="a",
                    kind=OrchestratorTaskKind.CHUNKING,
                    outcome="failed",
                    elapsed_seconds=0.0,
                    detail="err",
                ),
            )
        )
        self.assertTrue(summary.has_failure)


class TestOrchestratorTimer(unittest.TestCase):
    """This class tests Orchestrator.timer wiring."""

    def test_timer_defaults_when_missing(self):
        """Orchestrator.__init__: creates timers when callers omit shared clocks."""
        config = OrchestratorConfig(fail_fast=True, tasks=tuple())
        orchestrator_instance = Orchestrator(config, runners={})
        self.assertIsInstance(orchestrator_instance.timer, Timer)

    def test_timer_shares_provided_clock(self):
        """Orchestrator.__init__: exposes caller-supplied timers."""
        shared = Timer()
        config = OrchestratorConfig(fail_fast=True, tasks=tuple())
        orchestrator_instance = Orchestrator(config, timer=shared, runners={})
        self.assertIs(orchestrator_instance.timer, shared)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
