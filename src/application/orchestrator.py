"""Run declarative YAML pipeline tasks sequentially with timing and skip guards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

from src.application.task_runners import OrchestratorRunner, default_task_runner_map
from src.config.settings import (
    OrchestratorConfig,
    OrchestratorSkipWhen,
    OrchestratorTaskKind,
)
from src.utils.timer import Timer

_ORCHESTRATOR_TASK_PREFIX = "orchestrator.task"


@dataclass(frozen=True)
class OrchestratorTaskResult:
    """Outcome for a single orchestrated task invocation."""

    task_id: str
    kind: OrchestratorTaskKind
    outcome: str
    elapsed_seconds: Optional[float]
    detail: str


@dataclass(frozen=True)
class OrchestratorRunSummary:
    """Aggregated results from an orchestrator run."""

    task_results: Tuple[OrchestratorTaskResult, ...]

    @property
    def has_failure(self) -> bool:
        """Return True when any task ended in the failed outcome."""
        return any(result.outcome == "failed" for result in self.task_results)


def evaluate_orchestrator_skip_when(
    skip_when: Optional[OrchestratorSkipWhen],
) -> Tuple[bool, str]:
    """Return ``(skip, reason)`` when skip guards apply."""
    if skip_when is None or skip_when.missing_env_var is None:
        return False, ""
    env_name = skip_when.missing_env_var
    if not os.environ.get(env_name):
        return True, f"missing environment variable {env_name!r}"
    return False, ""


class Orchestrator:
    """Execute YAML-defined pipeline tasks using registered runners."""

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        configuration_root: Optional[Path] = None,
        timer: Optional[Timer] = None,
        runners: Optional[Mapping[OrchestratorTaskKind, OrchestratorRunner]] = None,
    ) -> None:
        self._config = config
        self._configuration_root = configuration_root
        self._timer = timer if timer is not None else Timer()
        self._runners = dict(runners) if runners is not None else default_task_runner_map()

    @property
    def timer(self) -> Timer:
        """Return the shared timer used for orchestration sections."""
        return self._timer

    def run(self) -> OrchestratorRunSummary:
        """Execute tasks in YAML order respecting ``fail_fast`` and skips."""
        results: List[OrchestratorTaskResult] = []
        halt = False
        for spec in self._config.tasks:
            if halt:
                results.append(
                    OrchestratorTaskResult(
                        task_id=spec.task_id,
                        kind=spec.kind,
                        outcome="skipped_fail_fast",
                        elapsed_seconds=None,
                        detail="halted due to earlier failure",
                    )
                )
                print(
                    f"[orchestrator] task={spec.task_id!r} kind={spec.kind.value} "
                    "SKIPPED (halted due to earlier failure)"
                )
                continue
            if not spec.enabled:
                results.append(
                    OrchestratorTaskResult(
                        task_id=spec.task_id,
                        kind=spec.kind,
                        outcome="skipped_disabled",
                        elapsed_seconds=None,
                        detail="disabled in configuration",
                    )
                )
                print(
                    f"[orchestrator] task={spec.task_id!r} kind={spec.kind.value} "
                    "SKIPPED (disabled)"
                )
                continue
            should_skip, skip_reason = evaluate_orchestrator_skip_when(spec.skip_when)
            if should_skip:
                results.append(
                    OrchestratorTaskResult(
                        task_id=spec.task_id,
                        kind=spec.kind,
                        outcome="skipped_predicate",
                        elapsed_seconds=None,
                        detail=skip_reason,
                    )
                )
                print(
                    f"[orchestrator] task={spec.task_id!r} kind={spec.kind.value} "
                    f"SKIPPED ({skip_reason})"
                )
                continue
            runner = self._runners.get(spec.kind)
            if runner is None:
                results.append(
                    OrchestratorTaskResult(
                        task_id=spec.task_id,
                        kind=spec.kind,
                        outcome="failed",
                        elapsed_seconds=None,
                        detail="no runner registered for task kind",
                    )
                )
                if self._config.fail_fast:
                    halt = True
                continue
            section_label = f"{_ORCHESTRATOR_TASK_PREFIX}.{spec.task_id}"
            self._timer.start(section_label)
            try:
                runner(spec, self._configuration_root, self._timer)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                elapsed = self._timer.stop(section_label)
                results.append(
                    OrchestratorTaskResult(
                        task_id=spec.task_id,
                        kind=spec.kind,
                        outcome="failed",
                        elapsed_seconds=elapsed,
                        detail=str(exc),
                    )
                )
                print(
                    f"[orchestrator] task={spec.task_id!r} kind={spec.kind.value} "
                    f"FAILED after {elapsed:.4f}s: {exc}"
                )
                if self._config.fail_fast:
                    halt = True
                continue
            elapsed = self._timer.stop(section_label)
            results.append(
                OrchestratorTaskResult(
                    task_id=spec.task_id,
                    kind=spec.kind,
                    outcome="success",
                    elapsed_seconds=elapsed,
                    detail="ok",
                )
            )
            print(
                f"[orchestrator] task={spec.task_id!r} kind={spec.kind.value} "
                f"SUCCESS in {elapsed:.4f}s"
            )
        return OrchestratorRunSummary(task_results=tuple(results))
