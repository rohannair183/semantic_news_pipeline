"""This module tests orchestrator helper functions in ``task_runners``."""

from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from src.application.task_runners import (
    resolve_normalizer_day_for_orchestrator,
    default_task_runner_map,
)
from src.config.settings import (
    OrchestratorTaskParams,
    OrchestratorTaskSpec,
)
from src.enums.orchestrator_normalizer_day_token import OrchestratorNormalizerDayToken
from src.enums.orchestrator_task_kind import OrchestratorTaskKind


class TestResolveNormalizerDayForOrchestrator(unittest.TestCase):
    """This class tests resolve_normalizer_day_for_orchestrator"""

    def _spec(self, params: OrchestratorTaskParams) -> OrchestratorTaskSpec:
        return OrchestratorTaskSpec(
            task_id="n",
            kind=OrchestratorTaskKind.ARTICLE_NORMALIZER,
            enabled=True,
            skip_when=None,
            params=params,
        )

    def test_none_uses_utc_today(self):
        """resolve_normalizer_day_for_orchestrator: None day defers to utc today."""
        with patch("src.application.task_runners.utc_today_date", return_value=date(2024, 1, 2)):
            resolved = resolve_normalizer_day_for_orchestrator(
                self._spec(OrchestratorTaskParams(normalizer_day_raw=None))
            )
        self.assertEqual(resolved, date(2024, 1, 2))

    def test_token_utc_today(self):
        """resolve_normalizer_day_for_orchestrator: honors utc_today token."""
        with patch("src.application.task_runners.utc_today_date", return_value=date(2024, 3, 4)):
            resolved = resolve_normalizer_day_for_orchestrator(
                self._spec(
                    OrchestratorTaskParams(
                        normalizer_day_raw=OrchestratorNormalizerDayToken.UTC_TODAY.value
                    )
                )
            )
        self.assertEqual(resolved, date(2024, 3, 4))

    def test_explicit_iso_day(self):
        """resolve_normalizer_day_for_orchestrator: parses explicit ISO days."""
        resolved = resolve_normalizer_day_for_orchestrator(
            self._spec(OrchestratorTaskParams(normalizer_day_raw="2022-06-15"))
        )
        self.assertEqual(resolved, date(2022, 6, 15))


class TestDefaultTaskRunnerMap(unittest.TestCase):
    """This class tests default_task_runner_map"""

    def test_default_task_runner_map_covers_all_kinds(self):
        """default_task_runner_map: contains every OrchestratorTaskKind."""
        mapping = default_task_runner_map()
        for kind in OrchestratorTaskKind:
            self.assertIn(kind, mapping)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
