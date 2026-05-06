"""This class tests OrchestratorTaskKind."""

import unittest

from src.enums.orchestrator_task_kind import OrchestratorTaskKind


class TestOrchestratorTaskKind(unittest.TestCase):
    """This class tests OrchestratorTaskKind helpers."""

    def test_from_value_accepts_article_ingestor(self):
        """from_value: parses article_ingestor."""
        self.assertEqual(
            OrchestratorTaskKind.from_value("article_ingestor"),
            OrchestratorTaskKind.ARTICLE_INGESTOR,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
