"""Unit tests for the Ingestor class in the src.ingestion module."""

import unittest

from src.ingestion.ingestor import Ingestor


class TestIngestorInit(unittest.TestCase):
    """This class tests __init__."""

    def test_init_with_no_client(self):
        """__init__: constructor accepts default argument without raising."""
        ingestor = Ingestor()
        self.assertIsInstance(ingestor, Ingestor)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
