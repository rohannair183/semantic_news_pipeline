"""Unit tests for ingestion logger behavior."""

import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.ingestion_logger import IngestionLogger


class TestIngestionLogger(unittest.TestCase):
    """This class tests log_api_call."""

    def test_logger_tracks_usage_when_disabled(self):
        """log_api_call: tracks usage counters even when file logging is disabled."""
        logger = IngestionLogger(
            enabled=False,
            logs_dir=Path("logs"),
            run_timestamp="20260429T000000Z",
        )
        logger.log_api_call(profile="main_daily", path="/search", status_code=200)
        logger.log_api_error(
            profile="main_daily",
            path="/search",
            error="Guardian API HTTP error 429",
            status_code=429,
        )
        self.assertIsNone(logger.log_path)
        self.assertEqual(
            logger.usage_counts,
            {
                "total_api_calls": 2,
                "error_api_calls": 1,
                "calls_by_profile": {"main_daily": 2},
            },
        )

    def test_logger_uses_unknown_profile_key_when_profile_missing(self):
        """log_api_call: uses unknown_profile when profile is missing."""
        logger = IngestionLogger(
            enabled=False,
            logs_dir=Path("logs"),
            run_timestamp="20260429T000000Z",
        )

        logger.log_api_call(profile=None, path="/search", status_code=200)

        self.assertEqual(
            logger.usage_counts["calls_by_profile"],
            {"unknown_profile": 1},
        )

    def test_logger_writes_jsonl_events_when_enabled(self):
        """log_ingestion_summary: writes minimal JSONL events when enabled."""
        with tempfile.TemporaryDirectory() as temp_directory:
            logger = IngestionLogger(
                enabled=True,
                logs_dir=Path(temp_directory),
                run_timestamp="20260429T000000Z",
            )
            logger.log_api_call(profile="main_daily", path="/search", status_code=200)
            logger.log_ingestion_summary({"profile_count": 1})

            log_path = Path(str(logger.log_path))
            self.assertTrue(log_path.is_file())
            with log_path.open("r", encoding="utf-8") as input_file:
                lines = [json.loads(line) for line in input_file if line.strip()]
            self.assertEqual(lines[0]["event"], "guardian_api_call")
            self.assertEqual(lines[1]["event"], "ingestion_summary")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
