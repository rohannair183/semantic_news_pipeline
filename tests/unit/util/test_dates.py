"""Unit tests for shared date helpers."""

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.dates import (
    coerce_day,
    format_day_compact,
    format_day_iso,
    parse_checkpoint_timestamp,
    parse_guardian_datetime,
    utc_now_checkpoint_token,
    utc_today_date,
)


class TestUTCTodayDate(unittest.TestCase):
    """This class tests utc_today_date."""

    @patch("src.util.dates.datetime")
    def test_utc_today_date_returns_utc_day(self, mock_datetime):
        """utc_today_date: returns today's date in UTC."""
        mock_now = Mock()
        mock_now.date.return_value = date(2026, 4, 29)
        mock_datetime.now.return_value = mock_now

        self.assertEqual(utc_today_date(), date(2026, 4, 29))
        mock_datetime.now.assert_called_once_with(timezone.utc)


class TestUTCNowCheckpointToken(unittest.TestCase):
    """This class tests utc_now_checkpoint_token."""

    @patch("src.util.dates.datetime")
    def test_utc_now_checkpoint_token_formats_timestamp(self, mock_datetime):
        """utc_now_checkpoint_token: returns checkpoint-formatted UTC timestamp."""
        mock_now = Mock()
        mock_now.strftime.return_value = "20260429T012456Z"
        mock_datetime.now.return_value = mock_now

        self.assertEqual(utc_now_checkpoint_token(), "20260429T012456Z")
        mock_datetime.now.assert_called_once_with(timezone.utc)


class TestCoerceDay(unittest.TestCase):
    """This class tests coerce_day."""

    def test_coerce_day_supports_date_datetime_and_strings(self):
        """coerce_day: normalizes supported inputs to a native date."""
        self.assertEqual(coerce_day(date(2026, 4, 29)), date(2026, 4, 29))
        self.assertEqual(
            coerce_day(datetime(2026, 4, 29, 5, 0, tzinfo=timezone.utc)),
            date(2026, 4, 29),
        )
        self.assertEqual(coerce_day("2026-04-29"), date(2026, 4, 29))
        self.assertEqual(coerce_day("20260429"), date(2026, 4, 29))

    def test_coerce_day_rejects_invalid_values(self):
        """coerce_day: raises for unsupported inputs."""
        with self.assertRaises(ValueError):
            coerce_day("not-a-date")
        with self.assertRaises(ValueError):
            coerce_day(123)  # type: ignore[arg-type]


class TestFormatDayHelpers(unittest.TestCase):
    """This class tests format_day_iso and format_day_compact."""

    def test_format_day_helpers_return_expected_shapes(self):
        """format_day helpers: serialize native dates to expected strings."""
        day = date(2026, 4, 29)
        self.assertEqual(format_day_iso(day), "2026-04-29")
        self.assertEqual(format_day_compact(day), "20260429")

    def test_format_day_helpers_reject_datetime(self):
        """format_day helpers: keep day-level interfaces strict."""
        with self.assertRaises(TypeError):
            format_day_iso(datetime(2026, 4, 29, 1, 2, 3))
        with self.assertRaises(TypeError):
            format_day_compact(datetime(2026, 4, 29, 1, 2, 3))


class TestParseGuardianDatetime(unittest.TestCase):
    """This class tests parse_guardian_datetime."""

    def test_parse_guardian_datetime_supports_current_api_formats(self):
        """parse_guardian_datetime: parses Guardian ISO values with Z and offsets."""
        self.assertEqual(
            parse_guardian_datetime("2026-04-29T01:24:23Z"),
            datetime(2026, 4, 29, 1, 24, 23, tzinfo=timezone.utc),
        )
        self.assertEqual(
            parse_guardian_datetime("2026-04-29T01:24:23+00:00"),
            datetime(2026, 4, 29, 1, 24, 23, tzinfo=timezone.utc),
        )

    def test_parse_guardian_datetime_returns_none_for_invalid_input(self):
        """parse_guardian_datetime: returns None for invalid or empty values."""
        self.assertIsNone(parse_guardian_datetime(None))
        self.assertIsNone(parse_guardian_datetime(""))
        self.assertIsNone(parse_guardian_datetime("not-a-date"))


class TestParseCheckpointTimestamp(unittest.TestCase):
    """This class tests parse_checkpoint_timestamp."""

    def test_parse_checkpoint_timestamp_supports_paths_and_tokens(self):
        """parse_checkpoint_timestamp: parses checkpoint filename tokens from strings or paths."""
        expected = datetime(2026, 4, 29, 1, 24, 56)
        self.assertEqual(
            parse_checkpoint_timestamp("technology_daily_20260429T012456Z.json"),
            expected,
        )
        self.assertEqual(
            parse_checkpoint_timestamp(
                Path(
                    "checkpoints/article_ingestor/"
                    "science_daily_20260429T012456Z.json"
                )
            ),
            expected,
        )
        self.assertEqual(parse_checkpoint_timestamp("20260429T012456Z"), expected)

    def test_parse_checkpoint_timestamp_returns_none_for_invalid_input(self):
        """parse_checkpoint_timestamp: returns None when no valid token is present."""
        self.assertIsNone(parse_checkpoint_timestamp("technology_daily_invalid.json"))
