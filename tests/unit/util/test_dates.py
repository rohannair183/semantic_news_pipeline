"""Unit tests for shared date helpers."""

import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.utils.dates import (
    coerce_day,
    date_range_last_n_calendar_days_inclusive,
    date_range_month_to_date,
    date_range_single_calendar_day,
    format_day_compact,
    format_day_iso,
    parse_checkpoint_timestamp,
    parse_guardian_datetime,
    parse_utc_instant_iso_z,
    utc_now_checkpoint_token,
    utc_today_date,
)


class TestUTCTodayDate(unittest.TestCase):
    """This class tests utc_today_date."""

    @patch("src.utils.dates.datetime")
    def test_utc_today_date_returns_utc_day(self, mock_datetime):
        """utc_today_date: returns today's date in UTC."""
        mock_now = Mock()
        mock_now.date.return_value = date(2026, 4, 29)
        mock_datetime.now.return_value = mock_now

        self.assertEqual(utc_today_date(), date(2026, 4, 29))
        mock_datetime.now.assert_called_once_with(timezone.utc)


class TestUTCNowCheckpointToken(unittest.TestCase):
    """This class tests utc_now_checkpoint_token."""

    @patch("src.utils.dates.datetime")
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
            coerce_day(datetime(2026, 4, 29, 5, 0)),
            date(2026, 4, 29),
        )
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


class TestDateRangeBriefingHelpers(unittest.TestCase):
    """This class tests date_range_single_calendar_day and related helpers."""

    def test_date_range_single_calendar_day(self):
        """date_range_single_calendar_day: returns the same day twice."""
        day = date(2026, 5, 10)
        self.assertEqual(date_range_single_calendar_day(day), (day, day))

    def test_date_range_last_n_calendar_days_inclusive(self):
        """date_range_last_n_calendar_days_inclusive: spans n days ending on end."""
        end = date(2026, 5, 10)
        self.assertEqual(
            date_range_last_n_calendar_days_inclusive(end, 7),
            (date(2026, 5, 4), end),
        )
        self.assertEqual(
            date_range_last_n_calendar_days_inclusive(end, 1),
            (end, end),
        )

    def test_date_range_last_n_rejects_zero(self):
        """date_range_last_n_calendar_days_inclusive: raises when n < 1."""
        with self.assertRaises(ValueError):
            date_range_last_n_calendar_days_inclusive(date(2026, 1, 1), 0)

    def test_date_range_month_to_date(self):
        """date_range_month_to_date: first of month through anchor."""
        self.assertEqual(
            date_range_month_to_date(date(2026, 5, 12)),
            (date(2026, 5, 1), date(2026, 5, 12)),
        )
        self.assertEqual(
            date_range_month_to_date(date(2026, 1, 1)),
            (date(2026, 1, 1), date(2026, 1, 1)),
        )


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


class TestParseUtcInstantIsoZ(unittest.TestCase):
    """This class tests parse_utc_instant_iso_z."""

    def test_parses_z_suffix_to_utc_aware(self) -> None:
        """parse_utc_instant_iso_z: returns UTC-aware datetime."""
        dt = parse_utc_instant_iso_z("2026-05-12T15:30:00Z")
        self.assertEqual(dt, datetime(2026, 5, 12, 15, 30, 0, tzinfo=timezone.utc))

    def test_strips_whitespace(self) -> None:
        """parse_utc_instant_iso_z: trims input."""
        dt = parse_utc_instant_iso_z("  2026-01-01T00:00:00Z  ")
        self.assertEqual(dt, datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc))

    def test_rejects_non_z_suffix(self) -> None:
        """parse_utc_instant_iso_z: requires Z UTC suffix."""
        with self.assertRaises(ValueError):
            parse_utc_instant_iso_z("2026-05-12T15:30:00+00:00")

    def test_rejects_empty(self) -> None:
        """parse_utc_instant_iso_z: rejects empty string."""
        with self.assertRaises(ValueError):
            parse_utc_instant_iso_z("")

    def test_rejects_invalid_iso(self) -> None:
        """parse_utc_instant_iso_z: raises on malformed timestamp."""
        with self.assertRaises(ValueError):
            parse_utc_instant_iso_z("2026-13-40T99:99:99Z")


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
