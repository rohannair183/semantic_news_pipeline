"""Unit tests for src.process.briefing_persistence._generated_at_to_utc_day.

This module contains focused tests for the internal helper used to normalise
persisted `generated_at` values into UTC calendar days and for the
`evaluate_briefing_persistence_skip` wrapper behaviour.
"""

from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from src.process.briefing_persistence import _generated_at_to_utc_day
from src.process import briefing_persistence as bp_module


class TestGeneratedAtToUtcDay(unittest.TestCase):
    """This class tests _generated_at_to_utc_day."""

    def test_naive_datetime_returns_date(self) -> None:
        """Naive datetime returns its date portion."""
        dt = datetime(2026, 5, 13, 12, 0, 0)
        out = _generated_at_to_utc_day(dt)
        self.assertEqual(out, date(2026, 5, 13))

    def test_aware_datetime_converted_to_utc_date(self) -> None:
        """Aware datetime converted to UTC then date returned."""
        dt = datetime(2026, 5, 13, 23, 0, 0, tzinfo=timezone.utc)
        out = _generated_at_to_utc_day(dt)
        self.assertEqual(out, date(2026, 5, 13))

    def test_date_passthrough(self) -> None:
        """Date value is returned unchanged."""
        d = date(2026, 5, 12)
        out = _generated_at_to_utc_day(d)
        self.assertEqual(out, d)

    def test_z_suffix_string_parsed(self) -> None:
        """ISO string ending with Z is parsed as UTC."""
        s = "2026-05-12T15:30:00Z"
        out = _generated_at_to_utc_day(s)
        self.assertEqual(out, date(2026, 5, 12))

    def test_offset_string_parsed(self) -> None:
        """ISO string with +00:00 is parsed as UTC."""
        s = "2026-05-12T15:30:00+00:00"
        out = _generated_at_to_utc_day(s)
        self.assertEqual(out, date(2026, 5, 12))

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with self.assertRaises(ValueError):
            _generated_at_to_utc_day("")

    def test_malformed_string_raises(self) -> None:
        """Malformed ISO string raises ValueError."""
        with self.assertRaises(ValueError):
            _generated_at_to_utc_day("2026-13-40T99:99:99+00:00")

    def test_wrong_type_raises_typeerror(self) -> None:
        """Non-date/datetime/string input raises TypeError."""
        with self.assertRaises(TypeError):
            _generated_at_to_utc_day(123)

    def test_naive_iso_string_parsed_returns_date(self) -> None:
        """ISO string without timezone parsed as naive datetime returns date."""
        s = "2026-05-12T15:30:00"
        out = _generated_at_to_utc_day(s)
        self.assertEqual(out, date(2026, 5, 12))


class TestEvaluateBriefingPersistenceSkip(unittest.TestCase):
    """Tests for evaluate_briefing_persistence_skip wrapper behaviour."""

    def test_no_previous_row_returns_false(self) -> None:
        """When fetch_latest_briefing_generated_at returns None, skip is False."""
        with unittest.mock.patch.object(
            bp_module,
            "fetch_latest_briefing_generated_at",
            return_value=None,
        ):
            skip, reason = bp_module.evaluate_briefing_persistence_skip(
                configuration_root=None, client_factory=lambda: None
            )
        self.assertFalse(skip)
        self.assertEqual(reason, "")

    def test_latest_on_same_day_skips(self) -> None:
        """If latest_generated_at is on the same UTC day, skip is True and includes date."""
        latest = "2026-05-13T00:00:00Z"
        with unittest.mock.patch.object(
            bp_module,
            "fetch_latest_briefing_generated_at",
            return_value=latest,
        ):
            skip, reason = bp_module.evaluate_briefing_persistence_skip(
                configuration_root=None,
                client_factory=lambda: None,
                current_date=date(2026, 5, 13),
            )
        self.assertTrue(skip)
        self.assertIn("2026-05-13", reason)

    def test_latest_before_today_does_not_skip(self) -> None:
        """If latest_generated_at is before today, do not skip."""
        latest = "2026-05-12T23:59:59Z"
        with unittest.mock.patch.object(
            bp_module,
            "fetch_latest_briefing_generated_at",
            return_value=latest,
        ):
            skip, reason = bp_module.evaluate_briefing_persistence_skip(
                configuration_root=None,
                client_factory=lambda: None,
                current_date=date(2026, 5, 13),
            )
        self.assertFalse(skip)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
