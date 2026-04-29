# pyright: reportPrivateUsage=false
"""Focused unit tests for Settings helper methods."""

import unittest
from datetime import date
from unittest.mock import patch

from src.config.settings import Settings
from src.enums.ingestion_timeframe_relative import IngestionTimeframeRelative


class TestSettingsProfileStringHelpers(unittest.TestCase):
    """This class tests profile string helper methods."""

    def test_load_required_profile_string_raises_for_blank(self):
        """_load_required_profile_string: raises when required string is blank."""
        with self.assertRaises(ValueError):
            Settings._load_required_profile_string(  # pylint: disable=protected-access
                profile_name="technology_daily",
                raw_profile={"topic": " "},
                field_name="topic",
            )

    def test_load_required_profile_string_returns_value_when_valid(self):
        """_load_required_profile_string: returns the value for valid strings."""
        resolved = Settings._load_required_profile_string(  # pylint: disable=protected-access
            profile_name="technology_daily",
            raw_profile={"topic": "technology"},
            field_name="topic",
        )
        self.assertEqual(resolved, "technology")

    def test_load_optional_profile_string_raises_for_non_string(self):
        """_load_optional_profile_string: raises when optional value is not a string."""
        with self.assertRaises(ValueError):
            Settings._load_optional_profile_string(  # pylint: disable=protected-access
                profile_name="technology_daily",
                raw_profile={"query": 123},
                field_name="query",
            )

    def test_load_optional_profile_string_raises_for_blank_when_not_allowed(self):
        """_load_optional_profile_string: rejects blank strings when allow_empty is false."""
        with self.assertRaises(ValueError):
            Settings._load_optional_profile_string(  # pylint: disable=protected-access
                profile_name="technology_daily",
                raw_profile={"query": " "},
                field_name="query",
            )

    def test_load_optional_profile_string_allows_empty_when_flag_enabled(self):
        """_load_optional_profile_string: returns empty strings when allow_empty is enabled."""
        resolved = Settings._load_optional_profile_string(  # pylint: disable=protected-access
            profile_name="technology_daily",
            raw_profile={"section": ""},
            field_name="section",
            allow_empty=True,
        )
        self.assertEqual(resolved, "")

    def test_load_profile_content_show_fields_defaults_when_helper_returns_none(self):
        """_load_profile_content_show_fields: falls back to 'all' when helper returns None."""
        with patch.object(
            Settings,
            "_load_optional_profile_string",
            return_value=None,
        ):
            resolved = Settings._load_profile_content_show_fields(  # pylint: disable=protected-access
                profile_name="technology_daily",
                raw_profile={},
            )
        self.assertEqual(resolved, "all")


class TestSettingsTimeframeHelpers(unittest.TestCase):
    """This class tests timeframe helper methods."""

    def test_load_timeframe_raises_when_missing_and_no_default(self):
        """_load_timeframe: raises when value is absent and no default is provided."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe=None,
                field_prefix="profiles.tech.timeframe",
                default_relative=None,
            )

    def test_load_timeframe_raises_for_non_mapping(self):
        """_load_timeframe: raises when raw timeframe is not a mapping."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe="past_week",
                field_prefix="profiles.tech.timeframe",
                default_relative=IngestionTimeframeRelative.PAST_DAY,
            )

    @patch("src.config.settings.utc_today_date", return_value=date(2026, 4, 29))
    def test_load_timeframe_uses_default_relative_when_absent(self, _mock_today):
        """_load_timeframe: uses default relative when timeframe mapping is absent."""
        timeframe = Settings._load_timeframe(  # pylint: disable=protected-access
            raw_timeframe=None,
            field_prefix="profiles.tech.timeframe",
            default_relative=IngestionTimeframeRelative.PAST_WEEK,
        )
        self.assertEqual(timeframe.from_date, date(2026, 4, 23))
        self.assertEqual(timeframe.to_date, date(2026, 4, 29))

    def test_load_timeframe_raises_for_invalid_mode(self):
        """_load_timeframe: raises when mode is unsupported."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe={"mode": "diagonal"},
                field_prefix="profiles.tech.timeframe",
                default_relative=IngestionTimeframeRelative.PAST_DAY,
            )

    def test_load_timeframe_raises_for_invalid_relative(self):
        """_load_timeframe: raises when relative value is unsupported."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe={"mode": "relative", "relative": "past_year"},
                field_prefix="profiles.tech.timeframe",
                default_relative=IngestionTimeframeRelative.PAST_DAY,
            )

    @patch("src.config.settings.utc_today_date", return_value=date(2026, 4, 29))
    def test_load_timeframe_resolves_relative_mode(self, _mock_today):
        """_load_timeframe: resolves relative mode into concrete date bounds."""
        timeframe = Settings._load_timeframe(  # pylint: disable=protected-access
            raw_timeframe={"mode": "relative", "relative": "past_day"},
            field_prefix="profiles.tech.timeframe",
            default_relative=IngestionTimeframeRelative.PAST_WEEK,
        )
        self.assertEqual(timeframe.from_date, date(2026, 4, 29))
        self.assertEqual(timeframe.to_date, date(2026, 4, 29))

    def test_load_timeframe_raises_for_missing_explicit_dates(self):
        """_load_timeframe: explicit mode requires from_date and to_date."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe={"mode": "explicit", "from_date": "2026-04-20"},
                field_prefix="profiles.tech.timeframe",
                default_relative=IngestionTimeframeRelative.PAST_DAY,
            )

    def test_load_timeframe_raises_for_reversed_explicit_dates(self):
        """_load_timeframe: explicit mode rejects reversed date ranges."""
        with self.assertRaises(ValueError):
            Settings._load_timeframe(  # pylint: disable=protected-access
                raw_timeframe={
                    "mode": "explicit",
                    "from_date": "2026-04-21",
                    "to_date": "2026-04-20",
                },
                field_prefix="profiles.tech.timeframe",
                default_relative=IngestionTimeframeRelative.PAST_DAY,
            )

    def test_resolve_profile_date_window_returns_explicit_from_to_dates(self):
        """_resolve_profile_date_window: returns explicit from/to when valid."""
        from_date, to_date = Settings._resolve_profile_date_window(  # pylint: disable=protected-access
            profile_name="technology_daily",
            raw_profile={"from_date": "2026-04-20", "to_date": "2026-04-28"},
            raw_run_date=None,
        )
        self.assertEqual(from_date, date(2026, 4, 20))
        self.assertEqual(to_date, date(2026, 4, 28))

    @patch("src.config.settings.utc_today_date", return_value=date(2026, 4, 29))
    def test_build_relative_timeframe_covers_week_and_month(self, _mock_today):
        """_build_relative_timeframe: returns expected windows for week and month."""
        week = Settings._build_relative_timeframe(  # pylint: disable=protected-access
            IngestionTimeframeRelative.PAST_WEEK
        )
        month = Settings._build_relative_timeframe(  # pylint: disable=protected-access
            IngestionTimeframeRelative.PAST_MONTH
        )

        self.assertEqual(week.from_date, date(2026, 4, 23))
        self.assertEqual(week.to_date, date(2026, 4, 29))
        self.assertEqual(month.from_date, date(2026, 3, 31))
        self.assertEqual(month.to_date, date(2026, 4, 29))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
