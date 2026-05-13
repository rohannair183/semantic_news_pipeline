"""This class tests BriefingDateFilter."""

import unittest

from src.enums.briefing_date_filter import BriefingDateFilter


class TestBriefingDateFilter(unittest.TestCase):
    """This class tests BriefingDateFilter helpers."""

    def test_from_value_accepts_daily_weekly_monthly(self):
        """from_value: parses daily, weekly, and monthly."""
        self.assertEqual(BriefingDateFilter.from_value("daily"), BriefingDateFilter.DAILY)
        self.assertEqual(BriefingDateFilter.from_value("weekly"), BriefingDateFilter.WEEKLY)
        self.assertEqual(BriefingDateFilter.from_value("monthly"), BriefingDateFilter.MONTHLY)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
