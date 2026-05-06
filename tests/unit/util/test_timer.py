"""Unit tests for Timer."""

import time
import unittest

from src.utils.timer import Timer


class TestTimerStartStop(unittest.TestCase):
    """This class tests start and stop."""

    def test_stop_returns_positive_elapsed(self) -> None:
        """start/stop: stop returns a positive elapsed time."""
        timer = Timer()
        timer.start("a")
        time.sleep(0.01)
        elapsed = timer.stop("a")
        self.assertGreater(elapsed, 0.0)

    def test_stop_records_section(self) -> None:
        """start/stop: completed section appears in records."""
        timer = Timer()
        timer.start("task")
        timer.stop("task")
        self.assertEqual(len(timer.records), 1)
        self.assertEqual(timer.records[0][0], "task")

    def test_double_start_raises(self) -> None:
        """start: raises ValueError when label is already active."""
        timer = Timer()
        timer.start("x")
        with self.assertRaises(ValueError) as ctx:
            timer.start("x")
        self.assertIn("already running", str(ctx.exception))

    def test_stop_unknown_raises(self) -> None:
        """stop: raises ValueError when label was never started."""
        timer = Timer()
        with self.assertRaises(ValueError) as ctx:
            timer.stop("nope")
        self.assertIn("not running", str(ctx.exception))


class TestTimerSection(unittest.TestCase):
    """This class tests section."""

    def test_section_records_timing(self) -> None:
        """section: context manager records a completed section."""
        timer = Timer()
        with timer.section("block"):
            time.sleep(0.01)
        self.assertEqual(len(timer.records), 1)
        self.assertEqual(timer.records[0][0], "block")
        self.assertGreater(timer.records[0][1], 0.0)

    def test_section_records_on_exception(self) -> None:
        """section: section is still recorded when the body raises."""
        timer = Timer()
        with self.assertRaises(RuntimeError):
            with timer.section("fail"):
                raise RuntimeError("boom")
        self.assertEqual(len(timer.records), 1)
        self.assertEqual(timer.records[0][0], "fail")


class TestTimerRecords(unittest.TestCase):
    """This class tests records."""

    def test_records_preserves_order(self) -> None:
        """records: entries are returned in insertion order."""
        timer = Timer()
        for label in ("first", "second", "third"):
            timer.start(label)
            timer.stop(label)
        labels = [r[0] for r in timer.records]
        self.assertEqual(labels, ["first", "second", "third"])

    def test_records_returns_copy(self) -> None:
        """records: returned list is a copy so mutations don't affect internal state."""
        timer = Timer()
        timer.start("a")
        timer.stop("a")
        snapshot = timer.records
        snapshot.clear()
        self.assertEqual(len(timer.records), 1)


class TestTimerSummary(unittest.TestCase):
    """This class tests summary."""

    def test_summary_aggregates_repeated_labels(self) -> None:
        """summary: repeated labels are summed together."""
        timer = Timer()
        for _ in range(3):
            timer.start("step")
            timer.stop("step")
        totals = timer.summary()
        self.assertIn("step", totals)
        individual_sum = sum(r[1] for r in timer.records)
        self.assertAlmostEqual(totals["step"], individual_sum)

    def test_summary_returns_empty_dict_when_no_records(self) -> None:
        """summary: returns empty dict when nothing has been timed."""
        timer = Timer()
        self.assertEqual(timer.summary(), {})

    def test_summary_distinct_labels(self) -> None:
        """summary: distinct labels each get their own entry."""
        timer = Timer()
        timer.start("a")
        timer.stop("a")
        timer.start("b")
        timer.stop("b")
        totals = timer.summary()
        self.assertEqual(set(totals.keys()), {"a", "b"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
