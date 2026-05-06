"""Reusable wall-clock timing utility."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Tuple


class Timer:
    """Accumulates named timing sections that can be shared across classes.

    Create a single ``Timer`` instance and pass it to any class that should
    contribute timing data.  All sections are recorded in declaration order
    and can be retrieved via :pyattr:`records` or aggregated with
    :pymeth:`summary`.
    """

    def __init__(self) -> None:
        self._records: List[Tuple[str, float]] = []
        self._active: Dict[str, float] = {}

    def start(self, label: str) -> None:
        """Begin a timing section called ``label``.

        Raises:
            ValueError: If ``label`` is already running.
        """
        if label in self._active:
            raise ValueError(f"Timer section already running: '{label}'")
        self._active[label] = time.monotonic()

    def stop(self, label: str) -> float:
        """End the timing section ``label`` and return elapsed seconds.

        Raises:
            ValueError: If ``label`` was never started.
        """
        start_time = self._active.pop(label, None)
        if start_time is None:
            raise ValueError(f"Timer section not running: '{label}'")
        elapsed = time.monotonic() - start_time
        self._records.append((label, elapsed))
        return elapsed

    @contextmanager
    def section(self, label: str) -> Iterator[None]:
        """Context manager that times the enclosed block under ``label``."""
        self.start(label)
        try:
            yield
        finally:
            self.stop(label)

    @property
    def records(self) -> List[Tuple[str, float]]:
        """Return all completed sections as ``(label, elapsed)`` pairs."""
        return list(self._records)

    def summary(self) -> Dict[str, float]:
        """Return total elapsed seconds per label, aggregated across repeats."""
        totals: Dict[str, float] = {}
        for label, elapsed in self._records:
            totals[label] = totals.get(label, 0.0) + elapsed
        return totals
