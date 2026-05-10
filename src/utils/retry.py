"""Shared retry helpers for transient external API failures."""

from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")


def retry_with_exponential_backoff(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_attempts: int = 3,
    initial_delay_seconds: float = 0.1,
    backoff_multiplier: float = 2.0,
    max_delay_seconds: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``operation`` with exponential backoff for retryable errors."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if initial_delay_seconds < 0:
        raise ValueError("initial_delay_seconds must be non-negative")
    if backoff_multiplier < 1.0:
        raise ValueError("backoff_multiplier must be at least 1.0")
    if max_delay_seconds < 0:
        raise ValueError("max_delay_seconds must be non-negative")

    delay_seconds = initial_delay_seconds
    for attempt_number in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt_number >= max_attempts or not is_retryable(exc):
                raise
            sleep_fn(delay_seconds)
            delay_seconds = min(delay_seconds * backoff_multiplier, max_delay_seconds)

    raise RuntimeError("unreachable retry state")
