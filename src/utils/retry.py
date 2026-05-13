"""Shared retry helpers for transient external API failures."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class ExponentialBackoffConfig:
    """Configuration for exponential backoff retry strategy."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.1
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 1.0
    sleep_fn: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be non-negative")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be at least 1.0")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be non-negative")


def retry_with_exponential_backoff(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    **kwargs: Any,
) -> T:
    """Run ``operation`` with exponential backoff for retryable errors.

    Keyword arguments: max_attempts, initial_delay_seconds, backoff_multiplier,
    max_delay_seconds, sleep_fn (see ExponentialBackoffConfig for defaults).
    """
    # Build config from kwargs with defaults
    config = ExponentialBackoffConfig(
        max_attempts=int(kwargs.get("max_attempts", 3)),
        initial_delay_seconds=float(kwargs.get("initial_delay_seconds", 0.1)),
        backoff_multiplier=float(kwargs.get("backoff_multiplier", 2.0)),
        max_delay_seconds=float(kwargs.get("max_delay_seconds", 1.0)),
        sleep_fn=kwargs.get("sleep_fn", time.sleep),
    )

    delay_seconds = config.initial_delay_seconds
    for attempt_number in range(1, config.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt_number >= config.max_attempts or not is_retryable(exc):
                raise
            config.sleep_fn(delay_seconds)
            delay_seconds = min(
                delay_seconds * config.backoff_multiplier,
                config.max_delay_seconds,
            )

    raise RuntimeError("unreachable retry state")  # pragma: no cover
