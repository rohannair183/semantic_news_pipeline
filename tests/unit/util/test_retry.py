"""Unit tests for exponential retry utilities."""

import unittest
from unittest.mock import Mock

from src.utils.retry import retry_with_exponential_backoff


class TestRetryWithExponentialBackoff(unittest.TestCase):
    """This class tests retry_with_exponential_backoff."""

    def test_returns_immediately_on_first_success(self):
        """retry_with_exponential_backoff: returns operation output without sleeping."""
        operation = Mock(return_value="ok")
        sleep_fn = Mock()

        result = retry_with_exponential_backoff(
            operation,
            is_retryable=lambda _: True,
            sleep_fn=sleep_fn,
        )

        self.assertEqual(result, "ok")
        operation.assert_called_once_with()
        sleep_fn.assert_not_called()

    def test_retries_then_succeeds(self):
        """retry_with_exponential_backoff: retries transient failures then returns."""
        operation = Mock(side_effect=[RuntimeError("temporary"), "ok"])
        sleep_fn = Mock()

        result = retry_with_exponential_backoff(
            operation,
            is_retryable=lambda exc: isinstance(exc, RuntimeError),
            initial_delay_seconds=0.2,
            backoff_multiplier=2.0,
            sleep_fn=sleep_fn,
        )

        self.assertEqual(result, "ok")
        self.assertEqual(operation.call_count, 2)
        sleep_fn.assert_called_once_with(0.2)

    def test_raises_immediately_for_non_retryable_error(self):
        """retry_with_exponential_backoff: does not retry non-retryable failures."""
        operation = Mock(side_effect=ValueError("bad input"))
        sleep_fn = Mock()

        with self.assertRaises(ValueError):
            retry_with_exponential_backoff(
                operation,
                is_retryable=lambda _: False,
                sleep_fn=sleep_fn,
            )

        operation.assert_called_once_with()
        sleep_fn.assert_not_called()

    def test_raises_after_max_attempts_with_exponential_sleeps(self):
        """retry_with_exponential_backoff: caps retries and applies exponential delays."""
        operation = Mock(side_effect=RuntimeError("still failing"))
        sleep_fn = Mock()

        with self.assertRaises(RuntimeError):
            retry_with_exponential_backoff(
                operation,
                is_retryable=lambda _: True,
                max_attempts=3,
                initial_delay_seconds=0.1,
                backoff_multiplier=2.0,
                max_delay_seconds=10.0,
                sleep_fn=sleep_fn,
            )

        self.assertEqual(operation.call_count, 3)
        self.assertEqual(sleep_fn.call_count, 2)
        self.assertEqual([call.args[0] for call in sleep_fn.call_args_list], [0.1, 0.2])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
