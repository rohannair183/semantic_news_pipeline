"""Unit tests for the article normalizer smoke script."""

import io
import unittest
from contextlib import redirect_stdout
from datetime import date
from unittest.mock import patch

import smoke_article_normalizer


class TestSmokeArticleNormalizer(unittest.TestCase):
    """Validate the smoke script uses a native date for normalization."""

    @patch("smoke_article_normalizer.utc_today_date", return_value=date(2026, 4, 29))
    @patch("smoke_article_normalizer.ArticleNormalizer")
    def test_main_uses_date_for_normalization(self, mock_normalizer_class, _mock_today):
        """main: passes a date object to the normalizer and logs the compact day token."""
        mock_normalizer = mock_normalizer_class.return_value
        mock_normalizer.normalize_day_to_parquet.return_value = {}

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            smoke_article_normalizer.main()

        mock_normalizer.normalize_day_to_parquet.assert_called_once_with(date(2026, 4, 29))
        self.assertIn("Normalizing checkpoints for 20260429...", stdout.getvalue())
