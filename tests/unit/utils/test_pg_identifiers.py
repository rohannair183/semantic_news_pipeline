"""Unit tests for pg_identifiers.validate_pg_identifier."""

import unittest

from src.utils.pg_identifiers import validate_pg_identifier


class TestValidatePgIdentifier(unittest.TestCase):
    """This class tests validate_pg_identifier."""

    def test_accepts_letters_digits_underscore(self) -> None:
        """validate_pg_identifier: accepts valid PostgreSQL-style names."""
        self.assertEqual(validate_pg_identifier("public", field_name="x"), "public")
        self.assertEqual(validate_pg_identifier("_t1", field_name="x"), "_t1")
        self.assertEqual(validate_pg_identifier("T_2", field_name="x"), "T_2")

    def test_strips_whitespace(self) -> None:
        """validate_pg_identifier: trims surrounding whitespace."""
        self.assertEqual(validate_pg_identifier("  foo  ", field_name="x"), "foo")

    def test_rejects_empty(self) -> None:
        """validate_pg_identifier: rejects empty string."""
        with self.assertRaises(ValueError):
            validate_pg_identifier("", field_name="f")
        with self.assertRaises(ValueError):
            validate_pg_identifier("   ", field_name="f")

    def test_rejects_invalid_characters(self) -> None:
        """validate_pg_identifier: rejects hyphens and other invalid tokens."""
        with self.assertRaises(ValueError):
            validate_pg_identifier("bad-name", field_name="f")
        with self.assertRaises(ValueError):
            validate_pg_identifier("1no", field_name="f")
