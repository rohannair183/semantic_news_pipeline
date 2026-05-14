"""Additional DNS-resolution variant tests for src.utils.supabase_db."""

from __future__ import annotations

import unittest

import psycopg

from src.utils import supabase_db as supabase_db_module


class TestDnsErrorVariants(unittest.TestCase):
    """Ensure different DNS error message variants are wrapped with a helpful hint."""

    def test_could_not_translate_host_name_variant(self) -> None:
        with unittest.mock.patch(
            "psycopg.connect",
            side_effect=psycopg.OperationalError("could not translate host name 'db.x.supabase.co'"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                supabase_db_module.ensure_briefing_persistence_table(
                    "public", "t", postgres_conninfo="postgresql://unused"
                )
        self.assertIn("SUPABASE_POSTGRES_URL", str(ctx.exception))

    def test_temporary_failure_in_name_resolution_variant(self) -> None:
        with unittest.mock.patch(
            "psycopg.connect",
            side_effect=psycopg.OperationalError("temporary failure in name resolution"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                supabase_db_module.ensure_briefing_persistence_table(
                    "public", "t", postgres_conninfo="postgresql://unused"
                )
        self.assertIn("SUPABASE_POSTGRES_URL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
