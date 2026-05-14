# pyright: reportPrivateUsage=false
"""Unit tests for src.utils.supabase_db."""

import os
import unittest
from unittest.mock import MagicMock, patch

import psycopg

from src.utils import supabase_db as supabase_db_module
from src.utils.supabase_db import (
    create_supabase_service_client,
    fetch_latest_briefing_generated_at,
    ensure_briefing_persistence_table,
    insert_briefing_row,
    parse_supabase_project_ref,
    postgres_conninfo_from_supabase_credentials,
)


class TestParseSupabaseProjectRef(unittest.TestCase):
    """This class tests parse_supabase_project_ref."""

    def test_extracts_ref_from_https_url(self) -> None:
        """parse_supabase_project_ref: reads project ref from API URL."""
        self.assertEqual(
            parse_supabase_project_ref("https://abcdefghijklmnop.supabase.co"),
            "abcdefghijklmnop",
        )

    def test_accepts_whitespace_and_trailing_slash(self) -> None:
        """parse_supabase_project_ref: tolerates surrounding whitespace."""
        self.assertEqual(
            parse_supabase_project_ref("  https://myproj.supabase.co/  "),
            "myproj",
        )

    def test_rejects_non_supabase_host(self) -> None:
        """parse_supabase_project_ref: raises when hostname is not *.supabase.co."""
        with self.assertRaises(ValueError) as ctx:
            parse_supabase_project_ref("https://example.com")
        self.assertIn("SUPABASE_URL", str(ctx.exception))

    def test_rejects_nested_subdomain(self) -> None:
        """parse_supabase_project_ref: raises when ref label contains a dot."""
        with self.assertRaises(ValueError):
            parse_supabase_project_ref("https://a.b.supabase.co")


class TestPostgresConninfoFromSupabaseCredentials(unittest.TestCase):
    """This class tests postgres_conninfo_from_supabase_credentials."""

    def test_uses_supabase_postgres_url_when_set(self) -> None:
        """postgres_conninfo_from_supabase_credentials: prefers SUPABASE_POSTGRES_URL."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_POSTGRES_URL": "postgresql://pool:6543/postgres",
                "SUPABASE_URL": "https://abcxyz.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "ignored",
            },
            clear=True,
        ):
            uri = postgres_conninfo_from_supabase_credentials(load_dotenv=False)
        self.assertEqual(uri, "postgresql://pool:6543/postgres")

    def test_uses_database_url_when_postgres_url_missing(self) -> None:
        """postgres_conninfo_from_supabase_credentials: uses DATABASE_URL when set."""
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://dbhost/postgres",
                "SUPABASE_URL": "https://abcxyz.supabase.co",
                "SUPABASE_SERVICE_ROLE_KEY": "k",
            },
            clear=True,
        ):
            uri = postgres_conninfo_from_supabase_credentials(load_dotenv=False)
        self.assertEqual(uri, "postgresql://dbhost/postgres")

    def test_builds_direct_postgres_uri(self) -> None:
        """postgres_conninfo_from_supabase_credentials: encodes key and sets sslmode."""
        with patch.object(
            supabase_db_module.Settings,
            "load_supabase_credentials",
            return_value=("https://abcxyz.supabase.co", "secret&token"),
        ):
            with patch.object(
                supabase_db_module.Settings,
                "load_optional_postgres_conninfo",
                return_value=None,
            ):
                uri = postgres_conninfo_from_supabase_credentials(load_dotenv=False)
        self.assertTrue(uri.startswith("postgresql://postgres:"))
        self.assertIn("secret%26token", uri)
        self.assertIn("@db.abcxyz.supabase.co:5432/postgres", uri)
        self.assertIn("sslmode=require", uri)


class TestCreateSupabaseServiceClient(unittest.TestCase):
    """This class tests create_supabase_service_client."""

    def test_builds_client_from_settings(self) -> None:
        """create_supabase_service_client: wires URL and service key into create_client."""
        sentinel = MagicMock()
        fake_create = MagicMock(return_value=sentinel)
        with patch.object(
            supabase_db_module.Settings,
            "load_supabase_credentials",
            return_value=("https://test.supabase.co", "srv"),
        ) as mocked_creds:
            with patch("supabase.create_client", fake_create):
                client = create_supabase_service_client()
        self.assertIs(client, sentinel)
        mocked_creds.assert_called_once_with()
        fake_create.assert_called_once_with("https://test.supabase.co", "srv")


class TestEnsureBriefingPersistenceTable(unittest.TestCase):
    """This class tests ensure_briefing_persistence_table."""

    @patch("psycopg.connect")
    def test_executes_create_table_ddl(self, mock_connect: MagicMock) -> None:
        """ensure_briefing_persistence_table: opens psycopg and runs CREATE TABLE IF NOT EXISTS."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        cur_ctx = MagicMock()
        cur_ctx.__enter__.return_value = mock_cursor
        cur_ctx.__exit__.return_value = False
        mock_conn.cursor.return_value = cur_ctx
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = mock_conn
        conn_ctx.__exit__.return_value = False
        mock_connect.return_value = conn_ctx

        ensure_briefing_persistence_table(
            "public",
            "news_briefings",
            postgres_conninfo="postgresql://u:p@localhost/db",
        )
        mock_connect.assert_called_once_with("postgresql://u:p@localhost/db")
        self.assertEqual(mock_cursor.execute.call_count, 2)
        first_sql = str(mock_cursor.execute.call_args_list[0][0][0])
        self.assertIn("CREATE TABLE IF NOT EXISTS", first_sql)
        self.assertIn("news_briefings", first_sql)
        second_sql = mock_cursor.execute.call_args_list[1][0][0]
        self.assertEqual(second_sql, "NOTIFY pgrst, 'reload schema'")
        mock_conn.commit.assert_called_once()

    @patch("psycopg.connect")
    @patch.object(
        supabase_db_module,
        "postgres_conninfo_from_supabase_credentials",
        return_value="postgresql://built",
    )
    def test_uses_built_conninfo_when_kwarg_omitted(
        self,
        mock_built: MagicMock,
        mock_connect: MagicMock,
    ) -> None:
        """ensure_briefing_persistence_table: derives conninfo from credentials when omitted."""
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        cur_ctx = MagicMock()
        cur_ctx.__enter__.return_value = mock_cursor
        cur_ctx.__exit__.return_value = False
        mock_conn.cursor.return_value = cur_ctx
        conn_ctx = MagicMock()
        conn_ctx.__enter__.return_value = mock_conn
        conn_ctx.__exit__.return_value = False
        mock_connect.return_value = conn_ctx

        ensure_briefing_persistence_table("public", "t")
        mock_built.assert_called_once()
        mock_connect.assert_called_once_with("postgresql://built")
        self.assertEqual(mock_cursor.execute.call_count, 2)

    def test_dns_resolution_failure_is_runtime_error_with_hint(self) -> None:
        """ensure_briefing_persistence_table: wraps unresolvable db.* Supabase host errors."""
        with patch(
            "psycopg.connect",
            side_effect=psycopg.OperationalError(
                "failed to resolve host 'db.x.supabase.co'",
            ),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                ensure_briefing_persistence_table(
                    "public",
                    "t",
                    postgres_conninfo="postgresql://unused",
                )
        self.assertIn("SUPABASE_POSTGRES_URL", str(ctx.exception))

    def test_other_operational_error_passes_through(self) -> None:
        """ensure_briefing_persistence_table: re-raises OperationalError without DNS pattern."""
        with patch(
            "psycopg.connect",
            side_effect=psycopg.OperationalError("password authentication failed"),
        ):
            with self.assertRaises(psycopg.OperationalError):
                ensure_briefing_persistence_table(
                    "public",
                    "t",
                    postgres_conninfo="postgresql://x",
                )


class TestInsertBriefingRow(unittest.TestCase):
    """This class tests insert_briefing_row."""

    def test_chains_schema_table_insert_execute(self) -> None:
        """insert_briefing_row: uses PostgREST chain with one row dict."""
        execute_ret = MagicMock()
        insert_chain = MagicMock()
        insert_chain.insert.return_value.execute.return_value = execute_ret
        table_chain = MagicMock()
        table_chain.table.return_value = insert_chain
        client = MagicMock()
        client.schema.return_value = table_chain

        out = insert_briefing_row(
            client,
            "public",
            "briefings",
            {"anchor_day_iso": "2026-05-12", "briefing_text": "x"},
        )
        self.assertIs(out, execute_ret)
        client.schema.assert_called_once_with("public")
        table_chain.table.assert_called_once_with("briefings")
        insert_chain.insert.assert_called_once_with(
            [{"anchor_day_iso": "2026-05-12", "briefing_text": "x"}],
        )


class TestFetchLatestBriefingGeneratedAt(unittest.TestCase):
    """This class tests fetch_latest_briefing_generated_at."""

    def test_returns_latest_generated_at_value(self) -> None:
        """fetch_latest_briefing_generated_at: reads the newest generated_at field."""
        execute_ret = MagicMock(data=[{"generated_at": "2026-05-13T00:00:00Z"}])
        query_chain = MagicMock()
        select_chain = query_chain.select.return_value
        order_chain = select_chain.order.return_value
        limit_chain = order_chain.limit.return_value
        limit_chain.execute.return_value = execute_ret
        table_chain = MagicMock()
        table_chain.table.return_value = query_chain
        client = MagicMock()
        client.schema.return_value = table_chain

        value = fetch_latest_briefing_generated_at(
            client,
            "public",
            "news_briefings",
        )

        self.assertEqual(value, "2026-05-13T00:00:00Z")
        client.schema.assert_called_once_with("public")
        table_chain.table.assert_called_once_with("news_briefings")
        query_chain.select.assert_called_once_with("generated_at")
        select_chain.order.assert_called_once_with("generated_at", desc=True)
        order_chain.limit.assert_called_once_with(1)

    def test_returns_none_when_no_rows(self) -> None:
        """fetch_latest_briefing_generated_at: returns None when no rows are found."""
        execute_ret = MagicMock(data=[])
        query_chain = MagicMock()
        select_chain = query_chain.select.return_value
        order_chain = select_chain.order.return_value
        limit_chain = order_chain.limit.return_value
        limit_chain.execute.return_value = execute_ret
        table_chain = MagicMock()
        table_chain.table.return_value = query_chain
        client = MagicMock()
        client.schema.return_value = table_chain

        value = fetch_latest_briefing_generated_at(
            client,
            "public",
            "news_briefings",
        )
        self.assertIsNone(value)

    def test_returns_none_when_first_row_not_dict(self) -> None:
        """fetch_latest_briefing_generated_at: returns None when first row is unexpected type."""
        execute_ret = MagicMock(data=["not-a-dict"])
        query_chain = MagicMock()
        select_chain = query_chain.select.return_value
        order_chain = select_chain.order.return_value
        limit_chain = order_chain.limit.return_value
        limit_chain.execute.return_value = execute_ret
        table_chain = MagicMock()
        table_chain.table.return_value = query_chain
        client = MagicMock()
        client.schema.return_value = table_chain

        value = fetch_latest_briefing_generated_at(
            client,
            "public",
            "news_briefings",
        )
        self.assertIsNone(value)
