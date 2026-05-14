"""Supabase service-role client helpers and Postgres DDL for briefing persistence."""

from __future__ import annotations

from typing import Any, Mapping, Optional


import psycopg
from psycopg import sql
from psycopg import OperationalError
import supabase

from src.config.settings import Settings
from src.utils.pg_identifiers import validate_pg_identifier


# Deriving a host from SUPABASE_URL (db.<ref>.supabase.co) was intentionally
# removed to avoid CI DNS/IPv6 reachability problems and unused helper code.


def postgres_conninfo_from_supabase_credentials(*, load_dotenv: bool = True) -> str:
    """Return Postgres libpq URI from SUPABASE_POSTGRES_URL.

    This function reads the explicit ``SUPABASE_POSTGRES_URL`` value via
    :meth:`Settings.load_optional_postgres_conninfo` and returns it. Automatic
    construction from ``SUPABASE_URL`` has been removed to avoid DNS/IPv6
    connectivity issues in CI.
    """
    optional = Settings.load_optional_postgres_conninfo(load_dotenv=load_dotenv)
    if optional:
        return optional
    raise RuntimeError(
        (
            "SUPABASE_POSTGRES_URL is not set. For DDL operations set it to the "
            "Supabase Dashboard 'Connection string' (pooler host, e.g. "
            "aws-*-pooler.supabase.com:6543). "
            "Automatic construction from SUPABASE_URL is disabled to avoid CI DNS/IPv6 issues."
        )
    )


def create_supabase_service_client() -> Any:
    """Build a Supabase client using URL + service role key from the environment."""
    url, service_key = Settings.load_supabase_credentials()
    return supabase.create_client(url, service_key)


def ensure_briefing_persistence_table(
    schema_name: str,
    table_name: str,
    *,
    postgres_conninfo: Optional[str] = None,
) -> None:
    """Create the briefing persistence table if it does not already exist.

    If a table with the same name already exists but with a different shape,
    PostgreSQL does not alter it; migrate manually in that case.

    On Supabase, after ``CREATE TABLE`` this sends ``NOTIFY pgrst, 'reload schema'`` so
    PostgREST picks up the new table before the next REST insert (avoids PGRST205).

    Args:
        schema_name: Schema for the table (validated identifier).
        table_name: Table name (validated identifier).
        postgres_conninfo: Optional libpq URI; when omitted, resolved via
            :func:`postgres_conninfo_from_supabase_credentials` (requires
            ``SUPABASE_POSTGRES_URL`` to be set).
    """
    conninfo = postgres_conninfo or postgres_conninfo_from_supabase_credentials()
    schema_sql = validate_pg_identifier(schema_name, field_name="schema_name")
    table_sql = validate_pg_identifier(table_name, field_name="table_name")
    ddl = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {schema}.{table} ("
        "id uuid PRIMARY KEY DEFAULT gen_random_uuid(), "
        "created_at timestamptz NOT NULL DEFAULT now(), "
        "anchor_day_iso text NOT NULL, "
        "generated_at timestamptz NOT NULL, "
        "gemini_model text NOT NULL, "
        "briefing_text text NOT NULL, "
        "llm_prompt text NOT NULL, "
        "topic_name text NULL, "
        "topic_date_filter text NULL, "
        "topics jsonb NOT NULL, "
        "record jsonb NOT NULL"
        ")",
    ).format(
        schema=sql.Identifier(schema_sql),
        table=sql.Identifier(table_sql),
    )
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
                # PostgREST (Supabase API) caches the schema; without this, inserts can fail
                # with PGRST205 until a reload or long cache TTL.
                cur.execute("NOTIFY pgrst, 'reload schema'")
            conn.commit()
            print("Table created successfully")
    except OperationalError as exc:
        err = str(exc)
        low = err.lower()
        print(f"OperationalError during table creation: {exc}")
        # Surface a clearer runtime hint about setting the explicit pooler URI
        # when connectivity fails.
        if (
            "failed to resolve host 'db." in low
            or "could not translate host name 'db." in low
            or "name or service not known" in low
            or "temporary failure in name resolution" in low
        ):
            raise RuntimeError(
                (
                    "Unable to resolve Supabase DB host. If running in CI or a network-"
                    "restricted environment, set SUPABASE_POSTGRES_URL to a reachable Postgres "
                    "URI from the Supabase Dashboard."
                )
            ) from exc
        raise


def insert_briefing_row(
    client: Any,
    schema_name: str,
    table_name: str,
    row: Mapping[str, Any],
) -> Any:
    """Insert one briefing row via PostgREST (``schema.table``)."""
    schema_sql = validate_pg_identifier(schema_name, field_name="schema_name")
    table_sql = validate_pg_identifier(table_name, field_name="table_name")
    return (
        client.schema(schema_sql)
        .table(table_sql)
        .insert([dict(row)])
        .execute()
    )


def fetch_latest_briefing_generated_at(
    client: Any,
    schema_name: str,
    table_name: str,
) -> Optional[Any]:
    """Return the latest persisted briefing ``generated_at`` value, if any."""
    schema_sql = validate_pg_identifier(schema_name, field_name="schema_name")
    table_sql = validate_pg_identifier(table_name, field_name="table_name")
    response = (
        client.schema(schema_sql)
        .table(table_sql)
        .select("generated_at")
        .order("generated_at", desc=True)
        .limit(1)
        .execute()
    )
    rows = getattr(response, "data", None)
    if not rows:
        return None
    first_row = rows[0]
    if not isinstance(first_row, dict):
        return None
    return first_row.get("generated_at")
