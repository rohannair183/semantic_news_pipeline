"""Supabase service-role client helpers and Postgres DDL for briefing persistence."""

from __future__ import annotations

from typing import Any, Mapping, Optional
from urllib.parse import quote_plus, urlparse

import psycopg
from psycopg import sql
from psycopg import OperationalError
import supabase

from src.config.settings import Settings
from src.utils.pg_identifiers import validate_pg_identifier


def parse_supabase_project_ref(supabase_api_url: str) -> str:
    """Extract the Supabase project ref from an API base URL.

    Expects a host shaped like ``<project_ref>.supabase.co`` (optionally with port).

    Raises:
        ValueError: When the URL does not match the expected Supabase host pattern.
    """
    parsed = urlparse(supabase_api_url.strip())
    host = (parsed.hostname or "").strip().lower()
    suffix = ".supabase.co"
    if not host.endswith(suffix):
        raise ValueError(
            "SUPABASE_URL hostname must end with .supabase.co (e.g. https://<ref>.supabase.co)",
        )
    ref = host[: -len(suffix)]
    if not ref or "." in ref:
        raise ValueError(
            "SUPABASE_URL must use host <project_ref>.supabase.co with a single project ref label",
        )
    return ref


def postgres_conninfo_from_supabase_credentials(*, load_dotenv: bool = True) -> str:
    """Resolve a libpq connection URI for direct Postgres access (DDL on Supabase).

    If ``SUPABASE_POSTGRES_URL`` or ``DATABASE_URL`` is set (non-empty), returns that
    string unchanged—use the Dashboard **Database** URI when ``db.<ref>.supabase.co``
    does not resolve or auth fails.

    Otherwise builds ``postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres``
    from :meth:`Settings.load_supabase_credentials` (``SUPABASE_URL`` +
    ``SUPABASE_SERVICE_ROLE_KEY``); the service role JWT is URL-encoded as the
    password and ``sslmode=require`` is appended.

    If your project rejects JWT-as-password for the ``postgres`` user, set one of the
    env URIs above from the dashboard, or set ``ensure_table: false`` in YAML and
    apply DDL manually.
    """
    optional = Settings.load_optional_postgres_conninfo(load_dotenv=load_dotenv)
    if optional:
        return optional
    api_url, service_key = Settings.load_supabase_credentials(load_dotenv=load_dotenv)
    ref = parse_supabase_project_ref(api_url)
    password = quote_plus(service_key)
    return (
        f"postgresql://postgres:{password}@db.{ref}.supabase.co:5432/postgres"
        "?sslmode=require"
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
            :func:`postgres_conninfo_from_supabase_credentials` (optional env URI, else
            derived from ``SUPABASE_URL`` + service role).
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
        err = str(exc).lower()
        # Common DNS failure when db.<ref>.supabase.co cannot be resolved; give
        # a clearer hint to configure SUPABASE_POSTGRES_URL or DATABASE_URL.
        if "failed to resolve host" in err and "db." in err and ".supabase.co" in err:
            raise RuntimeError(
                "Unable to resolve Supabase DB host. Set SUPABASE_POSTGRES_URL or "
                "DATABASE_URL to the dashboard connection string."
            ) from exc
        # Otherwise, re-raise the original OperationalError
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
