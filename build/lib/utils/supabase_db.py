"""Supabase service-role client helpers and Postgres DDL for briefing persistence."""

from __future__ import annotations

from typing import Any, Mapping

import psycopg
from psycopg import sql
from supabase import create_client

from src.config.settings import Settings
from src.utils.pg_identifiers import validate_pg_identifier


def create_supabase_service_client() -> Any:
    """Build a Supabase client using URL + service role key from the environment."""
    url, service_key = Settings.load_supabase_credentials()
    return create_client(url, service_key)


def ensure_briefing_persistence_table(
    database_url: str,
    schema_name: str,
    table_name: str,
) -> None:
    """Create the briefing persistence table if it does not already exist.

    If a table with the same name already exists but with a different shape,
    PostgreSQL does not alter it; migrate manually in that case.

    Args:
        database_url: Postgres connection URI (``DATABASE_URL``).
        schema_name: Schema for the table (validated identifier).
        table_name: Table name (validated identifier).
    """
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
        "topics jsonb NOT NULL, "
        "record jsonb NOT NULL"
        ")",
    ).format(
        schema=sql.Identifier(schema_sql),
        table=sql.Identifier(table_sql),
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()


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
