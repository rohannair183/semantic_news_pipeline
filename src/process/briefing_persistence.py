"""Run briefing generation and persist the result to Postgres (Supabase)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.config.settings import BriefingPersistenceConfig, Settings
from src.process.briefing_generator import BriefingGenerator
from src.process.briefing_result import BriefingGenerationResult
from src.utils.dates import format_day_iso, parse_utc_instant_iso_z, utc_today_date
from src.utils.supabase_db import (
    create_supabase_service_client,
    fetch_latest_briefing_generated_at,
    ensure_briefing_persistence_table,
    insert_briefing_row,
)

SupabaseClientFactory = Callable[[], Any]


def _generated_at_to_utc_day(value: Any) -> date:
    """Normalize a persisted ``generated_at`` value to a UTC calendar day."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.date()
        return value.astimezone(timezone.utc).date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return parse_utc_instant_iso_z(value).date()
    raise TypeError("generated_at must be a datetime or ISO-8601 string")


def briefing_row_from_result(result: BriefingGenerationResult) -> Dict[str, Any]:
    """Map a generation result to a PostgREST row dict (column names match DDL).

    Uses ISO strings for JSON-serializable fields (``generated_at``); Postgres
    accepts them for ``timestamptz`` columns.
    """
    payload = result.to_json_dict()
    return {
        "anchor_day_iso": result.anchor_day_iso,
        "generated_at": result.generated_at_iso,
        "gemini_model": result.gemini_model,
        "briefing_text": result.briefing_text,
        "llm_prompt": result.llm_prompt,
        "topics": payload["topics"],
        "record": payload,
    }


class BriefingPersistenceRunner:
    """Generate a briefing and insert it into the configured Postgres table."""

    def __init__(
        self,
        configuration_root: Optional[Path] = None,
        *,
        briefing_generator: Optional[BriefingGenerator] = None,
        supabase_client_factory: Optional[SupabaseClientFactory] = None,
        postgres_conninfo: Optional[str] = None,
    ) -> None:
        """Load persistence config; optional injectables for tests."""
        Settings.load_repository_dotenv()
        self._persist = Settings.load_briefing_persistence_config(
            configuration_root=configuration_root,
        )
        self._generator = briefing_generator or BriefingGenerator(
            configuration_root=configuration_root,
        )
        self._postgres_conninfo = postgres_conninfo
        self._client_factory: SupabaseClientFactory = (
            supabase_client_factory or create_supabase_service_client
        )

    @property
    def persistence_config(self) -> BriefingPersistenceConfig:
        """Parsed persistence YAML."""
        return self._persist

    def run(self) -> BriefingGenerationResult:
        """Generate, optionally ensure table, insert row, return the in-memory result."""
        result = self._generator.generate()
        if self._persist.ensure_table:
            ensure_briefing_persistence_table(
                self._persist.schema_name,
                self._persist.table_name,
                postgres_conninfo=self._postgres_conninfo,
            )
        client = self._client_factory()
        # If generator provided per-topic contexts, insert one row per topic
        # including the topic name and date_filter. Otherwise insert a single
        # legacy row for the whole briefing.
        if result.topics:
            for topic in result.topics:
                row = briefing_row_from_result(result)
                # enrich with topic-specific columns for relational queries
                row["topic_name"] = topic.topic_name
                row["topic_date_filter"] = topic.date_filter
                insert_briefing_row(
                    client,
                    self._persist.schema_name,
                    self._persist.table_name,
                    row,
                )
        else:
            row = briefing_row_from_result(result)
            insert_briefing_row(
                client,
                self._persist.schema_name,
                self._persist.table_name,
                row,
            )
        return result


def evaluate_briefing_persistence_skip(
    configuration_root: Optional[Path] = None,
    *,
    client_factory: Optional[SupabaseClientFactory] = None,
    current_date: Optional[date] = None,
) -> tuple[bool, str]:
    """Return ``(skip, reason)`` when briefing persistence already ran today."""
    Settings.load_repository_dotenv()
    persist = Settings.load_briefing_persistence_config(
        configuration_root=configuration_root,
    )
    client = (client_factory or create_supabase_service_client)()
    latest_generated_at = fetch_latest_briefing_generated_at(
        client,
        persist.schema_name,
        persist.table_name,
    )
    if latest_generated_at is None:
        return False, ""
    latest_day = _generated_at_to_utc_day(latest_generated_at)
    today = current_date or utc_today_date()
    if latest_day >= today:
        return True, f"briefing persistence already ran on {format_day_iso(latest_day)}"
    return False, ""
