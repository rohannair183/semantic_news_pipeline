"""Gemini-backed briefings using vector search context from the service layer."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

from google import genai

from src.config.settings import BriefingGeneratorConfig, BriefingTopicSpec, Settings
from src.enums.briefing_date_filter import BriefingDateFilter
from src.process.briefing_result import (
    BriefingGenerationResult,
    BriefingTopicContext,
    BriefingVectorHitRecord,
    utc_now_iso_z,
)
from src.service_layer.vector_search_service import VectorSearchService
from src.utils.dates import (
    date_range_last_n_calendar_days_inclusive,
    date_range_month_to_date,
    date_range_single_calendar_day,
    format_day_iso,
    utc_today_date,
)
from src.utils.timer import Timer


class BriefingGenerator:
    """Load briefing YAML, retrieve chunks via ``VectorSearchService``, call Gemini."""

    def __init__(
        self,
        configuration_root: Optional[Path] = None,
        *,
        vector_search: Optional[VectorSearchService] = None,
        timer: Optional[Timer] = None,
        reference_date: Optional[date] = None,
    ) -> None:
        """Load config; resolve API key from ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``.

        Repository ``.env`` is merged via ``Settings.load_repository_dotenv()`` before
        reading credentials (same behavior as other optional env bootstrap).

        Parameters:
            configuration_root: Optional YAML root (defaults to repo ``configuration/``).
            vector_search: Injectable search client (defaults to ``VectorSearchService``).
            timer: Shared timer for embed/query sections when using the default client.
            reference_date: Optional UTC calendar anchor for date filters (tests); default
                is :func:`src.utils.dates.utc_today_date`.
        """
        Settings.load_repository_dotenv()
        self._config = Settings.load_briefing_generator_config(
            configuration_root=configuration_root,
        )
        self._api_key = self._resolve_gemini_api_key()
        self._timer = timer or Timer()
        self._reference_date = reference_date
        self._vector_search = vector_search or VectorSearchService(
            configuration_root=configuration_root,
            timer=self._timer,
        )

    @property
    def config(self) -> BriefingGeneratorConfig:
        """Parsed briefing generator configuration."""
        return self._config

    @staticmethod
    def _resolve_gemini_api_key() -> str:
        raw = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not raw or not str(raw).strip():
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY must be set and non-empty",
            )
        return str(raw).strip()

    def _anchor_day(self) -> date:
        if self._reference_date is not None:
            return self._reference_date
        return utc_today_date()

    def _date_bounds_for_topic(self, topic: BriefingTopicSpec) -> Tuple[date, date]:
        anchor = self._anchor_day()
        if topic.date_filter == BriefingDateFilter.DAILY:
            return date_range_single_calendar_day(anchor)
        if topic.date_filter == BriefingDateFilter.WEEKLY:
            return date_range_last_n_calendar_days_inclusive(anchor, 7)
        if topic.date_filter == BriefingDateFilter.MONTHLY:
            return date_range_month_to_date(anchor)
        raise ValueError(f"unsupported briefing date filter: {topic.date_filter!r}")

    def generate(self) -> BriefingGenerationResult:
        """Retrieve context per topic, call Gemini, return structured result.

        Use :meth:`BriefingGenerationResult.to_json_dict` for JSONB / document storage, or map
        fields to relational tables (run header plus ``topics`` and nested ``hits``).
        """
        contexts: List[BriefingTopicContext] = []
        for topic in self._config.topics:
            date_from, date_to = self._date_bounds_for_topic(topic)
            response = self._vector_search.search_by_text(
                topic.vector_query,
                top_k=self._config.vector_top_k,
                date_from=date_from,
                date_to=date_to,
            )
            contexts.append(
                BriefingTopicContext.from_topic_search(
                    topic_name=topic.name,
                    vector_query=topic.vector_query,
                    date_filter=topic.date_filter,
                    date_from=date_from,
                    date_to=date_to,
                    response=response,
                ),
            )
        topic_tuple = tuple(contexts)
        prompt = self._build_prompt(topic_tuple)
        text = self._generate_with_gemini(prompt)
        return BriefingGenerationResult(
            briefing_text=text,
            llm_prompt=prompt,
            gemini_model=self._config.model,
            anchor_day_iso=format_day_iso(self._anchor_day()),
            generated_at_iso=utc_now_iso_z(),
            topics=topic_tuple,
        )

    def _format_hit_records(self, hits: Tuple[BriefingVectorHitRecord, ...]) -> str:
        lines: List[str] = []
        for hit in hits:
            meta = json.dumps(dict(hit.metadata), sort_keys=True) if hit.metadata else "{}"
            dist_part = "" if hit.distance is None else f", distance={hit.distance}"
            lines.append(f"  - key={hit.key}{dist_part}, metadata={meta}")
        return "\n".join(lines) if lines else "  (no hits)"

    def _build_prompt(self, contexts: Tuple[BriefingTopicContext, ...]) -> str:
        header = (
            "You are a news analyst. Using ONLY the retrieved chunk references below "
            "(keys and metadata), write a concise multi-topic briefing. "
            "Ground statements in the provided metadata when it contains text; "
            "do not invent facts not supported by the references.\n"
        )
        parts: List[str] = [header, "Retrieved context:", ""]
        for ctx in contexts:
            range_label = f"{ctx.date_from_iso} to {ctx.date_to_iso}"
            parts.append(
                f"## Topic: {ctx.topic_name} (date_filter={ctx.date_filter}, "
                f"source_day range {range_label})\n",
            )
            parts.append(self._format_hit_records(ctx.hits))
            parts.append("")
        parts.append(
            "Produce a structured briefing with one section per topic.",
        )
        return "\n".join(parts)

    def _generate_with_gemini(self, prompt: str) -> str:
        client = genai.Client(api_key=self._api_key)

        response = client.models.generate_content(
            model=self._config.model,
            contents=prompt,
        )
        try:
            text = response.text
        except ValueError as exc:
            raise RuntimeError("Gemini returned no usable text") from exc
        if text is None or not str(text).strip():
            raise RuntimeError("Gemini returned empty text")
        return str(text).strip()
