"""Structured outputs from briefing generation (DB / document-store friendly)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple

from src.enums.briefing_date_filter import BriefingDateFilter
from src.service_layer.vector_search_service import VectorSearchHit, VectorSearchResponse
from src.utils.dates import format_day_iso


def utc_now_iso_z() -> str:
    """UTC instant as ISO-8601 with ``Z`` suffix (stable for logs and DB)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BriefingVectorHitRecord:
    """One retrieved vector row, ready for JSON or relational child rows."""

    key: str
    distance: Optional[float]
    metadata: Mapping[str, Any]

    @classmethod
    def from_vector_hit(cls, hit: VectorSearchHit) -> "BriefingVectorHitRecord":
        """Normalize a :class:`VectorSearchHit` into a JSON-friendly record."""
        meta = dict(hit.metadata) if hit.metadata else {}
        return cls(key=hit.key, distance=hit.distance, metadata=meta)


@dataclass(frozen=True)
class BriefingTopicContext:
    """Per-topic retrieval context: query, date window, and hits."""

    topic_name: str
    vector_query: str
    date_filter: str
    date_from_iso: str
    date_to_iso: str
    hits: Tuple[BriefingVectorHitRecord, ...]

    @classmethod
    def from_topic_search(  # pylint: disable=too-many-arguments
        cls,
        *,
        topic_name: str,
        vector_query: str,
        date_filter: BriefingDateFilter,
        date_from: date,
        date_to: date,
        response: VectorSearchResponse,
    ) -> "BriefingTopicContext":
        """Build context from a topic label, bounds, filter, and vector response."""
        hit_records = tuple(
            BriefingVectorHitRecord.from_vector_hit(h) for h in response.hits
        )
        return cls(
            topic_name=topic_name,
            vector_query=vector_query,
            date_filter=date_filter.value,
            date_from_iso=format_day_iso(date_from),
            date_to_iso=format_day_iso(date_to),
            hits=hit_records,
        )


@dataclass(frozen=True)
class BriefingGenerationResult:
    """Full run output: LLM text, prompt, and per-topic retrieval (persist as JSON or tables)."""

    briefing_text: str
    llm_prompt: str
    gemini_model: str
    anchor_day_iso: str
    generated_at_iso: str
    topics: Tuple[BriefingTopicContext, ...]

    def to_json_dict(self) -> Dict[str, Any]:
        """Nested dict safe for ``json.dumps`` (e.g. JSONB column or document store)."""
        topics_payload: List[Dict[str, Any]] = []
        for topic in self.topics:
            hits_payload = [
                {"key": h.key, "distance": h.distance, "metadata": dict(h.metadata)}
                for h in topic.hits
            ]
            topics_payload.append(
                {
                    "topic_name": topic.topic_name,
                    "vector_query": topic.vector_query,
                    "date_filter": topic.date_filter,
                    "date_from_iso": topic.date_from_iso,
                    "date_to_iso": topic.date_to_iso,
                    "hits": hits_payload,
                },
            )
        return {
            "briefing_text": self.briefing_text,
            "llm_prompt": self.llm_prompt,
            "gemini_model": self.gemini_model,
            "anchor_day_iso": self.anchor_day_iso,
            "generated_at_iso": self.generated_at_iso,
            "topics": topics_payload,
        }
