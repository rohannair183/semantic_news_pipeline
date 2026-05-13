"""Semantic vector search: embed query text, then run Supabase Storage similarity query.

Date bounds on string metadata (e.g. ``source_day`` as ``YYYY-MM-DD``) are sent as a flat
``{"field": {"$in": ["YYYY-MM-DD", ...]}}`` filter. The live ``QueryVectors`` schema rejects
``$gte`` / ``$lte`` with string operands on those fields (it expects numeric bounds there).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.config.settings import Settings
from src.embeddings.providers import resolve_provider
from src.utils.dates import DayInput, coerce_day, format_day_iso
from src.utils.retry import retry_with_exponential_backoff
from src.utils.timer import Timer
from src.vector_sync.bucket_sync import (
    _default_supabase_client_factory,
    _is_retryable_supabase_api_exception,
)

SupabaseClientFactory = Callable[[], Any]

# ``$in`` payload size guard (inclusive calendar days).
_MAX_DAYS_IN_VECTOR_DATE_FILTER = 366


def _inclusive_iso_day_strings(
    date_from: Optional[DayInput],
    date_to: Optional[DayInput],
    *,
    max_days: int = _MAX_DAYS_IN_VECTOR_DATE_FILTER,
) -> List[str]:
    """Return each ``YYYY-MM-DD`` from ``date_from`` through ``date_to`` (inclusive).

    Open-ended bounds collapse to that single calendar day so the filter stays finite.

    Raises:
        ValueError: When ``date_from`` is after ``date_to`` or the span exceeds ``max_days``.
    """
    lo = coerce_day(date_from) if date_from is not None else None
    hi = coerce_day(date_to) if date_to is not None else None
    if lo is None and hi is None:
        return []
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if lo > hi:
        raise ValueError("date_from must be on or before date_to")
    span = (hi - lo).days + 1
    if span > max_days:
        raise ValueError(
            f"date range spans {span} days (from {format_day_iso(lo)} to {format_day_iso(hi)}); "
            f"max supported for vector date $in filter is {max_days} days",
        )
    out: List[str] = []
    cur = lo
    while cur <= hi:
        out.append(format_day_iso(cur))
        cur += timedelta(days=1)
    return out


def _metadata_has_only_logical_root(metadata_filter: Mapping[str, Any]) -> bool:
    return tuple(metadata_filter.keys()) in {("$and",), ("$or",)}


@dataclass(frozen=True)
class VectorSearchHit:
    """One row returned from a vector similarity query."""

    key: str
    distance: Optional[float]
    metadata: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class VectorSearchResponse:
    """Normalized vector query result."""

    hits: Tuple[VectorSearchHit, ...]


def _normalize_query_hits(response: Any) -> Tuple[VectorSearchHit, ...]:
    """Map a storage3 ``QueryVectorsResponse`` (or duck-typed equivalent) to hits."""
    vectors = getattr(response, "vectors", None)
    if not vectors:
        return ()
    hits: List[VectorSearchHit] = []
    for item in vectors:
        key = getattr(item, "key", None)
        if key is None:
            continue
        raw_distance = getattr(item, "distance", None)
        distance = float(raw_distance) if raw_distance is not None else None
        raw_meta = getattr(item, "metadata", None)
        metadata = dict(raw_meta) if isinstance(raw_meta, dict) else None
        hits.append(
            VectorSearchHit(
                key=str(key),
                distance=distance,
                metadata=metadata,
            )
        )
    return tuple(hits)


def _compose_query_filter(
    metadata_filter: Optional[Mapping[str, Any]],
    *,
    date_from: Optional[DayInput],
    date_to: Optional[DayInput],
    date_metadata_key: str,
) -> Optional[Dict[str, Any]]:
    """Merge optional metadata filter with inclusive calendar-day bounds.

    String day metadata (as stored by vector sync) is filtered with ``$in`` over the list
    of ``YYYY-MM-DD`` strings in the inclusive range. ``metadata_filter`` must be a **flat**
    mapping when combined with dates (no top-level ``$and`` / ``$or`` only).

    Raises:
        ValueError: When bounds are invalid, the range is too long, ``date_metadata_key`` is
            empty, the key collides with ``metadata_filter``, or logical-only metadata is
            combined with date bounds.
    """
    key = date_metadata_key.strip()
    if (date_from is not None or date_to is not None) and not key:
        raise ValueError("date_metadata_key must be non-empty when date_from or date_to is set")

    day_strings = _inclusive_iso_day_strings(date_from, date_to)
    if not day_strings:
        if not metadata_filter:
            return None
        return dict(metadata_filter)

    if metadata_filter and _metadata_has_only_logical_root(metadata_filter):
        raise ValueError(
            "metadata_filter with only top-level $and or $or cannot be combined with "
            "date_from/date_to; use a flat field map or omit date bounds",
        )

    date_pred = {key: {"$in": day_strings}}

    if not metadata_filter:
        return date_pred

    meta = dict(metadata_filter)
    if key in meta:
        raise ValueError(
            f"metadata_filter already sets {key!r}; "
            "omit that key or do not pass date_from/date_to",
        )
    return {**meta, **date_pred}


class VectorSearchService:
    """Semantic search: embed query text, run Storage vector index similarity search."""

    def __init__(
        self,
        configuration_root: Optional[Path] = None,
        *,
        supabase_client_factory: Optional[SupabaseClientFactory] = None,
        timer: Optional[Timer] = None,
    ) -> None:
        """Load vector search + embedding YAML; optionally inject client factory and timer.

        Query-time embedding uses the same provider and model as the embedding pipeline.
        Install optional dependency ``sentence-transformers`` (see ``pyproject.toml`` extras)
        when using the default ``sentence_transformers`` provider.
        """
        self._timer = timer or Timer()
        self._vector_config = Settings.load_vector_search_config(
            configuration_root=configuration_root,
        )
        self._embedding_config = Settings.load_embedding_config(
            configuration_root=configuration_root,
        )
        self._supabase_factory = (
            supabase_client_factory or _default_supabase_client_factory
        )

    @property
    def timer(self) -> Timer:
        """Timer used for embed/query sections."""
        return self._timer

    def _embed_query_text(self, text: str) -> List[float]:
        handler = resolve_provider(
            self._embedding_config.provider,
            self._embedding_config.model_name,
            timer=self._timer,
        )
        with self._timer.section("vector_search.embed_query"):
            vectors = handler.embed(
                [text],
                batch_size=self._embedding_config.batch_size,
            )
        if not vectors:
            raise RuntimeError("Embedding provider returned no vector for query text")
        return [float(x) for x in vectors[0]]

    def semantic_search(  # pylint: disable=too-many-arguments
        self,
        query: str,
        *,
        top_k: int = 10,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        date_from: Optional[DayInput] = None,
        date_to: Optional[DayInput] = None,
        date_metadata_key: Optional[str] = None,
        return_distance: bool = True,
        return_metadata: bool = True,
    ) -> VectorSearchResponse:
        """Semantic search: embed ``query``, then ``index.query`` (Supabase vector flow).

        Same behavior as :meth:`search_by_text`; name matches Supabase semantic search docs.
        """
        return self._semantic_vector_query(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            date_from=date_from,
            date_to=date_to,
            date_metadata_key=date_metadata_key,
            return_distance=return_distance,
            return_metadata=return_metadata,
        )

    def search_by_text(  # pylint: disable=too-many-arguments
        self,
        text: str,
        *,
        top_k: int = 10,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        date_from: Optional[DayInput] = None,
        date_to: Optional[DayInput] = None,
        date_metadata_key: Optional[str] = None,
        return_distance: bool = True,
        return_metadata: bool = True,
    ) -> VectorSearchResponse:
        """Semantic vector search: embed ``text``, then query the configured vector index.

        Parameters:
            text: Query string; must be non-empty after stripping.
            top_k: Maximum number of matches (passed as ``topK`` to Storage).
            metadata_filter: Optional filter; flat map when no date bounds (see Supabase docs).
            date_from: Optional inclusive lower bound (calendar day); with ``date_to`` omitted,
                only that day is included in the ``$in`` list.
            date_to: Optional inclusive upper bound; with ``date_from`` omitted, only that day
                is included in the ``$in`` list.
            date_metadata_key: Metadata field for date bounds; defaults to configured
                ``vector_search.date_metadata_key`` (normally ``source_day``).
            return_distance: Whether to request distance scores from Storage.
            return_metadata: Whether to request metadata from Storage.

        Returns:
            Normalized hits (key, distance, metadata).

        Raises:
            ValueError: When ``text`` is empty or ``top_k`` is not positive.
        """
        return self._semantic_vector_query(
            text,
            top_k=top_k,
            metadata_filter=metadata_filter,
            date_from=date_from,
            date_to=date_to,
            date_metadata_key=date_metadata_key,
            return_distance=return_distance,
            return_metadata=return_metadata,
        )

    def _semantic_vector_query(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        text: str,
        *,
        top_k: int = 10,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        date_from: Optional[DayInput] = None,
        date_to: Optional[DayInput] = None,
        date_metadata_key: Optional[str] = None,
        return_distance: bool = True,
        return_metadata: bool = True,
    ) -> VectorSearchResponse:
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self._embed_query_text(stripped)
        resolved_date_key = (
            date_metadata_key.strip()
            if isinstance(date_metadata_key, str) and date_metadata_key.strip()
            else self._vector_config.date_metadata_key
        )
        filter_payload = _compose_query_filter(
            metadata_filter,
            date_from=date_from,
            date_to=date_to,
            date_metadata_key=resolved_date_key,
        )

        client = self._supabase_factory()
        bucket_scope = client.storage.vectors().from_(self._vector_config.bucket_name)
        vector_index = bucket_scope.index(self._vector_config.index_name)

        def _query() -> Any:
            return vector_index.query(
                query_vector={"float32": query_vector},
                topK=top_k,
                filter=filter_payload,
                return_distance=return_distance,
                return_metadata=return_metadata,
            )

        with self._timer.section("vector_search.storage_query"):
            raw = retry_with_exponential_backoff(
                _query,
                is_retryable=_is_retryable_supabase_api_exception,
            )
        return VectorSearchResponse(hits=_normalize_query_hits(raw))
