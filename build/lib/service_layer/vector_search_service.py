"""Similarity search against a Supabase Storage vector index using configured embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from src.config.settings import Settings
from src.embeddings.providers import resolve_provider
from src.utils.retry import retry_with_exponential_backoff
from src.utils.timer import Timer
from src.vector_sync.bucket_sync import (
    _default_supabase_client_factory,
    _is_retryable_supabase_api_exception,
)

SupabaseClientFactory = Callable[[], Any]


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


class VectorSearchService:
    """Embed query text and run Storage vector index similarity search."""

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

    def search_by_text(  # pylint: disable=too-many-arguments
        self,
        text: str,
        *,
        top_k: int = 10,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        return_distance: bool = True,
        return_metadata: bool = True,
    ) -> VectorSearchResponse:
        """Embed ``text`` and query the configured vector index.

        Parameters:
            text: Query string; must be non-empty after stripping.
            top_k: Maximum number of matches (passed as ``topK`` to Storage).
            metadata_filter: Optional Storage vector filter mapping.
            return_distance: Whether to request distance scores from Storage.
            return_metadata: Whether to request metadata from Storage.

        Returns:
            Normalized hits (key, distance, metadata).

        Raises:
            ValueError: When ``text`` is empty or ``top_k`` is not positive.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("text must be non-empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_vector = self._embed_query_text(stripped)
        filter_payload: Optional[Dict[str, Any]] = None
        if metadata_filter is not None:
            filter_payload = dict(metadata_filter)

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
