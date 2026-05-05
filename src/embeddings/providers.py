"""Embedding provider registry and handlers used by Embedder."""

from __future__ import annotations

from typing import List, Optional, Protocol

from src.enums.embedding_provider import EmbeddingProvider


class EmbeddingProviderHandler(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for embedding provider handlers used by the registry."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""


class SentenceTransformerHandler:  # pylint: disable=too-few-public-methods
    """Handler that wraps the sentence-transformers library."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load_model(self) -> object:
        if self._model is None:
            # Lazy import so the dependency is only required at runtime.
            from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel,import-error
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Encode ``texts`` into embedding vectors."""
        model = self._load_model()
        embeddings = model.encode(texts, show_progress_bar=False)  # type: ignore[union-attr]
        return [list(map(float, vec)) for vec in embeddings]


def resolve_provider(
    provider: EmbeddingProvider,
    model_name: str,
) -> EmbeddingProviderHandler:
    """Return an initialised handler for ``provider`` or raise ValueError."""
    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerHandler(model_name)
    raise ValueError(
        f"No embedding handler registered for provider: {provider.value}"
    )
