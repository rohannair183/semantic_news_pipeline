"""Embedding provider registry and handlers used by Embedder."""
# pylint: disable=import-error,invalid-name

from __future__ import annotations

from typing import List, Optional, Protocol


from src.enums.embedding_provider import EmbeddingProvider
from src.utils.timer import Timer


class EmbeddingProviderHandler(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for embedding provider handlers used by the registry."""

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Return one embedding vector per input text."""


class SentenceTransformerHandler:  # pylint: disable=too-few-public-methods
    """Handler that wraps the sentence-transformers library."""

    def __init__(
        self,
        model_name: str,
        timer: Optional[Timer] = None,
    ) -> None:
        self._model_name = model_name
        self._model: Optional[object] = None
        self._timer = timer

    def _load_model(self) -> object:
        if self._model is None:
            # Import lazily so unit tests can inject a mock module via sys.modules.
            try:
                import sentence_transformers as _st  # type: ignore pylint: disable=import-outside-toplevel
            except Exception as exc:  # pragma: no cover - import errors tested elsewhere
                raise RuntimeError(
                    "sentence-transformers is required to load embedding models",
                ) from exc
            SentCls = getattr(_st, "SentenceTransformer", None)
            if SentCls is None:
                raise RuntimeError(
                    "sentence-transformers is required to load embedding models",
                )
            self._model = SentCls(self._model_name)
        return self._model

    def embed(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Encode ``texts`` into embedding vectors."""
        if self._timer is not None:
            return self._embed_timed(texts, batch_size)
        return self._embed_core(texts, batch_size)

    def _embed_timed(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Encode with timer sections for model loading and encoding."""
        with self._timer.section("provider.load_model"):  # type: ignore[union-attr]
            model = self._load_model()
        with self._timer.section("provider.encode"):  # type: ignore[union-attr]
            embeddings = model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return embeddings.tolist()  # type: ignore[union-attr]

    def _embed_core(self, texts: List[str], batch_size: int) -> List[List[float]]:
        """Encode without timing instrumentation."""
        model = self._load_model()
        embeddings = model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()  # type: ignore[union-attr]


def resolve_provider(
    provider: EmbeddingProvider,
    model_name: str,
    timer: Optional[Timer] = None,
) -> EmbeddingProviderHandler:
    """Return an initialised handler for ``provider`` or raise ValueError."""
    if provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
        return SentenceTransformerHandler(model_name, timer=timer)
    raise ValueError(
        f"No embedding handler registered for provider: {provider.value}"
    )
