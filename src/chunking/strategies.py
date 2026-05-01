"""Chunking strategy registry and handlers used by SemanticChunker."""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

from src.chunking.semantic_split import semantic_sentence_chunks
from src.config.settings import SemanticChunkingParams
from src.enums.chunking_strategy import ChunkingStrategy


class ChunkingStrategyHandler(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for chunking strategy handlers used by the registry."""

    def chunk(
        self,
        full_text: str,
        params: SemanticChunkingParams,
    ) -> List[Tuple[str, int, int]]:
        """Return chunk spans (text, start_char, end_char_exclusive) for ``full_text``."""


class SemanticSentenceHandler:  # pylint: disable=too-few-public-methods
    """Handler that delegates to semantic_sentence_chunks."""

    def chunk(
        self,
        full_text: str,
        params: SemanticChunkingParams,
    ) -> List[Tuple[str, int, int]]:
        """Run semantic sentence chunking with the supplied params."""
        return semantic_sentence_chunks(full_text, params)


STRATEGY_HANDLERS: Dict[ChunkingStrategy, ChunkingStrategyHandler] = {
    ChunkingStrategy.SEMANTIC_SENTENCE: SemanticSentenceHandler(),
}


def resolve_handler(strategy: ChunkingStrategy) -> ChunkingStrategyHandler:
    """Return the registered handler for ``strategy`` or raise ValueError."""
    handler = STRATEGY_HANDLERS.get(strategy)
    if handler is None:
        label = getattr(strategy, "value", str(strategy))
        raise ValueError(f"No chunking handler registered for strategy: {label}")
    return handler
