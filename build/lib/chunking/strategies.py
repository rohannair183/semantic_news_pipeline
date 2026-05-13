"""Chunking strategy registry and handlers used by Chunker."""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, Tuple

from src.chunking.semantic_chunker import SemanticChunker
from src.enums.chunking_strategy import ChunkingStrategy


class ChunkingStrategyHandler(Protocol):  # pylint: disable=too-few-public-methods
    """Protocol for chunking strategy handlers used by the registry."""

    def chunk(
        self,
        full_text: str,
        params: Dict[str, Any],
    ) -> List[Tuple[str, int, int]]:
        """Return chunk spans (text, start_char, end_char_exclusive) for ``full_text``."""


STRATEGY_HANDLERS: Dict[ChunkingStrategy, ChunkingStrategyHandler] = {
    ChunkingStrategy.SEMANTIC_SENTENCE: SemanticChunker(),
}


def resolve_handler(strategy: ChunkingStrategy) -> ChunkingStrategyHandler:
    """Return the registered handler for ``strategy`` or raise ValueError."""
    handler = STRATEGY_HANDLERS.get(strategy)
    if handler is None:
        label = getattr(strategy, "value", str(strategy))
        raise ValueError(f"No chunking handler registered for strategy: {label}")
    return handler
