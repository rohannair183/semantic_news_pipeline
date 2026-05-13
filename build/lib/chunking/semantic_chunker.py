"""Semantic sentence chunking strategy handler.

Implements ``ChunkingStrategyHandler`` for the ``semantic_sentence`` strategy.
Parses a raw ``params`` dict into typed ``SemanticChunkingParams`` and delegates
to ``semantic_sentence_chunks`` for the actual splitting logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.chunking.semantic_split import semantic_sentence_chunks
from src.config.settings import SemanticChunkingParams
from src.enums.sentence_splitter_mode import SentenceSplitterMode


class SemanticChunker:  # pylint: disable=too-few-public-methods
    """Strategy handler for semantic sentence chunking."""

    def chunk(
        self,
        full_text: str,
        params: Dict[str, Any],
    ) -> List[Tuple[str, int, int]]:
        """Split ``full_text`` into semantic sentence chunks.

        Returns (text, start_char, end_char_exclusive) spans.
        """
        parsed = self._parse_params(params)
        return semantic_sentence_chunks(full_text, parsed)

    @staticmethod
    def _parse_params(raw: Dict[str, Any]) -> SemanticChunkingParams:
        """Validate and convert a raw params dict into typed semantic params."""
        min_chars = _require_positive_int(raw.get("min_chars"), "min_chars")
        max_chars = _require_positive_int(raw.get("max_chars"), "max_chars")
        if max_chars < min_chars:
            raise ValueError("params 'max_chars' must be >= 'min_chars'")

        overlap_raw = raw.get("overlap_chars", 0)
        try:
            overlap_chars = int(overlap_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "params 'overlap_chars' must be an integer"
            ) from exc
        if overlap_chars < 0:
            raise ValueError("params 'overlap_chars' must be >= 0")
        if overlap_chars >= max_chars:
            raise ValueError("params 'overlap_chars' must be < 'max_chars'")

        threshold_raw = raw.get("similarity_threshold", 0.35)
        try:
            similarity_threshold = float(threshold_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "params 'similarity_threshold' must be a number"
            ) from exc
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                "params 'similarity_threshold' must be in [0, 1]"
            )

        splitter_raw = raw.get(
            "sentence_splitter",
            SentenceSplitterMode.SIMPLE_REGEX.value,
        )
        try:
            sentence_splitter = SentenceSplitterMode.from_value(
                str(splitter_raw)
            )
        except ValueError as exc:
            raise ValueError(
                f"params 'sentence_splitter': {exc}"
            ) from exc

        return SemanticChunkingParams(
            min_chars=min_chars,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            similarity_threshold=similarity_threshold,
            sentence_splitter=sentence_splitter,
        )


def _require_positive_int(value: Any, name: str) -> int:
    """Return ``value`` as a positive int or raise ``ValueError``."""
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"params '{name}' must be a positive integer"
        ) from exc
    if resolved < 1:
        raise ValueError(f"params '{name}' must be >= 1")
    return resolved
