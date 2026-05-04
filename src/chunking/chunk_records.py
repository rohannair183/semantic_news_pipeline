"""Build normalized chunk rows and config fingerprints for parquet output."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from src.enums.chunking_strategy import ChunkingStrategy

CHUNKING_VERSION = "1"


def chunking_params_fingerprint(
    strategy: ChunkingStrategy,
    params: Dict[str, Any],
) -> str:
    """Return a short stable hash of strategy + params for lineage."""
    strategy_label = getattr(strategy, "value", str(strategy))
    payload = {"strategy": strategy_label, **params}
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def build_chunk_row(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    source_day: str,
    source_row_index: int,
    chunk_index: int,
    chunk_text: str,
    chunk_start_char: int,
    chunk_end_char: int,
    source_text_column: str,
    strategy: ChunkingStrategy,
    params: Dict[str, Any],
    source_api_id: Optional[str],
    source_profile: Optional[str],
    passthrough: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble one chunk record dict aligned with the stable output schema."""
    fingerprint = chunking_params_fingerprint(strategy, params)
    row: Dict[str, Any] = {
        "source_day": source_day,
        "source_row_index": source_row_index,
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "chunk_start_char": chunk_start_char,
        "chunk_end_char": chunk_end_char,
        "chunk_char_len": max(0, chunk_end_char - chunk_start_char),
        "source_text_column": source_text_column,
        "chunking_strategy": strategy.value,
        "chunking_version": CHUNKING_VERSION,
        "chunking_params_hash": fingerprint,
        "source_api_id": source_api_id,
        "source_profile": source_profile,
    }
    for key, value in passthrough.items():
        row[key] = value
    return row
