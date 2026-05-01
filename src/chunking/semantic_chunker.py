# pylint: disable=duplicate-code
"""Chunk pre_chunk day parquet into semantic chunks.

``PreChunkPreprocessor`` writes ``{output_dir}/{YYYY-MM-DD}.parquet``; the
normalizer feeds it with ``{parquet_dir}/{YYYY-MM-DD}.parquet``. This module
reads the same ISO stem convention from ``chunking.input_dir`` and writes
``{chunking.output_dir}/{YYYY-MM-DD}.parquet`` with one row per chunk.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.chunking.chunk_records import build_chunk_row
from src.chunking.semantic_split import semantic_sentence_chunks
from src.config.settings import Settings
from src.enums.chunking_strategy import ChunkingStrategy


class SemanticChunker:
    """Run YAML-configured semantic chunking on pre_chunk parquet files."""

    def __init__(self, configuration_root: Path | None = None):
        """Load chunking settings from merged ingestion YAML."""
        self._config = Settings.load_chunking_config(
            configuration_root=configuration_root
        )

    @property
    def profile_names(self) -> list[str]:
        """Return profile names from merged ingestion config."""
        return list(self._config.profile_names)

    def _list_input_files(self) -> list[Path]:
        """Return sorted parquet paths under the chunking input directory."""
        paths = [
            path
            for path in self._config.input_dir.glob("*.parquet")
            if path.is_file()
        ]
        return sorted(paths)

    @staticmethod
    def _parse_day_from_filename(path: Path) -> str | None:
        """Parse ISO day from parquet stem; return None if invalid."""
        try:
            return date.fromisoformat(path.stem).isoformat()
        except ValueError:
            return None

    @staticmethod
    def _resolve_first_string(
        row: pd.Series,
        column_names: List[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Return (value, column_used) for the first present string column."""
        for column in column_names:
            if column not in row.index:
                continue
            raw = row[column]
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue
            text = str(raw).strip()
            if text:
                return text, column
        return None, None

    def _chunk_row(  # pylint: disable=too-many-locals
        self,
        row: pd.Series,
        *,
        source_day: str,
        source_row_index: int,
    ) -> List[Dict[str, Any]]:
        """Produce chunk dicts for one article row."""
        text_value, text_column = self._resolve_first_string(
            row,
            list(self._config.text_columns),
        )
        if text_value is None or text_column is None:
            return []
        api_id, _ = self._resolve_first_string(row, list(self._config.id_columns))
        profile, _ = self._resolve_first_string(row, list(self._config.profile_columns))
        passthrough: Dict[str, Any] = {}
        for column in self._config.passthrough_columns:
            if column in row.index:
                passthrough[column] = row[column]
        if self._config.strategy != ChunkingStrategy.SEMANTIC_SENTENCE:
            strategy = self._config.strategy
            label = getattr(strategy, "value", str(strategy))
            raise ValueError(f"Unsupported chunking strategy: {label}")
        chunk_spans = semantic_sentence_chunks(
            text_value,
            self._config.semantic,
        )
        records: List[Dict[str, Any]] = []
        for chunk_index, (chunk_text, start_char, end_char) in enumerate(chunk_spans):
            records.append(
                build_chunk_row(
                    source_day=source_day,
                    source_row_index=source_row_index,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    chunk_start_char=start_char,
                    chunk_end_char=end_char,
                    source_text_column=text_column,
                    strategy=self._config.strategy,
                    semantic=self._config.semantic,
                    source_api_id=api_id,
                    source_profile=profile,
                    passthrough=passthrough,
                )
            )
        return records

    def _combined_output_path(self, day_token: str) -> Path:
        """Destination parquet for one day."""
        return self._config.output_dir / f"{day_token}.parquet"

    def chunk_to_parquet(self) -> dict[str, str]:
        """Chunk all input day files and write per-day chunk parquet.

        Returns:
            Mapping of ISO day token to written parquet path.
        """
        chunked_by_day: Dict[str, List[pd.DataFrame]] = {}
        for source_path in self._list_input_files():
            day_token = self._parse_day_from_filename(source_path)
            if day_token is None:
                continue
            input_df = pd.read_parquet(source_path)
            if input_df.empty:
                continue
            day_frames: List[pd.DataFrame] = []
            for source_row_index in range(len(input_df)):
                row = input_df.iloc[source_row_index]
                records = self._chunk_row(
                    row,
                    source_day=day_token,
                    source_row_index=source_row_index,
                )
                if records:
                    day_frames.append(pd.DataFrame(records))
            if day_frames:
                chunked_by_day.setdefault(day_token, []).extend(day_frames)
        written: dict[str, str] = {}
        for day_token, frames in chunked_by_day.items():
            output_df = pd.concat(frames, ignore_index=True)
            output_path = self._combined_output_path(day_token)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_parquet(output_path, index=False)
            written[day_token] = str(output_path)
        return written
