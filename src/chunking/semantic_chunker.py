# pylint: disable=duplicate-code
"""Chunk pre_chunk day parquet into a per-profile combined parquet.

``PreChunkPreprocessor`` writes ``{output_dir}/{YYYY-MM-DD}.parquet``; the
normalizer feeds it with ``{parquet_dir}/{YYYY-MM-DD}.parquet``. This module
reads the same ISO stem convention from ``chunking.input_dir`` and, for each
chunking profile, writes a single combined parquet at
``{chunking.output_dir}/{profile}.parquet`` containing chunks from every
available input day.

Public API:

```
chunker = SemanticChunker()
chunker.chunk_to_parquet(profile="default")
```

The chunker is intentionally stateless across runs: every invocation rebuilds
the combined parquet from the current pre_chunk inputs. Idempotency at the
embedding/database layer is the consumer's responsibility (e.g. by upserting
on a deterministic chunk primary key derived from
``source_api_id``/``chunk_index``/``chunking_params_hash``).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.chunking.chunk_records import build_chunk_row
from src.chunking.strategies import resolve_handler
from src.config.settings import ChunkingProfileConfig, Settings


class SemanticChunker:
    """Run YAML-configured chunking on pre_chunk parquet via named profiles."""

    def __init__(self, configuration_root: Path | None = None):
        """Load chunking settings from merged ingestion YAML."""
        self._config = Settings.load_chunking_config(
            configuration_root=configuration_root
        )

    @property
    def profile_names(self) -> list[str]:
        """Return ingestion profile names from the merged config."""
        return list(self._config.profile_names)

    @property
    def chunking_profile_names(self) -> list[str]:
        """Return configured chunking profile names."""
        return list(self._config.chunking_profiles.keys())

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

    def _resolve_profile(self, profile_name: str) -> ChunkingProfileConfig:
        """Return the typed profile config for ``profile_name`` or raise."""
        profile = self._config.chunking_profiles.get(profile_name)
        if profile is None:
            available = ", ".join(sorted(self._config.chunking_profiles.keys()))
            raise ValueError(
                f"Unknown chunking profile '{profile_name}'. Available: {available}"
            )
        return profile

    def _chunk_row(  # pylint: disable=too-many-locals
        self,
        row: pd.Series,
        *,
        source_day: str,
        source_row_index: int,
        profile: ChunkingProfileConfig,
    ) -> List[Dict[str, Any]]:
        """Produce chunk dicts for one article row using ``profile``."""
        text_value, text_column = self._resolve_first_string(
            row,
            list(self._config.text_columns),
        )
        if text_value is None or text_column is None:
            return []
        api_id, _ = self._resolve_first_string(row, list(self._config.id_columns))
        source_profile, _ = self._resolve_first_string(
            row,
            list(self._config.profile_columns),
        )
        passthrough: Dict[str, Any] = {}
        for column in self._config.passthrough_columns:
            if column in row.index:
                passthrough[column] = row[column]
        handler = resolve_handler(profile.strategy)
        chunk_spans = handler.chunk(text_value, profile.semantic)
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
                    strategy=profile.strategy,
                    semantic=profile.semantic,
                    source_api_id=api_id,
                    source_profile=source_profile,
                    passthrough=passthrough,
                )
            )
        return records

    @staticmethod
    def _output_path(output_dir: Path, profile_name: str) -> Path:
        """Destination parquet path for the combined output of one profile."""
        safe_profile = profile_name.replace("/", "_")
        return output_dir / f"{safe_profile}.parquet"

    def _collect_day_records(
        self,
        source_path: Path,
        day_token: str,
        profile: ChunkingProfileConfig,
    ) -> List[Dict[str, Any]]:
        """Return all chunk records for one input day file."""
        input_df = pd.read_parquet(source_path)
        if input_df.empty:
            return []
        records: List[Dict[str, Any]] = []
        for source_row_index in range(len(input_df)):
            row = input_df.iloc[source_row_index]
            records.extend(
                self._chunk_row(
                    row,
                    source_day=day_token,
                    source_row_index=source_row_index,
                    profile=profile,
                )
            )
        return records

    def chunk_to_parquet(self, profile: str) -> Dict[str, str]:
        """Chunk every input day file for ``profile`` into one combined parquet.

        Parameters:
            profile: Name of a chunking profile defined in chunking.yaml.

        Returns:
            ``{profile_name: combined_path}`` when at least one chunk row was
            produced, otherwise an empty dict.
        """
        profile_config = self._resolve_profile(profile)
        all_records: List[Dict[str, Any]] = []
        for source_path in self._list_input_files():
            day_token = self._parse_day_from_filename(source_path)
            if day_token is None:
                continue
            all_records.extend(
                self._collect_day_records(
                    source_path=source_path,
                    day_token=day_token,
                    profile=profile_config,
                )
            )
        if not all_records:
            return {}
        output_path = self._output_path(self._config.output_dir, profile)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_records).to_parquet(output_path, index=False)
        return {profile: str(output_path)}
