"""Apply YAML-driven preprocessing to normalized parquet before chunking."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.settings import PreChunkOperationConfig, Settings
from src.enums.pre_chunk_operation import PreChunkOperation


class PreChunkPreprocessor:
    """Preprocess normalized parquet outputs into chunk-ready parquet files."""

    def __init__(self, configuration_root: Path | None = None):
        """Initialize the preprocessor from typed YAML settings."""
        self._config = Settings.load_pre_chunk_preprocessor_config(
            configuration_root=configuration_root
        )

    @property
    def profile_names(self) -> list[str]:
        """Return configured profile names."""
        return list(self._config.profile_names)

    def _list_input_files(self) -> list[Path]:
        parquet_files = [
            path for path in self._config.input_dir.glob("*.parquet") if path.is_file()
        ]
        return sorted(parquet_files)

    @staticmethod
    def _parse_day_from_filename(path: Path) -> str | None:
        """Parse yyyy-mm-dd day token from a parquet filename stem."""
        try:
            return date.fromisoformat(path.stem).isoformat()
        except ValueError:
            return None

    def _combined_output_path(self, day_token: str) -> Path:
        """Build output path for a day-level pre-chunk parquet."""
        return self._config.output_dir / f"{day_token}.parquet"

    @staticmethod
    def _ensure_columns_exist(df: pd.DataFrame, columns: list[str], operation: str) -> None:
        missing_columns = [column for column in columns if column not in df.columns]
        if missing_columns:
            missing_values = ", ".join(missing_columns)
            raise ValueError(
                f"Operation '{operation}' references missing columns: {missing_values}"
            )

    @staticmethod
    def _is_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    def _apply_operation(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        operation_handlers = {
            PreChunkOperation.DROP_COLUMNS: self._drop_columns,
            PreChunkOperation.RENAME_COLUMNS: self._rename_columns,
            PreChunkOperation.TRIM_WHITESPACE_COLUMNS: self._trim_whitespace_columns,
            PreChunkOperation.DROP_EMPTY_ROWS: self._drop_empty_rows,
            PreChunkOperation.COALESCE_COLUMNS: self._coalesce_columns,
            PreChunkOperation.NORMALIZE_TEXT_COLUMNS: self._normalize_text_columns,
        }
        handler = operation_handlers.get(operation.name)
        if handler is None:
            operation_name = getattr(operation.name, "value", str(operation.name))
            raise ValueError(f"Unsupported pre-chunk operation: {operation_name}")
        return handler(df, operation)

    def _drop_columns(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        columns = list(operation.args["columns"])
        existing_columns = [column for column in columns if column in df.columns]
        if not existing_columns:
            return df
        return df.drop(columns=existing_columns)

    def _rename_columns(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        mapping = dict(operation.args["mapping"])
        if not mapping:
            return df
        self._ensure_columns_exist(df, list(mapping.keys()), operation.name.value)
        return df.rename(columns=mapping)

    def _trim_whitespace_columns(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        columns = list(operation.args["columns"])
        self._ensure_columns_exist(df, columns, operation.name.value)
        for column in columns:
            df[column] = df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        return df

    def _drop_empty_rows(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        required_columns = list(operation.args["required_columns"])
        self._ensure_columns_exist(df, required_columns, operation.name.value)
        filtered_df = df.copy()
        for column in required_columns:
            present_mask = filtered_df[column].apply(self._is_present)
            filtered_df = filtered_df.loc[present_mask]
        return filtered_df.reset_index(drop=True)

    def _coalesce_columns(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        target = str(operation.args["target"])
        sources = list(operation.args["sources"])
        self._ensure_columns_exist(df, sources, operation.name.value)
        coalesced = df[sources].bfill(axis=1).iloc[:, 0]
        if target not in df.columns:
            df[target] = coalesced
            return df
        existing_target = df[target]
        df[target] = existing_target.where(existing_target.notna(), coalesced)
        return df

    def _normalize_text_columns(
        self,
        df: pd.DataFrame,
        operation: PreChunkOperationConfig,
    ) -> pd.DataFrame:
        columns = list(operation.args["columns"])
        self._ensure_columns_exist(df, columns, operation.name.value)
        whitespace_pattern = re.compile(r"\s+")
        for column in columns:
            df[column] = df[column].map(
                lambda value: whitespace_pattern.sub(" ", value).strip()
                if isinstance(value, str)
                else value
            )
        return df

    def _apply_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed_df = df.copy()
        for operation in self._config.operations:
            transformed_df = self._apply_operation(transformed_df, operation)
        return transformed_df

    def preprocess_to_parquet(self) -> dict[str, str]:
        """Apply operations and write combined day-level parquet outputs."""
        transformed_by_day: dict[str, list[pd.DataFrame]] = {}
        for source_path in self._list_input_files():
            day_token = self._parse_day_from_filename(source_path)
            if day_token is None:
                continue
            input_df = pd.read_parquet(source_path)
            if input_df.empty:
                continue
            transformed_df = self._apply_operations(input_df)
            if transformed_df.empty:
                continue
            transformed_by_day.setdefault(day_token, []).append(transformed_df)
        written: dict[str, str] = {}
        for day_token, transformed_frames in transformed_by_day.items():
            output_df = pd.concat(transformed_frames, ignore_index=True)
            output_path = self._combined_output_path(day_token)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_parquet(output_path, index=False)
            written[day_token] = str(output_path)
        return written
