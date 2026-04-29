"""Enum describing supported pre-chunk preprocessing operations."""

from src.enums.base import BaseEnum


class PreChunkOperation(BaseEnum):
    """Operations supported by the pre-chunk parquet preprocessor."""

    DROP_COLUMNS = "drop_columns"
    RENAME_COLUMNS = "rename_columns"
    TRIM_WHITESPACE_COLUMNS = "trim_whitespace_columns"
    DROP_EMPTY_ROWS = "drop_empty_rows"
    FILTER_MIN_NUMERIC = "filter_min_numeric"
    COALESCE_COLUMNS = "coalesce_columns"
    NORMALIZE_TEXT_COLUMNS = "normalize_text_columns"
