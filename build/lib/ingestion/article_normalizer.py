"""Normalize article checkpoints and write combined Parquet files."""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import (
    ArticleRowMappingConfig,
    ArticleRowSourceConfig,
    Settings,
)
from src.enums.article_row_source_kind import ArticleRowSourceKind
from src.enums.article_row_transform import ArticleRowTransform
from src.utils.dates import format_day_iso, parse_checkpoint_timestamp, parse_guardian_datetime


class ArticleNormalizer:
    """Normalize article checkpoints and write combined Parquet output.

    Configuration is read from ingestion YAML to determine checkpoint and parquet
    directories plus the row mapping rules used when converting checkpoint JSON
    into tabular output. Set `article_ingestor.checkpoint_dir` and optionally
    `article_ingestor.parquet_dir` (defaults to `checkpoints/parquet`) in
    `configuration/ingestion/article_ingestor.yaml`. Row mappings are controlled
    by `article_normalizer.row_mappings` in
    `configuration/ingestion/article_normalizer.yaml`.
    """

    def __init__(self, configuration_root: Optional[Path] = None):
        """Initialize normalizer from config.

        Parameters:
            configuration_root: Root directory for config files. If None, auto-resolved.
        """
        self._config = Settings.load_article_normalizer_config(
            configuration_root=configuration_root
        )

    @property
    def checkpoint_dir(self) -> Path:
        """Return the configured checkpoint directory."""
        return self._config.checkpoint_dir

    @property
    def parquet_dir(self) -> Path:
        """Return the configured parquet output directory."""
        return self._config.parquet_dir

    @property
    def profiles(self) -> List[str]:
        """Return the ordered profile names available for normalization."""
        return list(self._config.profile_names)

    @property
    def row_mappings(self) -> Dict[str, ArticleRowMappingConfig]:
        """Return the configured row mapping rules."""
        return self._config.row_mappings

    @staticmethod
    def _resolve_nested_value(source: Dict[str, Any], dotted_path: str) -> Any:
        """Resolve a dotted path from a mapping.

        Parameters:
            source: Source dict to traverse.
            dotted_path: Dot-separated path (e.g., 'fields.headline').

        Returns:
            Value at the path or None if not found or path is invalid.
        """
        current: Any = source
        for part in dotted_path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None
        return current

    def _resolve_source_value(
        self,
        source_config: ArticleRowSourceConfig,
        context: Dict[str, Any],
    ) -> Any:
        """Resolve one configured source path for a row mapping.

        Parameters:
            source_config: Typed source selector parsed from YAML.
            context: Dict with keys 'item', 'fields', 'payload', 'profile_value'.

        Returns:
            Resolved value from the source path or None if not found.
        """
        if source_config.kind == ArticleRowSourceKind.PROFILE:
            return context["profile_value"]
        if source_config.kind == ArticleRowSourceKind.DIRECT_KEY:
            return self._resolve_direct_key(
                source_name=str(source_config.path),
                source_mappings=[
                    context["item"],
                    context["fields"],
                    context["payload"],
                ],
            )
        return self._resolve_nested_value(
            context[source_config.kind.value],
            str(source_config.path),
        )

    @staticmethod
    def _resolve_direct_key(
        source_name: str,
        source_mappings: List[Dict[str, Any]],
    ) -> Any:
        """Resolve a direct key from the first source mapping that contains it."""
        for source_mapping in source_mappings:
            if source_name in source_mapping:
                return source_mapping[source_name]
        return None

    @staticmethod
    def _first_non_empty(values: List[Any]) -> Any:
        """Return the first value that is not None or an empty string.

        Parameters:
            values: List of values to search.

        Returns:
            First non-None, non-empty value or None if all values are empty.
        """
        for value in values:
            if value is None or value == "":
                continue
            return value
        return None

    def _apply_row_transform(
        self,
        value: Any,
        transform: Optional[ArticleRowTransform],
    ) -> Any:
        """Apply an optional row transform declared in YAML.

        Parameters:
            value: Value to transform.
            transform: Transform type or None.

        Returns:
            Transformed value or original value if transform is None.
        """
        if transform is None:
            return value
        if transform == ArticleRowTransform.PARSE_ISO:
            if value is None:
                return None
            return self._parse_iso(str(value))
        raise ValueError(f"Unsupported row mapping transform: {transform}")

    def _build_row(
        self,
        payload: Dict[str, Any],
        item: Dict[str, Any],
        profile: Optional[str],
    ) -> Dict[str, Any]:
        """Build one normalized row using YAML-driven field mappings.

        Parameters:
            payload: Guardian API response payload dict.
            item: Article item from payload.items.
            profile: Profile name to use in output row, or None to use payload profile.

        Returns:
            Row dict with keys from article_normalizer.row_mappings and resolved values.
        """
        fields = item.get("fields") or {}
        context = {
            "item": item,
            "fields": fields,
            "payload": payload,
            "profile_value": profile or payload.get("profile"),
        }
        return {
            output_field: self._resolve_row_value(
                output_field=output_field,
                field_config=field_config,
                context=context,
            )
            for output_field, field_config in self.row_mappings.items()
        }

    def _resolve_row_value(
        self,
        output_field: str,
        field_config: ArticleRowMappingConfig,
        context: Dict[str, Any],
    ) -> Any:
        """Resolve and transform one output field from configured row mappings."""
        if not field_config.sources:
            raise ValueError(
                f"Row mapping '{output_field}' must contain a non-empty 'sources' list"
            )

        value = self._first_non_empty(
            [
                self._resolve_source_value(source_config=source_config, context=context)
                for source_config in field_config.sources
            ]
        )
        return self._apply_row_transform(value, field_config.transform)

    def _parse_ts_from_filename(self, path: Path) -> Optional[datetime]:
        """Parse timestamp token from checkpoint filename.

        Parameters:
            path: Checkpoint file path.

        Returns:
            Parsed datetime or None if filename format is invalid.
        """
        return parse_checkpoint_timestamp(path)

    def _list_profile_files(self, profile: str):
        """List all JSON checkpoint files for a profile.

        Parameters:
            profile: Profile name.

        Returns:
            List of Path objects matching the checkpoint pattern.
        """
        pattern = f"{profile}_*.json"
        return list(self.checkpoint_dir.glob(pattern))

    def find_latest_checkpoints_for_date(self, day: date) -> Dict[str, Path]:
        """Find latest checkpoint per profile for the given day."""
        self._ensure_day(day)
        found: Dict[str, Path] = {}
        for profile in self.profiles:
            latest = self._find_latest_profile_checkpoint(profile=profile, day=day)
            if latest is not None:
                found[profile] = latest
        return found

    def _find_latest_profile_checkpoint(
        self,
        profile: str,
        day: date,
    ) -> Optional[Path]:
        """Return the latest checkpoint path for a profile on a given day."""
        latest_timestamp: Optional[datetime] = None
        latest_path: Optional[Path] = None
        for checkpoint_path in self._list_profile_files(profile):
            checkpoint_timestamp = self._parse_ts_from_filename(checkpoint_path)
            if checkpoint_timestamp is None or checkpoint_timestamp.date() != day:
                continue
            if latest_timestamp is None or checkpoint_timestamp > latest_timestamp:
                latest_timestamp = checkpoint_timestamp
                latest_path = checkpoint_path
        return latest_path

    @staticmethod
    def _parse_iso(value: Optional[str]) -> Optional[datetime]:
        """Parse ISO 8601 datetime string.

        Parameters:
            value: ISO datetime string or None.

        Returns:
            Parsed datetime or None if parsing fails.
        """
        return parse_guardian_datetime(value)

    @staticmethod
    def _ensure_day(day: date) -> None:
        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError("day must be a date instance")

    def normalize_checkpoint(
        self, checkpoint_path: Path, profile: Optional[str] = None
    ) -> pd.DataFrame:
        """Load and normalize one checkpoint JSON file.

        Extracts key fields from Guardian article items and returns a DataFrame
        with columns: profile, api_id, web_title, headline, byline, section,
        published_at, first_publication_date, url, body_text, trail_text,
        thumbnail, wordcount, pillar, last_modified.

        Parameters:
            checkpoint_path: Path to checkpoint JSON file.
            profile: Override profile name in output (defaults to checkpoint value).

        Returns:
            DataFrame with normalized article data.
        """
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        rows = [
            self._build_row(payload=payload, item=item, profile=profile)
            for item in payload.get("items", []) or []
        ]
        return pd.DataFrame(rows)

    def _list_checkpoints_for_date(self, day: date) -> List[Path]:
        """List all checkpoint files for configured profiles on a specific day."""
        checkpoints_with_timestamps: List[tuple[datetime, Path]] = []
        for profile in self.profiles:
            for checkpoint_path in self._list_profile_files(profile):
                checkpoint_timestamp = self._parse_ts_from_filename(checkpoint_path)
                if checkpoint_timestamp is None or checkpoint_timestamp.date() != day:
                    continue
                checkpoints_with_timestamps.append((checkpoint_timestamp, checkpoint_path))
        checkpoints_with_timestamps.sort(key=lambda item: item[0])
        return [path for _, path in checkpoints_with_timestamps]

    def _write_combined_parquet(self, df: pd.DataFrame, day: date) -> Path:
        """Write combined normalized DataFrame to a single parquet file.

        Parameters:
            df: DataFrame to write.
            day: Date (used in output path).

        Returns:
            Path to written Parquet file.
        """
        self._ensure_day(day)
        out_dir = self.parquet_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{format_day_iso(day)}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    def normalize_day_to_parquet(self, day: date) -> Dict[str, str]:
        """Normalize all same-day checkpoints and write a combined parquet file."""
        self._ensure_day(day)
        checkpoints = self._list_checkpoints_for_date(day)
        normalized_frames: List[pd.DataFrame] = []
        for path in checkpoints:
            df = self.normalize_checkpoint(path)
            if df.empty:
                continue
            normalized_frames.append(df)
        if not normalized_frames:
            return {}
        combined_df = pd.concat(normalized_frames, ignore_index=True)
        out = self._write_combined_parquet(combined_df, day=day)
        return {format_day_iso(day): str(out)}
