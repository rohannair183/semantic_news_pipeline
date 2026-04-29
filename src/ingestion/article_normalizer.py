"""Normalize per-profile article checkpoints and write Parquet files."""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import ArticleNormalizerConfig, Settings
from src.utils.dates import (
    format_day_compact,
    format_day_iso,
    parse_checkpoint_timestamp,
    parse_guardian_datetime,
)


class ArticleNormalizer:
    """Normalize per-profile article checkpoints and write Parquet output.

    Configuration is read from ingestion YAML to determine checkpoint and parquet
    directories plus the row mapping rules used when converting checkpoint JSON
    into tabular output. Set `article_ingestor.checkpoint_dir` and optionally
    `article_ingestor.parquet_dir` (defaults to `checkpoints/parquet`) in
    `configuration/ingestion/ingestion_config.yaml`. Row mappings are controlled
    by `article_normalizer.row_mappings` in the same YAML file.
    """

    def __init__(self, configuration_root: Optional[Path] = None):
        """Initialize normalizer from config.

        Parameters:
            configuration_root: Root directory for config files. If None, auto-resolved.
        """
        config = Settings.load_article_normalizer_config(configuration_root=configuration_root)
        self._config = config
        self._row_mappings = self._resolve_row_mappings(config)

        self.checkpoint_dir = config.checkpoint_dir
        self.parquet_dir = config.parquet_dir
        self.profiles = list(config.profile_names)

    def _resolve_row_mappings(
        self,
        config: ArticleNormalizerConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve required row mapping rules from config.

        Parameters:
            config: Typed ingestion configuration.

        Returns:
            Row mappings dict from article_normalizer.row_mappings in config.
        """
        return config.row_mappings

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
        source_name: str,
        context: Dict[str, Any],
    ) -> Any:
        """Resolve one configured source path for a row mapping.

        Parameters:
            source_name: Source name from YAML (e.g., 'profile', 'payload.profile',
                'item.headline').
            context: Dict with keys 'item', 'fields', 'payload', 'profile_value'.

        Returns:
            Resolved value from the source path or None if not found.
        """
        item = context["item"]
        fields = context["fields"]
        payload = context["payload"]
        profile_value = context["profile_value"]
        if source_name == "profile":
            return profile_value
        if source_name.startswith("payload."):
            return self._resolve_nested_value(payload, source_name.split(".", 1)[1])
        if source_name.startswith("fields."):
            return self._resolve_nested_value(fields, source_name.split(".", 1)[1])
        if source_name.startswith("item."):
            return self._resolve_nested_value(item, source_name.split(".", 1)[1])

        for source_mapping in (item, fields, payload):
            if source_name in source_mapping:
                return source_mapping.get(source_name)
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

    def _apply_row_transform(self, value: Any, transform: Optional[str]) -> Any:
        """Apply an optional row transform declared in YAML.

        Parameters:
            value: Value to transform.
            transform: Transform type or None. Supported: 'parse_iso' (parse ISO 8601 datetime).

        Returns:
            Transformed value or original value if transform is None.
        """
        if transform is None:
            return value
        if transform == "parse_iso":
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
        row: Dict[str, Any] = {}

        for output_field, field_config in self._row_mappings.items():
            sources = field_config.get("sources")
            if not isinstance(sources, list) or not sources:
                raise ValueError(
                    f"Row mapping '{output_field}' must contain a non-empty 'sources' list"
                )

            resolved_values = [
                self._resolve_source_value(source_name=str(source_name), context=context)
                for source_name in sources
            ]
            value = self._first_non_empty(resolved_values)
            value = self._apply_row_transform(value, field_config.get("transform"))
            row[output_field] = value

        return row

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
        """Find latest checkpoint per profile for the given day.

        Parameters:
            day: Date to search for.

        Returns:
            Dict mapping profile name to latest checkpoint Path on that day.
        """
        self._ensure_day(day)
        found: Dict[str, Path] = {}
        for profile in self.profiles:
            files = self._list_profile_files(profile)
            daily = []
            for f in files:
                ts = self._parse_ts_from_filename(f)
                if ts is None:
                    continue
                if ts.date() == day:
                    daily.append((ts, f))
            if not daily:
                continue
            latest = max(daily, key=lambda t_f: t_f[0])[1]
            found[profile] = latest
        return found

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

        items = payload.get("items", []) or []
        rows = []
        for it in items:
            row = self._build_row(payload=payload, item=it, profile=profile)
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return df

    def _write_parquet(self, df: pd.DataFrame, profile: str, day: date) -> Path:
        """Write normalized DataFrame to Parquet file.

        Parameters:
            df: DataFrame to write.
            profile: Profile name (used in output path).
            day: Date (used in output path).

        Returns:
            Path to written Parquet file.
        """
        self._ensure_day(day)
        out_dir = self.parquet_dir / profile / format_day_iso(day)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{profile}_{format_day_compact(day)}.parquet"
        df.to_parquet(out_path, index=False)
        return out_path

    def normalize_day_to_parquet(self, day: date) -> Dict[str, str]:
        """Normalize checkpoints for a day and write Parquet files.

        Finds the latest checkpoint for each profile on the given day, normalizes
        it, and writes per-profile Parquet files.

        Parameters:
            day: Date to process.

        Returns:
            Dict mapping profile name to written Parquet file path.
        """
        self._ensure_day(day)
        checkpoints = self.find_latest_checkpoints_for_date(day)
        written: Dict[str, str] = {}
        for profile, path in checkpoints.items():
            df = self.normalize_checkpoint(path, profile=profile)
            if df.empty:
                continue
            out = self._write_parquet(df, profile=profile, day=day)
            written[profile] = str(out)
        return written
