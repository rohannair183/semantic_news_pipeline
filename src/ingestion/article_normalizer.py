"""Normalize per-profile article checkpoints and write Parquet files."""

from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.yaml_config_type import YAMLConfigType


class ArticleNormalizer:
    """Normalize per-profile article checkpoints and write Parquet output.

    Configuration is read from ingestion YAML to determine checkpoint and parquet
    directories. Set `article_ingestor.checkpoint_dir` and optionally
    `article_ingestor.parquet_dir` (defaults to `checkpoints/parquet`) in
    `configuration/ingestion/ingestion_config.yaml`.
    """

    def __init__(self, configuration_root: Optional[Path] = None):
        """Initialize normalizer from config.

        Parameters:
            configuration_root: Root directory for config files. If None, auto-resolved.
        """
        parser = YAMLConfigParser(configuration_root=configuration_root)
        cfg_type = YAMLConfigType.INGESTION
        config = parser.parse(
            config_type=cfg_type, filename="ingestion_config.yaml"
        )
        self._config = config
        article_cfg = config.get("article_ingestor", {}) or {}

        checkpoint_dir = article_cfg.get("checkpoint_dir", "checkpoints/article_ingestor")
        parquet_dir = article_cfg.get("parquet_dir", "checkpoints/parquet")

        self.checkpoint_dir = Path(str(checkpoint_dir))
        self.parquet_dir = Path(str(parquet_dir))

        profiles = config.get("profiles") or {}
        if not isinstance(profiles, dict):
            raise ValueError("ingestion config must contain a 'profiles' mapping")
        self.profiles = list(profiles.keys())

    def _parse_ts_from_filename(self, path: Path) -> Optional[datetime]:
        """Parse timestamp token from checkpoint filename.

        Parameters:
            path: Checkpoint file path.

        Returns:
            Parsed datetime or None if filename format is invalid.
        """
        stem = path.stem  # e.g. technology_daily_20260428T221904Z
        if "_" not in stem:
            return None
        token = stem.split("_")[-1]
        try:
            return datetime.strptime(token, "%Y%m%dT%H%M%SZ")
        except ValueError:
            return None

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
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value.replace("Z", "+00:00")
            return datetime.fromisoformat(value)
        except ValueError:
            return None

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
            fields = it.get("fields") or {}
            row = {
                "profile": profile or payload.get("profile"),
                "api_id": it.get("id"),
                "web_title": it.get("webTitle") or fields.get("headline"),
                "headline": fields.get("headline"),
                "byline": fields.get("byline"),
                "section": it.get("sectionName"),
                "published_at": self._parse_iso(it.get("webPublicationDate")),
                "first_publication_date": self._parse_iso(
                    fields.get("firstPublicationDate")
                    or fields.get("firstPublicationDate")
                ),
                "url": it.get("webUrl"),
                "body_text": fields.get("bodyText") or fields.get("body"),
                "trail_text": fields.get("trailText"),
                "thumbnail": fields.get("thumbnail"),
                "wordcount": fields.get("wordcount"),
                "pillar": it.get("pillarName"),
                "last_modified": self._parse_iso(
                    fields.get("lastModified") or it.get("lastModified")
                ),
            }
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
        out_dir = self.parquet_dir / profile / day.strftime("%Y-%m-%d")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{profile}_{day.strftime('%Y%m%d')}.parquet"
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
        checkpoints = self.find_latest_checkpoints_for_date(day)
        written: Dict[str, str] = {}
        for profile, path in checkpoints.items():
            df = self.normalize_checkpoint(path, profile=profile)
            if df.empty:
                continue
            out = self._write_parquet(df, profile=profile, day=day)
            written[profile] = str(out)
        return written
