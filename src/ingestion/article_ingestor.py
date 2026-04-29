"""Article ingestion orchestration for Guardian API content."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.config.settings import Settings
from src.ingestion.guardian_client import GuardianClient
from src.utils.dates import utc_now_checkpoint_token


class ArticleIngestor:
    """Orchestrate configured profile ingestion and optional local checkpointing."""

    def __init__(self):
        """Initialize ingestion dependencies."""
        self.client = GuardianClient()
        self.config = Settings.load_article_ingestor_config()
        self.run_timestamp = utc_now_checkpoint_token()
        self._seen_ids: Set[str] = set()

    @property
    def profiles_to_run(self) -> List[str]:
        """Return the ordered profile names scheduled for ingestion."""
        return list(self.config.profiles_to_run)

    @property
    def resolved_limit(self) -> Optional[int]:
        """Return the optional per-profile item limit."""
        return self.config.limit_per_profile

    @property
    def checkpoint_directory(self) -> Optional[Path]:
        """Return the optional local checkpoint directory."""
        return self.config.checkpoint_dir

    @property
    def id_manifest_path(self) -> Optional[Path]:
        """Return the file path used to persist ingested IDs."""
        if self.checkpoint_directory is None:
            return None
        return self.checkpoint_directory.parent / "ingested_ids" / "ingested_ids.json"

    def _load_ingested_id_manifest(self) -> Set[str]:
        """Load persisted ingested IDs."""
        manifest_path = self.id_manifest_path
        if manifest_path is None or not manifest_path.is_file():
            return set()
        with manifest_path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        ids_payload = payload.get("ids", [])
        if not isinstance(ids_payload, list):
            return set()
        return {str(value) for value in ids_payload}

    def _write_ingested_id_manifest(self, ids: Set[str]) -> Optional[str]:
        """Persist ingested IDs as one global list."""
        manifest_path = self.id_manifest_path
        if manifest_path is None:
            return None
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": self.run_timestamp,
            "ids": sorted(ids),
        }
        with manifest_path.open("w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
        return str(manifest_path)

    def collect_profile_articles(self, profile: str, limit: Optional[int]) -> Dict[str, Any]:
        """Collect full article content for a configured profile.

        Parameters:
            profile: Profile name to ingest.
            limit: Optional max number of newly ingested items to fetch.

        Returns:
            Dict[str, Any]: Profile result with item and failure summaries.
        """
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        searched_count = 0
        skipped_existing_count = 0

        for topic_item in self.client.iter_topic_articles(
            profile=profile,
            limit=None,
        ):
            searched_count += 1
            content_id = topic_item.get("id")
            if not content_id:
                failures.append({"id": "", "error": "Missing id in topic search result"})
                continue
            content_id_str = str(content_id)
            if content_id_str in self._seen_ids:
                skipped_existing_count += 1
                continue
            try:
                full_item = self.client.get_article_by_id(
                    profile=profile,
                    content_id=content_id_str,
                )
                items.append(full_item)
                self._seen_ids.add(content_id_str)
                if limit is not None and len(items) >= limit:
                    break
            except Exception as exc:  # pylint: disable=broad-except
                failures.append({"id": content_id_str, "error": str(exc)})

        return {
            "profile": profile,
            "searched_count": searched_count,
            "fetched_count": len(items),
            "skipped_existing_count": skipped_existing_count,
            "failed_count": len(failures),
            "items": items,
            "failures": failures,
        }

    def write_profile_checkpoint(
        self,
        profile_result: Dict[str, Any],
        checkpoint_directory: Path,
        run_timestamp: str,
    ) -> str:
        """Write one profile result checkpoint as JSON.

        Parameters:
            profile_result: Structured ingestion result for one profile.
            checkpoint_directory: Destination directory for checkpoint files.
            run_timestamp: UTC timestamp token used in generated filenames.

        Returns:
            str: Path to the written checkpoint file.
        """
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        profile = str(profile_result.get("profile", "unknown_profile"))
        safe_profile = profile.replace("/", "_")
        checkpoint_path = checkpoint_directory / f"{safe_profile}_{run_timestamp}.json"
        with checkpoint_path.open("w", encoding="utf-8") as output_file:
            json.dump(profile_result, output_file, ensure_ascii=False, indent=2)
        return str(checkpoint_path)

    def run(self) -> Dict[str, Any]:
        """Run ingestion across selected profiles.

        Returns:
            Dict[str, Any]: Top-level ingestion summary and per-profile results.
        """
        profile_results: List[Dict[str, Any]] = []
        checkpoint_files: List[str] = []
        self._seen_ids = self._load_ingested_id_manifest()
        for profile in self.profiles_to_run:
            profile_result = self.collect_profile_articles(
                profile=profile,
                limit=self.resolved_limit,
            )
            profile_results.append(profile_result)
            if self.checkpoint_directory is not None:
                checkpoint_files.append(
                    self.write_profile_checkpoint(
                        profile_result=profile_result,
                        checkpoint_directory=self.checkpoint_directory,
                        run_timestamp=self.run_timestamp,
                    )
                )

        id_manifest_file = self._write_ingested_id_manifest(self._seen_ids)

        return {
            "profile_count": len(profile_results),
            "profiles_run": self.profiles_to_run,
            "limit_per_profile": self.resolved_limit,
            "checkpoint_files": checkpoint_files,
            "id_manifest_file": id_manifest_file,
            "searched_count": sum(result["searched_count"] for result in profile_results),
            "fetched_count": sum(result["fetched_count"] for result in profile_results),
            "skipped_existing_count": sum(
                result["skipped_existing_count"] for result in profile_results
            ),
            "failed_count": sum(result["failed_count"] for result in profile_results),
            "results": profile_results,
        }
