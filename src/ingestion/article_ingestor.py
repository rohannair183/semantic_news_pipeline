"""Article ingestion orchestration for Guardian API content."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    def collect_profile_articles(self, profile: str, limit: Optional[int]) -> Dict[str, Any]:
        """Collect full article content for a configured profile.

        Parameters:
            profile: Profile name to ingest.
            limit: Optional max number of search items to process.

        Returns:
            Dict[str, Any]: Profile result with item and failure summaries.
        """
        items: List[Dict[str, Any]] = []
        failures: List[Dict[str, str]] = []
        searched_count = 0

        for topic_item in self.client.iter_topic_articles(profile=profile, limit=limit):
            searched_count += 1
            content_id = topic_item.get("id")
            if not content_id:
                failures.append({"id": "", "error": "Missing id in topic search result"})
                continue
            try:
                full_item = self.client.get_article_by_id(
                    profile=profile,
                    content_id=str(content_id),
                )
                items.append(full_item)
            except Exception as exc:  # pylint: disable=broad-except
                failures.append({"id": str(content_id), "error": str(exc)})

        return {
            "profile": profile,
            "searched_count": searched_count,
            "fetched_count": len(items),
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

        return {
            "profile_count": len(profile_results),
            "profiles_run": self.profiles_to_run,
            "limit_per_profile": self.resolved_limit,
            "checkpoint_files": checkpoint_files,
            "searched_count": sum(result["searched_count"] for result in profile_results),
            "fetched_count": sum(result["fetched_count"] for result in profile_results),
            "failed_count": sum(result["failed_count"] for result in profile_results),
            "results": profile_results,
        }
