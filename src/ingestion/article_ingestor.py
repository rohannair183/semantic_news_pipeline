"""Article ingestion orchestration for Guardian API content."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.yaml_config_type import YAMLConfigType
from src.ingestion.guardian_client import GuardianClient


class ArticleIngestor:
    """Orchestrate configured profile ingestion and optional local checkpointing."""

    def __init__(self):
        """Initialize ingestion dependencies."""
        self.client = GuardianClient()
        self.parser = YAMLConfigParser()
        self.config = self.load_config()
        self.profiles_to_run = self._resolve_profiles_to_run(self.config)
        self.resolved_limit = self._resolve_limit(config=self.config)
        self.checkpoint_directory = self._resolve_checkpoint_directory(config=self.config)
        self.run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    def load_config(self) -> Dict[str, Any]:
        """Load ingestion configuration from YAML.

        Returns:
            Dict[str, Any]: Parsed ingestion configuration mapping.
        """
        return self.parser.parse(
            config_type=YAMLConfigType.INGESTION,
            filename="ingestion_config.yaml",
        )

    def _resolve_profiles_to_run(self, config: Dict[str, Any]) -> List[str]:
        """Resolve profile names to ingest.

        Parameters:
            config: Parsed ingestion configuration mapping.

        Returns:
            List[str]: Ordered profile names that should be ingested.
        """
        profiles = config.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("Ingestion config must define a non-empty 'profiles' mapping")

        article_ingestor_config = config.get("article_ingestor", {})
        if article_ingestor_config is None:
            article_ingestor_config = {}
        if not isinstance(article_ingestor_config, dict):
            raise ValueError("Ingestion config field 'article_ingestor' must be a mapping")

        selected_profiles = article_ingestor_config.get("profiles_to_run")
        if selected_profiles is None:
            return list(profiles.keys())
        if not isinstance(selected_profiles, list):
            raise ValueError(
                "Ingestion config field 'article_ingestor.profiles_to_run' must be a list"
            )
        if not selected_profiles:
            raise ValueError(
                "Ingestion config field 'article_ingestor.profiles_to_run' must not be empty"
            )

        unknown_profiles = [name for name in selected_profiles if name not in profiles]
        if unknown_profiles:
            unknown_values = ", ".join(str(name) for name in unknown_profiles)
            raise ValueError(f"Unknown ingestion profiles requested: {unknown_values}")
        return [str(name) for name in selected_profiles]

    def _resolve_limit(self, config: Dict[str, Any]) -> Optional[int]:
        """Resolve optional per-profile item limit from config.

        Parameters:
            config: Parsed ingestion configuration mapping.

        Returns:
            Optional[int]: Limit resolved from YAML config or None.
        """
        article_ingestor_config = config.get("article_ingestor", {})
        if article_ingestor_config is None:
            return None
        if not isinstance(article_ingestor_config, dict):
            raise ValueError("Ingestion config field 'article_ingestor' must be a mapping")

        configured_limit = article_ingestor_config.get("limit_per_profile")
        if configured_limit is None:
            return None

        resolved_limit = int(configured_limit)
        if resolved_limit < 1:
            raise ValueError(
                "Ingestion config field 'article_ingestor.limit_per_profile' must be >= 1"
            )
        return resolved_limit

    def _resolve_checkpoint_directory(self, config: Dict[str, Any]) -> Optional[Path]:
        """Resolve optional local checkpoint directory from config.

        Parameters:
            config: Parsed ingestion configuration mapping.

        Returns:
            Optional[Path]: Directory path where checkpoints are written, or None.
        """
        article_ingestor_config = config.get("article_ingestor", {})
        if article_ingestor_config is None:
            return None
        if not isinstance(article_ingestor_config, dict):
            raise ValueError("Ingestion config field 'article_ingestor' must be a mapping")

        save_local_checkpoint = article_ingestor_config.get("save_local_checkpoint", False)
        if not isinstance(save_local_checkpoint, bool):
            raise ValueError(
                "Ingestion config field 'article_ingestor.save_local_checkpoint' must be a boolean"
            )
        if not save_local_checkpoint:
            return None

        checkpoint_dir = article_ingestor_config.get(
            "checkpoint_dir",
            "checkpoints/article_ingestor",
        )
        return Path(str(checkpoint_dir))

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
                full_item = self.client.get_article_by_id(content_id=str(content_id))
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
