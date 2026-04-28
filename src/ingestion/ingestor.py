"""
Ingestor for handling data ingestion from the Open Guardian API. This module defines the
`Ingestor` class, which orchestrates configured profile runs and returns fetched articles.
"""

from typing import Any, Dict, List, Optional

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.yaml_config_type import YAMLConfigType
from src.ingestion.guardian_client import GuardianClient


class Ingestor:
    """
    Orchestrate Guardian ingestion for configured profile topics.

    This class loads YAML ingestion configuration, determines which profiles to run,
    iterates profile article ids from the Guardian search API, and fetches full content
    for each id.
    """

    def __init__(self):
        """Initialize ingestion dependencies."""
        self.client = GuardianClient()
        self.parser = YAMLConfigParser()

    def _load_config(self) -> Dict[str, Any]:
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

        ingestor_config = config.get("ingestor", {})
        if ingestor_config is None:
            ingestor_config = {}
        if not isinstance(ingestor_config, dict):
            raise ValueError("Ingestion config field 'ingestor' must be a mapping")

        selected_profiles = ingestor_config.get("profiles_to_run")
        if selected_profiles is None:
            return list(profiles.keys())
        if not isinstance(selected_profiles, list):
            raise ValueError("Ingestion config field 'ingestor.profiles_to_run' must be a list")
        if not selected_profiles:
            raise ValueError("Ingestion config field 'ingestor.profiles_to_run' must not be empty")

        unknown_profiles = [name for name in selected_profiles if name not in profiles]
        if unknown_profiles:
            unknown_values = ", ".join(str(name) for name in unknown_profiles)
            raise ValueError(f"Unknown ingestion profiles requested: {unknown_values}")
        return [str(name) for name in selected_profiles]

    def _resolve_limit(self, config: Dict[str, Any]) -> Optional[int]:
        """Resolve item limit precedence.

        Parameters:
            config: Parsed ingestion configuration mapping.

        Returns:
            Optional[int]: Limit resolved from YAML config or None.
        """
        ingestor_config = config.get("ingestor", {})
        if ingestor_config is None:
            return None
        if not isinstance(ingestor_config, dict):
            raise ValueError("Ingestion config field 'ingestor' must be a mapping")

        configured_limit = ingestor_config.get("limit_per_profile")
        if configured_limit is None:
            return None

        resolved_limit = int(configured_limit)
        if resolved_limit < 1:
            raise ValueError("Ingestion config field 'ingestor.limit_per_profile' must be >= 1")
        return resolved_limit

    def _collect_profile_articles(self, profile: str, limit: Optional[int]) -> Dict[str, Any]:
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

    def run(self) -> Dict[str, Any]:
        """Run ingestion across selected profiles.

        Returns:
            Dict[str, Any]: Top-level ingestion summary and per-profile results.
        """
        config = self._load_config()
        profiles_to_run = self._resolve_profiles_to_run(config)
        resolved_limit = self._resolve_limit(config=config)

        profile_results = [
            self._collect_profile_articles(profile=profile, limit=resolved_limit)
            for profile in profiles_to_run
        ]

        return {
            "profile_count": len(profile_results),
            "profiles_run": profiles_to_run,
            "limit_per_profile": resolved_limit,
            "searched_count": sum(result["searched_count"] for result in profile_results),
            "fetched_count": sum(result["fetched_count"] for result in profile_results),
            "failed_count": sum(result["failed_count"] for result in profile_results),
            "results": profile_results,
        }
