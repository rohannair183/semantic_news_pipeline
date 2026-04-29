"""Shared test config helpers for ingestion unit tests."""

from typing import Any, Dict, Optional


def build_ingestion_config(
    *,
    profiles_to_run: Optional[list[str]] = None,
    limit_per_profile: Optional[int] = None,
    save_local_checkpoint: bool = False,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical ingestion config for unit tests."""
    config: Dict[str, Any] = {
        "profiles": {
            "technology_daily": {"topic": "technology"},
            "science_daily": {"topic": "science"},
        },
        "article_ingestor": {"save_local_checkpoint": save_local_checkpoint},
    }
    if profiles_to_run is not None:
        config["article_ingestor"]["profiles_to_run"] = profiles_to_run
    if limit_per_profile is not None:
        config["article_ingestor"]["limit_per_profile"] = limit_per_profile
    if checkpoint_dir is not None:
        config["article_ingestor"]["checkpoint_dir"] = checkpoint_dir
    return config
