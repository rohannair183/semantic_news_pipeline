"""Shared test config helpers for ingestion unit tests."""

from typing import Any, Dict, Optional

NORMALIZER_ROW_MAPPINGS: Dict[str, Dict[str, Any]] = {
    "profile": {"sources": ["profile", "payload.profile"]},
    "api_id": {"sources": ["id"]},
    "web_title": {"sources": ["webTitle", "fields.headline"]},
    "headline": {"sources": ["fields.headline"]},
    "byline": {"sources": ["fields.byline"]},
    "section": {"sources": ["sectionName"]},
    "published_at": {"sources": ["webPublicationDate"], "transform": "parse_iso"},
    "first_publication_date": {
        "sources": ["fields.firstPublicationDate"],
        "transform": "parse_iso",
    },
    "url": {"sources": ["webUrl"]},
    "body_text": {"sources": ["fields.bodyText", "fields.body"]},
    "trail_text": {"sources": ["fields.trailText"]},
    "thumbnail": {"sources": ["fields.thumbnail"]},
    "wordcount": {"sources": ["fields.wordcount"]},
    "pillar": {"sources": ["pillarName"]},
    "last_modified": {
        "sources": ["fields.lastModified", "lastModified"],
        "transform": "parse_iso",
    },
}


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


def build_normalizer_config(
    *,
    checkpoint_dir: str,
    parquet_dir: str,
) -> Dict[str, Any]:
    """Build a canonical ingestion config for ArticleNormalizer tests."""
    config = build_ingestion_config(save_local_checkpoint=False, checkpoint_dir=checkpoint_dir)
    config["article_ingestor"]["parquet_dir"] = parquet_dir
    config["article_normalizer"] = {"row_mappings": NORMALIZER_ROW_MAPPINGS.copy()}
    return config
