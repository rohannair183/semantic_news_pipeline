"""Application settings helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, cast

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.yaml_config_type import YAMLConfigType


@dataclass(frozen=True)
class Settings:
    """Typed settings container for application secrets and config values."""

    api_key: str
    base_url: str
    default_page_size: int
    max_page_size: int
    timeout_seconds: int

    @classmethod
    def load_settings(cls, load_dotenv: bool = True) -> "Settings":
        """Build settings from environment variables and YAML configuration."""
        if load_dotenv:
            cls._load_env_file()

        resolved_key = os.getenv("GUARDIAN_API_KEY")
        if not resolved_key:
            raise ValueError("API key for Open Guardian API is not set")

        config_values = cls.load_ingestion_config()
        base_url = str(config_values.get("base_url"))
        default_page_size = int(cast(Any, config_values.get("default_page_size")))
        max_page_size = int(cast(Any, config_values.get("max_page_size")))
        timeout_seconds = int(cast(Any, config_values.get("timeout_seconds")))

        if default_page_size < 1:
            raise ValueError("default_page_size must be >= 1")
        if max_page_size < 1:
            raise ValueError("max_page_size must be >= 1")
        if default_page_size > max_page_size:
            raise ValueError("default_page_size must be <= max_page_size")
        if timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")

        return cls(
            api_key=resolved_key,
            base_url=base_url,
            default_page_size=default_page_size,
            max_page_size=max_page_size,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _load_env_file() -> None:
        """Load environment variables from the repository root .env file."""
        if os.getenv("GUARDIAN_API_KEY"):
            return

        module_path = Path(__file__).resolve()
        env_path = module_path.parents[2] / ".env"
        if not env_path.is_file():
            return

        with env_path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

    @classmethod
    def load_ingestion_config(cls) -> Dict[str, Any]:
        """Load Guardian ingestion settings from YAML config."""
        parser = YAMLConfigParser()
        return parser.parse(
            config_type=YAMLConfigType.INGESTION,
            filename="ingestion_config.yaml",
        )

    @classmethod
    def load_ingestion_config_from_root(
        cls,
        configuration_root: Optional[Path],
    ) -> Dict[str, Any]:
        """Load Guardian ingestion settings from a specific configuration root."""
        parser = YAMLConfigParser(configuration_root=configuration_root)
        return parser.parse(
            config_type=YAMLConfigType.INGESTION,
            filename="ingestion_config.yaml",
        )

    @classmethod
    def load_article_ingestor_config(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> ArticleIngestorConfig:
        """Load and validate typed article ingestor configuration."""
        config_values = (
            cls.load_ingestion_config()
            if configuration_root is None
            else cls.load_ingestion_config_from_root(configuration_root)
        )

        profile_names = cls._load_profile_names(config_values)
        article_ingestor_config = cls._load_optional_section(
            config_values,
            section_name="article_ingestor",
        )

        selected_profiles = article_ingestor_config.get("profiles_to_run")
        if selected_profiles is None:
            profiles_to_run = profile_names
        else:
            profiles_to_run = cls._load_selected_profiles(selected_profiles, profile_names)

        limit_per_profile = cls._load_optional_positive_int(
            article_ingestor_config.get("limit_per_profile"),
            field_name="article_ingestor.limit_per_profile",
        )

        save_local_checkpoint = article_ingestor_config.get("save_local_checkpoint", False)
        if not isinstance(save_local_checkpoint, bool):
            raise ValueError(
                "Ingestion config field 'article_ingestor.save_local_checkpoint' must be a boolean"
            )

        checkpoint_dir: Optional[Path] = None
        if save_local_checkpoint:
            checkpoint_dir_value = article_ingestor_config.get(
                "checkpoint_dir",
                "checkpoints/article_ingestor",
            )
            checkpoint_dir = Path(str(checkpoint_dir_value))

        return ArticleIngestorConfig(
            profile_names=profile_names,
            profiles_to_run=profiles_to_run,
            limit_per_profile=limit_per_profile,
            save_local_checkpoint=save_local_checkpoint,
            checkpoint_dir=checkpoint_dir,
        )

    @classmethod
    def load_article_normalizer_config(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> ArticleNormalizerConfig:
        """Load and validate typed article normalizer configuration."""
        config_values = (
            cls.load_ingestion_config()
            if configuration_root is None
            else cls.load_ingestion_config_from_root(configuration_root)
        )

        profile_names = cls._load_profile_names(config_values)
        article_ingestor_config = cls._load_optional_section(
            config_values,
            section_name="article_ingestor",
        )
        article_normalizer_config = cls._load_required_section(
            config_values,
            section_name="article_normalizer",
        )

        row_mappings = article_normalizer_config.get("row_mappings")
        if not isinstance(row_mappings, dict) or not row_mappings:
            raise ValueError("Ingestion config must contain a non-empty 'row_mappings' mapping")

        checkpoint_dir = Path(
            str(
                article_ingestor_config.get(
                    "checkpoint_dir",
                    "checkpoints/article_ingestor",
                )
            )
        )
        parquet_dir = Path(str(article_ingestor_config.get("parquet_dir", "checkpoints/parquet")))

        return ArticleNormalizerConfig(
            profile_names=profile_names,
            checkpoint_dir=checkpoint_dir,
            parquet_dir=parquet_dir,
            row_mappings=row_mappings,
        )

    @staticmethod
    def _load_required_section(config_values: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        section_value = config_values.get(section_name)
        if not isinstance(section_value, dict):
            raise ValueError(f"Ingestion config must contain an '{section_name}' mapping")
        return section_value

    @staticmethod
    def _load_optional_section(config_values: Dict[str, Any], section_name: str) -> Dict[str, Any]:
        section_value = config_values.get(section_name)
        if section_value is None:
            return {}
        if not isinstance(section_value, dict):
            raise ValueError(f"Ingestion config field '{section_name}' must be a mapping")
        return section_value

    @staticmethod
    def _load_profile_names(config_values: Dict[str, Any]) -> list[str]:
        profiles = config_values.get("profiles")
        if not isinstance(profiles, dict) or not profiles:
            raise ValueError("Ingestion config must define a non-empty 'profiles' mapping")
        return [str(profile_name) for profile_name in profiles.keys()]

    @staticmethod
    def _load_selected_profiles(selected_profiles: Any, profile_names: list[str]) -> list[str]:
        if not isinstance(selected_profiles, list):
            raise ValueError(
                "Ingestion config field 'article_ingestor.profiles_to_run' must be a list"
            )
        if not selected_profiles:
            raise ValueError(
                "Ingestion config field 'article_ingestor.profiles_to_run' must not be empty"
            )

        unknown_profiles = [name for name in selected_profiles if str(name) not in profile_names]
        if unknown_profiles:
            unknown_values = ", ".join(str(name) for name in unknown_profiles)
            raise ValueError(f"Unknown ingestion profiles requested: {unknown_values}")
        return [str(name) for name in selected_profiles]

    @staticmethod
    def _load_optional_positive_int(value: Any, field_name: str) -> Optional[int]:
        if value is None:
            return None
        try:
            resolved_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ingestion config field '{field_name}' must be an integer") from exc
        if resolved_value < 1:
            raise ValueError(f"Ingestion config field '{field_name}' must be >= 1")
        return resolved_value


@dataclass(frozen=True)
class ArticleIngestorConfig:
    """Typed configuration for article ingestion orchestration."""

    profile_names: list[str]
    profiles_to_run: list[str]
    limit_per_profile: Optional[int]
    save_local_checkpoint: bool
    checkpoint_dir: Optional[Path]


@dataclass(frozen=True)
class ArticleNormalizerConfig:
    """Typed configuration for article checkpoint normalization."""

    profile_names: list[str]
    checkpoint_dir: Path
    parquet_dir: Path
    row_mappings: Dict[str, Dict[str, Any]]
