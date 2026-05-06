# pylint: disable=too-many-lines
"""Application settings helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, cast

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.article_row_source_kind import ArticleRowSourceKind
from src.enums.article_row_transform import ArticleRowTransform
from src.enums.guardian_order_by import GuardianOrderBy
from src.enums.ingestion_timeframe_mode import IngestionTimeframeMode
from src.enums.ingestion_timeframe_relative import IngestionTimeframeRelative
from src.enums.chunking_strategy import ChunkingStrategy
from src.enums.embedding_provider import EmbeddingProvider
from src.enums.pre_chunk_operation import PreChunkOperation
from src.enums.sentence_splitter_mode import SentenceSplitterMode
from src.enums.orchestrator_normalizer_day_token import OrchestratorNormalizerDayToken
from src.enums.orchestrator_task_kind import OrchestratorTaskKind
from src.enums.yaml_config_type import YAMLConfigType
from src.utils.dates import coerce_day, utc_today_date


@dataclass(frozen=True)
class Settings:
    """Typed settings container for application secrets and config values."""

    _INGESTION_CONFIG_FILES: ClassVar[tuple[str, ...]] = (
        "base.yaml",
        "profiles.yaml",
        "article_ingestor.yaml",
        "article_normalizer.yaml",
        "pre_chunk_preprocessor.yaml",
    )
    _INGESTION_LEGACY_CONFIG_FILE: ClassVar[str] = "ingestion_config.yaml"

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
        return cls._load_and_merge_ingestion_configs(parser)

    @classmethod
    def load_ingestion_config_from_root(
        cls,
        configuration_root: Optional[Path],
    ) -> Dict[str, Any]:
        """Load Guardian ingestion settings from a specific configuration root."""
        parser = YAMLConfigParser(configuration_root=configuration_root)
        return cls._load_and_merge_ingestion_configs(parser)

    @classmethod
    def _load_and_merge_ingestion_configs(cls, parser: YAMLConfigParser) -> Dict[str, Any]:
        """Load ingestion settings from split files plus optional legacy overrides."""
        merged_config: Dict[str, Any] = {}
        for filename in cls._INGESTION_CONFIG_FILES:
            section_values = parser.parse(
                config_type=YAMLConfigType.INGESTION,
                filename=filename,
            )
            merged_config = cls._deep_merge_dicts(merged_config, section_values)
        chunking_values = parser.parse(
            config_type=YAMLConfigType.CHUNKING,
            filename="chunking.yaml",
        )
        merged_config = cls._deep_merge_dicts(merged_config, chunking_values)
        legacy_values = parser.parse(
            config_type=YAMLConfigType.INGESTION,
            filename=cls._INGESTION_LEGACY_CONFIG_FILE,
        )
        return cls._deep_merge_dicts(merged_config, legacy_values)

    @classmethod
    def _deep_merge_dicts(cls, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two mapping values with updates taking precedence."""
        merged: Dict[str, Any] = dict(base)
        for key, value in updates.items():
            existing_value = merged.get(key)
            if isinstance(existing_value, dict) and isinstance(value, dict):
                merged[key] = cls._deep_merge_dicts(existing_value, value)
                continue
            merged[key] = value
        return merged

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

        enable_usage_logging = article_ingestor_config.get("enable_usage_logging", False)
        if not isinstance(enable_usage_logging, bool):
            raise ValueError(
                "Ingestion config field 'article_ingestor.enable_usage_logging' must be a boolean"
            )
        logs_dir = Path(str(article_ingestor_config.get("logs_dir", "logs")))

        return ArticleIngestorConfig(
            profile_names=profile_names,
            profiles_to_run=profiles_to_run,
            limit_per_profile=limit_per_profile,
            save_local_checkpoint=save_local_checkpoint,
            checkpoint_dir=checkpoint_dir,
            enable_usage_logging=enable_usage_logging,
            logs_dir=logs_dir,
        )

    @classmethod
    def load_guardian_profile_configs(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> dict[str, GuardianProfileConfig]:
        """Load and validate typed Guardian search profile configuration."""
        config_values = (
            cls.load_ingestion_config()
            if configuration_root is None
            else cls.load_ingestion_config_from_root(configuration_root)
        )
        profile_values = config_values.get("profiles")
        if not isinstance(profile_values, dict) or not profile_values:
            raise ValueError("Ingestion config must define a non-empty 'profiles' mapping")
        default_page_size, max_page_size = cls._load_page_size_settings(config_values)

        resolved_profiles: dict[str, GuardianProfileConfig] = {}
        for raw_profile_name, raw_profile in profile_values.items():
            profile_name = str(raw_profile_name)
            resolved_profiles[profile_name] = cls._build_guardian_profile_config(
                profile_name=profile_name,
                raw_profile=raw_profile,
                default_page_size=default_page_size,
                max_page_size=max_page_size,
            )
        return resolved_profiles

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

        row_mappings = cls._load_row_mappings(article_normalizer_config.get("row_mappings"))

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

    @classmethod
    def load_pre_chunk_preprocessor_config(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> "PreChunkPreprocessorConfig":
        """Load and validate typed pre-chunk preprocessor configuration."""
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
        preprocessor_config = cls._load_required_section(
            config_values,
            section_name="pre_chunk_preprocessor",
        )
        raw_operations = preprocessor_config.get("operations")
        operations = cls._load_pre_chunk_operations(raw_operations)
        input_dir = Path(str(article_ingestor_config.get("parquet_dir", "checkpoints/parquet")))
        output_dir = Path(str(preprocessor_config.get("output_dir", "checkpoints/pre_chunk")))
        return PreChunkPreprocessorConfig(
            profile_names=profile_names,
            input_dir=input_dir,
            output_dir=output_dir,
            operations=operations,
        )

    @classmethod
    def load_chunking_config(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> "ChunkingConfig":
        """Load and validate typed chunking configuration with named profiles."""
        config_values = (
            cls.load_ingestion_config()
            if configuration_root is None
            else cls.load_ingestion_config_from_root(configuration_root)
        )
        profile_names = cls._load_profile_names(config_values)
        chunking_section = cls._load_required_section(
            config_values,
            section_name="chunking",
        )
        input_dir = Path(
            str(chunking_section.get("input_dir", "checkpoints/pre_chunk"))
        )
        output_dir = Path(
            str(chunking_section.get("output_dir", "checkpoints/chunked_parquet"))
        )
        text_columns = cls._load_non_empty_string_list(
            chunking_section.get("text_columns"),
            field_name="chunking.text_columns",
        )
        id_columns = cls._load_optional_string_list(
            chunking_section.get("id_columns"),
            field_name="chunking.id_columns",
        )
        profile_columns = cls._load_optional_string_list(
            chunking_section.get("profile_columns"),
            field_name="chunking.profile_columns",
        )
        passthrough_columns = cls._load_optional_string_list(
            chunking_section.get("passthrough_columns"),
            field_name="chunking.passthrough_columns",
        )
        chunking_profiles = cls._load_chunking_profiles(chunking_section.get("profiles"))
        return ChunkingConfig(
            profile_names=profile_names,
            input_dir=input_dir,
            output_dir=output_dir,
            text_columns=text_columns,
            id_columns=id_columns,
            profile_columns=profile_columns,
            passthrough_columns=passthrough_columns,
            chunking_profiles=chunking_profiles,
        )

    @classmethod
    def load_embedding_config(
        cls,
        configuration_root: Optional[Path] = None,
    ) -> "EmbeddingConfig":
        """Load and validate typed embedding configuration."""
        parser = (
            YAMLConfigParser()
            if configuration_root is None
            else YAMLConfigParser(configuration_root=configuration_root)
        )
        raw = parser.parse(
            config_type=YAMLConfigType.EMBEDDINGS,
            filename="embeddings.yaml",
        )
        section = cls._load_required_section(raw, section_name="embeddings")
        input_dir = Path(
            str(section.get("input_dir", "checkpoints/chunked_parquet"))
        )
        output_dir = Path(
            str(section.get("output_dir", "checkpoints/embeddings"))
        )
        text_column = cls._load_non_empty_string(
            section.get("text_column"),
            field_name="embeddings.text_column",
        )
        raw_provider = section.get("provider")
        if raw_provider is None:
            raise ValueError(
                "Embedding config field 'embeddings.provider' is required"
            )
        try:
            provider = EmbeddingProvider.from_value(str(raw_provider))
        except ValueError as exc:
            raise ValueError(
                f"Embedding config field 'embeddings.provider': {exc}"
            ) from exc
        model_name = cls._load_non_empty_string(
            section.get("model_name"),
            field_name="embeddings.model_name",
        )
        batch_size = cls._load_positive_int(
            section.get("batch_size"),
            field_name="embeddings.batch_size",
        )
        return EmbeddingConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            text_column=text_column,
            provider=provider,
            model_name=model_name,
            batch_size=batch_size,
        )

    @classmethod
    def load_orchestrator_config(
        cls,
        configuration_root: Optional[Path] = None,
        filename: str = "orchestrator.yaml",
    ) -> "OrchestratorConfig":
        """Load and validate typed YAML orchestrator pipeline configuration."""
        parser = (
            YAMLConfigParser()
            if configuration_root is None
            else YAMLConfigParser(configuration_root=configuration_root)
        )
        raw = parser.parse(
            config_type=YAMLConfigType.APPLICATION,
            filename=filename,
        )
        return cls._parse_orchestrator_config_mapping(raw)

    @classmethod
    def load_orchestrator_config_from_path(cls, config_path: Path) -> "OrchestratorConfig":
        """Load orchestrator configuration from an explicit YAML file path."""
        raw = YAMLConfigParser().parse_path(config_path)
        return cls._parse_orchestrator_config_mapping(raw)

    @classmethod
    def _parse_orchestrator_config_mapping(cls, raw: Dict[str, Any]) -> "OrchestratorConfig":
        if not raw:
            raise ValueError("Orchestrator YAML must be a non-empty mapping")
        fail_fast = raw.get("fail_fast", True)
        if not isinstance(fail_fast, bool):
            raise ValueError("Orchestrator field 'fail_fast' must be a boolean")
        raw_tasks = raw.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("Orchestrator field 'tasks' must be a non-empty list")
        tasks: list[OrchestratorTaskSpec] = []
        for index, raw_task in enumerate(raw_tasks):
            tasks.append(
                cls._parse_orchestrator_task_spec(
                    raw_task=raw_task,
                    index=index,
                )
            )
        return OrchestratorConfig(
            fail_fast=fail_fast,
            tasks=tuple(tasks),
        )

    @classmethod
    def _parse_orchestrator_task_spec(
        cls,
        raw_task: Any,
        index: int,
    ) -> OrchestratorTaskSpec:
        prefix = f"orchestrator.tasks[{index}]"
        if not isinstance(raw_task, dict):
            raise ValueError(f"{prefix} must be a mapping")
        raw_kind = raw_task.get("kind")
        if raw_kind is None:
            raise ValueError(f"{prefix} requires 'kind'")
        try:
            kind = OrchestratorTaskKind.from_value(str(raw_kind).strip())
        except ValueError as exc:
            raise ValueError(f"{prefix}.kind: {exc}") from exc
        raw_id = raw_task.get("id")
        if raw_id is None:
            task_id = kind.value
        else:
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"{prefix}.id must be a non-empty string when provided")
            task_id = raw_id.strip()
        enabled = raw_task.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"{prefix}.enabled must be a boolean")
        skip_when = cls._parse_orchestrator_skip_when(
            raw_task.get("skip_when"),
            field_prefix=f"{prefix}.skip_when",
        )
        params = cls._parse_orchestrator_task_params(
            kind=kind,
            raw_params=raw_task.get("params"),
            field_prefix=f"{prefix}.params",
        )
        return OrchestratorTaskSpec(
            task_id=task_id,
            kind=kind,
            enabled=enabled,
            skip_when=skip_when,
            params=params,
        )

    @classmethod
    def _parse_orchestrator_skip_when(
        cls,
        raw: Any,
        field_prefix: str,
    ) -> Optional[OrchestratorSkipWhen]:
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError(f"{field_prefix} must be a mapping when provided")
        allowed = {"missing_env_var"}
        unknown = set(raw.keys()) - allowed
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ValueError(f"{field_prefix} has unknown keys: {joined}")
        missing_env = raw.get("missing_env_var")
        if missing_env is None:
            return OrchestratorSkipWhen(missing_env_var=None)
        if not isinstance(missing_env, str) or not missing_env.strip():
            raise ValueError(
                f"{field_prefix}.missing_env_var must be a non-empty string when provided"
            )
        return OrchestratorSkipWhen(missing_env_var=missing_env.strip())

    @classmethod
    def _parse_orchestrator_task_params(
        cls,
        kind: OrchestratorTaskKind,
        raw_params: Any,
        field_prefix: str,
    ) -> OrchestratorTaskParams:
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ValueError(f"{field_prefix} must be a mapping when provided")
        profile = "default"
        if kind in (
            OrchestratorTaskKind.CHUNKING,
            OrchestratorTaskKind.EMBEDDINGS,
        ):
            allowed = {"profile"}
            unknown = set(raw_params.keys()) - allowed
            if unknown:
                joined = ", ".join(sorted(unknown))
                raise ValueError(f"{field_prefix} has unknown keys for {kind.value}: {joined}")
            raw_profile = raw_params.get("profile", "default")
            if not isinstance(raw_profile, str) or not raw_profile.strip():
                raise ValueError(f"{field_prefix}.profile must be a non-empty string")
            profile = raw_profile.strip()
            return OrchestratorTaskParams(
                profile=profile,
                normalizer_day_raw=None,
            )
        if kind == OrchestratorTaskKind.ARTICLE_NORMALIZER:
            allowed = {"day"}
            unknown = set(raw_params.keys()) - allowed
            if unknown:
                joined = ", ".join(sorted(unknown))
                raise ValueError(f"{field_prefix} has unknown keys for {kind.value}: {joined}")
            normalizer_day_raw = cls._parse_orchestrator_normalizer_day_value(
                raw_params.get("day"),
                field_prefix=f"{field_prefix}.day",
            )
            return OrchestratorTaskParams(
                profile=profile,
                normalizer_day_raw=normalizer_day_raw,
            )
        allowed_keys: set[str] = set()
        unknown_leaf = set(raw_params.keys()) - allowed_keys
        if unknown_leaf:
            joined = ", ".join(sorted(unknown_leaf))
            raise ValueError(f"{field_prefix} has unknown keys for {kind.value}: {joined}")
        return OrchestratorTaskParams(profile=profile, normalizer_day_raw=None)

    @staticmethod
    def _parse_orchestrator_normalizer_day_value(raw_day: Any, field_prefix: str) -> Optional[str]:
        if raw_day is None:
            return None
        if not isinstance(raw_day, str) or not raw_day.strip():
            raise ValueError(f"{field_prefix} must be a non-empty string when provided")
        token = raw_day.strip()
        if token == OrchestratorNormalizerDayToken.UTC_TODAY.value:
            return token
        coerce_day(token)
        return token

    @classmethod
    def _load_chunking_profiles(
        cls,
        raw_profiles: Any,
    ) -> Dict[str, "ChunkingProfileConfig"]:
        """Parse the chunking.profiles mapping into typed profile configs."""
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise ValueError(
                "Ingestion config must contain a non-empty 'chunking.profiles' mapping"
            )
        resolved: Dict[str, ChunkingProfileConfig] = {}
        for raw_name, raw_profile in raw_profiles.items():
            profile_name = str(raw_name)
            field_prefix = f"chunking.profiles.{profile_name}"
            if not isinstance(raw_profile, dict):
                raise ValueError(
                    f"Ingestion config field '{field_prefix}' must be a mapping"
                )
            raw_strategy = raw_profile.get(
                "strategy",
                ChunkingStrategy.SEMANTIC_SENTENCE.value,
            )
            try:
                strategy = ChunkingStrategy.from_value(str(raw_strategy))
            except ValueError as exc:
                raise ValueError(
                    f"Ingestion config field '{field_prefix}.strategy': {exc}"
                ) from exc
            raw_params = raw_profile.get("params")
            if not isinstance(raw_params, dict) or not raw_params:
                raise ValueError(
                    f"Ingestion config field '{field_prefix}.params' must be a non-empty mapping"
                )
            resolved[profile_name] = ChunkingProfileConfig(
                strategy=strategy,
                params=dict(raw_params),
            )
        return resolved

    @classmethod
    def _load_optional_string_list(cls, value: Any, field_name: str) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(f"Ingestion config field '{field_name}' must be a list")
        resolved: list[str] = []
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Ingestion config field '{field_name}[{index}]' must be a non-empty string"
                )
            resolved.append(str(item))
        return resolved

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
    def _load_positive_int(value: Any, field_name: str) -> int:
        """Parse ``value`` as a positive integer or raise ValueError."""
        try:
            resolved_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Config field '{field_name}' must be a positive integer"
            ) from exc
        if resolved_value < 1:
            raise ValueError(f"Config field '{field_name}' must be >= 1")
        return resolved_value

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

    @classmethod
    def _build_guardian_profile_config(
        cls,
        profile_name: str,
        raw_profile: Any,
        default_page_size: int,
        max_page_size: int,
    ) -> GuardianProfileConfig:
        if not isinstance(raw_profile, dict):
            raise ValueError(f"Profile '{profile_name}' must be a mapping")

        cls._validate_guardian_profile_keys(profile_name, raw_profile)
        topic = cls._load_profile_topic(profile_name=profile_name, raw_profile=raw_profile)
        page_size = cls._load_profile_page_size(
            raw_profile=raw_profile,
            default_page_size=default_page_size,
            max_page_size=max_page_size,
        )
        use_next_fallback = cls._load_profile_use_next_fallback(
            profile_name=profile_name,
            raw_profile=raw_profile,
        )
        order_by = cls._load_profile_order_by(profile_name=profile_name, raw_profile=raw_profile)
        resolved_run_date = cls._load_profile_run_date(raw_profile=raw_profile)
        from_date, to_date = cls._resolve_profile_date_window(
            profile_name=profile_name,
            raw_profile=raw_profile,
            raw_run_date=resolved_run_date,
        )
        content_show_fields = cls._load_profile_content_show_fields(
            profile_name=profile_name,
            raw_profile=raw_profile,
        )
        return GuardianProfileConfig(
            topic=topic,
            run_date=resolved_run_date,
            from_date=from_date,
            to_date=to_date,
            page_size=page_size,
            query=cls._load_optional_profile_string(
                profile_name=profile_name,
                raw_profile=raw_profile,
                field_name="query",
            ),
            section=cls._load_optional_profile_string(
                profile_name=profile_name,
                raw_profile=raw_profile,
                field_name="section",
                allow_empty=True,
            ),
            order_by=order_by,
            use_next_fallback=use_next_fallback,
            content_show_fields=content_show_fields,
        )

    @staticmethod
    def _load_profile_topic(profile_name: str, raw_profile: dict[str, Any]) -> str:
        raw_topic = raw_profile.get("topic", "")
        if not isinstance(raw_topic, str):
            raise ValueError(
                f"Profile '{profile_name}' field 'topic' must be a string when provided"
            )
        return raw_topic

    @classmethod
    def _load_profile_page_size(
        cls,
        raw_profile: dict[str, Any],
        default_page_size: int,
        max_page_size: int,
    ) -> int:
        raw_page_size = raw_profile.get("page_size", default_page_size)
        return cls._validate_page_size(
            page_size=int(raw_page_size),
            max_page_size=max_page_size,
        )

    @staticmethod
    def _load_profile_use_next_fallback(
        profile_name: str,
        raw_profile: dict[str, Any],
    ) -> bool:
        use_next_fallback = raw_profile.get("use_next_fallback", True)
        if not isinstance(use_next_fallback, bool):
            raise ValueError(
                f"Profile '{profile_name}' field 'use_next_fallback' must be a boolean"
            )
        return use_next_fallback

    @staticmethod
    def _load_profile_order_by(
        profile_name: str,
        raw_profile: dict[str, Any],
    ) -> GuardianOrderBy:
        raw_order_by = raw_profile.get("order_by", GuardianOrderBy.NEWEST.value)
        try:
            return GuardianOrderBy.from_value(str(raw_order_by))
        except ValueError as exc:
            raise ValueError(f"Profile '{profile_name}' field 'order_by': {exc}") from exc

    @classmethod
    def _load_profile_run_date(cls, raw_profile: dict[str, Any]) -> Optional[date]:
        raw_run_date = raw_profile.get("run_date")
        if raw_run_date is None:
            return None
        return cls._coerce_profile_run_date(raw_run_date)

    @classmethod
    def _load_profile_content_show_fields(
        cls,
        profile_name: str,
        raw_profile: dict[str, Any],
    ) -> str:
        content_show_fields = cls._load_optional_profile_string(
            profile_name=profile_name,
            raw_profile=raw_profile,
            field_name="content_show_fields",
            default="all",
        )
        if content_show_fields is None:
            return "all"
        return content_show_fields

    @staticmethod
    def _validate_guardian_profile_keys(
        profile_name: str,
        raw_profile: dict[Any, Any],
    ) -> None:
        allowed_keys = {
            "topic",
            "run_date",
            "from_date",
            "to_date",
            "timeframe",
            "page_size",
            "query",
            "section",
            "order_by",
            "use_next_fallback",
            "content_show_fields",
        }
        unknown_keys = sorted(
            str(key) for key in raw_profile.keys() if str(key) not in allowed_keys
        )
        if unknown_keys:
            unknown_values = ", ".join(unknown_keys)
            raise ValueError(
                f"Profile '{profile_name}' defines unsupported fields: {unknown_values}"
            )

    @staticmethod
    def _load_required_profile_string(
        profile_name: str,
        raw_profile: dict[str, Any],
        field_name: str,
    ) -> str:
        raw_value = raw_profile.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(
                f"Profile '{profile_name}' field '{field_name}' must be a non-empty string"
            )
        return raw_value

    @staticmethod
    def _load_optional_profile_string(
        profile_name: str,
        raw_profile: dict[str, Any],
        field_name: str,
        default: Optional[str] = None,
        allow_empty: bool = False,
    ) -> Optional[str]:
        raw_value = raw_profile.get(field_name, default)
        if raw_value is None:
            return None
        if not isinstance(raw_value, str):
            raise ValueError(
                f"Profile '{profile_name}' field '{field_name}' must be a non-empty string"
            )
        if not allow_empty and not raw_value.strip():
            raise ValueError(
                f"Profile '{profile_name}' field '{field_name}' must be a non-empty string"
            )
        return raw_value

    @classmethod
    def _resolve_profile_date_window(
        cls,
        profile_name: str,
        raw_profile: dict[str, Any],
        raw_run_date: Optional[date],
    ) -> tuple[Optional[date], Optional[date]]:
        raw_from_date = raw_profile.get("from_date")
        raw_to_date = raw_profile.get("to_date")
        raw_timeframe = raw_profile.get("timeframe")
        if raw_timeframe is not None and not isinstance(raw_timeframe, dict):
            raise ValueError(f"Profile '{profile_name}' field 'timeframe' must be a mapping")
        if raw_timeframe is not None and (
            raw_run_date is not None or raw_from_date is not None or raw_to_date is not None
        ):
            raise ValueError(
                f"Profile '{profile_name}' cannot define timeframe with run_date/from_date/to_date"
            )
        if raw_run_date is not None and (raw_from_date is not None or raw_to_date is not None):
            raise ValueError(
                f"Profile '{profile_name}' cannot define run_date with from_date/to_date"
            )

        if raw_timeframe is not None:
            timeframe = cls._load_timeframe(
                raw_timeframe=raw_timeframe,
                field_prefix=f"profiles.{profile_name}.timeframe",
                default_relative=None,
            )
            return timeframe.from_date, timeframe.to_date

        if raw_run_date is not None:
            return raw_run_date, raw_run_date

        profile_from_date = (
            None if raw_from_date is None else cls._coerce_profile_run_date(raw_from_date)
        )
        profile_to_date = None if raw_to_date is None else cls._coerce_profile_run_date(raw_to_date)
        if profile_from_date is None and profile_to_date is None:
            return None, None
        if profile_from_date is None or profile_to_date is None:
            raise ValueError(
                f"Profile '{profile_name}' must define both from_date and to_date together"
            )
        if profile_from_date > profile_to_date:
            raise ValueError(
                f"Profile '{profile_name}' field 'from_date' must be <= 'to_date'"
            )
        return profile_from_date, profile_to_date

    @classmethod
    def _load_timeframe(
        cls,
        raw_timeframe: Any,
        field_prefix: str,
        default_relative: Optional[IngestionTimeframeRelative],
    ) -> "IngestionTimeframe":
        if raw_timeframe is None:
            if default_relative is None:
                raise ValueError(f"Ingestion config field '{field_prefix}' must be provided")
            return cls._build_relative_timeframe(default_relative)
        if not isinstance(raw_timeframe, dict):
            raise ValueError(f"Ingestion config field '{field_prefix}' must be a mapping")

        raw_mode = str(raw_timeframe.get("mode", IngestionTimeframeMode.RELATIVE.value))
        try:
            mode = IngestionTimeframeMode.from_value(raw_mode)
        except ValueError as exc:
            raise ValueError(f"Ingestion config field '{field_prefix}.mode': {exc}") from exc

        if mode == IngestionTimeframeMode.RELATIVE:
            fallback_relative = (
                default_relative.value
                if default_relative is not None
                else IngestionTimeframeRelative.PAST_DAY.value
            )
            raw_relative = str(raw_timeframe.get("relative", fallback_relative))
            try:
                relative = IngestionTimeframeRelative.from_value(raw_relative)
            except ValueError as exc:
                raise ValueError(
                    f"Ingestion config field '{field_prefix}.relative': {exc}"
                ) from exc
            return cls._build_relative_timeframe(relative)

        raw_from_date = raw_timeframe.get("from_date")
        raw_to_date = raw_timeframe.get("to_date")
        if raw_from_date is None or raw_to_date is None:
            raise ValueError(
                f"Ingestion config field '{field_prefix}' explicit mode requires "
                "'from_date' and 'to_date'"
            )
        from_date = coerce_day(raw_from_date)
        to_date = coerce_day(raw_to_date)
        if from_date > to_date:
            raise ValueError(
                f"Ingestion config field '{field_prefix}.from_date' must be <= "
                f"'{field_prefix}.to_date'"
            )
        return IngestionTimeframe(from_date=from_date, to_date=to_date)

    @staticmethod
    def _build_relative_timeframe(relative: IngestionTimeframeRelative) -> "IngestionTimeframe":
        today = utc_today_date()
        if relative == IngestionTimeframeRelative.PAST_DAY:
            return IngestionTimeframe(from_date=today, to_date=today)
        if relative == IngestionTimeframeRelative.PAST_WEEK:
            return IngestionTimeframe(from_date=today - timedelta(days=6), to_date=today)
        return IngestionTimeframe(from_date=today - timedelta(days=29), to_date=today)

    @staticmethod
    def _coerce_profile_run_date(raw_run_date: Any) -> date:
        return coerce_day(raw_run_date)

    @staticmethod
    def _validate_page_size(page_size: int, max_page_size: int) -> int:
        if page_size < 1 or page_size > max_page_size:
            raise ValueError(f"page_size must be between 1 and {max_page_size}")
        return page_size

    @staticmethod
    def _load_page_size_settings(config_values: dict[str, Any]) -> tuple[int, int]:
        default_page_size = int(cast(Any, config_values.get("default_page_size")))
        max_page_size = int(cast(Any, config_values.get("max_page_size")))
        if default_page_size < 1:
            raise ValueError("default_page_size must be >= 1")
        if max_page_size < 1:
            raise ValueError("max_page_size must be >= 1")
        if default_page_size > max_page_size:
            raise ValueError("default_page_size must be <= max_page_size")
        return default_page_size, max_page_size

    @classmethod
    def _load_row_mappings(
        cls,
        row_mappings: Any,
    ) -> dict[str, ArticleRowMappingConfig]:
        if not isinstance(row_mappings, dict) or not row_mappings:
            raise ValueError("Ingestion config must contain a non-empty 'row_mappings' mapping")

        resolved_row_mappings: dict[str, ArticleRowMappingConfig] = {}
        for raw_output_field, raw_field_config in row_mappings.items():
            output_field = str(raw_output_field)
            if not isinstance(raw_field_config, dict):
                raise ValueError(f"Row mapping '{output_field}' must be a mapping")

            raw_sources = raw_field_config.get("sources")
            if not isinstance(raw_sources, list) or not raw_sources:
                raise ValueError(
                    f"Row mapping '{output_field}' must contain a non-empty 'sources' list"
                )

            sources = [
                cls._parse_row_source_config(output_field=output_field, raw_source=raw_source)
                for raw_source in raw_sources
            ]
            transform = cls._parse_row_transform(
                output_field=output_field,
                raw_transform=raw_field_config.get("transform"),
            )
            resolved_row_mappings[output_field] = ArticleRowMappingConfig(
                sources=sources,
                transform=transform,
            )
        return resolved_row_mappings

    @staticmethod
    def _parse_row_source_config(
        output_field: str,
        raw_source: Any,
    ) -> ArticleRowSourceConfig:
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError(
                f"Row mapping '{output_field}' sources must contain non-empty strings"
            )
        if raw_source == ArticleRowSourceKind.PROFILE.value:
            return ArticleRowSourceConfig(kind=ArticleRowSourceKind.PROFILE)

        prefix, has_separator, path = raw_source.partition(".")
        if has_separator:
            if prefix not in {
                ArticleRowSourceKind.PAYLOAD.value,
                ArticleRowSourceKind.FIELDS.value,
                ArticleRowSourceKind.ITEM.value,
            }:
                raise ValueError(
                    f"Row mapping '{output_field}' source '{raw_source}' uses unsupported "
                    f"namespace '{prefix}'"
                )
            if not path:
                raise ValueError(
                    f"Row mapping '{output_field}' source '{raw_source}' must include a "
                    f"non-empty path"
                )
            return ArticleRowSourceConfig(
                kind=ArticleRowSourceKind.from_value(prefix),
                path=path,
            )

        return ArticleRowSourceConfig(
            kind=ArticleRowSourceKind.DIRECT_KEY,
            path=raw_source,
        )

    @staticmethod
    def _parse_row_transform(
        output_field: str,
        raw_transform: Any,
    ) -> Optional[ArticleRowTransform]:
        if raw_transform is None:
            return None
        if not isinstance(raw_transform, str) or not raw_transform:
            raise ValueError(
                f"Row mapping '{output_field}' field 'transform' must be a non-empty string"
            )
        try:
            return ArticleRowTransform.from_value(raw_transform)
        except ValueError as exc:
            raise ValueError(f"Row mapping '{output_field}' field 'transform': {exc}") from exc

    @classmethod
    def _load_pre_chunk_operations(
        cls,
        raw_operations: Any,
    ) -> list["PreChunkOperationConfig"]:
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError(
                "Ingestion config must contain a non-empty "
                "'pre_chunk_preprocessor.operations' list"
            )
        resolved_operations: list[PreChunkOperationConfig] = []
        for index, raw_operation in enumerate(raw_operations):
            field_prefix = f"pre_chunk_preprocessor.operations[{index}]"
            if not isinstance(raw_operation, dict):
                raise ValueError(f"Ingestion config field '{field_prefix}' must be a mapping")
            raw_name = raw_operation.get("name")
            if not isinstance(raw_name, str) or not raw_name:
                raise ValueError(
                    f"Ingestion config field '{field_prefix}.name' must be a non-empty string"
                )
            try:
                operation_name = PreChunkOperation.from_value(raw_name)
            except ValueError as exc:
                raise ValueError(
                    f"Ingestion config field '{field_prefix}.name': {exc}"
                ) from exc
            args = cls._load_pre_chunk_operation_args(
                field_prefix=field_prefix,
                operation_name=operation_name,
                raw_args=raw_operation.get("args"),
            )
            resolved_operations.append(
                PreChunkOperationConfig(
                    name=operation_name,
                    args=args,
                )
            )
        return resolved_operations

    @classmethod
    def _load_pre_chunk_operation_args(
        cls,
        field_prefix: str,
        operation_name: PreChunkOperation,
        raw_args: Any,
    ) -> dict[str, Any]:
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise ValueError(f"Ingestion config field '{field_prefix}.args' must be a mapping")
        resolved_args = dict(raw_args)
        if operation_name == PreChunkOperation.DROP_COLUMNS:
            resolved_args["columns"] = cls._load_non_empty_string_list(
                resolved_args.get("columns"),
                field_name=f"{field_prefix}.args.columns",
            )
        elif operation_name == PreChunkOperation.RENAME_COLUMNS:
            resolved_args["mapping"] = cls._load_non_empty_string_mapping(
                resolved_args.get("mapping"),
                field_name=f"{field_prefix}.args.mapping",
            )
        elif operation_name == PreChunkOperation.TRIM_WHITESPACE_COLUMNS:
            resolved_args["columns"] = cls._load_non_empty_string_list(
                resolved_args.get("columns"),
                field_name=f"{field_prefix}.args.columns",
            )
        elif operation_name == PreChunkOperation.DROP_EMPTY_ROWS:
            resolved_args["required_columns"] = cls._load_non_empty_string_list(
                resolved_args.get("required_columns"),
                field_name=f"{field_prefix}.args.required_columns",
            )
        elif operation_name == PreChunkOperation.FILTER_MIN_NUMERIC:
            resolved_args["column"] = cls._load_non_empty_string(
                resolved_args.get("column"),
                field_name=f"{field_prefix}.args.column",
            )
            resolved_args["min_value"] = cls._load_numeric_value(
                resolved_args.get("min_value"),
                field_name=f"{field_prefix}.args.min_value",
            )
        elif operation_name == PreChunkOperation.COALESCE_COLUMNS:
            resolved_args["target"] = cls._load_non_empty_string(
                resolved_args.get("target"),
                field_name=f"{field_prefix}.args.target",
            )
            resolved_args["sources"] = cls._load_non_empty_string_list(
                resolved_args.get("sources"),
                field_name=f"{field_prefix}.args.sources",
            )
        elif operation_name == PreChunkOperation.NORMALIZE_TEXT_COLUMNS:
            resolved_args["columns"] = cls._load_non_empty_string_list(
                resolved_args.get("columns"),
                field_name=f"{field_prefix}.args.columns",
            )
        return resolved_args

    @staticmethod
    def _load_non_empty_string(value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Ingestion config field '{field_name}' must be a non-empty string")
        return value

    @staticmethod
    def _load_numeric_value(value: Any, field_name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Ingestion config field '{field_name}' must be a numeric value"
            ) from exc

    @classmethod
    def _load_non_empty_string_list(cls, value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list) or not value:
            raise ValueError(f"Ingestion config field '{field_name}' must be a non-empty list")
        return [cls._load_non_empty_string(item, field_name=field_name) for item in value]

    @classmethod
    def _load_non_empty_string_mapping(cls, value: Any, field_name: str) -> dict[str, str]:
        if not isinstance(value, dict) or not value:
            raise ValueError(f"Ingestion config field '{field_name}' must be a non-empty mapping")
        resolved_mapping: dict[str, str] = {}
        for raw_key, raw_mapping_value in value.items():
            key = cls._load_non_empty_string(raw_key, field_name=field_name)
            mapped_value = cls._load_non_empty_string(raw_mapping_value, field_name=field_name)
            resolved_mapping[key] = mapped_value
        return resolved_mapping


@dataclass(frozen=True)
class ArticleIngestorConfig:
    """Typed configuration for article ingestion orchestration."""

    profile_names: list[str]
    profiles_to_run: list[str]
    limit_per_profile: Optional[int]
    save_local_checkpoint: bool
    checkpoint_dir: Optional[Path]
    enable_usage_logging: bool = False
    logs_dir: Path = Path("logs")


@dataclass(frozen=True)
class ArticleNormalizerConfig:
    """Typed configuration for article checkpoint normalization."""

    profile_names: list[str]
    checkpoint_dir: Path
    parquet_dir: Path
    row_mappings: dict[str, "ArticleRowMappingConfig"]


@dataclass(frozen=True)
class ArticleRowSourceConfig:
    """Typed row-mapping source selector parsed from YAML."""

    kind: ArticleRowSourceKind
    path: Optional[str] = None


@dataclass(frozen=True)
class ArticleRowMappingConfig:
    """Typed ArticleNormalizer row mapping parsed from YAML."""

    sources: list[ArticleRowSourceConfig]
    transform: Optional[ArticleRowTransform] = None


@dataclass(frozen=True)
class PreChunkOperationConfig:
    """Typed operation config parsed for pre-chunk preprocessing."""

    name: PreChunkOperation
    args: dict[str, Any]


@dataclass(frozen=True)
class PreChunkPreprocessorConfig:
    """Typed configuration for the pre-chunk parquet preprocessor."""

    profile_names: list[str]
    input_dir: Path
    output_dir: Path
    operations: list[PreChunkOperationConfig]


@dataclass(frozen=True)
class SemanticChunkingParams:
    """YAML-driven parameters for semantic sentence chunking."""

    min_chars: int
    max_chars: int
    overlap_chars: int
    similarity_threshold: float
    sentence_splitter: SentenceSplitterMode


@dataclass(frozen=True)
class ChunkingProfileConfig:
    """Typed per-profile chunking strategy and parameters."""

    strategy: ChunkingStrategy
    params: Dict[str, Any]


@dataclass(frozen=True)
class ChunkingConfig:  # pylint: disable=too-many-instance-attributes
    """Typed configuration for post-pre_chunk chunking with named profiles."""

    profile_names: list[str]
    input_dir: Path
    output_dir: Path
    text_columns: list[str]
    id_columns: list[str]
    profile_columns: list[str]
    passthrough_columns: list[str]
    chunking_profiles: dict[str, ChunkingProfileConfig]


@dataclass(frozen=True)
class GuardianProfileConfig:  # pylint: disable=too-many-instance-attributes
    """Resolved query settings for a named Guardian search profile."""

    topic: str
    page_size: int
    run_date: Optional[date] = None
    from_date: Optional[date] = None
    to_date: Optional[date] = None
    query: Optional[str] = None
    section: Optional[str] = None
    order_by: GuardianOrderBy = GuardianOrderBy.NEWEST
    use_next_fallback: bool = True
    content_show_fields: str = "all"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Typed configuration for the embedding pipeline."""

    input_dir: Path
    output_dir: Path
    text_column: str
    provider: EmbeddingProvider
    model_name: str
    batch_size: int


@dataclass(frozen=True)
class IngestionTimeframe:
    """Resolved ingestion timeframe date window."""

    from_date: date
    to_date: date


@dataclass(frozen=True)
class OrchestratorSkipWhen:
    """Optional guard parsed from orchestrator YAML ``skip_when``."""

    missing_env_var: Optional[str] = None


@dataclass(frozen=True)
class OrchestratorTaskParams:
    """Per-task params for orchestrator YAML tasks."""

    profile: str = "default"
    normalizer_day_raw: Optional[str] = None


@dataclass(frozen=True)
class OrchestratorTaskSpec:
    """Single declarative orchestrator pipeline task."""

    task_id: str
    kind: OrchestratorTaskKind
    enabled: bool
    skip_when: Optional[OrchestratorSkipWhen]
    params: OrchestratorTaskParams


@dataclass(frozen=True)
class OrchestratorConfig:
    """Typed declarative orchestrator pipeline loaded from YAML."""

    fail_fast: bool
    tasks: tuple[OrchestratorTaskSpec, ...]
