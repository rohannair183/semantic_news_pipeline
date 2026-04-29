"""Application settings helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, cast

from src.config.yaml_config_parser import YAMLConfigParser
from src.enums.article_row_source_kind import ArticleRowSourceKind
from src.enums.article_row_transform import ArticleRowTransform
from src.enums.guardian_order_by import GuardianOrderBy
from src.enums.ingestion_timeframe_mode import IngestionTimeframeMode
from src.enums.ingestion_timeframe_relative import IngestionTimeframeRelative
from src.enums.yaml_config_type import YAMLConfigType
from src.utils.dates import coerce_day, utc_today_date


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
        raw_topic = raw_profile.get("topic", "")
        if not isinstance(raw_topic, str):
            raise ValueError(
                f"Profile '{profile_name}' field 'topic' must be a string when provided"
            )
        topic = raw_topic

        raw_page_size = raw_profile.get("page_size", default_page_size)
        page_size = cls._validate_page_size(
            page_size=int(raw_page_size),
            max_page_size=max_page_size,
        )

        use_next_fallback = raw_profile.get("use_next_fallback", True)
        if not isinstance(use_next_fallback, bool):
            raise ValueError(
                f"Profile '{profile_name}' field 'use_next_fallback' must be a boolean"
            )

        raw_order_by = raw_profile.get("order_by", GuardianOrderBy.NEWEST.value)
        try:
            order_by = GuardianOrderBy.from_value(str(raw_order_by))
        except ValueError as exc:
            raise ValueError(f"Profile '{profile_name}' field 'order_by': {exc}") from exc

        raw_run_date = raw_profile.get("run_date")
        resolved_run_date = (
            None if raw_run_date is None else cls._coerce_profile_run_date(raw_run_date)
        )
        profile_from_date, profile_to_date = cls._resolve_profile_date_window(
            profile_name=profile_name,
            raw_profile=raw_profile,
            raw_run_date=resolved_run_date,
        )
        return GuardianProfileConfig(
            topic=topic,
            run_date=resolved_run_date,
            from_date=profile_from_date,
            to_date=profile_to_date,
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
            content_show_fields=cls._load_optional_profile_string(
                profile_name=profile_name,
                raw_profile=raw_profile,
                field_name="content_show_fields",
                default="all",
            ),
        )

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

        profile_from_date = None if raw_from_date is None else cls._coerce_profile_run_date(raw_from_date)
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
                raise ValueError(f"Ingestion config field '{field_prefix}.relative': {exc}") from exc
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
class IngestionTimeframe:
    """Resolved ingestion timeframe date window."""

    from_date: date
    to_date: date
