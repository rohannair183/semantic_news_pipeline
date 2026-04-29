"""Application settings helpers."""

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
    def load_settings(
        cls,
        load_dotenv: bool = True,
    ) -> "Settings":
        """Build settings from environment variables and YAML configuration.

        Args:
            load_dotenv: Whether to load a root .env file before reading the
                environment.

        Returns:
            A validated Settings instance.
        """
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
        """Load environment variables from the repository root .env file.

        Returns:
            None.
        """
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
        """Load Guardian ingestion settings from YAML config.

        Returns:
            A dictionary of ingestion configuration values.
        """
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
        """Load Guardian ingestion settings from a specific configuration root.

        Args:
            configuration_root: Optional override for the configuration directory root.

        Returns:
            A dictionary of ingestion configuration values.
        """
        parser = YAMLConfigParser(configuration_root=configuration_root)
        return parser.parse(
            config_type=YAMLConfigType.INGESTION,
            filename="ingestion_config.yaml",
        )
