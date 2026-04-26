"""Application settings helpers."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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
        api_key: Optional[str] = None,
        env_var: str = "GUARDIAN_API_KEY",
        load_dotenv: bool = True,
        search_paths: Optional[Iterable[Path]] = None,
        config_path: Optional[Path] = None,
    ) -> "Settings":
        """Build settings from explicit args or environment variables."""
        if load_dotenv:
            cls._load_env_file(search_paths=search_paths)

        resolved_key = api_key or os.getenv(env_var)
        if not resolved_key:
            raise ValueError("API key for Open Guardian API is not set")

        config_values = cls._load_ingestion_config(config_path=config_path)
        base_url = str(config_values.get("base_url"))
        default_page_size = int(config_values.get("default_page_size"))
        max_page_size = int(config_values.get("max_page_size"))
        timeout_seconds = int(config_values.get("timeout_seconds"))

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
    def _load_env_file(search_paths: Optional[Iterable[Path]] = None) -> None:
        """Load environment variables from the first .env file found."""
        if os.getenv("GUARDIAN_API_KEY"):
            return

        if search_paths is None:
            module_path = Path(__file__).resolve()
            search_paths = [Path.cwd(), module_path.parent] + list(module_path.parents)

        seen_paths = set()
        for root in search_paths:
            env_path = (root / ".env").resolve()
            if env_path in seen_paths:
                continue
            seen_paths.add(env_path)
            if not env_path.is_file():
                continue

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
            break

    @staticmethod
    def _load_ingestion_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load Guardian ingestion settings from YAML config."""
        if config_path is None:
            module_path = Path(__file__).resolve()
            config_path = module_path.parents[2] / "configuration" / "ingestion" / "guardian_client.yaml"

        if not config_path.is_file():
            return {}

        parsed_values: Dict[str, Any] = {}
        with config_path.open("r", encoding="utf-8") as config_file:
            for raw_line in config_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if not key:
                    continue

                if value.startswith('"') and value.endswith('"'):
                    parsed_values[key] = value[1:-1]
                    continue
                if value.startswith("'") and value.endswith("'"):
                    parsed_values[key] = value[1:-1]
                    continue
                if value.isdigit():
                    parsed_values[key] = int(value)
                    continue

                parsed_values[key] = value

        return parsed_values
