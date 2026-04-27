"""Utilities for loading YAML configuration from the repository config directory."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.enums.yaml_config_type import YAMLConfigType


class YAMLConfigParser:
    """Parse YAML configuration files from the repository configuration directory."""

    def __init__(self, configuration_root: Optional[Path] = None):
        if configuration_root is None:
            module_path = Path(__file__).resolve()
            configuration_root = module_path.parents[2] / "configuration"
        self.configuration_root = configuration_root

    def get_config_path(self, config_type: YAMLConfigType, filename: str) -> Path:
        """Build a config file path from type + filename."""
        return self.configuration_root / str(config_type) / filename

    def parse(self, config_type: YAMLConfigType, filename: str) -> Dict[str, Any]:
        """Parse a typed config file under the configuration root."""
        return self.parse_path(self.get_config_path(config_type=config_type, filename=filename))

    def parse_path(self, config_path: Path) -> Dict[str, Any]:
        """Parse a YAML file path into a dictionary."""
        if not config_path.is_file():
            return {}

        with config_path.open("r", encoding="utf-8") as config_file:
            parsed_values = yaml.safe_load(config_file) or {}

        if not isinstance(parsed_values, dict):
            raise ValueError(f"YAML config at {config_path} must parse to a mapping")
        return parsed_values
