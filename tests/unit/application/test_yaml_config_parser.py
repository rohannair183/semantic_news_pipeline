"""Unit tests for YAMLConfigParser."""

import tempfile
import unittest
from pathlib import Path

from src.application.yaml_config_parser import YAMLConfigParser
from src.enums.yaml_config_type import YAMLConfigType


class TestYAMLConfigParser(unittest.TestCase):
    """This class tests YAML config parsing behavior."""

    def test_init_default_root_points_to_configuration_folder(self):
        """__init__: defaults to repository configuration folder when root not provided."""
        parser = YAMLConfigParser()
        self.assertEqual(parser.configuration_root.name, "configuration")
        self.assertTrue(parser.configuration_root.is_absolute())

    def test_get_config_path_uses_type_folder(self):
        """get_config_path: composes path from root, config type, and filename."""
        parser = YAMLConfigParser(configuration_root=Path("/tmp/config"))
        path = parser.get_config_path(YAMLConfigType.INGESTION, "guardian_client.yaml")
        self.assertEqual(path, Path("/tmp/config/ingestion/guardian_client.yaml"))

    def test_parse_and_parse_path_success_paths(self):
        """parse/parse_path: reads YAML mapping values for typed and explicit paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "ingestion"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "guardian_client.yaml"
            config_path.write_text(
                "base_url: https://content.guardianapis.com\ndefault_page_size: 10\n",
                encoding="utf-8",
            )

            parser = YAMLConfigParser(configuration_root=root)
            typed_values = parser.parse(YAMLConfigType.INGESTION, "guardian_client.yaml")
            explicit_values = parser.parse_path(config_path)

        self.assertEqual(typed_values["base_url"], "https://content.guardianapis.com")
        self.assertEqual(typed_values["default_page_size"], 10)
        self.assertEqual(explicit_values, typed_values)

    def test_parse_path_empty_and_missing_files(self):
        """parse_path: returns an empty mapping for empty or missing YAML files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            empty_yaml = root / "empty.yaml"
            empty_yaml.write_text("", encoding="utf-8")
            missing_yaml = root / "missing.yaml"

            parser = YAMLConfigParser(configuration_root=root)
            self.assertEqual(parser.parse_path(empty_yaml), {})
            self.assertEqual(parser.parse_path(missing_yaml), {})

    def test_parse_path_rejects_non_mapping_yaml(self):
        """parse_path: raises when the YAML root is not a dictionary mapping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "invalid.yaml"
            config_path.write_text("- one\n- two\n", encoding="utf-8")

            parser = YAMLConfigParser(configuration_root=Path(tmp_dir))
            with self.assertRaises(ValueError):
                parser.parse_path(config_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
