"""Unit tests for Settings.load_vector_search_config."""

import tempfile
import unittest
from pathlib import Path

from src.config.settings import Settings, VectorSearchConfig


def _write_vector_search_yaml(
    config_root: Path,
    content: str,
    filename: str = "vector_search.yaml",
) -> Path:
    """Write vector_search YAML under a temporary configuration root."""
    section_dir = config_root / "service_layer"
    section_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = section_dir / filename
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


_VALID_YAML = """\
vector_search:
  bucket_name: my-bucket
  index_name: my-index
"""


class TestLoadVectorSearchConfigHappyPath(unittest.TestCase):
    """This class tests load_vector_search_config."""

    def test_load_returns_typed_config(self) -> None:
        """load_vector_search_config: returns typed config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, _VALID_YAML)
            cfg = Settings.load_vector_search_config(configuration_root=root)

        self.assertIsInstance(cfg, VectorSearchConfig)
        self.assertEqual(cfg.bucket_name, "my-bucket")
        self.assertEqual(cfg.index_name, "my-index")
        self.assertEqual(cfg.date_metadata_key, "source_day")

    def test_load_parses_date_metadata_key(self) -> None:
        """load_vector_search_config: parses optional date_metadata_key."""
        yaml = """\
vector_search:
  bucket_name: b
  index_name: i
  date_metadata_key: published_at
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, yaml)
            cfg = Settings.load_vector_search_config(configuration_root=root)

        self.assertEqual(cfg.date_metadata_key, "published_at")

    def test_load_custom_filename(self) -> None:
        """load_vector_search_config: respects filename argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, _VALID_YAML, filename="alt.yaml")
            cfg = Settings.load_vector_search_config(
                configuration_root=root,
                filename="alt.yaml",
            )

        self.assertEqual(cfg.bucket_name, "my-bucket")
        self.assertEqual(cfg.index_name, "my-index")


class TestLoadVectorSearchConfigErrors(unittest.TestCase):
    """This class tests load_vector_search_config validation errors."""

    def test_raises_when_section_missing(self) -> None:
        """load_vector_search_config: requires vector_search section."""
        minimal = """other: {}\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, minimal)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_vector_search_config(configuration_root=root)
            self.assertIn("vector_search", str(ctx.exception))

    def test_raises_when_file_missing(self) -> None:
        """load_vector_search_config: requires YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "service_layer").mkdir()
            with self.assertRaises(ValueError) as ctx:
                Settings.load_vector_search_config(configuration_root=root)
            self.assertIn("vector_search", str(ctx.exception))

    def test_raises_when_bucket_name_missing(self) -> None:
        """load_vector_search_config: requires bucket_name."""
        yaml = """\
vector_search:
  index_name: i
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_vector_search_config(configuration_root=root)
            self.assertIn("bucket_name", str(ctx.exception))

    def test_raises_when_index_name_missing(self) -> None:
        """load_vector_search_config: requires index_name."""
        yaml = """\
vector_search:
  bucket_name: b
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_vector_search_config(configuration_root=root)
            self.assertIn("index_name", str(ctx.exception))

    def test_raises_when_date_metadata_key_empty(self) -> None:
        """load_vector_search_config: rejects blank date_metadata_key when set."""
        yaml = """\
vector_search:
  bucket_name: b
  index_name: i
  date_metadata_key: "   "
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_vector_search_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_vector_search_config(configuration_root=root)
            self.assertIn("date_metadata_key", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
