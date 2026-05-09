"""Unit tests for Settings.load_supabase_vector_sync_config."""

import tempfile
import unittest
from pathlib import Path

from src.config.settings import Settings, SupabaseVectorSyncConfig
from src.enums.vector_bucket_distance_metric import VectorBucketDistanceMetric


def _write_sync_yaml(config_root: Path, content: str) -> Path:
    """Write sync.yaml under a temporary configuration root."""
    section_dir = config_root / "vector_bucket"
    section_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = section_dir / "sync.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


_VALID_YAML = """\
supabase_vector_sync:
  input_dir: checkpoints/embeddings
  bucket_name: b1
  index_name: i1
  dimension: 384
  distance_metric: cosine
  embedding_column: embedding
  metadata_columns:
    - source_day
  batch_size: 100
  create_bucket_if_missing: true
  create_index_if_missing: false
"""


class TestLoadSupabaseVectorSyncConfigHappyPath(unittest.TestCase):
    """This class tests load_supabase_vector_sync_config."""

    def test_load_returns_typed_config(self) -> None:
        """load_supabase_vector_sync_config: returns typed config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, _VALID_YAML)
            cfg = Settings.load_supabase_vector_sync_config(configuration_root=root)

        self.assertIsInstance(cfg, SupabaseVectorSyncConfig)
        self.assertEqual(cfg.input_dir, Path("checkpoints/embeddings"))
        self.assertEqual(cfg.bucket_name, "b1")
        self.assertEqual(cfg.index_name, "i1")
        self.assertEqual(cfg.dimension, 384)
        self.assertEqual(cfg.distance_metric, VectorBucketDistanceMetric.COSINE)
        self.assertEqual(cfg.embedding_column, "embedding")
        self.assertEqual(cfg.key_columns, ("source_api_id", "chunk_index", "source_row_index"))
        self.assertEqual(cfg.metadata_columns, ("source_day",))
        self.assertEqual(cfg.batch_size, 100)
        self.assertTrue(cfg.create_bucket_if_missing)
        self.assertFalse(cfg.create_index_if_missing)

    def test_default_key_columns_when_omitted(self) -> None:
        """load_supabase_vector_sync_config: fills default key_columns."""
        yaml = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: euclidean
  embedding_column: embedding
  metadata_columns: []
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, yaml)
            cfg = Settings.load_supabase_vector_sync_config(configuration_root=root)
        self.assertEqual(
            cfg.key_columns,
            ("source_api_id", "chunk_index", "source_row_index"),
        )


class TestLoadSupabaseVectorSyncExplicitKeys(unittest.TestCase):
    """This class tests explicit key_columns parsing."""

    def test_custom_key_columns_parsed(self) -> None:
        """load_supabase_vector_sync_config: honors explicit key columns."""
        yaml = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 1
  distance_metric: l2
  embedding_column: e
  key_columns:
    - x
    - y
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, yaml)
            cfg = Settings.load_supabase_vector_sync_config(configuration_root=root)

        self.assertEqual(cfg.distance_metric, VectorBucketDistanceMetric.L2)
        self.assertEqual(cfg.key_columns, ("x", "y"))


class TestLoadSupabaseVectorSyncConfigErrors(unittest.TestCase):
    """This class tests load_supabase_vector_sync_config errors."""

    def test_raises_when_section_missing(self) -> None:
        """load_supabase_vector_sync_config: requires supabase_vector_sync."""
        minimal = """other: {}\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, minimal)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("supabase_vector_sync", str(ctx.exception))

    def test_raises_when_file_missing(self) -> None:
        """load_supabase_vector_sync_config: requires sync.yaml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "vector_bucket").mkdir()
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("supabase_vector_sync", str(ctx.exception))

    def test_requires_distance_metric(self) -> None:
        """load_supabase_vector_sync_config: rejects missing distance_metric."""
        yaml = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 1
  embedding_column: e
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("distance_metric", str(ctx.exception))

    def test_rejects_non_list_key_columns(self) -> None:
        """load_supabase_vector_sync_config: key_columns must list when present."""
        bad = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  key_columns: "not-a-list"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, bad)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("key_columns", str(ctx.exception))

    def test_rejects_blank_entry_in_key_columns(self) -> None:
        """load_supabase_vector_sync_config: entries must be strings."""
        bad = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  key_columns:
    - "   "
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, bad)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("key_columns[0]", str(ctx.exception))

    def test_rejects_unknown_distance_metric(self) -> None:
        """load_supabase_vector_sync_config: validates distance_metric."""
        bad = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 8
  distance_metric: cosmic
  embedding_column: e
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, bad)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("distance_metric", str(ctx.exception))

    def test_rejects_batch_over_500(self) -> None:
        """load_supabase_vector_sync_config: caps batch_size at 500."""
        bad = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  batch_size: 501
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, bad)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("batch_size", str(ctx.exception))

    def test_rejects_empty_explicit_key_columns(self) -> None:
        """load_supabase_vector_sync_config: empty key_columns list invalid."""
        bad = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  key_columns: []
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, bad)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("key_columns", str(ctx.exception))

    def test_rejects_bad_create_index_boolean(self) -> None:
        """create_index_if_missing must be strictly boolean."""
        yaml = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  create_bucket_if_missing: true
  create_index_if_missing: 1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("create_index_if_missing", str(ctx.exception))

    def test_rejects_bad_create_bucket_boolean(self) -> None:
        """load_supabase_vector_sync_config: create_bucket_if_missing strict."""
        yaml = """\
supabase_vector_sync:
  bucket_name: b
  index_name: i
  dimension: 2
  distance_metric: cosine
  embedding_column: e
  create_bucket_if_missing: "yes"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_sync_yaml(root, yaml)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_supabase_vector_sync_config(configuration_root=root)
            self.assertIn("create_bucket_if_missing", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
