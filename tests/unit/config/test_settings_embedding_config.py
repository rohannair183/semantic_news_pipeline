"""Unit tests for Settings.load_embedding_config."""

import tempfile
import unittest
from pathlib import Path

from src.config.settings import EmbeddingConfig, Settings
from src.enums.embedding_provider import EmbeddingProvider


def _write_embedding_yaml(config_root: Path, content: str) -> Path:
    """Write an embeddings.yaml under a temporary configuration root."""
    embeddings_dir = config_root / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = embeddings_dir / "embeddings.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


_VALID_YAML = """\
embeddings:
  input_dir: checkpoints/chunked_parquet
  output_dir: checkpoints/embeddings
  text_column: chunk_text
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  batch_size: 64
"""


class TestLoadEmbeddingConfigHappyPath(unittest.TestCase):
    """This class tests load_embedding_config."""

    def test_load_embedding_config_returns_typed_config(self) -> None:
        """load_embedding_config: returns EmbeddingConfig from valid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, _VALID_YAML)
            cfg = Settings.load_embedding_config(configuration_root=config_root)

        self.assertIsInstance(cfg, EmbeddingConfig)
        self.assertEqual(cfg.input_dir, Path("checkpoints/chunked_parquet"))
        self.assertEqual(cfg.output_dir, Path("checkpoints/embeddings"))
        self.assertEqual(cfg.text_column, "chunk_text")
        self.assertEqual(cfg.provider, EmbeddingProvider.SENTENCE_TRANSFORMERS)
        self.assertEqual(cfg.model_name, "all-MiniLM-L6-v2")
        self.assertEqual(cfg.batch_size, 64)


class TestLoadEmbeddingConfigMissingSection(unittest.TestCase):
    """This class tests load_embedding_config missing section."""

    def test_load_embedding_config_raises_when_section_missing(self) -> None:
        """load_embedding_config: raises ValueError when embeddings key absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, "other_key: true\n")
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("embeddings", str(ctx.exception))

    def test_load_embedding_config_raises_when_file_missing(self) -> None:
        """load_embedding_config: raises ValueError when YAML file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            (config_root / "embeddings").mkdir()
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("embeddings", str(ctx.exception))


class TestLoadEmbeddingConfigProviderValidation(unittest.TestCase):
    """This class tests load_embedding_config provider validation."""

    def test_load_embedding_config_raises_for_invalid_provider(self) -> None:
        """load_embedding_config: raises ValueError for unsupported provider."""
        yaml_content = _VALID_YAML.replace(
            "provider: sentence_transformers",
            "provider: unsupported_backend",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("embeddings.provider", str(ctx.exception))

    def test_load_embedding_config_raises_when_provider_missing(self) -> None:
        """load_embedding_config: raises ValueError when provider key is absent."""
        yaml_content = """\
embeddings:
  text_column: chunk_text
  model_name: all-MiniLM-L6-v2
  batch_size: 64
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("provider", str(ctx.exception))


class TestLoadEmbeddingConfigFieldValidation(unittest.TestCase):
    """This class tests load_embedding_config field validation."""

    def test_load_embedding_config_raises_for_missing_text_column(self) -> None:
        """load_embedding_config: raises ValueError when text_column absent."""
        yaml_content = """\
embeddings:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  batch_size: 64
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("text_column", str(ctx.exception))

    def test_load_embedding_config_raises_for_missing_model_name(self) -> None:
        """load_embedding_config: raises ValueError when model_name absent."""
        yaml_content = """\
embeddings:
  provider: sentence_transformers
  text_column: chunk_text
  batch_size: 64
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("model_name", str(ctx.exception))

    def test_load_embedding_config_raises_for_missing_batch_size(self) -> None:
        """load_embedding_config: raises ValueError when batch_size absent."""
        yaml_content = """\
embeddings:
  provider: sentence_transformers
  text_column: chunk_text
  model_name: all-MiniLM-L6-v2
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("batch_size", str(ctx.exception))

    def test_load_embedding_config_raises_for_zero_batch_size(self) -> None:
        """load_embedding_config: raises ValueError when batch_size is zero."""
        yaml_content = _VALID_YAML.replace("batch_size: 64", "batch_size: 0")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_embedding_config(configuration_root=config_root)
            self.assertIn("batch_size", str(ctx.exception))

    def test_load_embedding_config_uses_default_dirs(self) -> None:
        """load_embedding_config: uses defaults when input_dir/output_dir omitted."""
        yaml_content = """\
embeddings:
  text_column: chunk_text
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  batch_size: 32
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            cfg = Settings.load_embedding_config(configuration_root=config_root)

        self.assertEqual(cfg.input_dir, Path("checkpoints/chunked_parquet"))
        self.assertEqual(cfg.output_dir, Path("checkpoints/embeddings"))
        self.assertEqual(cfg.batch_size, 32)

    def test_load_embedding_config_accepts_openai_provider(self) -> None:
        """load_embedding_config: accepts openai as a valid provider value."""
        yaml_content = _VALID_YAML.replace(
            "provider: sentence_transformers",
            "provider: openai",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config_root = Path(tmpdir)
            _write_embedding_yaml(config_root, yaml_content)
            cfg = Settings.load_embedding_config(configuration_root=config_root)

        self.assertEqual(cfg.provider, EmbeddingProvider.OPENAI)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
