"""Unit tests for Settings.load_briefing_generator_config."""

import tempfile
import unittest
from pathlib import Path

from src.config.settings import BriefingGeneratorConfig, BriefingTopicSpec, Settings
from src.enums.briefing_date_filter import BriefingDateFilter


def _write_briefing_yaml(
    config_root: Path,
    content: str,
    filename: str = "briefing_generator.yaml",
) -> Path:
    """Write briefing_generator YAML under a temporary configuration root."""
    section_dir = config_root / "process"
    section_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = section_dir / filename
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


_VALID_YAML = """\
briefing_generator:
  model: gemini-2.0-flash
  vector_top_k: 10
  topics:
    - name: US politics
      vector_query: president congress
"""


class TestLoadBriefingGeneratorConfigHappyPath(unittest.TestCase):
    """This class tests load_briefing_generator_config."""

    def test_load_returns_typed_config(self) -> None:
        """load_briefing_generator_config: returns typed config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, _VALID_YAML)
            cfg = Settings.load_briefing_generator_config(configuration_root=root)

        self.assertIsInstance(cfg, BriefingGeneratorConfig)
        self.assertEqual(cfg.model, "gemini-2.0-flash")
        self.assertEqual(cfg.vector_top_k, 10)
        self.assertEqual(len(cfg.topics), 1)
        self.assertEqual(cfg.topics[0].name, "US politics")
        self.assertEqual(cfg.topics[0].vector_query, "president congress")
        self.assertEqual(cfg.topics[0].date_filter, BriefingDateFilter.DAILY)

    def test_topic_date_filter_weekly_and_monthly(self) -> None:
        """load_briefing_generator_config: parses per-topic date_filter."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - name: A
      vector_query: q1
      date_filter: weekly
    - name: B
      vector_query: q2
      date_filter: monthly
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            cfg = Settings.load_briefing_generator_config(configuration_root=root)

        self.assertEqual(cfg.topics[0].date_filter, BriefingDateFilter.WEEKLY)
        self.assertEqual(cfg.topics[1].date_filter, BriefingDateFilter.MONTHLY)

    def test_vector_top_k_defaults_to_ten(self) -> None:
        """load_briefing_generator_config: defaults vector_top_k when omitted."""
        yaml_text = """\
briefing_generator:
  model: gemini-pro
  topics:
    - name: A
      vector_query: q
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            cfg = Settings.load_briefing_generator_config(configuration_root=root)

        self.assertEqual(cfg.vector_top_k, 10)

    def test_load_custom_filename(self) -> None:
        """load_briefing_generator_config: respects filename argument."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, _VALID_YAML, filename="alt.yaml")
            cfg = Settings.load_briefing_generator_config(
                configuration_root=root,
                filename="alt.yaml",
            )

        self.assertEqual(cfg.model, "gemini-2.0-flash")


class TestLoadBriefingGeneratorConfigErrors(unittest.TestCase):
    """This class tests load_briefing_generator_config validation errors."""

    def test_raises_when_section_missing(self) -> None:
        """load_briefing_generator_config: requires briefing_generator section."""
        minimal = """other: {}\n"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, minimal)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("briefing_generator", str(ctx.exception))

    def test_raises_when_file_missing(self) -> None:
        """load_briefing_generator_config: requires YAML file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "process").mkdir()
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("briefing_generator", str(ctx.exception))

    def test_raises_when_model_missing(self) -> None:
        """load_briefing_generator_config: requires model."""
        yaml_text = """\
briefing_generator:
  vector_top_k: 5
  topics:
    - name: A
      vector_query: q
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("model", str(ctx.exception))

    def test_raises_when_topics_missing(self) -> None:
        """load_briefing_generator_config: requires non-empty topics."""
        yaml_text = """\
briefing_generator:
  model: m
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("topics", str(ctx.exception))

    def test_raises_when_topics_not_list(self) -> None:
        """load_briefing_generator_config: topics must be a list."""
        yaml_text = """\
briefing_generator:
  model: m
  topics: not_a_list
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("topics", str(ctx.exception))

    def test_raises_when_topic_entry_not_mapping(self) -> None:
        """load_briefing_generator_config: each topic must be a mapping."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - x
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("topics[0]", str(ctx.exception))

    def test_raises_when_date_filter_invalid(self) -> None:
        """load_briefing_generator_config: rejects unknown date_filter."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - name: A
      vector_query: q
      date_filter: yearly
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("BriefingDateFilter", str(ctx.exception))

    def test_raises_when_date_filter_empty_string(self) -> None:
        """load_briefing_generator_config: rejects empty date_filter string."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - name: A
      vector_query: q
      date_filter: "   "
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("date_filter", str(ctx.exception))

    def test_raises_when_date_filter_not_string(self) -> None:
        """load_briefing_generator_config: date_filter must be a string."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - name: A
      vector_query: q
      date_filter: 1
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("date_filter", str(ctx.exception))

    def test_raises_when_vector_top_k_invalid(self) -> None:
        """load_briefing_generator_config: vector_top_k must be >= 1."""
        yaml_text = """\
briefing_generator:
  model: m
  vector_top_k: 0
  topics:
    - name: A
      vector_query: q
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            with self.assertRaises(ValueError) as ctx:
                Settings.load_briefing_generator_config(configuration_root=root)
            self.assertIn("vector_top_k", str(ctx.exception))


class TestBriefingTopicSpec(unittest.TestCase):
    """This class tests BriefingTopicSpec usage from YAML."""

    def test_multiple_topics_order(self) -> None:
        """load_briefing_generator_config: preserves topic order."""
        yaml_text = """\
briefing_generator:
  model: m
  topics:
    - name: First
      vector_query: a
    - name: Second
      vector_query: b
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_briefing_yaml(root, yaml_text)
            cfg = Settings.load_briefing_generator_config(configuration_root=root)

        self.assertEqual(
            cfg.topics,
            (
                BriefingTopicSpec(name="First", vector_query="a"),
                BriefingTopicSpec(name="Second", vector_query="b"),
            ),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
