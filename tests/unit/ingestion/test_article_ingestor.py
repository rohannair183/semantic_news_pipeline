"""Unit tests for the ArticleIngestor class in the src.ingestion module."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock
from unittest.mock import patch

from src.config.settings import ArticleIngestorConfig
from src.ingestion.article_ingestor import ArticleIngestor
from tests.unit.ingestion.test_config_helpers import build_ingestion_config


def _build_article_ingestor_config(
    config: Optional[Dict[str, Any]] = None,
) -> ArticleIngestorConfig:
    if config is None:
        config = build_ingestion_config(save_local_checkpoint=False)

    article_ingestor_config = config.get("article_ingestor", {}) or {}
    profile_names = list(config["profiles"].keys())
    selected_profiles = article_ingestor_config.get("profiles_to_run")
    if selected_profiles is None:
        selected_profiles = profile_names

    checkpoint_dir = None
    if article_ingestor_config.get("save_local_checkpoint", False):
        checkpoint_dir_value = article_ingestor_config.get("checkpoint_dir")
        if checkpoint_dir_value is not None:
            checkpoint_dir = Path(str(checkpoint_dir_value))

    return ArticleIngestorConfig(
        profile_names=profile_names,
        profiles_to_run=[str(profile_name) for profile_name in selected_profiles],
        limit_per_profile=article_ingestor_config.get("limit_per_profile"),
        save_local_checkpoint=bool(article_ingestor_config.get("save_local_checkpoint", False)),
        checkpoint_dir=checkpoint_dir,
    )


def _build_article_ingestor(config: Optional[Dict[str, Any]] = None) -> ArticleIngestor:
    if config is None:
        config = build_ingestion_config(save_local_checkpoint=False)
    typed_config = _build_article_ingestor_config(config)
    with patch("src.ingestion.article_ingestor.GuardianClient") as mock_client_class, patch(
        "src.ingestion.article_ingestor.Settings.load_article_ingestor_config",
        return_value=typed_config,
    ):
        mock_client_class.return_value = Mock()
        return ArticleIngestor()


class TestArticleIngestorInit(unittest.TestCase):
    """This class tests __init__."""

    @patch("src.ingestion.article_ingestor.GuardianClient")
    @patch("src.ingestion.article_ingestor.Settings.load_article_ingestor_config")
    @patch("src.ingestion.article_ingestor.utc_now_checkpoint_token")
    def test_init_with_defaults(self, mock_run_timestamp, mock_load_config, mock_client_class):
        """__init__: constructor creates dependencies and caches resolved config state."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_run_timestamp.return_value = "20260428T010203Z"
        mock_load_config.return_value = _build_article_ingestor_config(
            build_ingestion_config(
                profiles_to_run=["science_daily"],
                limit_per_profile=3,
                save_local_checkpoint=True,
                checkpoint_dir="checkpoints/custom",
            )
        )

        article_ingestor = ArticleIngestor()

        self.assertIsInstance(article_ingestor, ArticleIngestor)
        self.assertIs(article_ingestor.client, mock_client)
        self.assertEqual(
            article_ingestor.config.profile_names,
            ["technology_daily", "science_daily"],
        )
        self.assertEqual(article_ingestor.profiles_to_run, ["science_daily"])
        self.assertEqual(article_ingestor.resolved_limit, 3)
        self.assertEqual(article_ingestor.checkpoint_directory, Path("checkpoints/custom"))
        self.assertEqual(article_ingestor.run_timestamp, "20260428T010203Z")
        mock_load_config.assert_called_once_with()


class TestArticleIngestorLoadConfig(unittest.TestCase):
    """This class tests load_config."""

    def test_load_config_uses_settings_loader(self):
        """load_config: Settings loader is called for ingestion config."""
        article_ingestor = _build_article_ingestor()
        with patch(
            "src.ingestion.article_ingestor.Settings.load_article_ingestor_config",
            return_value=_build_article_ingestor_config(
                {"profiles": {"technology_daily": {"topic": "technology"}}}
            ),
        ) as mock_load_config:
            result = article_ingestor.load_config()

        self.assertEqual(result.profile_names, ["technology_daily"])
        mock_load_config.assert_called_once_with()


class TestArticleIngestorResolveProfilesToRun(unittest.TestCase):
    """This class tests _resolve_profiles_to_run."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_resolve_profiles_defaults_to_all_profiles(self):
        """_resolve_profiles_to_run: returns all profiles when selection is absent."""
        config = _build_article_ingestor_config(
            {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                }
            }
        )
        resolved = self.article_ingestor._resolve_profiles_to_run(config)  # pylint: disable=protected-access
        self.assertEqual(resolved, ["technology_daily", "science_daily"])

    def test_resolve_profiles_returns_selected_profiles(self):
        """_resolve_profiles_to_run: returns selected profile names."""
        config = _build_article_ingestor_config(
            {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                },
                "article_ingestor": {"profiles_to_run": ["science_daily"]},
            }
        )
        resolved = self.article_ingestor._resolve_profiles_to_run(config)  # pylint: disable=protected-access
        self.assertEqual(resolved, ["science_daily"])

class TestArticleIngestorCollectProfileArticles(unittest.TestCase):
    """This class tests collect_profile_articles."""

    def test_collect_profile_articles_handles_successes_and_failures(self):
        """collect_profile_articles: collects full items and records failures."""
        article_ingestor = _build_article_ingestor()
        client = Mock()
        client.iter_topic_articles.return_value = [
            {"id": "article-1"},
            {"id": "article-2"},
            {"headline": "missing id"},
        ]
        client.get_article_by_id.side_effect = [
            {"id": "article-1", "fields": {"body": "a"}},
            RuntimeError("boom"),
        ]
        article_ingestor.client = client

        result = article_ingestor.collect_profile_articles(
            profile="technology_daily",
            limit=5,
        )

        client.iter_topic_articles.assert_called_once_with(profile="technology_daily", limit=5)
        self.assertEqual(result["searched_count"], 3)
        self.assertEqual(result["fetched_count"], 1)
        self.assertEqual(result["failed_count"], 2)
        self.assertEqual(result["items"][0]["id"], "article-1")
        self.assertEqual(result["failures"][0]["id"], "article-2")
        self.assertEqual(result["failures"][1]["error"], "Missing id in topic search result")


class TestArticleIngestorResolveLimit(unittest.TestCase):
    """This class tests _resolve_limit."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_resolve_limit_handles_none_mapping_and_missing_value(self):
        """_resolve_limit: returns None when config or limit value is missing."""
        self.assertIsNone(
            self.article_ingestor._resolve_limit(  # pylint: disable=protected-access
                config=_build_article_ingestor_config(
                    {
                        "profiles": {"technology_daily": {"topic": "technology"}},
                        "article_ingestor": {"save_local_checkpoint": False},
                    }
                )
            )
        )
        self.assertEqual(
            self.article_ingestor._resolve_limit(  # pylint: disable=protected-access
                config=_build_article_ingestor_config(
                    {
                        "profiles": {"technology_daily": {"topic": "technology"}},
                        "article_ingestor": {"limit_per_profile": 4},
                    }
                )
            ),
            4,
        )


class TestArticleIngestorResolveCheckpointDirectory(unittest.TestCase):
    """This class tests _resolve_checkpoint_directory."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_resolve_checkpoint_directory_returns_none_by_default(self):
        """_resolve_checkpoint_directory: defaults to no checkpointing."""
        result = self.article_ingestor._resolve_checkpoint_directory(  # pylint: disable=protected-access
            config=_build_article_ingestor_config(
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {"save_local_checkpoint": False},
                }
            )
        )
        self.assertIsNone(result)

    def test_resolve_checkpoint_directory_validates_and_resolves_directory(self):
        """_resolve_checkpoint_directory: validates config and resolves path."""
        result = self.article_ingestor._resolve_checkpoint_directory(  # pylint: disable=protected-access
            config=_build_article_ingestor_config(
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {
                        "save_local_checkpoint": True,
                        "checkpoint_dir": "checkpoints/custom",
                    },
                }
            )
        )
        self.assertEqual(result, Path("checkpoints/custom"))


class TestArticleIngestorWriteProfileCheckpoint(unittest.TestCase):
    """This class tests write_profile_checkpoint."""

    def test_write_profile_checkpoint_writes_json_file(self):
        """write_profile_checkpoint: writes one JSON file and returns its path."""
        article_ingestor = _build_article_ingestor()
        with tempfile.TemporaryDirectory() as temp_directory:
            checkpoint_path = article_ingestor.write_profile_checkpoint(
                profile_result={"profile": "technology_daily", "items": [{"id": "a"}]},
                checkpoint_directory=Path(temp_directory) / "nested",
                run_timestamp="20260428T000000Z",
            )
            self.assertTrue(Path(checkpoint_path).is_file())
            with Path(checkpoint_path).open("r", encoding="utf-8") as checkpoint_file:
                parsed = json.load(checkpoint_file)
            self.assertEqual(parsed["profile"], "technology_daily")


class TestArticleIngestorRun(unittest.TestCase):
    """This class tests run."""

    def test_run_happy_path_uses_limit_and_writes_checkpoints(self):
        """run: aggregates profiles and writes checkpoints when enabled."""
        article_ingestor = _build_article_ingestor()
        client = Mock()
        article_ingestor.client = client
        article_ingestor.profiles_to_run = ["technology_daily", "science_daily"]
        article_ingestor.resolved_limit = 2
        article_ingestor.checkpoint_directory = Path("checkpoints/article_ingestor")
        article_ingestor.run_timestamp = "20260428T010203Z"
        article_ingestor.collect_profile_articles = Mock(
            side_effect=[
                {
                    "profile": "technology_daily",
                    "searched_count": 2,
                    "fetched_count": 2,
                    "failed_count": 0,
                    "items": [{"id": "a"}, {"id": "b"}],
                    "failures": [],
                },
                {
                    "profile": "science_daily",
                    "searched_count": 2,
                    "fetched_count": 1,
                    "failed_count": 1,
                    "items": [{"id": "c"}],
                    "failures": [{"id": "d", "error": "x"}],
                },
            ]
        )
        article_ingestor.write_profile_checkpoint = Mock(
            side_effect=["checkpoint_1.json", "checkpoint_2.json"]
        )

        result = article_ingestor.run()

        self.assertEqual(result["profile_count"], 2)
        self.assertEqual(result["profiles_run"], ["technology_daily", "science_daily"])
        self.assertEqual(result["limit_per_profile"], 2)
        self.assertEqual(result["searched_count"], 4)
        self.assertEqual(result["fetched_count"], 3)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["checkpoint_files"], ["checkpoint_1.json", "checkpoint_2.json"])
        self.assertEqual(len(result["results"]), 2)
        article_ingestor.collect_profile_articles.assert_any_call(
            profile="technology_daily",
            limit=2,
        )
        article_ingestor.collect_profile_articles.assert_any_call(
            profile="science_daily",
            limit=2,
        )

    def test_run_uses_configured_limit_without_checkpoints(self):
        """run: uses configured limit and skips checkpointing when disabled."""
        article_ingestor = _build_article_ingestor(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"limit_per_profile": 2, "save_local_checkpoint": False},
            }
        )
        article_ingestor.collect_profile_articles = Mock(
            return_value={
                "profile": "technology_daily",
                "searched_count": 1,
                "fetched_count": 1,
                "failed_count": 0,
                "items": [{"id": "x"}],
                "failures": [],
            }
        )

        result = article_ingestor.run()

        self.assertEqual(result["limit_per_profile"], 2)
        self.assertEqual(result["checkpoint_files"], [])
        article_ingestor.collect_profile_articles.assert_called_once_with(
            profile="technology_daily",
            limit=2,
        )

    def test_init_rejects_invalid_configured_limit(self):
        """__init__: rejects invalid configured limit values."""
        with patch("src.ingestion.article_ingestor.GuardianClient"), patch(
            "src.ingestion.article_ingestor.Settings.load_article_ingestor_config",
            side_effect=ValueError(
                "Ingestion config field 'article_ingestor.limit_per_profile' must be >= 1"
            ),
        ):
            with self.assertRaises(ValueError):
                ArticleIngestor()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
