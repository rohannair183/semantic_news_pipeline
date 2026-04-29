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
        enable_usage_logging=bool(article_ingestor_config.get("enable_usage_logging", False)),
        logs_dir=Path(str(article_ingestor_config.get("logs_dir", "logs"))),
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
        """__init__: constructor creates dependencies and keeps config-backed state available."""
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
        self.assertFalse(article_ingestor.config.enable_usage_logging)
        self.assertEqual(article_ingestor.config.logs_dir, Path("logs"))
        mock_load_config.assert_called_once_with()
        mock_client_class.assert_called_once_with(usage_logger=article_ingestor.logger)


class TestArticleIngestorProfilesToRun(unittest.TestCase):
    """This class tests profiles_to_run."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_profiles_to_run_returns_all_profiles_when_selection_is_absent(self):
        """profiles_to_run: returns all profiles when selection is absent."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                }
            }
        )
        self.assertEqual(
            self.article_ingestor.profiles_to_run,
            ["technology_daily", "science_daily"],
        )

    def test_profiles_to_run_returns_selected_profiles(self):
        """profiles_to_run: returns selected profile names."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                },
                "article_ingestor": {"profiles_to_run": ["science_daily"]},
            }
        )
        self.assertEqual(self.article_ingestor.profiles_to_run, ["science_daily"])

    def test_profiles_to_run_returns_new_list_each_time(self):
        """profiles_to_run: returns a copy so callers cannot mutate config state."""
        profiles_to_run = self.article_ingestor.profiles_to_run
        profiles_to_run.append("mutated")

        self.assertEqual(
            self.article_ingestor.profiles_to_run,
            ["technology_daily", "science_daily"],
        )


class TestArticleIngestorCollectProfileArticles(unittest.TestCase):
    """This class tests collect_profile_articles."""

    @staticmethod
    def _set_seen_ids(article_ingestor: ArticleIngestor, ids: set[str]) -> None:
        """Seed seen ids for dedupe-focused test cases."""
        article_ingestor._seen_ids = ids  # pylint: disable=protected-access

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

        client.iter_topic_articles.assert_called_once_with(
            profile="technology_daily",
            limit=None,
        )
        self.assertEqual(result["searched_count"], 3)
        self.assertEqual(result["fetched_count"], 1)
        self.assertEqual(result["skipped_existing_count"], 0)
        self.assertEqual(result["failed_count"], 2)
        self.assertEqual(result["items"][0]["id"], "article-1")
        client.get_article_by_id.assert_any_call(
            profile="technology_daily",
            content_id="article-1",
        )
        client.get_article_by_id.assert_any_call(
            profile="technology_daily",
            content_id="article-2",
        )
        self.assertEqual(result["failures"][0]["id"], "article-2")
        self.assertEqual(result["failures"][1]["error"], "Missing id in topic search result")

    def test_collect_profile_articles_skips_existing_ids(self):
        """collect_profile_articles: skips detail fetches for previously ingested ids."""
        article_ingestor = _build_article_ingestor()
        self._set_seen_ids(article_ingestor, {"article-1"})
        client = Mock()
        client.iter_topic_articles.return_value = [{"id": "article-1"}, {"id": "article-2"}]
        client.get_article_by_id.return_value = {"id": "article-2"}
        article_ingestor.client = client

        result = article_ingestor.collect_profile_articles("technology_daily", limit=None)

        self.assertEqual(result["searched_count"], 2)
        self.assertEqual(result["skipped_existing_count"], 1)
        self.assertEqual(result["fetched_count"], 1)
        client.get_article_by_id.assert_called_once_with(
            profile="technology_daily",
            content_id="article-2",
        )

    def test_collect_profile_articles_limit_counts_only_newly_fetched_items(self):
        """collect_profile_articles: applies limit to newly fetched items, not skipped ids."""
        article_ingestor = _build_article_ingestor()
        self._set_seen_ids(article_ingestor, {"article-1", "article-2"})
        client = Mock()
        client.iter_topic_articles.return_value = [
            {"id": "article-1"},
            {"id": "article-2"},
            {"id": "article-3"},
            {"id": "article-4"},
        ]
        client.get_article_by_id.side_effect = [
            {"id": "article-3"},
            {"id": "article-4"},
        ]
        article_ingestor.client = client

        result = article_ingestor.collect_profile_articles("technology_daily", limit=1)

        client.iter_topic_articles.assert_called_once_with(
            profile="technology_daily",
            limit=None,
        )
        self.assertEqual(result["searched_count"], 3)
        self.assertEqual(result["skipped_existing_count"], 2)
        self.assertEqual(result["fetched_count"], 1)
        client.get_article_by_id.assert_called_once_with(
            profile="technology_daily",
            content_id="article-3",
        )

    def test_collect_profile_articles_skips_ids_seen_in_other_profiles(self):
        """collect_profile_articles: skips ids already ingested globally across profiles."""
        article_ingestor = _build_article_ingestor()
        self._set_seen_ids(article_ingestor, {"shared-article"})
        client = Mock()
        client.iter_topic_articles.return_value = [{"id": "shared-article"}, {"id": "new-article"}]
        client.get_article_by_id.return_value = {"id": "new-article"}
        article_ingestor.client = client

        result = article_ingestor.collect_profile_articles("science_daily", limit=None)

        self.assertEqual(result["searched_count"], 2)
        self.assertEqual(result["skipped_existing_count"], 1)
        self.assertEqual(result["fetched_count"], 1)
        client.get_article_by_id.assert_called_once_with(
            profile="science_daily",
            content_id="new-article",
        )


class TestArticleIngestorResolvedLimit(unittest.TestCase):
    """This class tests resolved_limit."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_resolved_limit_returns_none_when_value_is_missing(self):
        """resolved_limit: returns None when limit value is missing."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"save_local_checkpoint": False},
            }
        )
        self.assertIsNone(self.article_ingestor.resolved_limit)

    def test_resolved_limit_returns_configured_value(self):
        """resolved_limit: returns configured limit value."""
        self.assertEqual(
            _build_article_ingestor(
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {"limit_per_profile": 4},
                }
            ).resolved_limit,
            4,
        )


class TestArticleIngestorCheckpointDirectory(unittest.TestCase):
    """This class tests checkpoint_directory."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_checkpoint_directory_returns_none_by_default(self):
        """checkpoint_directory: defaults to no checkpointing."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"save_local_checkpoint": False},
            }
        )
        self.assertIsNone(self.article_ingestor.checkpoint_directory)

    def test_checkpoint_directory_returns_configured_directory(self):
        """checkpoint_directory: returns configured directory."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {
                    "save_local_checkpoint": True,
                    "checkpoint_dir": "checkpoints/custom",
                },
            }
        )
        self.assertEqual(self.article_ingestor.checkpoint_directory, Path("checkpoints/custom"))


class TestArticleIngestorIdManifestPath(unittest.TestCase):
    """This class tests id_manifest_path."""

    def setUp(self):
        self.article_ingestor = _build_article_ingestor()

    def test_id_manifest_path_returns_none_when_checkpointing_disabled(self):
        """id_manifest_path: returns None when local checkpoints are disabled."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {"save_local_checkpoint": False},
            }
        )
        self.assertIsNone(self.article_ingestor.id_manifest_path)

    def test_id_manifest_path_uses_dedicated_ingested_ids_directory(self):
        """id_manifest_path: resolves to sibling checkpoints/ingested_ids path."""
        self.article_ingestor.config = _build_article_ingestor_config(
            {
                "profiles": {"technology_daily": {"topic": "technology"}},
                "article_ingestor": {
                    "save_local_checkpoint": True,
                    "checkpoint_dir": "checkpoints/article_ingestor",
                },
            }
        )
        self.assertEqual(
            self.article_ingestor.id_manifest_path,
            Path("checkpoints/ingested_ids/ingested_ids.json"),
        )


class TestArticleIngestorIdManifestPersistence(unittest.TestCase):
    """This class tests ingested id manifest persistence helpers."""

    def test_load_ingested_id_manifest_returns_empty_for_non_list_ids(self):
        """_load_ingested_id_manifest: returns empty set when ids payload is malformed."""
        with tempfile.TemporaryDirectory() as temp_directory:
            article_ingestor = _build_article_ingestor(
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {
                        "save_local_checkpoint": True,
                        "checkpoint_dir": str(Path(temp_directory) / "article_ingestor"),
                    },
                }
            )
            manifest_path = Path(temp_directory) / "ingested_ids" / "ingested_ids.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("w", encoding="utf-8") as output_file:
                json.dump({"ids": {"invalid": True}}, output_file)

            loaded_ids = article_ingestor._load_ingested_id_manifest()  # pylint: disable=protected-access

            self.assertEqual(loaded_ids, set())

    def test_write_ingested_id_manifest_writes_global_sorted_ids(self):
        """_write_ingested_id_manifest: writes ids as one sorted global list."""
        with tempfile.TemporaryDirectory() as temp_directory:
            article_ingestor = _build_article_ingestor(
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "article_ingestor": {
                        "save_local_checkpoint": True,
                        "checkpoint_dir": str(Path(temp_directory) / "article_ingestor"),
                    },
                }
            )
            article_ingestor.run_timestamp = "20260429T000000Z"

            manifest_file = article_ingestor._write_ingested_id_manifest(  # pylint: disable=protected-access
                {"b", "a"},
                {"total_api_calls": 2, "error_api_calls": 0, "calls_by_profile": {}},
            )

            self.assertIsNotNone(manifest_file)
            with Path(str(manifest_file)).open("r", encoding="utf-8") as input_file:
                payload = json.load(input_file)
            self.assertEqual(payload["updated_at"], "20260429T000000Z")
            self.assertEqual(payload["ids"], ["a", "b"])
            self.assertEqual(payload["api_usage"]["total_api_calls"], 2)


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
        article_ingestor = _build_article_ingestor(
            {
                "profiles": {
                    "technology_daily": {"topic": "technology"},
                    "science_daily": {"topic": "science"},
                },
                "article_ingestor": {
                    "profiles_to_run": ["technology_daily", "science_daily"],
                    "limit_per_profile": 2,
                    "save_local_checkpoint": True,
                    "checkpoint_dir": "checkpoints/article_ingestor",
                },
            }
        )
        client = Mock()
        client.get_usage_counts.return_value = {
            "total_api_calls": 5,
            "error_api_calls": 1,
            "calls_by_profile": {"technology_daily": 3, "science_daily": 2},
        }
        article_ingestor.client = client
        article_ingestor.logger = Mock()
        article_ingestor.logger.log_path = "logs/ingestion_usage.jsonl"
        article_ingestor.run_timestamp = "20260428T010203Z"
        article_ingestor.collect_profile_articles = Mock(
            side_effect=[
                {
                    "profile": "technology_daily",
                    "searched_count": 2,
                    "fetched_count": 2,
                    "skipped_existing_count": 0,
                    "failed_count": 0,
                    "items": [{"id": "a"}, {"id": "b"}],
                    "failures": [],
                },
                {
                    "profile": "science_daily",
                    "searched_count": 2,
                    "fetched_count": 1,
                    "skipped_existing_count": 1,
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
        self.assertEqual(result["skipped_existing_count"], 1)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["checkpoint_files"], ["checkpoint_1.json", "checkpoint_2.json"])
        self.assertEqual(result["usage_log_file"], "logs/ingestion_usage.jsonl")
        self.assertEqual(result["api_usage"]["total_api_calls"], 5)
        self.assertEqual(len(result["results"]), 2)
        article_ingestor.collect_profile_articles.assert_any_call(
            profile="technology_daily",
            limit=2,
        )
        article_ingestor.collect_profile_articles.assert_any_call(
            profile="science_daily",
            limit=2,
        )
        article_ingestor.logger.log_ingestion_summary.assert_called_once()

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
                "skipped_existing_count": 0,
                "failed_count": 0,
                "items": [{"id": "x"}],
                "failures": [],
            }
        )
        article_ingestor.client = Mock()
        article_ingestor.client.get_usage_counts.return_value = {
            "total_api_calls": 1,
            "error_api_calls": 0,
            "calls_by_profile": {"technology_daily": 1},
        }
        article_ingestor.logger = Mock()
        article_ingestor.logger.log_path = None

        result = article_ingestor.run()

        self.assertEqual(result["limit_per_profile"], 2)
        self.assertEqual(result["checkpoint_files"], [])
        self.assertEqual(result["api_usage"]["total_api_calls"], 1)
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
