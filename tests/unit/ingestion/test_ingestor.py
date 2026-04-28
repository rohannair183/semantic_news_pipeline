"""Unit tests for the Ingestor class in the src.ingestion module."""

import unittest
from unittest.mock import Mock
from unittest.mock import patch

from src.ingestion.ingestor import Ingestor


def _build_ingestor() -> Ingestor:
    with patch("src.ingestion.ingestor.GuardianClient"), patch(
        "src.ingestion.ingestor.YAMLConfigParser"
    ):
        return Ingestor()


class TestIngestorInit(unittest.TestCase):
    """This class tests __init__."""

    @patch("src.ingestion.ingestor.GuardianClient")
    @patch("src.ingestion.ingestor.YAMLConfigParser")
    def test_init_with_defaults(self, mock_parser_class, mock_client_class):
        """__init__: constructor creates default dependencies when not injected."""
        mock_client = Mock()
        mock_parser = Mock()
        mock_client_class.return_value = mock_client
        mock_parser_class.return_value = mock_parser

        ingestor = Ingestor()

        self.assertIsInstance(ingestor, Ingestor)
        self.assertIs(ingestor.client, mock_client)
        self.assertIs(ingestor.parser, mock_parser)

class TestIngestorLoadConfig(unittest.TestCase):
    """This class tests _load_config."""

    def test_load_config_uses_yaml_parser(self):
        """_load_config: parser.parse is called with ingestion config target."""
        ingestor = _build_ingestor()
        parser = Mock()
        parser.parse.return_value = {"profiles": {"technology_daily": {"topic": "technology"}}}
        ingestor.parser = parser

        result = ingestor._load_config()  # pylint: disable=protected-access

        self.assertEqual(result, {"profiles": {"technology_daily": {"topic": "technology"}}})
        parser.parse.assert_called_once()


class TestIngestorResolveProfilesToRun(unittest.TestCase):
    """This class tests _resolve_profiles_to_run."""

    def setUp(self):
        self.ingestor = _build_ingestor()

    def test_resolve_profiles_defaults_to_all_profiles(self):
        """_resolve_profiles_to_run: returns all profiles when selection is absent."""
        config = {
            "profiles": {
                "technology_daily": {"topic": "technology"},
                "science_daily": {"topic": "science"},
            }
        }
        resolved = self.ingestor._resolve_profiles_to_run(config)  # pylint: disable=protected-access
        self.assertEqual(resolved, ["technology_daily", "science_daily"])

    def test_resolve_profiles_returns_selected_profiles(self):
        """_resolve_profiles_to_run: returns explicitly selected profile names."""
        config = {
            "profiles": {
                "technology_daily": {"topic": "technology"},
                "science_daily": {"topic": "science"},
            },
            "ingestor": {"profiles_to_run": ["science_daily"]},
        }
        resolved = self.ingestor._resolve_profiles_to_run(config)  # pylint: disable=protected-access
        self.assertEqual(resolved, ["science_daily"])

    def test_resolve_profiles_allows_none_ingestor_config(self):
        """_resolve_profiles_to_run: treats null ingestor config as empty mapping."""
        config = {
            "profiles": {"technology_daily": {"topic": "technology"}},
            "ingestor": None,
        }
        resolved = self.ingestor._resolve_profiles_to_run(config)  # pylint: disable=protected-access
        self.assertEqual(resolved, ["technology_daily"])

    def test_resolve_profiles_rejects_invalid_config_shapes(self):
        """_resolve_profiles_to_run: raises ValueError for malformed profile-selection config."""
        with self.assertRaises(ValueError):
            self.ingestor._resolve_profiles_to_run({})  # pylint: disable=protected-access

        with self.assertRaises(ValueError):
            self.ingestor._resolve_profiles_to_run(  # pylint: disable=protected-access
                {"profiles": {"technology_daily": {"topic": "technology"}}, "ingestor": []}
            )

        with self.assertRaises(ValueError):
            self.ingestor._resolve_profiles_to_run(  # pylint: disable=protected-access
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "ingestor": {"profiles_to_run": "technology_daily"},
                }
            )

        with self.assertRaises(ValueError):
            self.ingestor._resolve_profiles_to_run(  # pylint: disable=protected-access
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "ingestor": {"profiles_to_run": []},
                }
            )

        with self.assertRaises(ValueError):
            self.ingestor._resolve_profiles_to_run(  # pylint: disable=protected-access
                {
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "ingestor": {"profiles_to_run": ["science_daily"]},
                }
            )


class TestIngestorCollectProfileArticles(unittest.TestCase):
    """This class tests _collect_profile_articles."""

    def test_collect_profile_articles_handles_successes_and_failures(self):
        """_collect_profile_articles: collects full items and records per-id failures."""
        ingestor = _build_ingestor()
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
        ingestor.client = client

        result = ingestor._collect_profile_articles(  # pylint: disable=protected-access
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


class TestIngestorResolveLimit(unittest.TestCase):
    """This class tests _resolve_limit."""

    def setUp(self):
        self.ingestor = _build_ingestor()

    def test_resolve_limit_handles_none_mapping_and_missing_value(self):
        """_resolve_limit: returns None when ingestor config or limit value is missing."""
        self.assertIsNone(
            self.ingestor._resolve_limit(  # pylint: disable=protected-access
                config={"profiles": {"technology_daily": {"topic": "technology"}}, "ingestor": None},
            )
        )
        self.assertIsNone(
            self.ingestor._resolve_limit(  # pylint: disable=protected-access
                config={
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "ingestor": {},
                },
            )
        )

    def test_resolve_limit_rejects_non_mapping_and_invalid_values(self):
        """_resolve_limit: raises ValueError for malformed or invalid limit settings."""
        with self.assertRaises(ValueError):
            self.ingestor._resolve_limit(  # pylint: disable=protected-access
                config={"profiles": {"technology_daily": {"topic": "technology"}}, "ingestor": []},
            )

        with self.assertRaises(ValueError):
            self.ingestor._resolve_limit(  # pylint: disable=protected-access
                config={
                    "profiles": {"technology_daily": {"topic": "technology"}},
                    "ingestor": {"limit_per_profile": 0},
                },
            )


class TestIngestorRun(unittest.TestCase):
    """This class tests run."""

    def test_run_happy_path_uses_yaml_limit_and_multiple_profiles(self):
        """run: executes selected profiles and aggregates counts with YAML default limit."""
        ingestor = _build_ingestor()
        client = Mock()
        parser = Mock()
        parser.parse.return_value = {
            "profiles": {
                "technology_daily": {"topic": "technology"},
                "science_daily": {"topic": "science"},
            },
            "ingestor": {"profiles_to_run": ["technology_daily", "science_daily"], "limit_per_profile": 2},
        }
        ingestor.client = client
        ingestor.parser = parser
        ingestor._collect_profile_articles = Mock(  # pylint: disable=protected-access
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

        result = ingestor.run()

        self.assertEqual(result["profile_count"], 2)
        self.assertEqual(result["profiles_run"], ["technology_daily", "science_daily"])
        self.assertEqual(result["limit_per_profile"], 2)
        self.assertEqual(result["searched_count"], 4)
        self.assertEqual(result["fetched_count"], 3)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(len(result["results"]), 2)
        ingestor._collect_profile_articles.assert_any_call(  # pylint: disable=protected-access
            profile="technology_daily",
            limit=2,
        )
        ingestor._collect_profile_articles.assert_any_call(  # pylint: disable=protected-access
            profile="science_daily",
            limit=2,
        )

    def test_run_uses_configured_limit(self):
        """run: uses configured limit_per_profile value from YAML config."""
        ingestor = _build_ingestor()
        parser = Mock()
        parser.parse.return_value = {
            "profiles": {"technology_daily": {"topic": "technology"}},
            "ingestor": {"limit_per_profile": 2},
        }
        ingestor.parser = parser
        ingestor._collect_profile_articles = Mock(  # pylint: disable=protected-access
            return_value={
                "profile": "technology_daily",
                "searched_count": 1,
                "fetched_count": 1,
                "failed_count": 0,
                "items": [{"id": "x"}],
                "failures": [],
            }
        )

        result = ingestor.run()

        self.assertEqual(result["limit_per_profile"], 2)
        ingestor._collect_profile_articles.assert_called_once_with(  # pylint: disable=protected-access
            profile="technology_daily",
            limit=2,
        )

    def test_run_rejects_invalid_configured_limit(self):
        """run: rejects invalid configured limit values."""
        ingestor = _build_ingestor()
        parser = Mock()
        parser.parse.return_value = {
            "profiles": {"technology_daily": {"topic": "technology"}},
            "ingestor": {"limit_per_profile": 0},
        }
        ingestor.parser = parser

        with self.assertRaises(ValueError):
            ingestor.run()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
