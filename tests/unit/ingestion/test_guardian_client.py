# pyright: reportPrivateUsage=false
"""Unit tests for the GuardianClient class in the src.ingestion module."""

import unittest
from datetime import date
from unittest.mock import Mock
from unittest.mock import patch

import requests

from src.config.settings import GuardianProfileConfig, Settings
from src.enums.guardian_order_by import GuardianOrderBy
from src.ingestion.guardian_client import GuardianClient


class GuardianClientTestCase(unittest.TestCase):
    """Base test case that mocks settings and YAML profile loading."""

    def setUp(self):
        super().setUp()
        self.mock_settings = Settings(
            api_key="test_api_key",
            base_url="https://content.guardianapis.com",
            default_page_size=10,
            max_page_size=50,
            timeout_seconds=15,
        )
        self.mock_profiles = {
            "technology_profile": GuardianProfileConfig(
                topic="technology",
                run_date=date(2026, 4, 26),
                page_size=2,
                query="technology",
                section="technology",
                order_by=GuardianOrderBy.NEWEST,
                use_next_fallback=True,
                content_show_fields="headline,bodyText",
            ),
            "science_profile": GuardianProfileConfig(
                topic="science",
                run_date=date(2026, 4, 26),
                page_size=2,
                order_by=GuardianOrderBy.NEWEST,
                use_next_fallback=True,
            ),
            "x_profile": GuardianProfileConfig(
                topic="x",
                run_date=date(2026, 4, 26),
                page_size=2,
                order_by=GuardianOrderBy.NEWEST,
                use_next_fallback=True,
            ),
            "no_fallback_profile": GuardianProfileConfig(
                topic="x",
                run_date=date(2026, 4, 26),
                page_size=2,
                order_by=GuardianOrderBy.NEWEST,
                use_next_fallback=False,
            ),
        }

        self.load_settings_patcher = patch(
            "src.ingestion.guardian_client.Settings.load_settings",
            return_value=self.mock_settings,
        )
        self.load_profiles_patcher = patch(
            "src.ingestion.guardian_client.Settings.load_guardian_profile_configs",
            return_value=self.mock_profiles,
        )

        self.mock_load_settings = self.load_settings_patcher.start()
        self.mock_load_profiles = self.load_profiles_patcher.start()

    def tearDown(self):
        self.load_profiles_patcher.stop()
        self.load_settings_patcher.stop()
        super().tearDown()


class TestGuardianClientInit(GuardianClientTestCase):
    """This class tests __init__."""

    def test_init_with_secret_from_settings(self):
        """__init__: initialization succeeds when Settings provides an API key."""
        client = GuardianClient()
        profiles = getattr(client, "_profiles")
        self.assertIsNotNone(client)
        self.assertEqual(client.api_key, "test_api_key")
        self.assertEqual(profiles["technology_profile"].run_date, date(2026, 4, 26))
        self.assertEqual(profiles["technology_profile"].order_by, GuardianOrderBy.NEWEST)
        self.mock_load_settings.assert_called_once_with()
        self.mock_load_profiles.assert_called_once_with()

    @patch(
        "src.ingestion.guardian_client.Settings.load_settings",
        side_effect=ValueError("Missing required setting: GUARDIAN_API_KEY"),
    )
    def test_init_without_secret(self, _mock_load_settings):
        """__init__: initialization fails when settings loading fails."""
        with self.assertRaises(ValueError):
            GuardianClient()

    @patch("src.ingestion.guardian_client.Settings.load_settings")
    def test_init_uses_settings_values(self, mock_load_settings):
        """__init__: values are taken from settings resolved at construction time."""
        mock_load_settings.return_value = Settings(
            api_key="k",
            base_url="https://cfg.test",
            default_page_size=11,
            max_page_size=22,
            timeout_seconds=9,
        )
        client = GuardianClient()
        self.assertEqual(client.api_key, "k")
        self.assertEqual(client.base_url, "https://cfg.test")
        self.assertEqual(client.default_page_size, 11)
        self.assertEqual(client.max_page_size, 22)
        self.assertEqual(client.timeout_seconds, 9)

    def test_init_rejects_non_positive_request_rate(self):
        """__init__: request rate must be strictly positive."""
        with self.assertRaises(ValueError):
            GuardianClient(requests_per_second=0)

    @patch(
        "src.ingestion.guardian_client.Settings.load_guardian_profile_configs",
        side_effect=ValueError("bad profiles"),
    )
    def test_init_propagates_profile_loading_error(self, _mock_load_profiles):
        """__init__: surfaces typed profile-loading errors from Settings."""
        with self.assertRaises(ValueError):
            GuardianClient()


class TestGuardianClientValidatePageSize(GuardianClientTestCase):
    """This class tests _validate_page_size."""

    def test_validate_page_size_errors(self):
        """_validate_page_size: enforces configured min/max bounds."""
        client = GuardianClient()
        validate_page_size = getattr(client, "_validate_page_size")
        self.assertEqual(validate_page_size(1), 1)
        with self.assertRaises(ValueError):
            validate_page_size(0)
        with self.assertRaises(ValueError):
            validate_page_size(client.max_page_size + 1)


class TestGuardianClientThrottle(GuardianClientTestCase):
    """This class tests _throttle."""

    def test_throttle_sleep_and_no_sleep_paths(self):
        """_throttle: sleeps when needed and skips sleep when interval already elapsed."""
        client = GuardianClient(requests_per_second=2.0)
        with patch("src.ingestion.guardian_client.time.monotonic", side_effect=[1.0, 1.01]), patch(
            "src.ingestion.guardian_client.time.sleep"
        ) as mock_sleep:
            setattr(client, "_last_request_time", 0.9)
            getattr(client, "_throttle")()
            mock_sleep.assert_called_once()

        with patch("src.ingestion.guardian_client.time.monotonic", side_effect=[3.0, 3.0]), patch(
            "src.ingestion.guardian_client.time.sleep"
        ) as mock_sleep:
            setattr(client, "_last_request_time", 1.0)
            getattr(client, "_throttle")()
            mock_sleep.assert_not_called()


class TestGuardianClientRequestJson(GuardianClientTestCase):
    """This class tests _request_json."""

    def test_request_json_success_and_error_paths(self):
        """_request_json: returns JSON on success and wraps HTTP/request errors."""
        usage_logger = Mock()
        client = GuardianClient(usage_logger=usage_logger)

        with patch.object(client, "_throttle"), patch(
            "src.ingestion.guardian_client.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.json.return_value = {"response": {"status": "ok"}}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            payload = getattr(client, "_request_json")("/search", {"a": 1})
        self.assertEqual(payload["response"]["status"], "ok")
        mock_get.assert_called_once_with(
            url="https://content.guardianapis.com/search",
            params={"a": 1},
            timeout=15,
        )
        usage_logger.log_api_call.assert_called_once_with(
            profile=None,
            path="/search",
            status_code=mock_response.status_code,
        )

        http_response = Mock()
        http_response.status_code = 400
        http_response.text = "detail"
        http_err = requests.HTTPError(
            "bad request",
            response=http_response,
        )
        with patch.object(client, "_throttle"), patch(
            "src.ingestion.guardian_client.requests.get"
        ) as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = http_err
            mock_get.return_value = mock_response
            with self.assertRaises(RuntimeError):
                getattr(client, "_request_json")("/search")
        usage_logger.log_api_error.assert_called_with(
            profile=None,
            path="/search",
            error="Guardian API HTTP error 400",
            status_code=400,
        )

        with patch.object(client, "_throttle"), patch(
            "src.ingestion.guardian_client.requests.get",
            side_effect=requests.RequestException("down"),
        ):
            with self.assertRaises(RuntimeError):
                getattr(client, "_request_json")("/search")
        usage_logger.log_api_error.assert_called_with(
            profile=None,
            path="/search",
            error="Guardian API connection error: down",
            status_code=None,
        )

    def test_request_json_retries_transient_failures(self):
        """_request_json: retries transient Guardian failures before succeeding."""
        client = GuardianClient(usage_logger=Mock())
        response = Mock()
        response.raise_for_status.return_value = None
        response.status_code = 200
        response.json.return_value = {"response": {"status": "ok"}}

        with patch.object(client, "_throttle"), patch(
            "src.ingestion.guardian_client.requests.get",
            side_effect=[requests.Timeout("slow"), response],
        ) as mock_get:
            payload = getattr(client, "_request_json")("/search")

        self.assertEqual(payload["response"]["status"], "ok")
        self.assertEqual(mock_get.call_count, 2)

    def test_request_json_does_not_retry_non_retryable_http_status(self):
        """_request_json: does not retry deterministic client HTTP errors."""
        client = GuardianClient(usage_logger=Mock())
        http_response = Mock()
        http_response.status_code = 400
        http_response.text = "detail"
        http_err = requests.HTTPError("bad request", response=http_response)
        response = Mock()
        response.raise_for_status.side_effect = http_err

        with patch.object(client, "_throttle"), patch(
            "src.ingestion.guardian_client.time.sleep"
        ) as mock_sleep, patch(
            "src.ingestion.guardian_client.requests.get",
            return_value=response,
        ) as mock_get:
            with self.assertRaises(RuntimeError):
                getattr(client, "_request_json")("/search")

        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()


class TestGuardianClientExtractResponseOrRaise(GuardianClientTestCase):
    """This class tests _extract_response_or_raise."""

    def test_extract_response_validation(self):
        """_extract_response_or_raise: validates payload shape and API status."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            getattr(client, "_extract_response_or_raise")({})
        with self.assertRaises(RuntimeError):
            getattr(client, "_extract_response_or_raise")(
                {"response": {"status": "error"}})
        self.assertEqual(
            getattr(client, "_extract_response_or_raise")(
                {"response": {"status": "ok", "total": 1}})[
                "total"
            ],
            1,
        )


class TestGuardianClientBuildSearchParams(GuardianClientTestCase):
    """This class tests _build_search_params."""

    def test_build_search_params_validation_and_query(self):
        """_build_search_params: rejects missing topic and uses typed query and section fields."""
        client = GuardianClient()
        build_search_params = getattr(client, "_build_search_params")
        request = GuardianProfileConfig(
            topic="",
            from_date=date(2026, 4, 1),
            to_date=date(2026, 4, 1),
            page_size=10,
        )
        params = build_search_params(request, page=1)
        self.assertEqual(params["q"], "")

        request = GuardianProfileConfig(
            topic="technology",
            from_date=date(2026, 4, 1),
            to_date=date(2026, 4, 1),
            page_size=10,
            query="chips",
            section="technology",
            order_by=GuardianOrderBy.OLDEST,
        )
        params = build_search_params(request, page=2)
        self.assertEqual(params["q"], "chips")
        self.assertEqual(params["section"], "technology")
        self.assertEqual(params["page"], 2)
        self.assertEqual(params["order-by"], "oldest")
        self.assertEqual(params["use-date"], "published")

        request = GuardianProfileConfig(
            topic="science",
            from_date=date(2026, 4, 1),
            to_date=date(2026, 4, 1),
            page_size=10,
        )
        params = build_search_params(request, page=1)
        self.assertEqual(params["q"], "science")
        self.assertNotIn("section", params)

    def test_build_search_params_raises_when_date_window_missing(self):
        """_build_search_params: raises when no run_date or from/to window exists."""
        client = GuardianClient()
        build_search_params = getattr(client, "_build_search_params")
        request = GuardianProfileConfig(
            topic="science",
            run_date=None,
            from_date=None,
            to_date=None,
            page_size=10,
        )
        with self.assertRaises(ValueError):
            build_search_params(request, page=1)


class TestGuardianClientSearchNextPage(GuardianClientTestCase):
    """This class tests _search_next_page."""

    def test_search_next_page_requires_last_id(self):
        """_search_next_page: rejects empty continuation id."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            getattr(client, "_search_next_page")("", {})

    def test_search_next_page_success_path(self):
        """_search_next_page: builds the continuation path and parses the response."""
        client = GuardianClient()
        with patch.object(
            client,
            "_request_json",
            return_value={"response": {"status": "ok", "results": []}},
        ):
            response = getattr(client, "_search_next_page")(
                "technology/1", {"order-by": "newest"})

        self.assertEqual(response["status"], "ok")


class TestGuardianClientSearchPage(GuardianClientTestCase):
    """This class tests _search_page."""

    def test_search_page_success_path(self):
        """_search_page: wrapper calls the request parser pipeline correctly."""
        client = GuardianClient()
        request = GuardianProfileConfig(
            topic="technology",
            run_date=date(2026, 4, 1),
            page_size=10,
        )
        with patch.object(
            client,
            "_request_json",
            return_value={"response": {"status": "ok", "results": []}},
        ):
            response = getattr(client, "_search_page")(request=request, page=1)

        self.assertEqual(response["status"], "ok")

class TestGuardianClientIterTopicArticles(GuardianClientTestCase):
    """This class tests iter_topic_articles."""

    def test_iter_topic_articles_respects_limit_across_pages(self):
        """iter_topic_articles: paginates and respects requested item limits."""
        client = GuardianClient()
        with patch.object(client, "_search_page") as mock_search_page:
            mock_search_page.side_effect = [
                {
                    "status": "ok",
                    "currentPage": 1,
                    "pages": 2,
                    "results": [{"id": "technology/1"}, {"id": "technology/2"}],
                },
                {
                    "status": "ok",
                    "currentPage": 2,
                    "pages": 2,
                    "results": [{"id": "technology/3"}, {"id": "technology/4"}],
                },
            ]

            items = list(client.iter_topic_articles(
                profile="technology_profile", limit=3))

        self.assertEqual([item["id"] for item in items], [
                         "technology/1", "technology/2", "technology/3"])

    def test_iter_topic_articles_uses_next_fallback(self):
        """iter_topic_articles: continues via /next when normal pagination is exhausted."""
        client = GuardianClient()
        with patch.object(client, "_search_page") as mock_search_page, patch.object(
            client, "_search_next_page"
        ) as mock_next:
            mock_search_page.return_value = {
                "status": "ok",
                "currentPage": 1,
                "pages": 1,
                "results": [{"id": "science/1"}, {"id": "science/2"}],
            }
            mock_next.return_value = {
                "status": "ok",
                "results": [{"id": "science/3"}],
            }

            items = list(client.iter_topic_articles(
                profile="science_profile", limit=3))

        self.assertEqual([item["id"] for item in items], [
                         "science/1", "science/2", "science/3"])
        self.assertEqual(mock_next.call_count, 1)

    def test_iter_topic_articles_limit_error_and_empty_path(self):
        """iter_topic_articles: rejects invalid limits and exits cleanly on empty responses."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            list(client.iter_topic_articles(profile="x_profile", limit=0))
        with patch.object(client, "_search_page", return_value={"results": []}):
            self.assertEqual(
                list(client.iter_topic_articles(profile="x_profile")), [])

    def test_iter_topic_articles_next_fallback_early_exits(self):
        """
        iter_topic_articles: next-fallback guard clauses return without extra calls when
        needed.
        """
        client = GuardianClient()

        with patch.object(
            client,
            "_search_page",
            return_value={"currentPage": 1,
                          "pages": 1, "results": [{"id": "a"}]},
        ), patch.object(client, "_search_next_page") as mock_next:
            items = list(client.iter_topic_articles(
                profile="no_fallback_profile"))
            self.assertEqual(len(items), 1)
            mock_next.assert_not_called()

        with patch.object(
            client,
            "_search_page",
            return_value={"currentPage": 1,
                          "pages": 1, "results": [{"id": "a"}]},
        ), patch.object(client, "_search_next_page") as mock_next:
            list(client.iter_topic_articles(profile="x_profile", limit=1))
            mock_next.assert_not_called()

        with patch.object(
            client,
            "_search_page",
            return_value={"currentPage": 1,
                          "pages": 1, "results": [{"id": "a"}]},
        ), patch.object(client, "_search_next_page") as mock_next:
            list(client.iter_topic_articles(profile="x_profile", limit=2))
            mock_next.assert_not_called()

    def test_iter_topic_articles_next_fallback_result_break_paths(self):
        """iter_topic_articles: next-fallback loop exits on empty and short result batches."""
        client = GuardianClient()

        with patch.object(
            client,
            "_search_page",
            return_value={"currentPage": 1, "pages": 1,
                          "results": [{"id": "a"}, {"id": "b"}]},
        ), patch.object(client, "_search_next_page", return_value={"results": []}):
            items = list(client.iter_topic_articles(
                profile="x_profile", limit=5))
            self.assertEqual(len(items), 2)

        with patch.object(
            client,
            "_search_page",
            return_value={"currentPage": 1, "pages": 1,
                          "results": [{"id": "a"}, {"id": "b"}]},
        ), patch.object(client, "_search_next_page", return_value={"results": [{"id": "c"}]}):
            items = list(client.iter_topic_articles(
                profile="x_profile", limit=5))
            self.assertEqual(len(items), 3)

    def test_iter_topic_articles_profile_not_found(self):
        """iter_topic_articles: raises when profile is not configured."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            list(client.iter_topic_articles(profile="missing_profile"))

    def test_iter_topic_articles_profile_empty_error(self):
        """iter_topic_articles: rejects empty profile names."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            list(client.iter_topic_articles(profile=""))

    def test_iter_topic_articles_requires_window_pair(self):
        """iter_topic_articles: from_date and to_date must be provided together."""
        client = GuardianClient()
        with self.assertRaises(ValueError):
            list(
                client.iter_topic_articles(
                    profile="technology_profile",
                    from_date=date(2026, 4, 20),
                    to_date=None,
                )
            )

    def test_iter_topic_articles_applies_override_window(self):
        """iter_topic_articles: override window is applied to profile request."""
        client = GuardianClient()
        with patch.object(
            client,
            "_iter_search_responses",
            return_value=iter([{"results": []}]),
        ) as mock_iter:
            list(
                client.iter_topic_articles(
                    profile="technology_profile",
                    from_date=date(2026, 4, 20),
                    to_date=date(2026, 4, 22),
                )
            )
        request_arg = mock_iter.call_args.args[0]
        self.assertEqual(request_arg.from_date, date(2026, 4, 20))
        self.assertEqual(request_arg.to_date, date(2026, 4, 22))


class TestGuardianClientGetArticlesForTopicDay(GuardianClientTestCase):
    """This class tests get_articles_for_topic_day."""

    def test_get_articles_for_topic_day_uses_profile_config(self):
        """get_articles_for_topic_day: summary fields come from selected profile settings."""
        client = GuardianClient()
        with patch.object(client, "iter_topic_articles", return_value=iter([{"id": "a"}])):
            payload = client.get_articles_for_topic_day(profile="x_profile")
            self.assertEqual(payload["fetched_count"], 1)
            self.assertEqual(payload["pagination_summary"]["page_size"], 2)
            self.assertEqual(payload["pagination_summary"]
                             ["used_next_fallback"], True)
            self.assertEqual(payload["profile"], "x_profile")


class TestGuardianClientGetArticleById(GuardianClientTestCase):
    """This class tests get_article_by_id."""

    def test_get_article_by_id_returns_content(self):
        """get_article_by_id: returns content payload using profile-defined show-fields."""
        client = GuardianClient()
        with patch.object(client, "_request_json") as mock_request:
            mock_request.return_value = {
                "response": {
                    "status": "ok",
                    "content": {"id": "world/1", "webTitle": "Title"},
                }
            }
            article = client.get_article_by_id(
                profile="technology_profile",
                content_id="world/1",
            )

        self.assertEqual(article["id"], "world/1")
        self.assertEqual(article["webTitle"], "Title")
        mock_request.assert_called_once_with(
            "/world/1",
            {
                "api-key": "test_api_key",
                "format": "json",
                "show-fields": "headline,bodyText",
            },
            profile="technology_profile",
        )

    def test_get_article_by_id_errors(self):
        """get_article_by_id: rejects bad profile inputs and missing content payloads."""
        client = GuardianClient()

        with self.assertRaises(ValueError):
            client.get_article_by_id(profile="technology_profile", content_id="")

        with self.assertRaises(ValueError):
            client.get_article_by_id(profile="missing_profile", content_id="x/y")

        with patch.object(client, "_request_json", return_value={"response": {"status": "ok"}}):
            with self.assertRaises(ValueError):
                client.get_article_by_id(profile="technology_profile", content_id="x/y")


class TestGuardianClientGetArticlesByIds(GuardianClientTestCase):
    """This class tests get_articles_by_ids."""

    def test_get_articles_by_ids_collects_failures(self):
        """get_articles_by_ids: captures per-id failures without aborting the whole batch."""
        client = GuardianClient()
        with patch.object(client, "_get_article_by_id_for_request") as mock_get_by_id:
            mock_get_by_id.side_effect = [
                {"id": "world/1"},
                RuntimeError("boom"),
                {"id": "world/3"},
            ]
            result = client.get_articles_by_ids(
                profile="technology_profile",
                content_ids=["world/1", "world/2", "world/3"],
            )

        self.assertEqual(result["fetched_count"], 2)
        self.assertEqual(result["failed_count"], 1)
        self.assertEqual(result["failures"][0]["id"], "world/2")

    def test_get_articles_by_ids_all_success(self):
        """get_articles_by_ids: success path returns expected counters."""
        client = GuardianClient()
        with patch.object(client, "_get_article_by_id_for_request", return_value={"id": "ok"}):
            result = client.get_articles_by_ids(
                profile="technology_profile",
                content_ids=["a"],
            )
        self.assertEqual(result["fetched_count"], 1)
        self.assertEqual(result["failed_count"], 0)


class TestGuardianClientGetUsageCounts(GuardianClientTestCase):
    """This class tests get_usage_counts."""

    def test_get_usage_counts_returns_logger_counters(self):
        """get_usage_counts: proxies tracked counters from the usage logger."""
        usage_logger = Mock()
        usage_logger.usage_counts = {
            "total_api_calls": 4,
            "error_api_calls": 1,
            "calls_by_profile": {"technology_profile": 4},
        }
        client = GuardianClient(usage_logger=usage_logger)
        self.assertEqual(
            client.get_usage_counts(),
            {
                "total_api_calls": 4,
                "error_api_calls": 1,
                "calls_by_profile": {"technology_profile": 4},
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
