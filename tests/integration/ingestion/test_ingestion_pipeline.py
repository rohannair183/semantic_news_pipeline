"""Integration tests for the ingestion pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import pandas as pd
import requests
import yaml

from src.config.settings import Settings
from src.ingestion.article_ingestor import ArticleIngestor
from src.chunking.chunker import Chunker
from src.ingestion.article_normalizer import ArticleNormalizer
from src.ingestion.prechunk_processing import PreChunkPreprocessor
from tests.unit.ingestion.test_config_helpers import NORMALIZER_ROW_MAPPINGS


ROW_MAPPINGS: Dict[str, Dict[str, Any]] = NORMALIZER_ROW_MAPPINGS.copy()

INGESTION_DAY = date(2026, 4, 28)


def _build_article(
    content_id: str,
    web_title: str,
    profile_name: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build a realistic Guardian article payload for integration tests."""
    resolved_metadata = {
        "headline": web_title,
        "body_text": "Body text",
        "published_at": "2026-04-28T10:00:00Z",
        "last_modified": "2026-04-28T10:30:00Z",
    }
    if metadata is not None:
        resolved_metadata.update(metadata)
    return {
        "id": content_id,
        "webTitle": web_title,
        "webPublicationDate": str(resolved_metadata["published_at"]),
        "webUrl": f"https://example.com/{content_id}",
        "sectionName": "Technology" if "technology" in profile_name else "Science",
        "pillarName": "News",
        "fields": {
            "headline": str(resolved_metadata["headline"]),
            "byline": "Reporter Name",
            "bodyText": str(resolved_metadata["body_text"]),
            "trailText": f"Trail for {content_id}",
            "thumbnail": f"https://example.com/{content_id}.jpg",
            "wordcount": "500",
            "lastModified": str(resolved_metadata["last_modified"]),
            "firstPublicationDate": str(resolved_metadata["published_at"]),
        },
    }


def _build_search_response(
    results: list[dict[str, Any]],
    *,
    current_page: int = 1,
    pages: int = 1,
) -> Dict[str, Any]:
    """Build a Guardian /search response payload."""
    return {
        "status": "ok",
        "currentPage": current_page,
        "pages": pages,
        "results": results,
    }


class _FakeResponse:
    """Minimal requests-like response object for offline integration tests."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: Optional[Dict[str, Any]] = None,
        text: str = "",
    ):
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text

    def raise_for_status(self) -> None:
        """Raise requests.HTTPError for non-success status codes."""
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"HTTP {self.status_code}",
                response=self,
            )

    def json(self) -> Dict[str, Any]:
        """Return the configured response body."""
        return self._json_body


@dataclass
class _TransportFailure:
    """Represent a detail endpoint failure."""

    status_code: int
    text: str


class FakeGuardianTransport:  # pylint: disable=too-few-public-methods
    """Offline fake Guardian API transport for integration tests."""

    def __init__(
        self,
        *,
        search_payloads: Dict[tuple[str, int], Dict[str, Any]],
        detail_payloads: Dict[str, Dict[str, Any] | _TransportFailure],
    ):
        self.search_payloads = search_payloads
        self.detail_payloads = detail_payloads
        self.calls: list[dict[str, Any]] = []

    def get(self, *, url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        """Serve a fake Guardian API response and record the request."""
        recorded_params = dict(params)
        self.calls.append(
            {
                "url": url,
                "params": recorded_params,
                "timeout": timeout,
            }
        )

        if url.endswith("/search"):
            query = str(recorded_params.get("q"))
            page = int(recorded_params.get("page", 1))
            response = self.search_payloads.get((query, page))
            if response is None:
                raise AssertionError(f"Unexpected search request for query={query}, page={page}")
            return _FakeResponse(json_body={"response": response})

        content_id = url.replace("https://content.guardianapis.com/", "", 1)
        content_payload = self.detail_payloads.get(content_id)
        if content_payload is None:
            raise AssertionError(f"Unexpected detail request for content_id={content_id}")
        if isinstance(content_payload, _TransportFailure):
            return _FakeResponse(
                status_code=content_payload.status_code,
                text=content_payload.text,
            )
        return _FakeResponse(
            json_body={"response": {"status": "ok", "content": content_payload}}
        )


class IngestionPipelineIntegrationTestCase(unittest.TestCase):
    """This class tests shared ingestion pipeline integration setup."""

    def setUp(self) -> None:
        """setUp: creates a temporary config root and shared patch stack."""
        super().setUp()
        temp_root = Path(
            self.enterContext(tempfile.TemporaryDirectory())  # pylint: disable=consider-using-with
        )
        self.configuration_root = temp_root / "configuration"
        self.checkpoint_dir = temp_root / "checkpoints" / "article_ingestor"
        self.parquet_dir = temp_root / "checkpoints" / "parquet"
        self._write_ingestion_config()

        self._original_load_article_ingestor_config = Settings.load_article_ingestor_config
        self._original_load_guardian_profile_configs = Settings.load_guardian_profile_configs
        self.transport: Optional[FakeGuardianTransport] = None

        self.enterContext(
            patch(
                "src.ingestion.guardian_client.Settings.load_settings",
                side_effect=self._load_settings_for_guardian_client,
            )
        )
        self.enterContext(
            patch(
                "src.ingestion.guardian_client.Settings.load_guardian_profile_configs",
                side_effect=self._load_profiles_for_guardian_client,
            )
        )
        self.enterContext(
            patch(
                "src.ingestion.article_ingestor.Settings.load_article_ingestor_config",
                side_effect=self._load_article_ingestor_config,
            )
        )
        self.enterContext(
            patch(
                "src.ingestion.guardian_client.requests.get",
                side_effect=self._dispatch_request,
            )
        )
        self.enterContext(
            patch("src.ingestion.guardian_client.time.sleep", return_value=None)
        )

    def _write_ingestion_config(self) -> None:
        """Write a temp ingestion YAML used by the integration suite."""
        ingestion_config_path = self.configuration_root / "ingestion" / "ingestion_config.yaml"
        ingestion_config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "base_url": "https://content.guardianapis.com",
            "default_page_size": 2,
            "max_page_size": 50,
            "timeout_seconds": 15,
            "profiles": {
                "technology_daily": {
                    "topic": "technology",
                    "run_date": INGESTION_DAY.isoformat(),
                    "page_size": 2,
                    "query": "chips",
                    "section": "technology",
                    "order_by": "newest",
                    "use_next_fallback": False,
                    "content_show_fields": "headline,bodyText",
                },
                "science_daily": {
                    "topic": "science",
                    "run_date": INGESTION_DAY.isoformat(),
                    "page_size": 2,
                    "order_by": "newest",
                    "use_next_fallback": False,
                    "content_show_fields": "headline,bodyText",
                },
            },
            "article_ingestor": {
                "profiles_to_run": ["technology_daily", "science_daily"],
                "limit_per_profile": 2,
                "save_local_checkpoint": True,
                "checkpoint_dir": str(self.checkpoint_dir),
                "parquet_dir": str(self.parquet_dir),
            },
            "article_normalizer": {
                "row_mappings": ROW_MAPPINGS,
            },
            "pre_chunk_preprocessor": {
                "output_dir": str(self.parquet_dir.parent / "pre_chunk"),
                "operations": [
                    {"name": "drop_columns", "args": {"columns": ["thumbnail", "trail_text"]}},
                    {"name": "trim_whitespace_columns", "args": {"columns": ["web_title"]}},
                    {
                        "name": "drop_empty_rows",
                        "args": {"required_columns": ["body_text"]},
                    },
                    {
                        "name": "normalize_text_columns",
                        "args": {"columns": ["body_text"]},
                    },
                ],
            },
            "chunking": {
                "input_dir": str(self.parquet_dir.parent / "pre_chunk"),
                "output_dir": str(self.parquet_dir.parent / "chunked_parquet"),
                "text_columns": ["body_text"],
                "id_columns": ["api_id"],
                "profile_columns": ["profile"],
                "passthrough_columns": ["headline", "web_title"],
                "profiles": {
                    "default": {
                        "strategy": "semantic_sentence",
                        "params": {
                            "min_chars": 5,
                            "max_chars": 200,
                            "overlap_chars": 0,
                            "similarity_threshold": 0.25,
                            "sentence_splitter": "simple_regex",
                        },
                    },
                },
            },
        }
        with ingestion_config_path.open("w", encoding="utf-8") as config_file:
            yaml.safe_dump(config, config_file, sort_keys=False)

    def _load_settings_for_guardian_client(self) -> Settings:
        """Load typed Settings values from the temp YAML config."""
        raw_config = Settings.load_ingestion_config_from_root(self.configuration_root)
        return Settings(
            api_key="integration-test-api-key",
            base_url=str(raw_config["base_url"]),
            default_page_size=int(raw_config["default_page_size"]),
            max_page_size=int(raw_config["max_page_size"]),
            timeout_seconds=int(raw_config["timeout_seconds"]),
        )

    def _load_profiles_for_guardian_client(self) -> Dict[str, Any]:
        """Load typed Guardian profile configs from the temp YAML config."""
        return self._original_load_guardian_profile_configs(
            configuration_root=self.configuration_root
        )

    def _load_article_ingestor_config(self) -> Any:
        """Load typed ArticleIngestor config from the temp YAML config."""
        return self._original_load_article_ingestor_config(
            configuration_root=self.configuration_root
        )

    def _dispatch_request(self, *, url: str, params: Dict[str, Any], timeout: int) -> _FakeResponse:
        """Route a request through the active fake transport."""
        if self.transport is None:
            raise AssertionError("FakeGuardianTransport must be configured before running a test")
        return self.transport.get(url=url, params=params, timeout=timeout)

    def _create_ingestor(self, *, run_timestamp: str) -> ArticleIngestor:
        """Create a real ArticleIngestor bound to the temp config."""
        article_ingestor = ArticleIngestor()
        article_ingestor.run_timestamp = run_timestamp
        return article_ingestor

    def _create_normalizer(self) -> ArticleNormalizer:
        """Create a real ArticleNormalizer bound to the temp config."""
        return ArticleNormalizer(configuration_root=self.configuration_root)

    def _create_preprocessor(self) -> PreChunkPreprocessor:
        """Create a real PreChunkPreprocessor bound to the temp config."""
        return PreChunkPreprocessor(configuration_root=self.configuration_root)

    def _set_transport(
        self,
        *,
        search_payloads: Dict[tuple[str, int], Dict[str, Any]],
        detail_payloads: Dict[str, Dict[str, Any] | _TransportFailure],
    ) -> FakeGuardianTransport:
        """Attach the fake Guardian transport for the current test."""
        self.transport = FakeGuardianTransport(
            search_payloads=search_payloads,
            detail_payloads=detail_payloads,
        )
        return self.transport


class TestArticleIngestorRunIntegration(IngestionPipelineIntegrationTestCase):
    """This class tests run."""

    def test_run_ingests_profiles_and_writes_checkpoints(self) -> None:
        """run: ingests configured profiles, records request params, and writes checkpoints."""
        transport = self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response(
                    [{"id": "technology/article-1"}, {"id": "technology/article-2"}]
                ),
                ("science", 1): _build_search_response(
                    [{"id": "science/article-1"}]
                ),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Technology Article 1",
                    "technology_daily",
                ),
                "technology/article-2": _build_article(
                    "technology/article-2",
                    "Technology Article 2",
                    "technology_daily",
                    metadata={
                        "published_at": "2026-04-28T11:00:00Z",
                        "last_modified": "2026-04-28T11:30:00Z",
                    },
                ),
                "science/article-1": _build_article(
                    "science/article-1",
                    "Science Article 1",
                    "science_daily",
                ),
            },
        )

        article_ingestor = self._create_ingestor(run_timestamp="20260428T120000Z")
        result = article_ingestor.run()

        self.assertEqual(result["profile_count"], 2)
        self.assertEqual(result["profiles_run"], ["technology_daily", "science_daily"])
        self.assertEqual(result["limit_per_profile"], 2)
        self.assertEqual(result["searched_count"], 3)
        self.assertEqual(result["fetched_count"], 3)
        self.assertEqual(result["failed_count"], 0)
        self.assertEqual(len(result["checkpoint_files"]), 2)
        for checkpoint_file in result["checkpoint_files"]:
            self.assertTrue(Path(checkpoint_file).is_file())
        self.assertEqual(
            Path(str(result["id_manifest_file"])),
            self.checkpoint_dir.parent / "ingested_ids" / "ingested_ids.json",
        )
        self.assertTrue(Path(str(result["id_manifest_file"])).is_file())

        technology_checkpoint = self.checkpoint_dir / "technology_daily_20260428T120000Z.json"
        with technology_checkpoint.open("r", encoding="utf-8") as checkpoint_file:
            payload = json.load(checkpoint_file)
        self.assertEqual(payload["profile"], "technology_daily")
        self.assertEqual(payload["fetched_count"], 2)
        self.assertEqual(payload["items"][0]["id"], "technology/article-1")

        search_calls = [call for call in transport.calls if call["url"].endswith("/search")]
        self.assertEqual(len(search_calls), 2)
        technology_search = next(
            call for call in search_calls if call["params"]["q"] == "chips"
        )
        self.assertEqual(technology_search["params"]["section"], "technology")
        self.assertEqual(technology_search["params"]["page-size"], 2)
        self.assertEqual(technology_search["params"]["order-by"], "newest")
        self.assertEqual(technology_search["params"]["use-date"], "published")
        self.assertEqual(technology_search["timeout"], 15)

        detail_calls = [call for call in transport.calls if not call["url"].endswith("/search")]
        self.assertEqual(len(detail_calls), 3)
        self.assertTrue(
            any(
                call["params"]["show-fields"] == "headline,bodyText"
                and call["url"].endswith("/technology/article-1")
                for call in detail_calls
            )
        )

    def test_run_skips_already_ingested_ids_on_second_run(self) -> None:
        """run: second execution skips detail fetches for already ingested ids."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "technology/article-1"}]),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Technology Article 1",
                    "technology_daily",
                ),
            },
        )
        first_ingestor = self._create_ingestor(run_timestamp="20260428T120000Z")
        first_result = first_ingestor.run()
        self.assertEqual(first_result["fetched_count"], 1)

        second_transport = self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "technology/article-1"}]),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Technology Article 1",
                    "technology_daily",
                ),
            },
        )
        second_ingestor = self._create_ingestor(run_timestamp="20260428T130000Z")
        second_result = second_ingestor.run()

        self.assertEqual(second_result["fetched_count"], 0)
        self.assertEqual(second_result["skipped_existing_count"], 1)
        self.assertEqual(
            Path(str(second_result["id_manifest_file"])),
            self.checkpoint_dir.parent / "ingested_ids" / "ingested_ids.json",
        )
        second_detail_calls = [
            call for call in second_transport.calls if not call["url"].endswith("/search")
        ]
        self.assertEqual(second_detail_calls, [])

    def test_run_skips_shared_ids_across_profiles_in_same_run(self) -> None:
        """run: shared ids from one profile are skipped when another profile returns them."""
        transport = self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "shared/article-1"}]),
                ("science", 1): _build_search_response([{"id": "shared/article-1"}]),
            },
            detail_payloads={
                "shared/article-1": _build_article(
                    "shared/article-1",
                    "Shared Article",
                    "technology_daily",
                ),
            },
        )
        article_ingestor = self._create_ingestor(run_timestamp="20260428T140000Z")
        result = article_ingestor.run()

        self.assertEqual(result["fetched_count"], 1)
        self.assertEqual(result["skipped_existing_count"], 1)

        profile_results = {item["profile"]: item for item in result["results"]}
        self.assertEqual(profile_results["technology_daily"]["fetched_count"], 1)
        self.assertEqual(profile_results["technology_daily"]["skipped_existing_count"], 0)
        self.assertEqual(profile_results["science_daily"]["fetched_count"], 0)
        self.assertEqual(profile_results["science_daily"]["skipped_existing_count"], 1)

        detail_calls = [call for call in transport.calls if not call["url"].endswith("/search")]
        self.assertEqual(len(detail_calls), 1)


class TestArticleNormalizerIntegration(IngestionPipelineIntegrationTestCase):
    """This class tests normalize_day_to_parquet."""

    def test_normalize_day_to_parquet_writes_expected_rows(self) -> None:
        """normalize_day_to_parquet: converts ingested checkpoints into combined parquet output."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response(
                    [{"id": "technology/article-1"}]
                ),
                ("science", 1): _build_search_response(
                    [{"id": "science/article-1"}]
                ),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Technology Article 1",
                    "technology_daily",
                ),
                "science/article-1": _build_article(
                    "science/article-1",
                    "Science Article 1",
                    "science_daily",
                    metadata={
                        "published_at": "2026-04-28T12:00:00Z",
                        "last_modified": "2026-04-28T12:15:00Z",
                    },
                ),
            },
        )

        article_ingestor = self._create_ingestor(run_timestamp="20260428T130000Z")
        article_ingestor.run()

        normalizer = self._create_normalizer()
        written = normalizer.normalize_day_to_parquet(INGESTION_DAY)

        self.assertEqual(set(written.keys()), {INGESTION_DAY.isoformat()})
        combined_df = pd.read_parquet(Path(written[INGESTION_DAY.isoformat()]))

        self.assertEqual(
            set(combined_df["api_id"]),
            {"technology/article-1", "science/article-1"},
        )
        technology_row = combined_df[combined_df["api_id"] == "technology/article-1"].iloc[0]
        self.assertEqual(technology_row["profile"], "technology_daily")
        self.assertEqual(technology_row["web_title"], "Technology Article 1")
        self.assertEqual(
            technology_row["published_at"].isoformat(),
            "2026-04-28T10:00:00+00:00",
        )
        self.assertEqual(
            technology_row["last_modified"].isoformat(),
            "2026-04-28T10:30:00+00:00",
        )
        science_row = combined_df[combined_df["api_id"] == "science/article-1"].iloc[0]
        self.assertEqual(science_row["profile"], "science_daily")

    def test_normalize_day_to_parquet_uses_all_same_day_checkpoints(self) -> None:
        """normalize_day_to_parquet: includes rows from all same-day checkpoints."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response(
                    [{"id": "technology/article-1"}]
                ),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Old Technology Title",
                    "technology_daily",
                ),
            },
        )
        first_ingestor = self._create_ingestor(run_timestamp="20260428T090000Z")
        first_ingestor.run()

        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response(
                    [{"id": "technology/article-2"}]
                ),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-2": _build_article(
                    "technology/article-2",
                    "New Technology Title",
                    "technology_daily",
                    metadata={
                        "published_at": "2026-04-28T14:00:00Z",
                        "last_modified": "2026-04-28T14:45:00Z",
                    },
                ),
            },
        )
        second_ingestor = self._create_ingestor(run_timestamp="20260428T210000Z")
        second_ingestor.run()

        normalizer = self._create_normalizer()
        written = normalizer.normalize_day_to_parquet(INGESTION_DAY)

        combined_df = pd.read_parquet(Path(written[INGESTION_DAY.isoformat()]))
        self.assertEqual(
            set(combined_df["web_title"]),
            {"Old Technology Title", "New Technology Title"},
        )

    def test_normalize_day_to_parquet_skips_empty_profiles_after_failures(self) -> None:
        """normalize_day_to_parquet: writes successful rows while skipping empty failed profiles."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response(
                    [
                        {"id": "technology/article-1"},
                        {"headline": "missing id"},
                    ]
                ),
                ("science", 1): _build_search_response(
                    [{"id": "science/article-1"}]
                ),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "Technology Success",
                    "technology_daily",
                ),
                "science/article-1": _TransportFailure(
                    status_code=503,
                    text="science unavailable",
                ),
            },
        )

        article_ingestor = self._create_ingestor(run_timestamp="20260428T150000Z")
        result = article_ingestor.run()

        self.assertEqual(result["searched_count"], 3)
        self.assertEqual(result["fetched_count"], 1)
        self.assertEqual(result["failed_count"], 2)

        science_checkpoint = self.checkpoint_dir / "science_daily_20260428T150000Z.json"
        with science_checkpoint.open("r", encoding="utf-8") as checkpoint_file:
            science_payload = json.load(checkpoint_file)
        self.assertEqual(science_payload["items"], [])
        self.assertEqual(science_payload["failed_count"], 1)

        technology_checkpoint = self.checkpoint_dir / "technology_daily_20260428T150000Z.json"
        with technology_checkpoint.open("r", encoding="utf-8") as checkpoint_file:
            technology_payload = json.load(checkpoint_file)
        self.assertEqual(technology_payload["fetched_count"], 1)
        self.assertEqual(len(technology_payload["failures"]), 1)
        self.assertEqual(
            technology_payload["failures"][0]["error"],
            "Missing id in topic search result",
        )

        normalizer = self._create_normalizer()
        written = normalizer.normalize_day_to_parquet(INGESTION_DAY)

        self.assertEqual(set(written.keys()), {INGESTION_DAY.isoformat()})
        combined_df = pd.read_parquet(Path(written[INGESTION_DAY.isoformat()]))
        self.assertEqual(list(combined_df["api_id"]), ["technology/article-1"])
        self.assertTrue((self.parquet_dir / f"{INGESTION_DAY.isoformat()}.parquet").is_file())


class TestPreChunkPreprocessorIntegration(IngestionPipelineIntegrationTestCase):
    """This class tests preprocess_all_to_parquet."""

    def test_preprocess_all_to_parquet_creates_chunk_ready_outputs(self) -> None:
        """preprocess_all_to_parquet: writes transformed parquet outputs for chunking."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "technology/article-1"}]),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "  Technology Article 1  ",
                    "technology_daily",
                    metadata={"body_text": "Body\n text"},
                ),
            },
        )
        article_ingestor = self._create_ingestor(run_timestamp="20260428T170000Z")
        article_ingestor.run()

        normalizer = self._create_normalizer()
        normalized = normalizer.normalize_day_to_parquet(INGESTION_DAY)
        self.assertIn(INGESTION_DAY.isoformat(), normalized)

        preprocessor = self._create_preprocessor()
        written = preprocessor.preprocess_to_parquet()
        self.assertEqual(set(written.keys()), {INGESTION_DAY.isoformat()})
        output_df = pd.read_parquet(Path(written[INGESTION_DAY.isoformat()]))
        self.assertEqual(output_df.iloc[0]["web_title"], "Technology Article 1")
        self.assertEqual(output_df.iloc[0]["body_text"], "Body text")
        self.assertNotIn("thumbnail", output_df.columns)
        self.assertNotIn("trail_text", output_df.columns)


class TestChunkerIntegration(IngestionPipelineIntegrationTestCase):
    """This class tests chunk_to_parquet."""

    def _prepare_pipeline_for_chunking(self, run_timestamp: str) -> None:
        """Run ingestion + normalization + pre-chunk preprocessing for chunking tests."""
        article_ingestor = self._create_ingestor(run_timestamp=run_timestamp)
        article_ingestor.run()
        normalizer = self._create_normalizer()
        normalizer.normalize_day_to_parquet(INGESTION_DAY)
        preprocessor = self._create_preprocessor()
        preprocessor.preprocess_to_parquet()

    def test_chunk_to_parquet_writes_combined_per_profile_parquet(self) -> None:
        """chunk_to_parquet: writes a single combined per-profile parquet for all input days."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "technology/article-1"}]),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "  Technology Article 1  ",
                    "technology_daily",
                    metadata={
                        "body_text": "First sentence here. Second sentence unrelated xyz.",
                    },
                ),
            },
        )
        self._prepare_pipeline_for_chunking("20260428T180000Z")

        chunker = Chunker(configuration_root=self.configuration_root)
        written = chunker.chunk_to_parquet(profile="default")
        self.assertEqual(set(written.keys()), {"default"})
        chunk_path = Path(written["default"])
        self.assertTrue(chunk_path.is_file())
        self.assertEqual(
            chunk_path,
            self.parquet_dir.parent / "chunked_parquet" / "default.parquet",
        )
        chunk_df = pd.read_parquet(chunk_path)
        self.assertIn("chunk_text", chunk_df.columns)
        self.assertIn("source_day", chunk_df.columns)
        self.assertIn("chunk_index", chunk_df.columns)
        self.assertGreater(len(chunk_df), 0)
        self.assertEqual(chunk_df.iloc[0]["source_api_id"], "technology/article-1")

    def test_chunk_to_parquet_rebuilds_on_repeated_runs(self) -> None:
        """chunk_to_parquet: re-running rebuilds the same combined parquet (no skip state)."""
        self._set_transport(
            search_payloads={
                ("chips", 1): _build_search_response([{"id": "technology/article-1"}]),
                ("science", 1): _build_search_response([]),
            },
            detail_payloads={
                "technology/article-1": _build_article(
                    "technology/article-1",
                    "  Technology Article 1  ",
                    "technology_daily",
                    metadata={
                        "body_text": "First sentence here. Second sentence unrelated xyz.",
                    },
                ),
            },
        )
        self._prepare_pipeline_for_chunking("20260428T180000Z")

        chunker = Chunker(configuration_root=self.configuration_root)
        first = chunker.chunk_to_parquet(profile="default")
        second = chunker.chunk_to_parquet(profile="default")
        self.assertEqual(first, second)
        self.assertEqual(set(first.keys()), {"default"})
        chunk_df = pd.read_parquet(first["default"])
        self.assertGreater(len(chunk_df), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
