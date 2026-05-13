"""Unit tests for vector search service layer."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config.settings import EmbeddingConfig, Settings, VectorSearchConfig
from src.enums.embedding_provider import EmbeddingProvider
from src.service_layer.vector_search_service import (
    VectorSearchResponse,
    VectorSearchService,
    _compose_query_filter,
    _inclusive_iso_day_strings,
    _normalize_query_hits,
)


def _embedding_config_fixture() -> EmbeddingConfig:
    return EmbeddingConfig(
        input_dir=Path("checkpoints/chunked_parquet"),
        output_dir=Path("checkpoints/embeddings"),
        text_column="chunk_text",
        provider=EmbeddingProvider.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )


class FakeEmbedHandler:  # pylint: disable=too-few-public-methods
    """Stub embedding handler for unit tests."""

    def __init__(self, vectors=None):
        self._vectors = vectors if vectors is not None else [[0.25, 0.75]]

    def embed(self, texts, batch_size=64):
        """Return fixed vectors (ignores texts)."""
        _, _ = texts, batch_size
        return list(self._vectors)


class TestNormalizeQueryHits(unittest.TestCase):
    """This class tests _normalize_query_hits."""

    def test_empty_vectors(self):
        """_normalize_query_hits: returns empty tuple when no vectors."""
        raw = SimpleNamespace(vectors=[])
        self.assertEqual(_normalize_query_hits(raw), ())

    def test_skips_vector_without_key(self):
        """_normalize_query_hits: skips entries with missing key."""
        raw = SimpleNamespace(
            vectors=[
                SimpleNamespace(key=None, distance=1.0, metadata={}),
                SimpleNamespace(key="ok", distance=0.5, metadata=None),
            ]
        )
        hits = _normalize_query_hits(raw)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].key, "ok")
        self.assertEqual(hits[0].distance, 0.5)
        self.assertIsNone(hits[0].metadata)

    def test_metadata_must_be_dict(self):
        """_normalize_query_hits: non-dict metadata becomes None."""
        raw = SimpleNamespace(
            vectors=[SimpleNamespace(key="k", distance=None, metadata="bad")]
        )
        hits = _normalize_query_hits(raw)
        self.assertIsNone(hits[0].metadata)


class TestInclusiveIsoDayStrings(unittest.TestCase):
    """This class tests _inclusive_iso_day_strings."""

    def test_empty_when_no_bounds(self):
        """_inclusive_iso_day_strings: returns empty when both bounds missing."""
        self.assertEqual(_inclusive_iso_day_strings(None, None), [])

    def test_single_day_when_only_lower_bound(self):
        """_inclusive_iso_day_strings: open upper bound collapses to date_from day."""
        self.assertEqual(
            _inclusive_iso_day_strings("2026-03-05", None),
            ["2026-03-05"],
        )

    def test_single_day_when_only_upper_bound(self):
        """_inclusive_iso_day_strings: open lower bound collapses to date_to day."""
        self.assertEqual(
            _inclusive_iso_day_strings(None, "20260305"),
            ["2026-03-05"],
        )

    def test_inclusive_range(self):
        """_inclusive_iso_day_strings: lists every day in closed range."""
        self.assertEqual(
            _inclusive_iso_day_strings("2026-05-01", "2026-05-03"),
            ["2026-05-01", "2026-05-02", "2026-05-03"],
        )

    def test_raises_when_from_after_to(self):
        """_inclusive_iso_day_strings: raises when date_from is after date_to."""
        with self.assertRaises(ValueError) as ctx:
            _inclusive_iso_day_strings("2026-05-10", "2026-05-01")
        self.assertIn("on or before", str(ctx.exception))

    def test_raises_when_span_exceeds_max_days(self):
        """_inclusive_iso_day_strings: raises when range longer than max_days."""
        with self.assertRaises(ValueError) as ctx:
            _inclusive_iso_day_strings(
                "2026-01-01",
                "2026-01-20",
                max_days=5,
            )
        self.assertIn("max supported", str(ctx.exception))


class TestComposeQueryFilter(unittest.TestCase):
    """This class tests _compose_query_filter."""

    def test_none_when_no_dates_and_no_metadata(self):
        """_compose_query_filter: returns None when nothing to merge."""
        self.assertIsNone(
            _compose_query_filter(
                None,
                date_from=None,
                date_to=None,
                date_metadata_key="source_day",
            )
        )

    def test_metadata_only(self):
        """_compose_query_filter: passes flat metadata when no dates (Supabase doc style)."""
        out = _compose_query_filter(
            {"k": 1},
            date_from=None,
            date_to=None,
            date_metadata_key="source_day",
        )
        self.assertEqual(out, {"k": 1})

    def test_date_range_merged_with_metadata(self):
        """_compose_query_filter: merges flat metadata with $in day list."""
        out = _compose_query_filter(
            {"source_profile": "main"},
            date_from="20260501",
            date_to="2026-05-04",
            date_metadata_key="source_day",
        )
        self.assertEqual(
            out,
            {
                "source_profile": "main",
                "source_day": {
                    "$in": [
                        "2026-05-01",
                        "2026-05-02",
                        "2026-05-03",
                        "2026-05-04",
                    ],
                },
            },
        )

    def test_single_bound(self):
        """_compose_query_filter: single-sided bound uses one calendar day."""
        only_from = _compose_query_filter(
            None,
            date_from="2026-01-02",
            date_to=None,
            date_metadata_key="d",
        )
        self.assertEqual(only_from, {"d": {"$in": ["2026-01-02"]}})
        only_to = _compose_query_filter(
            None,
            date_from=None,
            date_to="2026-01-02",
            date_metadata_key="d",
        )
        self.assertEqual(only_to, {"d": {"$in": ["2026-01-02"]}})

    def test_raises_logical_only_metadata_with_dates(self):
        """_compose_query_filter: raises when top-level $and only with date bounds."""
        with self.assertRaises(ValueError) as ctx:
            _compose_query_filter(
                {"$and": [{"source_profile": "p"}]},
                date_from="2026-05-01",
                date_to=None,
                date_metadata_key="source_day",
            )
        self.assertIn("flat", str(ctx.exception))

    def test_raises_when_from_after_to(self):
        """_compose_query_filter: raises when date_from is after date_to."""
        with self.assertRaises(ValueError) as ctx:
            _compose_query_filter(
                None,
                date_from="2026-05-10",
                date_to="2026-05-01",
                date_metadata_key="source_day",
            )
        self.assertIn("on or before", str(ctx.exception))

    def test_raises_on_metadata_key_collision(self):
        """_compose_query_filter: raises when metadata_filter already sets date key."""
        with self.assertRaises(ValueError) as ctx:
            _compose_query_filter(
                {"source_day": {"$eq": "2026-05-01"}},
                date_from="2026-05-01",
                date_to=None,
                date_metadata_key="source_day",
            )
        self.assertIn("already sets", str(ctx.exception))

    def test_raises_when_and_clause_sets_date_key(self):
        """_compose_query_filter: rejects logical-only metadata when dates are set."""
        with self.assertRaises(ValueError) as ctx:
            _compose_query_filter(
                {"$and": [{"source_profile": "p"}, {"source_day": "2026-05-01"}]},
                date_from="2026-05-02",
                date_to=None,
                date_metadata_key="source_day",
            )
        self.assertIn("flat", str(ctx.exception))

    def test_raises_when_key_empty_with_dates(self):
        """_compose_query_filter: raises when date bounds set but key is blank."""
        with self.assertRaises(ValueError):
            _compose_query_filter(
                None,
                date_from="2026-05-01",
                date_to=None,
                date_metadata_key="   ",
            )


class TestVectorSearchServiceSearchByText(unittest.TestCase):
    """This class tests VectorSearchService.search_by_text."""

    def test_search_empty_text_raises(self):
        """search_by_text: raises ValueError when text empty."""
        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="b", index_name="i"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ):
            service = VectorSearchService()
        with self.assertRaises(ValueError) as ctx:
            service.search_by_text("   ")
        self.assertIn("non-empty", str(ctx.exception))

    def test_search_top_k_invalid_raises(self):
        """search_by_text: raises when top_k < 1."""
        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="b", index_name="i"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ):
            service = VectorSearchService()
        with self.assertRaises(ValueError):
            service.search_by_text("hello", top_k=0)

    def test_search_happy_path_forwards_sdk_args(self):
        """search_by_text: embeds, queries storage, normalizes hits."""
        mock_query = MagicMock(
            return_value=SimpleNamespace(
                vectors=[
                    SimpleNamespace(key="row-1", distance=0.1, metadata={"url": "u"}),
                ]
            )
        )
        mock_index = MagicMock()
        mock_index.query = mock_query
        mock_bucket = MagicMock()
        mock_bucket.index.return_value = mock_index
        mock_vectors = MagicMock()
        mock_vectors.from_.return_value = mock_bucket

        mock_client = MagicMock()
        mock_client.storage.vectors.return_value = mock_vectors

        def factory():
            return mock_client

        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="my-bucket", index_name="my-index"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ), patch(
            "src.service_layer.vector_search_service.resolve_provider",
            return_value=FakeEmbedHandler([[0.1, 0.2, 0.3]]),
        ):
            service = VectorSearchService(supabase_client_factory=factory)
            self.assertIsNotNone(service.timer)
            out = service.search_by_text(
                "economy news ",
                top_k=7,
                metadata_filter={"source_profile": "main"},
                return_distance=False,
                return_metadata=False,
            )

        self.assertIsInstance(out, VectorSearchResponse)
        self.assertEqual(len(out.hits), 1)
        self.assertEqual(out.hits[0].key, "row-1")
        self.assertEqual(out.hits[0].distance, 0.1)
        self.assertEqual(out.hits[0].metadata, {"url": "u"})
        mock_bucket.index.assert_called_once_with("my-index")
        mock_vectors.from_.assert_called_once_with("my-bucket")
        mock_query.assert_called_once()
        kwargs = mock_query.call_args.kwargs
        self.assertEqual(kwargs["topK"], 7)
        self.assertEqual(kwargs["filter"], {"source_profile": "main"})
        self.assertFalse(kwargs["return_distance"])
        self.assertFalse(kwargs["return_metadata"])
        self.assertEqual(kwargs["query_vector"]["float32"], [0.1, 0.2, 0.3])

    def test_search_forwards_date_filter(self):
        """search_by_text: merges date bounds into Storage filter."""
        mock_query = MagicMock(
            return_value=SimpleNamespace(vectors=[]),
        )
        mock_index = MagicMock()
        mock_index.query = mock_query
        mock_bucket = MagicMock()
        mock_bucket.index.return_value = mock_index
        mock_vectors = MagicMock()
        mock_vectors.from_.return_value = mock_bucket
        mock_client = MagicMock()
        mock_client.storage.vectors.return_value = mock_vectors

        def factory():
            return mock_client

        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(
                bucket_name="b",
                index_name="i",
                date_metadata_key="source_day",
            ),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ), patch(
            "src.service_layer.vector_search_service.resolve_provider",
            return_value=FakeEmbedHandler([[0.1, 0.2, 0.3]]),
        ):
            service = VectorSearchService(supabase_client_factory=factory)
            service.search_by_text(
                "q",
                top_k=3,
                metadata_filter={"source_profile": "p"},
                date_from="2026-05-01",
                date_to="2026-05-02",
                date_metadata_key="first_publication_date",
            )

        kwargs = mock_query.call_args.kwargs
        self.assertEqual(
            kwargs["filter"],
            {
                "source_profile": "p",
                "first_publication_date": {
                    "$in": ["2026-05-01", "2026-05-02"],
                },
            },
        )

    def test_search_raises_when_date_conflicts_with_metadata_filter(self):
        """search_by_text: raises when metadata_filter already sets the date field."""
        mock_index = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.index.return_value = mock_index
        mock_vectors = MagicMock()
        mock_vectors.from_.return_value = mock_bucket
        mock_client = MagicMock()
        mock_client.storage.vectors.return_value = mock_vectors

        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="b", index_name="i"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ), patch(
            "src.service_layer.vector_search_service.resolve_provider",
            return_value=FakeEmbedHandler([[0.1]]),
        ):
            service = VectorSearchService(supabase_client_factory=lambda: mock_client)
            with self.assertRaises(ValueError):
                service.search_by_text(
                    "q",
                    metadata_filter={"source_day": "2026-05-01"},
                    date_from="2026-05-01",
                )
        mock_index.query.assert_not_called()

    def test_search_raises_when_embedding_returns_no_rows(self):
        """search_by_text: raises when provider returns zero vectors."""
        mock_index = MagicMock()
        mock_index.query = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.index.return_value = mock_index
        mock_vectors = MagicMock()
        mock_vectors.from_.return_value = mock_bucket
        mock_client = MagicMock()
        mock_client.storage.vectors.return_value = mock_vectors

        def factory():
            return mock_client

        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="b", index_name="i"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ), patch(
            "src.service_layer.vector_search_service.resolve_provider",
            return_value=FakeEmbedHandler(vectors=[]),
        ):
            service = VectorSearchService(supabase_client_factory=factory)
            with self.assertRaises(RuntimeError) as ctx:
                service.search_by_text("x")
            self.assertIn("no vector", str(ctx.exception))
        mock_index.query.assert_not_called()


class TestVectorSearchServiceSemanticSearch(unittest.TestCase):
    """This class tests VectorSearchService.semantic_search."""

    def test_semantic_search_delegates_to_search_path(self):
        """semantic_search: uses same implementation as search_by_text."""
        with patch.object(
            Settings,
            "load_vector_search_config",
            return_value=VectorSearchConfig(bucket_name="b", index_name="i"),
        ), patch.object(
            Settings,
            "load_embedding_config",
            return_value=_embedding_config_fixture(),
        ), patch.object(
            VectorSearchService,
            "_semantic_vector_query",
            return_value=VectorSearchResponse(hits=()),
        ) as mock_impl:
            service = VectorSearchService()
            out = service.semantic_search("hello", top_k=5)

        self.assertEqual(out, VectorSearchResponse(hits=()))
        mock_impl.assert_called_once()
        _, kwargs = mock_impl.call_args
        self.assertEqual(kwargs["top_k"], 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
