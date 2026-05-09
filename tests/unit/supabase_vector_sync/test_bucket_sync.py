"""Unit tests for Supabase vector bucket sync helpers and client."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
from storage3.exceptions import StorageApiError

from src.config.settings import SupabaseVectorSyncConfig
from src.enums.vector_bucket_distance_metric import VectorBucketDistanceMetric
from src.supabase_vector_sync import bucket_sync as bucket_sync_module
from src.supabase_vector_sync.bucket_sync import (
    SupabaseVectorBucketSync,
    _build_metadata,
    _coerce_leaf_metadata,
    _default_supabase_client_factory,
    _embedding_to_float_vector,
    _looks_like_duplicate_resource,
    _run_idempotent_create,
    _stable_row_key,
)
from src.utils.timer import Timer


def _patch_sync_config(return_value: SupabaseVectorSyncConfig):
    """Patch Settings vector sync YAML loader."""
    return patch.object(
        bucket_sync_module.Settings,
        "load_supabase_vector_sync_config",
        return_value=return_value,
    )


def _named_client_factory(client: MagicMock):
    """Return factory bound to ``client`` (avoid lambda warnings in tests)."""

    def factory() -> MagicMock:
        return client

    return factory


def _cfg(**kwargs: Any) -> SupabaseVectorSyncConfig:
    merged: dict[str, Any] = {
        "input_dir": Path("/tmp/embed"),
        "bucket_name": "b",
        "index_name": "i",
        "dimension": 3,
        "distance_metric": VectorBucketDistanceMetric.COSINE,
        "embedding_column": "embedding",
        "key_columns": ("source_api_id", "chunk_index", "source_row_index"),
        "metadata_columns": ("source_day",),
        "batch_size": 500,
        "create_bucket_if_missing": True,
        "create_index_if_missing": True,
    }
    merged.update(kwargs)
    return SupabaseVectorSyncConfig(**merged)


class TestLooksLikeDuplicateResource(unittest.TestCase):
    """This class tests _looks_like_duplicate_resource."""

    def test_detects_conflict_message_from_storage_api_error(self) -> None:
        """_looks_like_duplicate_resource: detects Storage conflicts."""
        err = StorageApiError("duplicate", "Conflict", "409")
        self.assertTrue(_looks_like_duplicate_resource(err))

    def test_false_for_unrelated_errors(self) -> None:
        """_looks_like_duplicate_resource: rejects unrelated failures."""
        self.assertFalse(_looks_like_duplicate_resource(ValueError("broken")))

    def test_non_storage_exception_can_match_conflict_shape(self) -> None:
        """_looks_like_duplicate_resource: honors conflict signals on generic exceptions."""
        self.assertTrue(
            _looks_like_duplicate_resource(RuntimeError("status 409 for conflict")),
        )


class TestRunIdempotentCreate(unittest.TestCase):
    """This class tests _run_idempotent_create."""

    def test_swallows_storage_duplicate(self) -> None:
        """_run_idempotent_create: ignores duplicate Storage failures."""
        def boom() -> None:
            raise StorageApiError("exists", "Conflict", "409")

        _run_idempotent_create(boom)

    def test_reraises_non_duplicate_storage(self) -> None:
        """_run_idempotent_create: propagates actionable Storage errors."""

        def boom() -> None:
            raise StorageApiError("nope", "BadRequest", "400")

        with self.assertRaises(StorageApiError):
            _run_idempotent_create(boom)

    def test_swallows_duplicate_generic_exceptions(self) -> None:
        """_run_idempotent_create: maps duplicate-ish generic failures."""

        def boom() -> None:
            raise RuntimeError("bucket already exists")

        _run_idempotent_create(boom)

    def test_reraises_non_duplicate_generic(self) -> None:
        """_run_idempotent_create: propagates unrelated generic failures."""

        def boom() -> None:
            raise RuntimeError("forbidden")

        with self.assertRaises(RuntimeError):
            _run_idempotent_create(boom)


class TestSupabaseVectorBucketSync(unittest.TestCase):
    """This class tests SupabaseVectorBucketSync."""

    def test_timer_property(self) -> None:
        """SupabaseVectorBucketSync.timer: echoes injected timer."""
        timer = Timer()
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(input_dir=Path(tmp))
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    timer=timer,
                    client_factory=_named_client_factory(MagicMock()),
                )
        self.assertIs(sync.timer, timer)

    def test_raises_when_parquet_missing(self) -> None:
        """sync_profile_to_bucket: raises FileNotFoundError when parquet absent."""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _cfg(input_dir=Path(tmp))
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(client_factory=MagicMock, timer=Timer())
                with self.assertRaises(FileNotFoundError):
                    sync.sync_profile_to_bucket(profile="missing")

    def _make_client_mock(self, put_batches: list) -> MagicMock:
        index_scope = MagicMock()
        def capture_batch(batch: Any) -> None:
            put_batches.append(list(batch))

        index_scope.put.side_effect = capture_batch

        bucket_scope = MagicMock()
        bucket_scope.index.return_value = index_scope

        vectors = MagicMock()
        vectors.from_.return_value = bucket_scope
        vectors.create_bucket = MagicMock()

        client = MagicMock()
        client.storage.vectors.return_value = vectors
        return client

    def test_sync_writes_batches_under_cap(self) -> None:
        """sync_profile_to_bucket: emits capped batches."""

        batches: list[list] = []
        embedding = [0.0, 1.0, 2.0]
        df = pd.DataFrame(
            [
                {
                    "source_api_id": "a",
                    "chunk_index": 0,
                    "source_row_index": 1,
                    "source_day": "2026-01-01",
                    "embedding": embedding,
                },
                {
                    "source_api_id": "b",
                    "chunk_index": 0,
                    "source_row_index": 2,
                    "source_day": "2026-01-02",
                    "embedding": embedding,
                },
                {
                    "source_api_id": "c",
                    "chunk_index": 0,
                    "source_row_index": 3,
                    "source_day": "2026-01-03",
                    "embedding": embedding,
                },
            ]
        )

        client = self._make_client_mock(batches)

        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            path = parquet_dir / "myprof.parquet"
            df.to_parquet(path, index=False)
            cfg = _cfg(
                input_dir=parquet_dir,
                batch_size=2,
                dimension=3,
                metadata_columns=("source_day",),
            )

            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    client_factory=_named_client_factory(client),
                    timer=Timer(),
                )
                with patch("builtins.print") as mocked_print:
                    out = sync.sync_profile_to_bucket(profile="myprof")

            self.assertEqual(out["vectors_uploaded"], 3)
            self.assertEqual(len(batches), 2)
            self.assertEqual(len(batches[0]), 2)
            self.assertEqual(len(batches[1]), 1)
            prefix = "[supabase_vector_sync] uploaded batch"
            batch_messages = [
                call.args[0]
                for call in mocked_print.call_args_list
                if call.args and str(call.args[0]).startswith(prefix)
            ]
            self.assertEqual(len(batch_messages), 2)
            self.assertIn("uploaded batch 1/2", batch_messages[0])
            self.assertIn("uploaded batch 2/2", batch_messages[1])
            vectors_api = client.storage.vectors.return_value
            vectors_api.create_bucket.assert_called_once_with("b")
            bucket_scope = vectors_api.from_.return_value
            bucket_scope.create_index.assert_called_once()

    def test_raises_when_dimension_mismatch(self) -> None:
        """sync_profile_to_bucket: rejects wrong embedding widths."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            df = pd.DataFrame(
                [
                    {
                        "source_api_id": "a",
                        "chunk_index": 0,
                        "source_row_index": 1,
                        "source_day": "d",
                        "embedding": [0.1, 0.2],
                    },
                ]
            )
            df.to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir, dimension=3)
            batches: list = []
            client = self._make_client_mock(batches)
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    client_factory=_named_client_factory(client),
                    timer=Timer(),
                )
                with self.assertRaises(ValueError) as ctx:
                    sync.sync_profile_to_bucket(profile="p")
                self.assertIn("dimension", str(ctx.exception))

    def test_empty_after_null_embeddings_returns_empty(self) -> None:
        """sync_profile_to_bucket: returns {} when all embeddings absent."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            df = pd.DataFrame(
                [
                    {
                        "source_api_id": "a",
                        "chunk_index": 0,
                        "source_row_index": 1,
                        "source_day": "d",
                        "embedding": None,
                    },
                ]
            )
            df.to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir)
            client = self._make_client_mock([])
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    client_factory=_named_client_factory(client),
                    timer=Timer(),
                )
                result = sync.sync_profile_to_bucket(profile="p")
            self.assertEqual(result, {})
            client.storage.vectors.return_value.create_bucket.assert_not_called()

    def test_skips_provision_when_disabled(self) -> None:
        """sync_profile_to_bucket: omits provisioning when configured off."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            df = pd.DataFrame(
                [
                    {
                        "source_api_id": "a",
                        "chunk_index": 0,
                        "source_row_index": 1,
                        "source_day": "d",
                        "embedding": [0.0, 1.0, 2.0],
                    },
                ]
            )
            df.to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(
                input_dir=parquet_dir,
                create_bucket_if_missing=False,
                create_index_if_missing=False,
            )
            batches: list = []
            client = self._make_client_mock(batches)
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    client_factory=_named_client_factory(client),
                    timer=Timer(),
                )
                sync.sync_profile_to_bucket(profile="p")

            vectors_api = client.storage.vectors.return_value
            vectors_api.create_bucket.assert_not_called()
            bucket_scope = vectors_api.from_.return_value
            bucket_scope.create_index.assert_not_called()

    def test_empty_frame_returns_before_client(self) -> None:
        """sync_profile_to_bucket: short-circuit on empty parquet rows."""
        sentinels = {"called": False}

        def factory() -> MagicMock:
            """Track whether provisioning attempted to instantiate a Supabase client."""
            sentinels["called"] = True
            raise AssertionError()

        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            empty = pd.DataFrame(
                columns=[
                    "source_api_id",
                    "chunk_index",
                    "source_row_index",
                    "source_day",
                    "embedding",
                ],
            )
            empty.to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir, metadata_columns=("source_day",))
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(client_factory=factory, timer=Timer())
                result = sync.sync_profile_to_bucket(profile="p")
        self.assertEqual(result, {})
        self.assertFalse(sentinels["called"])

    def test_raises_when_embedding_column_missing(self) -> None:
        """sync_profile_to_bucket: requires embedding column."""

        def factory() -> MagicMock:
            """Block Supabase uploads if validation fails to short-circuit."""
            raise AssertionError()

        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "source_api_id": "x",
                        "chunk_index": 0,
                        "source_row_index": 0,
                        "source_day": "d",
                    },
                ],
            ).to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir, metadata_columns=("source_day",))
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(client_factory=factory, timer=Timer())
                with self.assertRaises(ValueError) as ctx:
                    sync.sync_profile_to_bucket(profile="p")
                self.assertIn("embedding", str(ctx.exception))

    def test_raises_when_metadata_column_missing(self) -> None:
        """sync_profile_to_bucket: requires declared metadata selectors."""

        def factory() -> MagicMock:
            """Block Supabase uploads if validation fails to short-circuit."""
            raise AssertionError()

        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "source_api_id": "x",
                        "chunk_index": 0,
                        "source_row_index": 0,
                        "embedding": [0.5],
                    },
                ],
            ).to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir, dimension=1, metadata_columns=("source_day",))
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(client_factory=factory, timer=Timer())
                with self.assertRaises(ValueError) as ctx:
                    sync.sync_profile_to_bucket(profile="p")
                self.assertIn("metadata", str(ctx.exception))

    def test_raises_when_key_columns_missing_from_parquet(self) -> None:
        """sync_profile_to_bucket: requires configured id columns."""

        def factory() -> MagicMock:
            """Block Supabase uploads if validation fails to short-circuit."""
            raise AssertionError()

        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "chunk_index": 0,
                        "source_row_index": 0,
                        "source_day": "d",
                        "embedding": [1.0, 2.0, 3.0],
                    },
                ],
            ).to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(input_dir=parquet_dir)
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(client_factory=factory, timer=Timer())
                with self.assertRaises(ValueError) as ctx:
                    sync.sync_profile_to_bucket(profile="p")
                self.assertIn("source_api_id", str(ctx.exception))


class TestDefaultSupabaseClientFactory(unittest.TestCase):
    """This class tests _default_supabase_client_factory."""

    def test_returns_client_when_settings_returns_credentials(self) -> None:
        """_default_supabase_client_factory: uses Settings-managed secrets."""
        sentinel = MagicMock()
        fake_create = MagicMock(return_value=sentinel)
        with patch.object(
            bucket_sync_module.Settings,
            "load_supabase_credentials",
            return_value=("https://test.supabase.co", "srv"),
        ) as mocked_settings:
            with patch("supabase.create_client", fake_create):
                client = _default_supabase_client_factory()
        self.assertIs(client, sentinel)
        mocked_settings.assert_called_once_with()
        fake_create.assert_called_once_with("https://test.supabase.co", "srv")

    def test_raises_when_settings_raise(self) -> None:
        """_default_supabase_client_factory: forwards Settings validation errors."""
        with patch.object(
            bucket_sync_module.Settings,
            "load_supabase_credentials",
            side_effect=ValueError("SUPABASE_SERVICE_ROLE_KEY is not set or empty"),
        ):
            with self.assertRaises(ValueError) as ctx:
                _default_supabase_client_factory()
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY", str(ctx.exception))


class TestCoercionHelpers(unittest.TestCase):
    """This class tests metadata and embedding coercion."""

    def test_coerce_leaf_supports_numbers_and_dates(self) -> None:
        """_coerce_leaf_metadata: preserves standard scalar conversions."""
        self.assertIsNone(_coerce_leaf_metadata(None))
        self.assertIsNone(_coerce_leaf_metadata(pd.NA))
        self.assertTrue(_coerce_leaf_metadata(True) is True)
        self.assertEqual(_coerce_leaf_metadata(2), 2.0)
        self.assertEqual(
            _coerce_leaf_metadata(date(2024, 1, 5)),
            "2024-01-05",
        )
        weird = object()
        self.assertEqual(_coerce_leaf_metadata(weird), str(weird))

    def test_build_metadata_returns_none_when_all_missing(self) -> None:
        """_build_metadata: yields None when filters remove every field."""
        series = pd.Series({"source_day": pd.NA})
        self.assertIsNone(_build_metadata(series, ("source_day",)))

    def test_build_metadata_skips_unknown_columns(self) -> None:
        """_build_metadata: ignores absent metadata selectors."""
        series = pd.Series({"source_day": "x"})
        self.assertEqual(
            _build_metadata(series, ("source_day", "missing_meta")),
            {"source_day": "x"},
        )

    def test_embedding_handles_custom_iterables(self) -> None:
        """_embedding_to_float_vector: consumes generic iterables."""

        class Yielder:  # pylint: disable=too-few-public-methods
            """Yields deterministic embedding components."""

            def __iter__(self) -> Any:
                yield 0.1
                yield 0.9

        self.assertEqual(_embedding_to_float_vector(Yielder()), [0.1, 0.9])

    def test_embedding_rejects_nan_scalars(self) -> None:
        """_embedding_to_float_vector: treats NaN scalars as missing."""
        with self.assertRaises(ValueError) as ctx:
            _embedding_to_float_vector(float("nan"))
        self.assertIn("NaN", str(ctx.exception))

    def test_embedding_rejects_none(self) -> None:
        """_embedding_to_float_vector: rejects absent payload."""
        with self.assertRaises(ValueError) as ctx:
            _embedding_to_float_vector(None)
        self.assertIn("missing", str(ctx.exception))

    def test_embedding_typeerror_na_branch(self) -> None:
        """_embedding_to_float_vector: survives ``pd.isna`` TypeErrors on scalars."""

        with patch.object(bucket_sync_module.pd, "isna", side_effect=TypeError):
            self.assertEqual(_embedding_to_float_vector(range(3)), [0.0, 1.0, 2.0])

    @patch.object(bucket_sync_module.pd, "isna", side_effect=TypeError)
    def test_coerce_ignores_na_errors(self, _: MagicMock) -> None:
        """_coerce_leaf_metadata: survives ``pd.isna`` TypeErrors safely."""
        self.assertEqual(_coerce_leaf_metadata("legacy"), "legacy")


class TestStableRowKey(unittest.TestCase):
    """This class tests _stable_row_key."""

    def test_replaces_na_with_none(self) -> None:
        """_stable_row_key: normalizes missing ids deterministically."""
        series = pd.Series(
            {"source_api_id": pd.NA, "chunk_index": 2, "source_row_index": 3},
        )
        key_first = _stable_row_key(series, ("source_api_id", "chunk_index"))
        key_second = _stable_row_key(series, ("source_api_id", "chunk_index"))
        self.assertEqual(key_first, key_second)

    def test_series_without_get_uses_mapping_index(self) -> None:
        """_stable_row_key: supports Mapping-like rows without ``get``."""

        class PlainRow:  # pylint: disable=too-few-public-methods
            """Mapping-like slice without SQLAlchemy row helpers."""

            def __init__(self, mapping: dict[str, Any]) -> None:
                self._mapping = mapping

            def __getitem__(self, key: str) -> Any:
                return self._mapping[key]

        row = PlainRow({"source_api_id": "story", "chunk_index": 0})
        digest = _stable_row_key(row, ("source_api_id", "chunk_index"))
        dup = _stable_row_key(row, ("source_api_id", "chunk_index"))
        self.assertEqual(digest, dup)


class TestBucketSyncTimers(unittest.TestCase):
    """This class tests Timer integration."""

    def test_timer_records_sections(self) -> None:
        """sync_profile_to_bucket: emits timer sections."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "source_api_id": "a",
                        "chunk_index": 0,
                        "source_row_index": 1,
                        "source_day": "d",
                        "embedding": [0.1, 0.2, 0.3],
                    },
                ]
            ).to_parquet(parquet_dir / "p.parquet", index=False)
            cfg = _cfg(
                input_dir=parquet_dir,
                dimension=3,
                create_bucket_if_missing=False,
                create_index_if_missing=False,
            )
            client = MagicMock()
            vectors = MagicMock()
            vectors.from_.return_value.index.return_value = MagicMock()
            client.storage.vectors.return_value = vectors

            timer = Timer()
            with _patch_sync_config(cfg):
                sync = SupabaseVectorBucketSync(
                    client_factory=_named_client_factory(client),
                    timer=timer,
                )
                sync.sync_profile_to_bucket(profile="p")

            labels = [record[0] for record in timer.records]
            self.assertIn("supabase_vector_sync.read_parquet", labels)
            self.assertIn("supabase_vector_sync.put_vectors", labels)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
