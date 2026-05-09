"""Sync embedding parquet files into Supabase Storage vector bucket indexes."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd
from storage3.exceptions import StorageApiError
from storage3.types import VectorData, VectorObject

from src.config.settings import Settings
from src.enums.vector_bucket_distance_metric import VectorBucketDistanceMetric
from src.utils.timer import Timer

MetadataScalar = Union[str, bool, float]

SupabaseClientFactory = Callable[[], Any]


def _default_supabase_client_factory() -> Any:
    """Build a Supabase client from runtime environment secrets."""
    from supabase import (  # pylint: disable=import-outside-toplevel
        create_client,
    )

    url, service_key = Settings.load_supabase_credentials()
    return create_client(url, service_key)


def _looks_like_duplicate_resource(exc: BaseException) -> bool:
    """Return True when the Storage API error suggests the bucket or index exists."""
    if isinstance(exc, StorageApiError):
        parts = [
            str(exc.message).lower(),
            str(exc.code).lower(),
            str(exc.status),
        ]
    else:
        parts = []
    parts.append(str(exc).lower())
    blob = " ".join(parts)
    needle_substrings = (
        "already exists",
        "alreadyexist",
        "duplicate",
        "conflict",
        "resource_already_exists",
        "bucket_already_exists",
        "index_already_exists",
        "409",
    )
    return any(part in blob for part in needle_substrings)


def _run_idempotent_create(operation: Callable[[], None]) -> None:
    """Run a create call; tolerate duplicate-resource Storage errors."""
    try:
        operation()
    except StorageApiError as exc:
        if _looks_like_duplicate_resource(exc):
            return
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Supabase stacks may translate HTTP conflicts into generic errors.
        if _looks_like_duplicate_resource(exc):
            return
        raise


def _coerce_leaf_metadata(raw: Any) -> Optional[MetadataScalar]:
    """Normalize a cell into vector metadata scalars Storage accepts."""
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except TypeError:
        pass
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (float, int)) and not isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (date, datetime, pd.Timestamp)):
        return str(raw.isoformat())
    return str(raw)


def _build_metadata(
    series: pd.Series,
    columns: Sequence[str],
) -> Optional[Dict[str, MetadataScalar]]:
    payload: Dict[str, MetadataScalar] = {}
    for column in columns:
        if column not in series.index:
            continue
        coerced = _coerce_leaf_metadata(series[column])
        if coerced is not None:
            payload[column] = coerced
    if not payload:
        return None
    return payload


def _embedding_to_float_vector(raw: Any) -> List[float]:
    """Coerce an embedding cell into a dense float vector."""
    if raw is None:
        raise ValueError("embedding cell is missing")

    if hasattr(raw, "tolist") and callable(raw.tolist):
        raw = raw.tolist()

    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]

    try:
        if pd.isna(raw):
            raise ValueError("embedding cell is NaN")
    except TypeError:
        pass

    iterator_any: Any = list(raw)
    return [float(x) for x in iterator_any]


def _stable_row_key(series: pd.Series, columns: Sequence[str]) -> str:
    """Stable, bounded-length key derived from declared id columns."""
    payload: Dict[str, Any] = {}
    for column in columns:
        payload[column] = series.get(column) if hasattr(series, "get") else series[column]
        if column in payload and pd.isna(payload[column]):
            payload[column] = None
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SupabaseVectorBucketSync:
    """Upload rows from embedding parquet profiles into Storage vector indexes."""

    def __init__(
        self,
        configuration_root: Optional[Path] = None,
        timer: Optional[Timer] = None,
        *,
        client_factory: SupabaseClientFactory = _default_supabase_client_factory,
    ) -> None:
        self._timer = timer or Timer()
        self._config = Settings.load_supabase_vector_sync_config(
            configuration_root=configuration_root,
        )
        self._client_factory = client_factory

    @property
    def timer(self) -> Timer:
        """Return shared timer."""
        return self._timer

    def _prepare_vector_upload_frame(self, frame: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Return rows carrying embeddings or ``None`` when there is nothing to upload."""
        if frame.empty:
            return None

        missing_key = sorted(
            column
            for column in self._config.key_columns
            if column not in frame.columns
        )
        if missing_key:
            raise ValueError(
                "Parquet missing key columns required for stable vector ids: "
                + ", ".join(missing_key),
            )

        embed_col = self._config.embedding_column
        if embed_col not in frame.columns:
            raise ValueError(f"Parquet missing embedding column {embed_col!r}")

        for meta_column in self._config.metadata_columns:
            if meta_column not in frame.columns:
                raise ValueError(f"Parquet missing metadata column {meta_column!r}")

        framed = frame.loc[frame[embed_col].notna()].copy()
        if framed.empty:
            return None
        return framed

    def _put_vector_batches(
        self,
        client: Any,
        payloads: List[VectorObject],
    ) -> int:
        bucket_scope = client.storage.vectors().from_(self._config.bucket_name)
        vector_index = bucket_scope.index(self._config.index_name)
        uploaded = 0
        stride = min(self._config.batch_size, 500)
        total_batches = (len(payloads) + stride - 1) // stride
        with self._timer.section("supabase_vector_sync.put_vectors"):
            for batch_number, offset in enumerate(range(0, len(payloads), stride), start=1):
                batch = payloads[offset : offset + stride]
                vector_index.put(batch)
                uploaded += len(batch)
                print(
                    "[supabase_vector_sync] "
                    f"uploaded batch {batch_number}/{total_batches} "
                    f"(batch_size={len(batch)}, total_uploaded={uploaded})"
                )
        return uploaded

    def _resolve_parquet_path(self, profile: str) -> Path:
        safe_profile = profile.replace("/", "_")
        return self._config.input_dir / f"{safe_profile}.parquet"

    def _ensure_bucket_and_index(self, client: Any) -> None:
        """Create bucket and vector index when configured and absent."""
        vectors_root = client.storage.vectors()

        bucket_name = self._config.bucket_name
        index_name = self._config.index_name

        if self._config.create_bucket_if_missing:
            _run_idempotent_create(
                lambda: vectors_root.create_bucket(bucket_name),
            )

        bucket_scope = vectors_root.from_(bucket_name)
        dimension = self._config.dimension

        if self._config.create_index_if_missing:
            metric = self._distance_metric_for_sdk(self._config.distance_metric)

            def create_index_operation() -> None:
                bucket_scope.create_index(
                    index_name=index_name,
                    dimension=dimension,
                    distance_metric=metric,
                    data_type="float32",
                    metadata=None,
                )

            _run_idempotent_create(create_index_operation)

    @staticmethod
    def _distance_metric_for_sdk(
        metric: VectorBucketDistanceMetric,
    ) -> str:
        """Map configured metrics to literals accepted by the Storage vectors client."""
        # storage3 type hints omit ``l2`` while product docs advertise it; strings are forwarded.
        return str(metric.value)

    def _dataframe_to_vectors(  # pylint: disable=too-many-locals
        self,
        dataframe: pd.DataFrame,
    ) -> List[VectorObject]:
        """Transform dataframe rows into vector upload payloads."""
        embedding_column = self._config.embedding_column
        vectors: List[VectorObject] = []
        dimension = self._config.dimension

        for _, row in dataframe.iterrows():
            embedding = _embedding_to_float_vector(row.get(embedding_column))
            if len(embedding) != dimension:
                raise ValueError(
                    f"Embedding length {len(embedding)} does not match configured dimension "
                    f"{dimension}"
                )
            key = _stable_row_key(row, self._config.key_columns)
            meta = _build_metadata(row, self._config.metadata_columns)
            vectors.append(
                VectorObject(
                    key=key,
                    data=VectorData(float32=embedding),
                    metadata=meta,
                )
            )
        return vectors

    def sync_profile_to_bucket(self, profile: str) -> Dict[str, Any]:
        """Read parquet for ``profile`` and upsert vectors to the configured index."""
        parquet_path = self._resolve_parquet_path(profile)
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Embedding parquet not found for profile '{profile}': {parquet_path}"
            )

        with self._timer.section("supabase_vector_sync.read_parquet"):
            frame = pd.read_parquet(parquet_path)

        framed_ready = self._prepare_vector_upload_frame(frame)
        if framed_ready is None:
            return {}

        with self._timer.section("supabase_vector_sync.provision"):
            client = self._client_factory()
            self._ensure_bucket_and_index(client)

        payloads = self._dataframe_to_vectors(framed_ready)
        uploaded = self._put_vector_batches(client, payloads)

        return {profile: str(parquet_path), "vectors_uploaded": uploaded}
