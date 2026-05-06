"""Dispatch orchestrator task kinds to concrete pipeline modules."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.chunking.chunker import Chunker
from src.config.settings import OrchestratorTaskSpec
from src.embeddings.embedder import Embedder
from src.enums.orchestrator_task_kind import OrchestratorTaskKind
from src.enums.orchestrator_normalizer_day_token import OrchestratorNormalizerDayToken
from src.ingestion.article_ingestor import ArticleIngestor
from src.ingestion.article_normalizer import ArticleNormalizer
from src.ingestion.prechunk_processing import PreChunkPreprocessor
from src.utils.dates import coerce_day, utc_today_date
from src.utils.timer import Timer

OrchestratorRunner = Callable[[OrchestratorTaskSpec, Optional[Path], Timer], Any]


def resolve_normalizer_day_for_orchestrator(spec: OrchestratorTaskSpec) -> date:
    """Resolve the normalizer calendar day from declarative orchestrator params."""
    raw = spec.params.normalizer_day_raw
    if raw is None or raw == OrchestratorNormalizerDayToken.UTC_TODAY.value:
        return utc_today_date()
    return coerce_day(raw)


def _run_article_ingestor(
    _spec: OrchestratorTaskSpec,
    _configuration_root: Optional[Path],
    _timer: Timer,
) -> Dict[str, Any]:
    return ArticleIngestor().run()


def _run_article_normalizer(
    spec: OrchestratorTaskSpec,
    configuration_root: Optional[Path],
    _timer: Timer,
) -> Any:
    normalizer = ArticleNormalizer(configuration_root=configuration_root)
    day = resolve_normalizer_day_for_orchestrator(spec)
    return normalizer.normalize_day_to_parquet(day)


def _run_pre_chunk_preprocessor(
    _spec: OrchestratorTaskSpec,
    configuration_root: Optional[Path],
    _timer: Timer,
) -> Any:
    preprocessor = PreChunkPreprocessor(configuration_root=configuration_root)
    return preprocessor.preprocess_to_parquet()


def _run_chunking(
    spec: OrchestratorTaskSpec,
    configuration_root: Optional[Path],
    _timer: Timer,
) -> Any:
    chunker = Chunker(configuration_root=configuration_root)
    return chunker.chunk_to_parquet(profile=spec.params.profile)


def _run_embeddings(
    spec: OrchestratorTaskSpec,
    configuration_root: Optional[Path],
    timer: Timer,
) -> Any:
    embedder = Embedder(configuration_root=configuration_root, timer=timer)
    return embedder.embed_to_parquet(profile=spec.params.profile)


_DEFAULT_RUNNERS: Dict[OrchestratorTaskKind, OrchestratorRunner] = {
    OrchestratorTaskKind.ARTICLE_INGESTOR: _run_article_ingestor,
    OrchestratorTaskKind.ARTICLE_NORMALIZER: _run_article_normalizer,
    OrchestratorTaskKind.PRE_CHUNK_PREPROCESSOR: _run_pre_chunk_preprocessor,
    OrchestratorTaskKind.CHUNKING: _run_chunking,
    OrchestratorTaskKind.EMBEDDINGS: _run_embeddings,
}


def default_task_runner_map() -> Dict[OrchestratorTaskKind, OrchestratorRunner]:
    """Return a fresh copy of the built-in task runners."""
    return dict(_DEFAULT_RUNNERS)
