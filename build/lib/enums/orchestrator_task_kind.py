"""Orchestrator task kind selectors declared in YAML pipeline specs."""

from src.enums.base import BaseEnum


class OrchestratorTaskKind(BaseEnum):
    """Pipeline stages the YAML orchestrator can run."""

    ARTICLE_INGESTOR = "article_ingestor"
    ARTICLE_NORMALIZER = "article_normalizer"
    PRE_CHUNK_PREPROCESSOR = "pre_chunk_preprocessor"
    CHUNKING = "chunking"
    EMBEDDINGS = "embeddings"
    VECTOR_SYNC = "vector_sync"
