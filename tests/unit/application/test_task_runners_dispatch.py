"""This class tests orchestrator-backed module dispatch stubs."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from src.application.task_runners import (
    OrchestratorRunner,
    default_task_runner_map,
)
from src.config.settings import (
    OrchestratorTaskParams,
    OrchestratorTaskSpec,
)
from src.enums.orchestrator_task_kind import OrchestratorTaskKind
from src.utils.timer import Timer


def _minimal_spec(kind: OrchestratorTaskKind) -> OrchestratorTaskSpec:
    return OrchestratorTaskSpec(
        task_id=kind.value,
        kind=kind,
        enabled=True,
        skip_when=None,
        params=OrchestratorTaskParams(),
    )


class TestOrchestratorTaskDispatcher(unittest.TestCase):
    """This class tests default orchestrator-backed module dispatch."""

    def test_each_runner_invokes_backing_module(self):  # pylint: disable=too-many-locals
        """default_task_runner_map: each handler reaches its backing class."""
        mappings = {
            OrchestratorTaskKind.ARTICLE_INGESTOR: "src.application.task_runners.ArticleIngestor",
            OrchestratorTaskKind.ARTICLE_NORMALIZER: (
                "src.application.task_runners.ArticleNormalizer"
            ),
            OrchestratorTaskKind.PRE_CHUNK_PREPROCESSOR: (
                "src.application.task_runners.PreChunkPreprocessor"
            ),
            OrchestratorTaskKind.CHUNKING: "src.application.task_runners.Chunker",
            OrchestratorTaskKind.EMBEDDINGS: "src.application.task_runners.Embedder",
            OrchestratorTaskKind.VECTOR_SYNC: (
                "src.application.task_runners.VectorBucketSync"
            ),
            OrchestratorTaskKind.BRIEFING_PERSISTENCE: (
                "src.application.task_runners.BriefingPersistenceRunner"
            ),
        }
        timer = Timer()
        runners = default_task_runner_map()
        fake_root = Path("/tmp/orch_configuration_root_fake")

        normalizer_spec = OrchestratorTaskSpec(
            task_id="norm",
            kind=OrchestratorTaskKind.ARTICLE_NORMALIZER,
            enabled=True,
            skip_when=None,
            params=OrchestratorTaskParams(normalizer_day_raw="2020-01-01"),
        )

        chunk_spec = OrchestratorTaskSpec(
            task_id="chunk",
            kind=OrchestratorTaskKind.CHUNKING,
            enabled=True,
            skip_when=None,
            params=OrchestratorTaskParams(profile="x"),
        )
        embed_spec = OrchestratorTaskSpec(
            task_id="emb",
            kind=OrchestratorTaskKind.EMBEDDINGS,
            enabled=True,
            skip_when=None,
            params=OrchestratorTaskParams(profile="y"),
        )
        sync_spec = OrchestratorTaskSpec(
            task_id="syn",
            kind=OrchestratorTaskKind.VECTOR_SYNC,
            enabled=True,
            skip_when=None,
            params=OrchestratorTaskParams(profile="z"),
        )
        briefing_spec = _minimal_spec(OrchestratorTaskKind.BRIEFING_PERSISTENCE)

        spec_by_kind = {
            OrchestratorTaskKind.ARTICLE_INGESTOR: _minimal_spec(
                OrchestratorTaskKind.ARTICLE_INGESTOR,
            ),
            OrchestratorTaskKind.ARTICLE_NORMALIZER: normalizer_spec,
            OrchestratorTaskKind.PRE_CHUNK_PREPROCESSOR: _minimal_spec(
                OrchestratorTaskKind.PRE_CHUNK_PREPROCESSOR,
            ),
            OrchestratorTaskKind.CHUNKING: chunk_spec,
            OrchestratorTaskKind.EMBEDDINGS: embed_spec,
            OrchestratorTaskKind.VECTOR_SYNC: sync_spec,
            OrchestratorTaskKind.BRIEFING_PERSISTENCE: briefing_spec,
        }

        for kind, target in mappings.items():
            spec = spec_by_kind[kind]
            runner: OrchestratorRunner = runners[kind]
            mock_instance = MagicMock()
            with mock.patch(target) as ctor:
                ctor.return_value = mock_instance
                runner(spec, fake_root, timer)
                if kind == OrchestratorTaskKind.ARTICLE_INGESTOR:
                    ctor.assert_called_once_with()
                    mock_instance.run.assert_called_once_with()
                elif kind == OrchestratorTaskKind.ARTICLE_NORMALIZER:
                    ctor.assert_called_once_with(configuration_root=fake_root)
                    mock_instance.normalize_day_to_parquet.assert_called_once()
                elif kind == OrchestratorTaskKind.PRE_CHUNK_PREPROCESSOR:
                    ctor.assert_called_once_with(configuration_root=fake_root)
                    mock_instance.preprocess_to_parquet.assert_called_once_with()
                elif kind == OrchestratorTaskKind.CHUNKING:
                    ctor.assert_called_once_with(configuration_root=fake_root)
                    mock_instance.chunk_to_parquet.assert_called_once_with(profile="x")
                elif kind == OrchestratorTaskKind.EMBEDDINGS:
                    ctor.assert_called_once_with(
                        configuration_root=fake_root,
                        timer=timer,
                    )
                    mock_instance.embed_to_parquet.assert_called_once_with(profile="y")
                elif kind == OrchestratorTaskKind.VECTOR_SYNC:
                    ctor.assert_called_once_with(
                        configuration_root=fake_root,
                        timer=timer,
                    )
                    mock_instance.sync_profile_to_bucket.assert_called_once_with(profile="z")
                else:
                    ctor.assert_called_once_with(configuration_root=fake_root)
                    mock_instance.run.assert_called_once_with()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
