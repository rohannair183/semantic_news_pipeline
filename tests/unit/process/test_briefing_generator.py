"""Unit tests for BriefingGenerator."""

import os
import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config.settings import (
    BriefingGeneratorConfig,
    BriefingTopicSpec,
    Settings,
)
from src.enums.briefing_date_filter import BriefingDateFilter
from src.process.briefing_generator import BriefingGenerator
from src.service_layer.vector_search_service import (
    VectorSearchHit,
    VectorSearchResponse,
    VectorSearchService,
)


class _BlockedResponse:  # pylint: disable=too-few-public-methods
    """Stub ``generate_content`` result whose ``text`` property raises ``ValueError``."""

    @property
    def text(self) -> str:
        """Raise ``ValueError`` like blocked Gemini responses."""
        raise ValueError("blocked")


def _config_fixture() -> BriefingGeneratorConfig:
    return BriefingGeneratorConfig(
        model="gemini-2.0-flash",
        topics=(
            BriefingTopicSpec(
                name="Topic A",
                vector_query="query one",
                date_filter=BriefingDateFilter.DAILY,
            ),
            BriefingTopicSpec(
                name="Topic B",
                vector_query="query two",
                date_filter=BriefingDateFilter.DAILY,
            ),
        ),
        vector_top_k=3,
    )


class TestBriefingGeneratorInit(unittest.TestCase):
    """This class tests BriefingGenerator.__init__."""

    def test_raises_when_api_key_missing(self) -> None:
        """__init__: raises ValueError when Gemini API key is unset."""
        cfg = _config_fixture()
        fake_vs = MagicMock(spec=VectorSearchService)
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}):
            with self.assertRaises(ValueError) as ctx:
                BriefingGenerator(vector_search=fake_vs)
            self.assertIn("GEMINI_API_KEY", str(ctx.exception))

    def test_accepts_google_api_key_when_gemini_empty(self) -> None:
        """__init__: falls back to GOOGLE_API_KEY when GEMINI_API_KEY is empty."""
        cfg = _config_fixture()
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.return_value = VectorSearchResponse(hits=())
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(
                    os.environ,
                    {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google-only"},
                ), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure") as gen_configure:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="Ok")
            gen_model.return_value = mock_model
            gen = BriefingGenerator(vector_search=fake_vs)
            gen.generate()
            gen_configure.assert_called_with(api_key="google-only")

    def test_prefers_gemini_key_over_google(self) -> None:
        """__init__: uses GEMINI_API_KEY when both are set."""
        cfg = _config_fixture()
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.return_value = VectorSearchResponse(hits=())
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(
                    os.environ,
                    {"GEMINI_API_KEY": "gemini-key", "GOOGLE_API_KEY": "google-key"},
                ), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure") as gen_configure:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="Out")
            gen_model.return_value = mock_model
            gen = BriefingGenerator(vector_search=fake_vs)
            gen.generate()
            gen_configure.assert_called_with(api_key="gemini-key")


class TestBriefingGeneratorGenerate(unittest.TestCase):
    """This class tests BriefingGenerator.generate."""

    def test_generate_calls_search_and_returns_gemini_text(self) -> None:
        """generate: searches per topic and returns model text."""
        cfg = _config_fixture()
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.side_effect = [
            VectorSearchResponse(
                hits=(
                    VectorSearchHit(
                        key="k1",
                        distance=0.1,
                        metadata={"chunk_text": "hello"},
                    ),
                    VectorSearchHit(
                        key="k2",
                        distance=None,
                        metadata=None,
                    ),
                ),
            ),
            VectorSearchResponse(hits=()),
        ]
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "x"}), \
                patch("src.process.briefing_generator.utc_now_iso_z", return_value="2026-05-10T00:00:00Z"), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure"):
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="  Final briefing.\n")
            gen_model.return_value = mock_model
            anchor = date(2026, 5, 10)
            gen = BriefingGenerator(vector_search=fake_vs, reference_date=anchor)
            result = gen.generate()

        self.assertEqual(result.briefing_text, "Final briefing.")
        self.assertEqual(fake_vs.search_by_text.call_count, 2)
        fake_vs.search_by_text.assert_any_call(
            "query one",
            top_k=3,
            date_from=anchor,
            date_to=anchor,
        )
        fake_vs.search_by_text.assert_any_call(
            "query two",
            top_k=3,
            date_from=anchor,
            date_to=anchor,
        )
        prompt = mock_model.generate_content.call_args[0][0]
        self.assertEqual(result.llm_prompt, prompt)
        doc = result.to_json_dict()
        self.assertEqual(doc["briefing_text"], "Final briefing.")
        self.assertEqual(doc["gemini_model"], "gemini-2.0-flash")
        self.assertEqual(len(doc["topics"]), 2)
        self.assertEqual(doc["topics"][0]["vector_query"], "query one")
        self.assertEqual(doc["topics"][0]["hits"][0]["key"], "k1")
        self.assertEqual(doc["topics"][1]["hits"], [])
        self.assertEqual(result.generated_at_iso, "2026-05-10T00:00:00Z")
        self.assertEqual(result.anchor_day_iso, "2026-05-10")
        self.assertIn("Topic A", prompt)
        self.assertIn("Topic B", prompt)
        self.assertIn("date_filter=daily", prompt)
        self.assertIn("2026-05-10 to 2026-05-10", prompt)
        self.assertIn("k1", prompt)
        self.assertIn("chunk_text", prompt)
        self.assertIn("(no hits)", prompt)

    def test_generate_passes_weekly_and_monthly_date_bounds(self) -> None:
        """generate: weekly uses 7-day window; monthly uses month-to-date."""
        cfg = BriefingGeneratorConfig(
            model="m",
            topics=(
                BriefingTopicSpec(
                    name="W",
                    vector_query="qw",
                    date_filter=BriefingDateFilter.WEEKLY,
                ),
                BriefingTopicSpec(
                    name="M",
                    vector_query="qm",
                    date_filter=BriefingDateFilter.MONTHLY,
                ),
            ),
            vector_top_k=5,
        )
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.return_value = VectorSearchResponse(hits=())
        anchor = date(2026, 5, 10)
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "x"}), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure"):
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="ok")
            gen_model.return_value = mock_model
            gen = BriefingGenerator(vector_search=fake_vs, reference_date=anchor)
            gen.generate()

        fake_vs.search_by_text.assert_any_call(
            "qw",
            top_k=5,
            date_from=date(2026, 5, 4),
            date_to=anchor,
        )
        fake_vs.search_by_text.assert_any_call(
            "qm",
            top_k=5,
            date_from=date(2026, 5, 1),
            date_to=anchor,
        )

    def test_generate_runtime_error_when_gemini_empty(self) -> None:
        """generate: raises RuntimeError when Gemini returns empty text."""
        cfg = BriefingGeneratorConfig(
            model="m",
            topics=(BriefingTopicSpec(name="T", vector_query="q"),),
            vector_top_k=2,
        )
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.return_value = VectorSearchResponse(hits=())
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "x"}), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure"):
            mock_model = MagicMock()
            mock_model.generate_content.return_value = MagicMock(text="   ")
            gen_model.return_value = mock_model
            gen = BriefingGenerator(vector_search=fake_vs)
            with self.assertRaises(RuntimeError) as ctx:
                gen.generate()
            self.assertIn("empty", str(ctx.exception))

    def test_generate_runtime_error_when_response_text_raises(self) -> None:
        """generate: raises RuntimeError when response.text raises ValueError."""
        cfg = BriefingGeneratorConfig(
            model="m",
            topics=(BriefingTopicSpec(name="T", vector_query="q"),),
            vector_top_k=2,
        )
        fake_vs = MagicMock(spec=VectorSearchService)
        fake_vs.search_by_text.return_value = VectorSearchResponse(hits=())

        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "x"}), \
                patch("google.generativeai.GenerativeModel") as gen_model, \
                patch("google.generativeai.configure"):
            mock_model = MagicMock()
            mock_model.generate_content.return_value = _BlockedResponse()
            gen_model.return_value = mock_model
            gen = BriefingGenerator(vector_search=fake_vs)
            with self.assertRaises(RuntimeError) as ctx:
                gen.generate()
            self.assertIn("usable text", str(ctx.exception))


class TestBriefingGeneratorConfigProperty(unittest.TestCase):
    """This class tests BriefingGenerator.config."""

    def test_config_property(self) -> None:
        """config: exposes parsed BriefingGeneratorConfig."""
        cfg = _config_fixture()
        fake_vs = MagicMock(spec=VectorSearchService)
        with patch.object(Settings, "load_briefing_generator_config", return_value=cfg), \
                patch.object(Settings, "load_repository_dotenv"), \
                patch.dict(os.environ, {"GEMINI_API_KEY": "x"}):
            gen = BriefingGenerator(vector_search=fake_vs)
        self.assertIs(gen.config, cfg)


class TestBriefingGeneratorDateBounds(unittest.TestCase):
    """This class tests BriefingGenerator._date_bounds_for_topic."""

    def test_date_bounds_raises_for_unsupported_filter(self) -> None:
        """_date_bounds_for_topic: raises ValueError when filter is not recognized."""
        gen = BriefingGenerator.__new__(BriefingGenerator)
        gen._reference_date = date(2026, 1, 1)
        bad_topic = SimpleNamespace(date_filter="bogus")
        with self.assertRaises(ValueError) as ctx:
            BriefingGenerator._date_bounds_for_topic(gen, bad_topic)  # type: ignore[arg-type]
        self.assertIn("unsupported", str(ctx.exception).lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
