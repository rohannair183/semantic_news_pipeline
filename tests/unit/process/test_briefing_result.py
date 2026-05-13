"""Unit tests for briefing result DTOs."""

import json
import unittest
from datetime import date

from src.enums.briefing_date_filter import BriefingDateFilter
from src.process.briefing_result import (
    BriefingGenerationResult,
    BriefingTopicContext,
    BriefingVectorHitRecord,
    utc_now_iso_z,
)
from src.service_layer.vector_search_service import VectorSearchHit, VectorSearchResponse


class TestBriefingVectorHitRecord(unittest.TestCase):
    """This class tests BriefingVectorHitRecord.from_vector_hit."""

    def test_from_vector_hit_copies_metadata(self) -> None:
        """from_vector_hit: preserves key, distance, and metadata dict."""
        hit = VectorSearchHit(key="a", distance=0.5, metadata={"x": 1})
        rec = BriefingVectorHitRecord.from_vector_hit(hit)
        self.assertEqual(rec.key, "a")
        self.assertEqual(rec.distance, 0.5)
        self.assertEqual(dict(rec.metadata), {"x": 1})

    def test_from_vector_hit_empty_metadata(self) -> None:
        """from_vector_hit: uses empty dict when metadata is None."""
        hit = VectorSearchHit(key="b", distance=None, metadata=None)
        rec = BriefingVectorHitRecord.from_vector_hit(hit)
        self.assertEqual(dict(rec.metadata), {})


class TestBriefingTopicContext(unittest.TestCase):
    """This class tests BriefingTopicContext.from_topic_search."""

    def test_from_topic_search(self) -> None:
        """from_topic_search: maps bounds and hits."""
        resp = VectorSearchResponse(
            hits=(VectorSearchHit(key="k", distance=1.0, metadata=None),),
        )
        ctx = BriefingTopicContext.from_topic_search(
            topic_name="T",
            vector_query="q",
            date_filter=BriefingDateFilter.WEEKLY,
            date_from=date(2026, 5, 1),
            date_to=date(2026, 5, 7),
            response=resp,
        )
        self.assertEqual(ctx.topic_name, "T")
        self.assertEqual(ctx.vector_query, "q")
        self.assertEqual(ctx.date_filter, "weekly")
        self.assertEqual(ctx.date_from_iso, "2026-05-01")
        self.assertEqual(ctx.date_to_iso, "2026-05-07")
        self.assertEqual(len(ctx.hits), 1)
        self.assertEqual(ctx.hits[0].key, "k")


class TestBriefingGenerationResult(unittest.TestCase):
    """This class tests BriefingGenerationResult.to_json_dict."""

    def test_to_json_dict_round_trip(self) -> None:
        """to_json_dict: produces json.dumps-serializable payload."""
        hit = BriefingVectorHitRecord(key="k", distance=None, metadata={"m": True})
        topic = BriefingTopicContext(
            topic_name="A",
            vector_query="vq",
            date_filter="daily",
            date_from_iso="2026-05-10",
            date_to_iso="2026-05-10",
            hits=(hit,),
        )
        result = BriefingGenerationResult(
            briefing_text="body",
            llm_prompt="prompt",
            gemini_model="gemini-pro",
            anchor_day_iso="2026-05-10",
            generated_at_iso="2026-05-10T12:00:00Z",
            topics=(topic,),
        )
        raw = result.to_json_dict()
        encoded = json.dumps(raw)
        self.assertIn("body", encoded)
        self.assertIn("topics", raw)
        self.assertEqual(raw["topics"][0]["hits"][0]["key"], "k")
        self.assertEqual(raw["topics"][0]["hits"][0]["metadata"]["m"], True)


class TestUtcNowIsoZ(unittest.TestCase):
    """This class tests utc_now_iso_z."""

    def test_ends_with_z(self) -> None:
        """utc_now_iso_z: returns ISO-like UTC string ending with Z."""
        token = utc_now_iso_z()
        self.assertTrue(token.endswith("Z"))
        self.assertIn("T", token)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
