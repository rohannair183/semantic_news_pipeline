# pyright: reportPrivateUsage=false
"""Unit tests for briefing_persistence runner behavior."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.process.briefing_persistence import (
    BriefingPersistenceRunner,
    evaluate_briefing_persistence_skip,
    briefing_row_from_result,
)
from src.process.briefing_result import BriefingGenerationResult, BriefingTopicContext


class TestBriefingRowFromResult(unittest.TestCase):
    """This class tests briefing_row_from_result."""

    def test_maps_columns_and_json_payloads(self) -> None:
        """briefing_row_from_result: builds row dict for PostgREST."""
        result = BriefingGenerationResult(
            briefing_text="body",
            llm_prompt="prompt",
            gemini_model="gemini-x",
            anchor_day_iso="2026-05-12",
            generated_at_iso="2026-05-12T15:30:00Z",
            topics=(),
        )
        row = briefing_row_from_result(result)
        self.assertEqual(row["briefing_text"], "body")
        self.assertEqual(row["llm_prompt"], "prompt")
        self.assertEqual(row["gemini_model"], "gemini-x")
        self.assertEqual(row["anchor_day_iso"], "2026-05-12")
        self.assertEqual(row["generated_at"], "2026-05-12T15:30:00Z")
        self.assertEqual(row["topics"], [])
        self.assertEqual(row["record"]["briefing_text"], "body")


class TestBriefingPersistenceRunner(unittest.TestCase):
    """This class tests BriefingPersistenceRunner.run."""

    def test_persistence_config_property_returns_loaded_config(self) -> None:
        """BriefingPersistenceRunner.persistence_config: returns BriefingPersistenceConfig."""
        yaml_text = """\
briefing_persistence:
  table_name: t99
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")
            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                runner = BriefingPersistenceRunner(
                    configuration_root=root,
                    briefing_generator=MagicMock(),
                    supabase_client_factory=MagicMock(),
                )
        self.assertEqual(runner.persistence_config.table_name, "t99")

    def test_run_generates_inserts_skips_ddl_when_disabled(self) -> None:
        """run: when ensure_table false, skips DDL and inserts via client factory."""
        yaml_text = """\
briefing_persistence:
  table_name: t1
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")

            mock_result = BriefingGenerationResult(
                briefing_text="b",
                llm_prompt="p",
                gemini_model="m",
                anchor_day_iso="2026-05-12",
                generated_at_iso="2026-05-12T12:00:00Z",
                topics=(),
            )
            mock_gen = MagicMock()
            mock_gen.generate.return_value = mock_result

            mock_client = MagicMock()
            chain = mock_client.schema.return_value.table.return_value.insert
            chain.return_value.execute.return_value = MagicMock(data=[{"id": "x"}])

            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                runner = BriefingPersistenceRunner(
                    configuration_root=root,
                    briefing_generator=mock_gen,
                    supabase_client_factory=lambda: mock_client,
                )

            with patch(
                "src.process.briefing_persistence.ensure_briefing_persistence_table",
            ) as mock_ensure:
                out = runner.run()

            self.assertIs(out, mock_result)
            mock_ensure.assert_not_called()
            mock_gen.generate.assert_called_once()
            mock_client.schema.assert_called_once_with("public")
            mock_client.schema.return_value.table.assert_called_once_with("t1")

    def test_run_calls_ddl_when_ensure_table_true(self) -> None:
        """run: when ensure_table true, calls ensure_briefing_persistence_table with conninfo."""
        yaml_text = """\
briefing_persistence:
  table_name: t2
  schema_name: public
  ensure_table: true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")

            mock_result = BriefingGenerationResult(
                briefing_text="b",
                llm_prompt="p",
                gemini_model="m",
                anchor_day_iso="2026-05-12",
                generated_at_iso="2026-05-12T12:00:00Z",
                topics=(),
            )
            mock_gen = MagicMock()
            mock_gen.generate.return_value = mock_result
            mock_client = MagicMock()
            table_chain = mock_client.schema.return_value.table.return_value
            exec_chain = table_chain.insert.return_value.execute
            exec_chain.return_value = MagicMock()

            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                runner = BriefingPersistenceRunner(
                    configuration_root=root,
                    briefing_generator=mock_gen,
                    supabase_client_factory=lambda: mock_client,
                    postgres_conninfo="postgresql://local/db",
                )

            with patch(
                "src.process.briefing_persistence.ensure_briefing_persistence_table",
            ) as mock_ensure:
                runner.run()

            mock_ensure.assert_called_once_with(
                "public",
                "t2",
                postgres_conninfo="postgresql://local/db",
            )

    def test_run_inserts_one_row_per_topic(self) -> None:  # pylint: disable=too-many-locals
        """run: when result has topics, inserts one row per topic with topic columns."""
        yaml_text = """\
briefing_persistence:
  table_name: t_topics
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")

            topic = BriefingTopicContext(
                topic_name="T1",
                vector_query="q",
                date_filter="daily",
                date_from_iso="2026-05-10",
                date_to_iso="2026-05-12",
                hits=(),
            )
            mock_result = BriefingGenerationResult(
                briefing_text="b",
                llm_prompt="p",
                gemini_model="m",
                anchor_day_iso="2026-05-12",
                generated_at_iso="2026-05-12T12:00:00Z",
                topics=(topic,),
            )

            mock_gen = MagicMock()
            mock_gen.generate.return_value = mock_result
            mock_client = MagicMock()
            insert_chain = mock_client.schema.return_value.table.return_value.insert
            insert_chain.return_value.execute.return_value = MagicMock()

            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                runner = BriefingPersistenceRunner(
                    configuration_root=root,
                    briefing_generator=mock_gen,
                    supabase_client_factory=lambda: mock_client,
                )

            out = runner.run()
            self.assertIs(out, mock_result)
            # insert called once for single topic
            self.assertEqual(insert_chain.call_count, 1)
            # verify the inserted payload included topic_name and topic_date_filter
            insert_args = insert_chain.call_args_list[0][0][0]
            self.assertIsInstance(insert_args, list)
            inserted_row = insert_args[0]
            self.assertEqual(inserted_row.get("topic_name"), "T1")
            self.assertEqual(inserted_row.get("topic_date_filter"), "daily")


class TestBriefingPersistenceSkipEvaluation(unittest.TestCase):
    """This class tests evaluate_briefing_persistence_skip."""

    def test_returns_skip_when_latest_run_is_today(self) -> None:
        """evaluate_briefing_persistence_skip: skips when latest row is fresh."""
        yaml_text = """\
briefing_persistence:
  table_name: t_daily
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")

            mock_client = MagicMock()
            query_chain = mock_client.schema.return_value.table.return_value
            select_chain = query_chain.select.return_value
            order_chain = select_chain.order.return_value
            limit_chain = order_chain.limit.return_value
            limit_chain.execute.return_value = MagicMock(
                data=[{"generated_at": "2026-05-13T08:30:00Z"}],
            )

            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                skip, reason = evaluate_briefing_persistence_skip(
                    configuration_root=root,
                    client_factory=lambda: mock_client,
                    current_date=date(2026, 5, 13),
                )

        self.assertTrue(skip)
        self.assertIn("2026-05-13", reason)

    def test_returns_run_when_latest_run_is_stale(self) -> None:
        """evaluate_briefing_persistence_skip: runs when latest row is older than today."""
        yaml_text = """\
briefing_persistence:
  table_name: t_stale
  ensure_table: false
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            proc = root / "process"
            proc.mkdir(parents=True)
            (proc / "briefing_persistence.yaml").write_text(yaml_text, encoding="utf-8")

            mock_client = MagicMock()
            query_chain = mock_client.schema.return_value.table.return_value
            select_chain = query_chain.select.return_value
            order_chain = select_chain.order.return_value
            limit_chain = order_chain.limit.return_value
            limit_chain.execute.return_value = MagicMock(
                data=[{"generated_at": "2026-05-12T23:59:00Z"}],
            )

            with patch("src.process.briefing_persistence.Settings.load_repository_dotenv"):
                skip, reason = evaluate_briefing_persistence_skip(
                    configuration_root=root,
                    client_factory=lambda: mock_client,
                    current_date=date(2026, 5, 13),
                )

        self.assertFalse(skip)
        self.assertEqual(reason, "")
