"""This module exercises ``python -m src.application`` argparse wiring."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.application import __main__ as orchestrator_cli
from src.config.settings import OrchestratorConfig, Settings


class TestOrchestratorCliParseArgs(unittest.TestCase):
    """This class tests parse_args."""

    def test_parse_args_defaults(self):
        """parse_args: provides default bundled orchestrator path."""
        resolved = orchestrator_cli.parse_args([])
        expected = orchestrator_cli.DEFAULT_ORCHESTRATOR_PATH
        self.assertEqual(resolved.config, expected)


class TestOrchestratorCliMain(unittest.TestCase):
    """This class tests main."""

    def test_main_returns_one_on_failure(self):
        """main: propagates failures from orchestrator summary."""
        config = OrchestratorConfig(fail_fast=True, tasks=tuple())

        fake_summary = mock.Mock()
        fake_summary.has_failure = True
        fake_config_path = Path(tempfile.gettempdir()) / "orch-test-config.yaml"

        with mock.patch.object(
            Settings,
            "load_orchestrator_config_from_path",
            return_value=config,
        ) as mocked_load:
            with mock.patch.object(orchestrator_cli, "Orchestrator") as orchestrator_cls:
                orchestrator_cls.return_value.run.return_value = fake_summary
                exit_code = orchestrator_cli.main(
                    ["--config", str(fake_config_path)],
                )
        mocked_load.assert_called_once()
        resolved_arg = mocked_load.call_args.args[0]
        self.assertEqual(resolved_arg, fake_config_path.resolve())
        self.assertEqual(exit_code, 1)

    def test_main_returns_zero_when_successful(self):
        """main: exits zero when summaries report no failures."""
        config = OrchestratorConfig(fail_fast=True, tasks=tuple())

        fake_summary = mock.Mock()
        fake_summary.has_failure = False
        fake_config_path = Path(tempfile.mktemp(suffix=".yaml"))

        with mock.patch.object(
            Settings,
            "load_orchestrator_config_from_path",
            return_value=config,
        ):
            with mock.patch.object(orchestrator_cli, "Orchestrator") as orchestrator_cls:
                orchestrator_cls.return_value.run.return_value = fake_summary
                exit_code = orchestrator_cli.main(
                    ["--config", str(fake_config_path)],
                )
        self.assertEqual(exit_code, 0)

    def test_main_forwards_configuration_root(self):
        """main: passes resolved optional configuration roots."""
        config = OrchestratorConfig(fail_fast=True, tasks=tuple())
        fake_summary = mock.Mock()
        fake_summary.has_failure = False
        fake_config_path = Path(tempfile.mktemp(suffix=".yaml"))
        root = Path(tempfile.mkdtemp())
        resolved_root = root.resolve()

        with mock.patch.object(
            Settings,
            "load_orchestrator_config_from_path",
            return_value=config,
        ):
            with mock.patch.object(orchestrator_cli, "Orchestrator") as orchestrator_cls:
                orchestrator_cls.return_value.run.return_value = fake_summary
                orchestrator_cli.main(
                    ["--config", str(fake_config_path), "--configuration-root", str(root)],
                )
                _, kwargs = orchestrator_cls.call_args
                self.assertEqual(kwargs["configuration_root"], resolved_root)

    def test_module_entrypoint_handles_exit_code_zero(self):
        """``python -m src.application``: honors ``__main__`` guard wiring."""

        repo_root = Path(__file__).resolve().parents[3]
        yaml_text = (
            "fail_fast: true\n"
            "tasks:\n"
            "  - kind: chunking\n"
            "    enabled: false\n"
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "cli_orchestrator.yaml"
            config_path.write_text(yaml_text, encoding="utf-8")

            outcome = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.application",
                    "--config",
                    str(config_path),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(outcome.returncode, 0, outcome.stderr + outcome.stdout)
            self.assertIn("chunking", outcome.stdout.lower())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
