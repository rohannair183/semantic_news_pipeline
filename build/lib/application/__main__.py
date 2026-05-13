"""CLI entry for YAML-driven orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.application.orchestrator import Orchestrator
from src.config.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORCHESTRATOR_PATH = REPO_ROOT / "configuration" / "orchestration" / "orchestrator.yaml"
TEST_ORCHESTRATOR_PATH = REPO_ROOT / "configuration" / "orchestration" / "orchestrator_ci.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build an argument namespace for orchestrator CLI execution."""
    parser = argparse.ArgumentParser(description="Run YAML-defined pipeline tasks.")
    parser.add_argument(
        "--mode",
        choices=("production", "test"),
        default="production",
        help=(
            "Preset mode. production uses orchestrator.yaml (default), "
            "test uses orchestrator_ci.yaml."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to orchestrator YAML. "
            "Overrides --mode preset selection."
        ),
    )
    parser.add_argument(
        "--configuration-root",
        type=Path,
        default=None,
        help="Optional repository configuration root for downstream YAML loaders.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load orchestrator YAML, run tasks, and exit non-zero on failures."""
    args = parse_args(argv)
    if args.config is not None:
        config_path = args.config.resolve()
    elif args.mode == "test":
        config_path = TEST_ORCHESTRATOR_PATH
    else:
        config_path = DEFAULT_ORCHESTRATOR_PATH
    config = Settings.load_orchestrator_config_from_path(config_path)
    root_path = args.configuration_root.resolve() if args.configuration_root else None
    orchestrator = Orchestrator(config, configuration_root=root_path)
    summary = orchestrator.run()
    if summary.has_failure:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
