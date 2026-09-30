from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import run_evaluation, run_generation, run_pipeline, run_selection

COMMANDS = {
    "generate": ("run dataset generation", run_generation),
    "evaluate": ("evaluate generated samples with evaluator models", run_evaluation),
    "select": ("filter evaluated samples into the final dataset", run_selection),
    "pipeline": ("run generation, evaluation, and selection", run_pipeline),
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    args.run(config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic repair datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, (help_text, run) in COMMANDS.items():
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--config", required=True, help="YAML config path")
        command_parser.set_defaults(run=run)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
