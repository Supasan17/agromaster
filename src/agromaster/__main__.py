"""Command-line entry point: ``python -m agromaster <stage> [<stage> ...]``.

Runs one or more pipeline stages from :mod:`agromaster.pipeline`, in the
order given, or every stage in order with ``all``.
"""
import argparse
import sys
from collections.abc import Sequence

from agromaster import config
from agromaster.pipeline import STAGES


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the pipeline command."""
    parser = argparse.ArgumentParser(
        prog="python -m agromaster",
        description="Run AgroMaster pipeline stages in order.",
    )
    parser.add_argument(
        "stages",
        nargs="+",
        choices=[*config.PIPELINE_STAGES, "all"],
        help="stage(s) to run; 'all' runs: " + ", ".join(config.PIPELINE_STAGES),
    )
    return parser


def resolve_stages(requested: Sequence[str]) -> list[str]:
    """Expand ``all`` into every stage and keep the requested order otherwise."""
    if "all" in requested:
        return list(config.PIPELINE_STAGES)
    return list(requested)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested stages; return a process exit code."""
    args = build_parser().parse_args(argv)
    for stage in resolve_stages(args.stages):
        print(f"[agromaster] {stage}")
        try:
            STAGES[stage]()
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"[agromaster] {stage} failed: {error}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
