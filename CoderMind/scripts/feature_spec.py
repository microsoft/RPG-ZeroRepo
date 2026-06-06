#!/usr/bin/env python3
"""CLI wrapper for the ``feature_spec`` stage.

This is the entry point invoked by ``cmind script feature_spec.py`` and
by the ``feature_construct`` orchestrator.  All real work happens in
:mod:`feature.spec`; this module only translates CLI arguments and
formats human-readable output.

Examples
--------

Generate from ``docs/*.md`` (auto-detected)::

    cmind script feature_spec.py

Generate from inline requirement text::

    cmind script feature_spec.py --input-text "Build a CLI for managing Docker containers"

Probe current state without modifying anything::

    cmind script feature_spec.py --check-only --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from common.paths import FEATURE_SPEC_FILE

from feature.spec import (
    DEFAULT_DOCS_DIR,
    NoInputAvailable,
    generate_feature_spec,
    probe,
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="feature_spec.py",
        description=(
            "Generate feature_spec.json directly from requirements "
            "(inline text or docs/*.md), via a strict Pydantic-validated "
            "LLM call."
        ),
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing requirement Markdown files "
            f"(default: {DEFAULT_DOCS_DIR})."
        ),
    )
    parser.add_argument(
        "--input-text",
        default=None,
        metavar="TEXT",
        help=(
            "Inline requirement description.  Overrides --docs."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURE_SPEC_FILE,
        metavar="PATH",
        help=f"Output path (default: {FEATURE_SPEC_FILE}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite even if a valid feature_spec.json already exists.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Probe the output state and exit without invoking the LLM.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --check-only, emit JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Disable trajectory recording.",
    )
    return parser.parse_args(argv)


def _emit_probe(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    status = payload["status"]
    marker = {
        "valid": "[OK]",
        "missing": "[--]",
        "invalid": "[!!]",
    }.get(status, "[??]")
    print(f"{marker} feature_spec: {payload['message']}")


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    if args.check_only:
        _emit_probe(probe(args.output), as_json=args.json)
        return 0

    try:
        result = generate_feature_spec(
            input_text=args.input_text,
            docs_dir=args.docs,
            output_path=args.output,
            force=args.force,
            enable_trajectory=not args.no_trajectory,
        )
    except NoInputAvailable as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    if result.skipped:
        print(f"[SKIP] feature_spec: {result.reason}")
        print(f"       Output: {result.output_path}")
        return 0

    assert result.spec is not None  # generated path always sets spec
    spec = result.spec
    print(f"[OK]   feature_spec written to {result.output_path}")
    print(f"       Repository: {spec.repository_name}")
    print(f"       Top-level features: {len(spec.functional_requirements)}")
    print(f"       Background items: {len(spec.background_and_overview)}")
    print(f"       NFR items: {len(spec.non_functional_requirements)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
