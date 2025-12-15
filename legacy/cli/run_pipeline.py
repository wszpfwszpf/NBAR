#!/usr/bin/env python3
"""
已归档的命令行入口脚本，仅供历史参考。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from nbar.io import load_npz, save_npz
from nbar.stages.s1 import run as run_s1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal unified pipeline entrypoint")
    parser.add_argument("--input", required=True, help="Path to input NPZ file")
    parser.add_argument("--outdir", required=True, help="Directory to store outputs")
    parser.add_argument(
        "--stage",
        choices=("raw", "s1"),
        default="raw",
        help="Pipeline stage to run (default: raw)",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Optional suffix appended to output filename before extension",
    )
    return parser.parse_args()


def build_output_path(outdir: Path, stage: str, input_path: Path, suffix: str) -> Path:
    stage_dir = outdir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ext = "".join(input_path.suffixes) or ".npz"
    final_name = f"{stem}{suffix}{ext}"
    return stage_dir / final_name


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_root = Path(args.outdir)
    suffix = args.suffix or ""

    start_time = time.perf_counter()
    events = load_npz(str(input_path))
    input_count = len(events.t)
    print(f"Loaded {input_count} events from {input_path}")

    if args.stage == "s1":
        events = run_s1(events)

    output_count = len(events.t)
    retention = (output_count / input_count) if input_count else 0.0

    output_path = build_output_path(output_root, args.stage, input_path, suffix)
    save_npz(str(output_path), events)

    elapsed = time.perf_counter() - start_time
    print(
        f"Stage: {args.stage}; Input events: {input_count}; Output events: {output_count}; "
        f"Retention: {retention:.6f}; Elapsed: {elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()
