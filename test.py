"""Regression entrypoint for PyCharm one-click run.

This script runs two regression checks:
1) Raw pass-through hash verification
2) S1 stage regression (including w1 hash)

Configure ``INPUT_NPZ`` and ``OUTDIR`` below, then run the script directly.
Outputs are written under ``OUTDIR`` (default: ``outputs_regression/``).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from nbar.io import load_npz, save_npz
from nbar.stages.s1 import run as run_s1

# =========================
# User configuration
# =========================
# Path to the input NPZ file containing (t, x, y, p).
INPUT_NPZ = "data/mvsec_clip_2s.npz"

# Root directory for regression outputs.
OUTDIR = "outputs_regression"


def _hash_arrays(arrays: Iterable[np.ndarray]) -> str:
    hasher = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(arr)
        hasher.update(arr.dtype.str.encode())
        hasher.update(str(arr.shape).encode())
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def compute_event_hash(events, *, include_w1: bool = False) -> str:
    arrays: Tuple[np.ndarray, ...] = (events.t, events.x, events.y, events.p)
    extra: Tuple[np.ndarray, ...] = ()
    if include_w1 and events.meta is not None and "w1" in events.meta:
        extra = (np.asarray(events.meta["w1"]),)
    return _hash_arrays(arrays + extra)


def build_output_path(outdir: Path, stage: str, input_path: Path) -> Path:
    stage_dir = outdir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    ext = "".join(input_path.suffixes) or ".npz"
    return stage_dir / f"{stem}{ext}"


def summarize(stage: str, count: int, hash_value: str, output_path: Path) -> None:
    print(f"[{stage}] events: {count:,}; hash: {hash_value}; saved to: {output_path}")


def run_raw(events, input_path: Path, outdir: Path) -> Dict[str, str]:
    output_path = build_output_path(outdir, "raw", input_path)
    save_npz(str(output_path), events)
    hash_value = compute_event_hash(events)
    summarize("raw", len(events.t), hash_value, output_path)
    return {"stage": "raw", "hash": hash_value, "output": str(output_path)}


def run_s1_stage(events, input_path: Path, outdir: Path) -> Dict[str, str]:
    events_s1 = run_s1(events)
    output_path = build_output_path(outdir, "s1", input_path)
    save_npz(str(output_path), events_s1)
    hash_value = compute_event_hash(events_s1, include_w1=True)
    summarize("s1", len(events_s1.t), hash_value, output_path)
    return {
        "stage": "s1",
        "hash": hash_value,
        "output": str(output_path),
        "retention": f"{len(events_s1.t) / len(events.t):.6f}" if len(events.t) else "n/a",
    }


def main() -> None:
    input_path = Path(INPUT_NPZ)
    outdir = Path(OUTDIR)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input NPZ not found: {input_path}. Update INPUT_NPZ at the top of test.py."
        )

    print("== Regression entrypoint ==")
    print(f"Input: {input_path}")
    print(f"Output directory: {outdir}")

    events = load_npz(str(input_path))
    print(f"Loaded events: {len(events.t):,}")

    results = [run_raw(events, input_path, outdir), run_s1_stage(events, input_path, outdir)]

    print("\nSummary: ")
    for res in results:
        suffix = f" (retention={res['retention']})" if res["stage"] == "s1" else ""
        print(f" - {res['stage']}: hash={res['hash']}; output={res['output']}{suffix}")


if __name__ == "__main__":
    main()
