# -*- coding: utf-8 -*-
import argparse
import os
from typing import Any, Dict, Optional

from nbar.io import load_npz, save_npz
from nbar.stages.s1 import run as s1_stage_run
from nbar.types import EventStream
from utils.config import S1 as S1_DEFAULT


def s1_run_npz(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
    store_bin_arrays: bool = True,
) -> str:
    """
    CLI wrapper for S1.

    Loads events from ``in_npz``, runs :mod:`nbar.stages.s1`, and writes the
    output to ``out_npz`` while preserving the original file naming rules.
    """

    events: EventStream = load_npz(in_npz)

    result = s1_stage_run(
        events,
        cfg=cfg or S1_DEFAULT,
        override=override,
        store_bin_arrays=store_bin_arrays,
    )

    in_dir = os.path.dirname(os.path.abspath(in_npz))
    in_stem = os.path.splitext(os.path.basename(in_npz))[0]

    if out_npz is None or out_npz == "":
        out_npz = os.path.join(in_dir, f"{in_stem}_s1.npz")
    else:
        if os.path.dirname(out_npz) == "":
            out_npz = os.path.join(in_dir, out_npz)
        else:
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    save_npz(out_npz, result)
    return out_npz


def main():
    parser = argparse.ArgumentParser(description="Run S1 stage on an NPZ event file")
    parser.add_argument("in_npz", nargs="?", default=r"mvsec_clip_2s.npz", help="input NPZ path")
    parser.add_argument("out_npz", nargs="?", default=None, help="output NPZ path (optional)")
    args = parser.parse_args()

    out = s1_run_npz(args.in_npz, out_npz=args.out_npz, override=None)
    print("Saved:", out)


if __name__ == "__main__":
    main()
