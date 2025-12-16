# -*- coding: utf-8 -*-
"""
PyCharm-friendly entry point for Stage-1 (S1) processing.

Run directly to load an input NPZ, execute S1, and write outputs to the
standard ``outputs/`` directory tree while preserving the original
algorithms.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

from nbar.io import load_npz, save_npz
from nbar.stages.s1 import run as s1_stage_run
from nbar.types import EventStream
from utils.config import S1 as S1_DEFAULT

# =========================================================
# User configuration (PyCharm-friendly)
# =========================================================
STAGE = "s1"               # reserved for future extension (e.g., "s2")
INPUT_NPZ = "data/mvsec_clip_2s.npz"
OUT_ROOT = Path("outputs")
COPY_RAW = True             # save a normalized raw copy to outputs/raw/
STORE_BIN_ARRAYS = True     # keep Ck/Sk/zk arrays when small enough
OVERRIDE: Optional[Dict[str, Any]] = None  # e.g., {"lam": 0.4}


def _default_output_paths(input_npz: Path, out_root: Path) -> tuple[Path, Path]:
    stem = input_npz.stem
    raw_dir = out_root / "raw"
    s1_dir = out_root / STAGE
    return raw_dir / f"{stem}_raw.npz", s1_dir / f"{stem}_{STAGE}.npz"


def s1_run_npz(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
    store_bin_arrays: bool = True,
    copy_raw: bool = True,
    out_root: Path | str = OUT_ROOT,
) -> str:
    """
    Load ``in_npz``, run Stage-1, and save the result.
    """

    in_path = Path(in_npz)
    out_root = Path(out_root)
    raw_default, s1_default = _default_output_paths(in_path, out_root)

    events: EventStream = load_npz(str(in_path))

    # keep a normalized raw copy for convenience
    if copy_raw:
        raw_default.parent.mkdir(parents=True, exist_ok=True)
        save_npz(str(raw_default), events)

    result = s1_stage_run(
        events,
        cfg=cfg or S1_DEFAULT,
        override=override,
        store_bin_arrays=store_bin_arrays,
    )

    if out_npz is None or out_npz == "":
        out_npz_path = s1_default
    else:
        out_npz_path = Path(out_npz)
        if out_npz_path.parent == Path(""):
            out_npz_path = s1_default.parent / out_npz_path

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(str(out_npz_path), result)
    return str(out_npz_path)


def main():
    out = s1_run_npz(
        INPUT_NPZ,
        out_npz=None,
        override=OVERRIDE,
        store_bin_arrays=STORE_BIN_ARRAYS,
        copy_raw=COPY_RAW,
        out_root=OUT_ROOT,
    )
    print("Saved:", out)


if __name__ == "__main__":
    main()
