# -*- coding: utf-8 -*-
"""
PyCharm-friendly entry for Stage-2 placeholder.
"""
import os
import numpy as np

from utils.config import get_meta_base
from utils.s2_global_sync import s2_compute_w2
from utils.util import load_events_npz2, save_events_npz2

# =========================
# CONFIG
# =========================
_DEFAULT_WITH_S1 = "mvsec_clip_2s_s1.npz"
_DEFAULT_RAW = "mvsec_clip_2s.npz"

INPUT_NPZ = _DEFAULT_WITH_S1 if os.path.exists(_DEFAULT_WITH_S1) else _DEFAULT_RAW
OUT_DIR = "outputs/s2"
SUFFIX = "_s2"


def _resolve_out_path(in_path: str) -> str:
    stem = os.path.splitext(os.path.basename(in_path))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f"{stem}{SUFFIX}.npz")


def main():
    t, x, y, p, meta, extra = load_events_npz2(INPUT_NPZ)

    w2, meta2 = s2_compute_w2(t, x, y, p, meta, extra, cfg=None)

    extra_out = dict(extra) if extra else {}
    extra_out["w2"] = np.asarray(w2, dtype=np.float32)
    if extra_out["w2"].shape != t.shape:
        raise ValueError(f"w2 shape mismatch: {extra_out['w2'].shape} vs {t.shape}")

    meta_out = dict(get_meta_base())
    if meta:
        meta_out.update(meta)
    meta_out.update(meta2)

    out_path = _resolve_out_path(INPUT_NPZ)
    save_events_npz2(out_path, t=t, x=x, y=y, p=p, meta=meta_out, extra=extra_out)

    N = int(t.size)
    w2_min = float(extra_out["w2"].min()) if N > 0 else 0.0
    w2_mean = float(extra_out["w2"].mean()) if N > 0 else 0.0
    w2_max = float(extra_out["w2"].max()) if N > 0 else 0.0

    print(f"Input N: {N}; Output N: {extra_out['w2'].shape[0]}")
    print(f"w2 stats -> min: {w2_min:.6f}, mean: {w2_mean:.6f}, max: {w2_max:.6f}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
