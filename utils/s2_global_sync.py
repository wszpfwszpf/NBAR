# -*- coding: utf-8 -*-
# s2_global_sync.py

from __future__ import annotations
import os
from typing import Optional, Dict, Any, Tuple

import numpy as np

from utils.config import get_meta_base, get_s2_cfg_dict
from utils.util import load_events_npz2, save_events_npz2


def s2_compute_w2(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    meta: Optional[Dict[str, Any]],
    extra: Optional[Dict[str, np.ndarray]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stage-2 placeholder: generate per-event weights w2.

    当前版本：
      - 不删除事件，w2 全为 1.0
      - meta2 写入基础统计，便于后续替换算法
    """
    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError("t/x/y/p shape mismatch")

    N = int(t.size)
    cfg_dict = dict(cfg) if cfg is not None else get_s2_cfg_dict()

    w2 = np.ones((N,), dtype=np.float32)

    stats = {
        "num_events": N,
        "w2_mean": float(w2.mean()) if N > 0 else 0.0,
        "w2_min": float(w2.min()) if N > 0 else 0.0,
        "w2_max": float(w2.max()) if N > 0 else 0.0,
    }

    meta2 = {
        "s2_cfg": cfg_dict,
        "s2_stats": stats,
    }

    return w2, meta2


def s2_run_npz(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """
    S2: placeholder global sync stage that keeps all events.

    - 加载：使用 load_events_npz2，完整保留 meta/extra
    - 计算：生成 w2 全 1，并记录基础统计
    - 保存：仅新增/覆盖 w2，其他 meta/extra 不变
    """
    t, x, y, p, meta_in, extra_in = load_events_npz2(in_npz)

    cfg_dict = dict(cfg) if cfg is not None else get_s2_cfg_dict()
    w2, meta2 = s2_compute_w2(t, x, y, p, meta_in, extra_in, cfg=cfg_dict)

    meta_out = dict(meta_in) if meta_in else {}
    for k, v in get_meta_base().items():
        meta_out.setdefault(k, v)
    meta_out.update(meta2)

    extra_out = dict(extra_in) if extra_in else {}
    extra_out["w2"] = w2

    in_dir = os.path.dirname(os.path.abspath(in_npz))
    in_stem = os.path.splitext(os.path.basename(in_npz))[0]

    if out_npz is None or out_npz == "":
        out_npz = os.path.join(in_dir, f"{in_stem}_s2.npz")
    else:
        if os.path.dirname(out_npz) == "":
            out_npz = os.path.join(in_dir, out_npz)
        else:
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    save_events_npz2(
        out_npz,
        t=t, x=x, y=y, p=p,
        meta=meta_out,
        extra=extra_out,
    )

    return out_npz


if __name__ == "__main__":
    # Quick manual run
    in_path = r"mvsec_clip_2s.npz"
    saved = s2_run_npz(in_path)
    print("Saved:", saved)
