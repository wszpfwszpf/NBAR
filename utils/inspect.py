# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

def inspect_w2_distribution(
    npz_path: str,
    *,
    key: str = "w2",
    bins: int = 50,
    clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    print_head: int = 8,
) -> Dict[str, Any]:
    """
    读取 npz 中的 w2，输出分布统计、分位数、直方图，以及一些一致性检查。

    返回 dict 便于你在 analyse.py 里复用/记录。
    """
    data = np.load(npz_path, allow_pickle=True)
    if key not in data.files:
        raise KeyError(f"'{key}' not found in {npz_path}. keys={data.files}")

    w2 = np.asarray(data[key], dtype=np.float64).reshape(-1)
    N = int(w2.size)
    if N == 0:
        return {"path": npz_path, "key": key, "N": 0}

    finite_mask = np.isfinite(w2)
    n_nan_inf = int(N - finite_mask.sum())
    w2_f = w2[finite_mask]
    if w2_f.size == 0:
        return {"path": npz_path, "key": key, "N": N, "all_nonfinite": True}

    # 可选裁剪（只是用于统计/直方图，原数组不改）
    if clip_range is not None:
        lo, hi = clip_range
        w2_c = np.clip(w2_f, lo, hi)
    else:
        w2_c = w2_f

    qs = [0.0, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999, 1.0]
    # quantiles = {f"q{q:g}": float(np.quantile(w2_c, q)) for q in qs}
    quantiles = {f"q{q:.2f}": float(np.quantile(w2_c, q)) for q in qs}

    mean = float(w2_c.mean())
    std = float(w2_c.std())
    vmin = float(w2_c.min())
    vmax = float(w2_c.max())

    # 饱和度：接近 0/1 的比例（默认按 1e-3 判定）
    eps = 1e-3
    near0 = float(np.mean(w2_c <= (0.0 + eps)))
    near1 = float(np.mean(w2_c >= (1.0 - eps)))

    # 简单直方图（用于快速看形状）
    hist_counts, hist_edges = np.histogram(w2_c, bins=bins, range=clip_range if clip_range else None)
    hist_counts = hist_counts.astype(np.int64)

    report: Dict[str, Any] = {
        "path": npz_path,
        "key": key,
        "N": N,
        "n_nonfinite": n_nan_inf,
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
        "near0_ratio": near0,
        "near1_ratio": near1,
        "quantiles": quantiles,
        "hist": {
            "bins": int(bins),
            "edges": hist_edges.astype(np.float64),
            "counts": hist_counts,
        },
    }

    # 控制台输出（简洁、够用）
    print("=" * 72)
    print(f"[W2 INSPECT] {npz_path}")
    print(f"key={key}  N={N}  nonfinite={n_nan_inf}")
    print(f"min/mean/max = {vmin:.6f} / {mean:.6f} / {vmax:.6f}   std={std:.6f}")
    print(f"near0(<=1e-3)={near0:.4f}   near1(>=1-1e-3)={near1:.4f}")
    q_show = ["q0.01", "q0.05", "q0.10", "q0.50", "q0.90", "q0.95", "q0.99"]
    print("quantiles:", "  ".join([f"{k}={quantiles[k]:.4f}" for k in q_show]))
    print("hist counts head:", hist_counts[:print_head].tolist(), " ...")
    print("=" * 72)
    return report
