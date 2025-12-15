# -*- coding: utf-8 -*-
# s1_global_sync.py

from __future__ import annotations
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from utils.util import load_events_npz2, save_events_npz2

# load/save 的行为：保留额外字段 + 完整meta，且避免覆盖 t/x/y/p【:contentReference[oaicite:2]{index=2}】【:contentReference[oaicite:3]{index=3}】

def _sigmoid(x: np.ndarray) -> np.ndarray:
    # 数值稳健：避免exp溢出
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))

def s1_compute_w1(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    W: int = 346,
    H: int = 260,
    dt_s: float = 0.0005,          # 0.5ms
    Gx: int = 16,
    Gy: int = 12,
    rho: float = 0.02,             # EMA更新率（可后续调）
    alpha: float = 8.0,            # sigmoid斜率（可后续调）
    z_th: float = 1.0,             # z阈值（可后续调）
    lam: float = 0.8,              # 抑制强度：w1 = 1 - lam*S
    freq_prior_gk: Optional[np.ndarray] = None,  # 可选：bin级先验(0~1)
    normalize_t0: bool = True,     # 安全：强制t从0开始
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    返回：
      w1: (N,) float32
      aux: 一些bin级统计与配置，用于写入meta或debug
    """
    assert t.ndim == 1 and t.shape == x.shape == y.shape
    N = t.size
    if N == 0:
        return np.zeros((0,), dtype=np.float32), {"num_bins": 0}

    t0 = float(t[0])
    if normalize_t0:
        tt = t - t0
    else:
        tt = t.copy()

    # bin index
    bin_idx = np.floor(tt / dt_s).astype(np.int64)
    num_bins = int(bin_idx.max()) + 1

    # 将像素映射到(Gx,Gy)网格cell
    bx = int(np.ceil(W / Gx))
    by = int(np.ceil(H / Gy))
    u = np.minimum(x // bx, Gx - 1)
    v = np.minimum(y // by, Gy - 1)
    cell = v * Gx + u                      # [0, Gx*Gy)
    G = Gx * Gy

    # 统计每个bin的“活跃cell数”：对 (bin, cell) 去重
    key = bin_idx * G + cell
    uniq_key = np.unique(key)
    uniq_bin = (uniq_key // G).astype(np.int64)
    active_cells = np.bincount(uniq_bin, minlength=num_bins).astype(np.float32)
    Ck = active_cells / float(G)           # 覆盖率 in [0,1]

    # 在线统计：EMA的均值方差
    mu = np.zeros((num_bins,), dtype=np.float32)
    var = np.zeros((num_bins,), dtype=np.float32)

    mu_k = float(Ck[0])
    var_k = 1e-6
    mu[0] = mu_k
    var[0] = var_k

    for k in range(1, num_bins):
        ck = float(Ck[k])
        mu_k = (1.0 - rho) * mu_k + rho * ck
        # EMA方差（简单稳健版）
        diff = ck - mu_k
        var_k = (1.0 - rho) * var_k + rho * (diff * diff)
        mu[k] = mu_k
        var[k] = var_k

    sigma = np.sqrt(np.maximum(var, 1e-8))
    zk = (Ck - mu) / sigma

    Sk = _sigmoid(alpha * (zk - z_th)).astype(np.float32)  # [0,1]

    # 可选：频率先验（例如120Hz附近bin给更大权重）
    if freq_prior_gk is not None:
        freq_prior_gk = np.asarray(freq_prior_gk, dtype=np.float32)
        if freq_prior_gk.shape[0] != num_bins:
            raise ValueError(f"freq_prior_gk length mismatch: {freq_prior_gk.shape[0]} vs {num_bins}")
        Sk = np.clip(Sk * freq_prior_gk, 0.0, 1.0)

    # 事件级权重
    w1 = 1.0 - lam * Sk[bin_idx]
    w1 = np.clip(w1, 0.0, 1.0).astype(np.float32)

    aux = {
        "num_bins": num_bins,
        "dt_s": float(dt_s),
        "grid": (int(Gx), int(Gy)),
        "bx_by": (int(bx), int(by)),
        "rho": float(rho),
        "alpha": float(alpha),
        "z_th": float(z_th),
        "lam": float(lam),
        "t0_in": t0,
        "normalize_t0": bool(normalize_t0),
        "Ck_mean": float(Ck.mean()),
        "Ck_p95": float(np.quantile(Ck, 0.95)),
        "Sk_mean": float(Sk.mean()),
        "Sk_p95": float(np.quantile(Sk, 0.95)),
    }
    return w1, aux

def s1_run_npz(
    in_npz: str,
    out_npz: str,
    *,
    dt_s: float = 0.0005,
    Gx: int = 16,
    Gy: int = 12,
    rho: float = 0.02,
    alpha: float = 4.0,
    z_th: float = 1.0,
    lam: float = 0.3,
    store_bin_arrays: bool = False,
) -> str:


    t, x, y, p, meta, extra = load_events_npz2(in_npz)

    # 分辨率：你明确就是346x260，建议强制写死，避免历史文件污染
    W, H = 346, 260
    meta["resolution"] = [W, H]
    meta["time_unit"] = "seconds"

    w1, aux = s1_compute_w1(
        t, x, y,
        W=W, H=H,
        dt_s=dt_s, Gx=Gx, Gy=Gy,
        rho=rho, alpha=alpha, z_th=z_th, lam=lam,
        freq_prior_gk=None,
        normalize_t0=True,
    )

    # meta 只放轻量配置/摘要
    meta["s1_cfg"] = {"dt_s": dt_s, "grid": [Gx, Gy], "rho": rho, "alpha": alpha, "z_th": z_th, "lam": lam}
    meta["s1_stats"] = {k: v for k, v in aux.items() if k not in ["Ck", "Sk", "zk"]}

    # extra 放 per-event 长通道
    extra = dict(extra) if extra is not None else {}
    extra["w1"] = w1

    # 可选：把bin数组作为“中等数组”放extra还是meta？
    # 建议：默认不存；要存就另存debug文件，别污染主npz
    if store_bin_arrays and aux["num_bins"] <= 1000:
        meta["s1_bin_debug"] = True
        # 如果你坚持存，也可以放到NPZ顶层字段（但注意文件大小）
        # extra["s1_Ck"] = aux["Ck"]
        # extra["s1_Sk"] = aux["Sk"]
        # extra["s1_zk"] = aux["zk"]
    else:
        meta["s1_bin_debug"] = False

    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    save_events_npz2(out_npz, t, x, y, p, meta=meta, extra=extra)
    return out_npz


if __name__ == "__main__":
    # 示例：
    # python s1_global_sync.py input.npz output_s1.npz
    import sys
    if len(sys.argv) >= 3:
        s1_run_npz(sys.argv[1], sys.argv[2])
        print("Done.")
    else:
        print("Usage: python s1_global_sync.py <in_npz> <out_npz>")
