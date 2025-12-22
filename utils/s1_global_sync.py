# -*- coding: utf-8 -*-
# s1_global_sync.py

from __future__ import annotations
import numpy as np

import os
from typing import Optional, Dict, Any, Tuple

from utils.config import RESOLUTION, get_meta_base, get_s1_cfg_dict, S1 as S1_DEFAULT
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
        # ===== Debug bin arrays (optional) =====
        "Ck": Ck if num_bins <= 4000 else None,
        "Sk": Sk if num_bins <= 4000 else None,
        "zk": zk if num_bins <= 4000 else None,
    }
    return w1, aux

def s1_run_npz(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
    store_bin_arrays: bool = True,
) -> str:
    """
    S1: Global sync suppression (Mode-1)

    - 默认参数来自 config.py
    - override: 临时参数覆盖（不污染 config）
    - w1 写入 extra
    - meta 仅存常量/配置/小统计量
    - 若 out_npz 未指定目录，则保存到输入文件目录
    """

    # ---------- 1. Load ----------
    t, x, y, p, meta_in, extra_in = load_events_npz2(in_npz)

    # ---------- 2. Resolve config ----------
    cfg = cfg or S1_DEFAULT
    params = get_s1_cfg_dict()
    if override:
        params.update(override)

    W, H = RESOLUTION

    # ---------- 3. Compute w1 ----------
    w1, aux = s1_compute_w1(
        t, x, y,
        W=W, H=H,
        dt_s=params["dt_s"],
        Gx=params["Gx"],
        Gy=params["Gy"],
        rho=params["rho"],
        alpha=params["alpha"],
        z_th=params["z_th"],
        lam=params["lam"],
        freq_prior_gk=None,
        normalize_t0=params["normalize_t0"],
    )

    # ---------- 4. Build meta / extra ----------
    meta_out = get_meta_base()
    meta_out["s1_cfg"] = params
    meta_out["s1_stats"] = {
        k: v for k, v in aux.items()
        if k not in ("Ck", "Sk", "zk")
    }

    if store_bin_arrays and aux["num_bins"] <= 4000:
        meta_out["s1_debug"] = {
            "Ck": aux["Ck"],
            "Sk": aux["Sk"],
            "zk": aux["zk"],
        }

    extra_out = dict(extra_in) if extra_in else {}
    extra_out["w1"] = w1

    # ---------- 5. Resolve output path ----------
    in_dir = os.path.dirname(os.path.abspath(in_npz))
    in_stem = os.path.splitext(os.path.basename(in_npz))[0]

    if out_npz is None or out_npz == "":
        out_npz = os.path.join(in_dir, f"{in_stem}_s1.npz")
    else:
        # 如果只给了文件名，不含目录 → 存到输入目录
        if os.path.dirname(out_npz) == "":
            out_npz = os.path.join(in_dir, out_npz)
        else:
            os.makedirs(os.path.dirname(out_npz), exist_ok=True)

    # ---------- 6. Save ----------
    save_events_npz2(
        out_npz,
        t=t, x=x, y=y, p=p,
        meta=meta_out,
        extra=extra_out,
    )

    return out_npz



if __name__ == "__main__":
    IN = r"D:\event_data\night_drive_raw.npz"
    # out_npz=None -> 自动保存到输入目录，文件名加 _s1
    out = s1_run_npz(IN, out_npz=None, override=None)
    print("Saved:", out)

