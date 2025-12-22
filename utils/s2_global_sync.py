# -*- coding: utf-8 -*-
# s2_global_sync.py

from __future__ import annotations
import numpy as np

import os
from typing import Optional
from typing import Any, Dict, Tuple
from utils.config import RESOLUTION, get_meta_base, get_s2_cfg_dict, S2 as S2_DEFAULT
from utils.util import load_events_npz2, save_events_npz2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))



def s2_compute_c2(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    W: int,
    H: int,
    r: int = 2,            # 5x5
    Tn: float = 0.003,     # 3ms
    normalize_t0: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    S2 (count-only): local support count via last-event map (time-window existence).

    计数定义（同极性）：
      对每个事件 i，在 (2r+1)x(2r+1) 邻域内统计有多少像素的 last-event 满足：
         0 < (t_i - last_t) <= Tn
      由于每个像素只存一条 last-event，所以这是“像素支持数”，不是事件数。

    Returns:
      c2: (N,) int16, in [0, (2r+1)^2] (实际中心像素 dt=0 不计入，最大一般 <= (2r+1)^2-1)
      aux: stats dict
    """

    assert t.ndim == 1
    assert t.shape == x.shape == y.shape == p.shape
    N = int(t.size)
    if N == 0:
        return np.zeros((0,), dtype=np.int16), {"num_events": 0}

    # Optional normalization (keep consistent with pipeline habit)
    if normalize_t0:
        t0 = float(t[0])
        t = t - t0

    # last-event timestamp maps per polarity
    neg_inf = -1e30
    last_pos = np.full((H, W), neg_inf, dtype=np.float64)
    last_neg = np.full((H, W), neg_inf, dtype=np.float64)

    c2 = np.empty((N,), dtype=np.int16)

    Tn_f = float(Tn)

    for i in range(N):
        xi = int(x[i]); yi = int(y[i])
        ti = float(t[i])
        pi = int(p[i])

        # bounds check (robust)
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            c2[i] = 0
            continue

        # choose polarity map
        last_map = last_pos if pi > 0 else last_neg

        x0 = max(0, xi - r); x1 = min(W - 1, xi + r)
        y0 = max(0, yi - r); y1 = min(H - 1, yi + r)

        patch = last_map[y0:y1 + 1, x0:x1 + 1]
        dt = ti - patch  # dt may be huge for neg_inf

        # valid neighbors: 0 < dt <= Tn
        cnt = int(np.sum((dt > 0.0) & (dt <= Tn_f)))
        c2[i] = cnt

        # update current pixel last-event time for its polarity
        last_map[yi, xi] = ti

    # stats
    aux: Dict[str, Any] = {
        "num_events": N,
        "normalize_t0": bool(normalize_t0),
        "W": int(W),
        "H": int(H),
        "r": int(r),
        "Tn": float(Tn),
        "c2_min": int(c2.min()),
        "c2_max": int(c2.max()),
        "c2_mean": float(c2.mean()),
        "c2_p05": int(np.quantile(c2, 0.05)),
        "c2_p50": int(np.quantile(c2, 0.50)),
        "c2_p95": int(np.quantile(c2, 0.95)),
    }
    return c2, aux


def s2_run_npz_c2(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
) -> str:
    """
    S2-C2: count-only local support stage (no filtering)

    - 默认参数来自 config.py
    - override: 临时参数覆盖（不污染 config）
    - c2 写入 extra
    - meta 仅存常量/配置/小统计量
    - 若 out_npz 未指定目录，则保存到输入文件目录
    """

    # ---------- 1. Load ----------
    t, x, y, p, meta_in, extra_in = load_events_npz2(in_npz)

    # ---------- 2. Resolve config ----------
    cfg = cfg or S2_DEFAULT
    params = get_s2_cfg_dict()
    if override:
        params.update(override)

    # ---------- 3. Compute c2 ----------
    W, H = RESOLUTION  # (W,H)

    c2, aux = s2_compute_c2(
        t=t, x=x, y=y, p=p,
        W=W, H=H,
        r=params.get("r", 2),
        Tn=params.get("Tn", 0.003),
        normalize_t0=params.get("normalize_t0", True),
    )

    # ---------- 4. Build meta / extra ----------
    meta_out = get_meta_base()
    meta_out["s2_cfg"] = params
    meta_out["s2_stats"] = aux

    extra_out = dict(extra_in) if extra_in else {}
    extra_out["c2"] = c2

    # ---------- 5. Resolve output path ----------
    in_dir = os.path.dirname(os.path.abspath(in_npz))
    in_stem = os.path.splitext(os.path.basename(in_npz))[0]

    if out_npz is None or out_npz == "":
        out_npz = os.path.join(in_dir, f"{in_stem}_s2c2.npz")
    else:
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



def s2_compute_w2(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    *,
    W: int,
    H: int,
    r: int = 2,            # 5x5
    Tn: float = 0.003,     # 3ms
    tau: float = 0.001,    # 1ms
    theta: float = 1.0,    # support threshold
    beta: float = 2.0,     # sigmoid slope
    normalize_t0: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    S2: local consistency support via last-event map (time-surface-like).
    Returns:
      w2: (N,) float32 in [0,1]
      aux: stats dict
    """

    assert t.ndim == 1
    assert t.shape == x.shape == y.shape == p.shape
    N = int(t.size)
    if N == 0:
        return np.zeros((0,), dtype=np.float32), {"num_events": 0}

    # Optional normalization (keep consistent with your pipeline habit)
    if normalize_t0:
        t0 = float(t[0])
        t = t - t0

    # last-event timestamp maps per polarity
    neg_inf = -1e30
    last_pos = np.full((H, W), neg_inf, dtype=np.float64)
    last_neg = np.full((H, W), neg_inf, dtype=np.float64)

    w2 = np.empty((N,), dtype=np.float32)

    # Precompute for speed
    inv_tau = 1.0 / float(tau)
    Tn_f = float(Tn)



    # Optional counters for debugging
    support_vals = np.empty((N,), dtype=np.float32)

    # -------- pass-1: compute Ki only --------
    for i in range(N):
        xi = int(x[i]); yi = int(y[i])
        ti = float(t[i])
        pi = int(p[i])

        # bounds check (robust)
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            support_vals[i] = 0.0
            continue

        # choose polarity map
        last_map = last_pos if pi > 0 else last_neg

        x0 = max(0, xi - r); x1 = min(W - 1, xi + r)
        y0 = max(0, yi - r); y1 = min(H - 1, yi + r)

        patch = last_map[y0:y1 + 1, x0:x1 + 1]
        dt = ti - patch  # dt may be huge for neg_inf

        # valid neighbors: 0 < dt <= Tn
        mask = (dt > 0.0) & (dt <= Tn_f)
        if np.any(mask):
            Ki = float(np.exp(-dt[mask] * inv_tau).sum())
        else:
            Ki = 0.0

        support_vals[i] = Ki

        # update current pixel last-event time for its polarity
        last_map[yi, xi] = ti

    # -------- pass-2: adapt theta/beta (optional) --------
    K_p10 = float(np.quantile(support_vals, 0.10))
    K_p90 = float(np.quantile(support_vals, 0.90))
    K_p95 = float(np.quantile(support_vals, 0.95))

    auto = (theta == 1.0 and beta == 2.0)

    if auto:
        # 你可以调这两个目标值：越小 w_low 越“压噪”
        w_low = 0.10
        w_high = 0.90

        def logit(w: float) -> float:
            w = min(max(w, 1e-6), 1.0 - 1e-6)
            return float(np.log(w / (1.0 - w)))

        a = logit(w_high) - logit(w_low)  # 对应 K_p90 - K_p10
        denom = max(K_p90 - K_p10, 1e-6)
        beta_eff = float(a / denom)
        beta_eff = float(np.clip(beta_eff, 0.5, 10.0))

        theta_eff = float(K_p10 - logit(w_low) / beta_eff)
    else:
        theta_eff = float(theta)
        beta_eff = float(beta)

    # -------- pass-3: vectorized w2 --------
    w2 = 1.0 / (1.0 + np.exp(-beta_eff * (support_vals.astype(np.float64) - theta_eff)))
    w2 = w2.astype(np.float32)

    aux = {
        "num_events": N,
        "normalize_t0": bool(normalize_t0),
        "W": int(W),
        "H": int(H),
        "r": int(r),
        "Tn": float(Tn),
        "tau": float(tau),

        "theta_in": float(theta),
        "beta_in": float(beta),
        "auto_theta_beta": bool(auto),
        "theta_eff": float(theta_eff),
        "beta_eff": float(beta_eff),

        "w2_mean": float(w2.mean()),
        "w2_p05": float(np.quantile(w2, 0.05)),
        "w2_p50": float(np.quantile(w2, 0.50)),
        "w2_p95": float(np.quantile(w2, 0.95)),
        "w2_p99": float(np.quantile(w2, 0.99)),

        "K_mean": float(support_vals.mean()),
        # "K_p50": K_p50,
        "K_p90": K_p90,
        "K_p95": K_p95,
    }
    return w2, aux




def s2_run_npz(
    in_npz: str,
    out_npz: Optional[str] = None,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
) -> str:
    """
    S2: (placeholder) global sync / global constraint stage

    - 默认参数来自 config.py
    - override: 临时参数覆盖（不污染 config）
    - w2 写入 extra
    - meta 仅存常量/配置/小统计量
    - 若 out_npz 未指定目录，则保存到输入文件目录
    """

    # ---------- 1. Load ----------
    t, x, y, p, meta_in, extra_in = load_events_npz2(in_npz)

    # ---------- 2. Resolve config ----------
    cfg = cfg or S2_DEFAULT
    params = get_s2_cfg_dict()
    if override:
        params.update(override)

    # ---------- 3. Compute w2 ----------
    # w2, aux = s2_compute_w2(
    #     t, x, y,
    #     normalize_t0=params.get("normalize_t0", True),
    # )



    W, H = RESOLUTION  # (W,H)

    w2, aux = s2_compute_w2(
        t=t, x=x, y=y, p=p,
        W=W, H=H,
        r=params.get("r", 2),
        Tn=params.get("Tn", 0.003),
        tau=params.get("tau", 0.001),
        theta=params.get("theta", 1.0),
        beta=params.get("beta", 2.0),
        normalize_t0=params.get("normalize_t0", True),
    )

    # ---------- 4. Build meta / extra ----------
    meta_out = get_meta_base()
    meta_out["s2_cfg"] = params
    meta_out["s2_stats"] = aux

    extra_out = dict(extra_in) if extra_in else {}
    print(extra_out.keys())
    extra_out["w2"] = w2

    # ---------- 5. Resolve output path ----------
    in_dir = os.path.dirname(os.path.abspath(in_npz))
    in_stem = os.path.splitext(os.path.basename(in_npz))[0]

    if out_npz is None or out_npz == "":
        out_npz = os.path.join(in_dir, f"{in_stem}_s2.npz")
    else:
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
