"""S1 global sync utilities (minimal runtime surface).

The only exported function :func:`s1_compute_w1` is used by
``nbar.stages.s1``. CLI helpers and legacy loaders were removed to keep the
runtime dependency surface small and focused on the regression path driven by
``test.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid used by :func:`s1_compute_w1`."""

    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def s1_compute_w1(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    W: int = 346,
    H: int = 260,
    dt_s: float = 0.0005,  # 0.5ms
    Gx: int = 16,
    Gy: int = 12,
    rho: float = 0.02,  # EMA 更新率
    alpha: float = 8.0,  # sigmoid 斜率
    z_th: float = 1.0,  # z 阈值
    lam: float = 0.8,  # 抑制强度：w1 = 1 - lam * S
    freq_prior_gk: Optional[np.ndarray] = None,  # 可选：bin 级先验 (0~1)
    normalize_t0: bool = True,  # 安全：强制 t 从 0 开始
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute per-event weights for the S1 stage."""

    assert t.ndim == 1 and t.shape == x.shape == y.shape
    N = t.size
    if N == 0:
        return np.zeros((0,), dtype=np.float32), {"num_bins": 0}

    t0 = float(t[0])
    tt = t - t0 if normalize_t0 else t.copy()

    bin_idx = np.floor(tt / dt_s).astype(np.int64)
    num_bins = int(bin_idx.max()) + 1

    bx = int(np.ceil(W / Gx))
    by = int(np.ceil(H / Gy))
    u = np.minimum(x // bx, Gx - 1)
    v = np.minimum(y // by, Gy - 1)
    cell = v * Gx + u
    G = Gx * Gy

    key = bin_idx * G + cell
    uniq_key = np.unique(key)
    uniq_bin = (uniq_key // G).astype(np.int64)
    active_cells = np.bincount(uniq_bin, minlength=num_bins).astype(np.float32)
    Ck = active_cells / float(G)

    mu = np.zeros((num_bins,), dtype=np.float32)
    var = np.zeros((num_bins,), dtype=np.float32)

    mu_k = float(Ck[0])
    var_k = 1e-6
    mu[0] = mu_k
    var[0] = var_k

    for k in range(1, num_bins):
        ck = float(Ck[k])
        mu_k = (1.0 - rho) * mu_k + rho * ck
        diff = ck - mu_k
        var_k = (1.0 - rho) * var_k + rho * (diff * diff)
        mu[k] = mu_k
        var[k] = var_k

    sigma = np.sqrt(np.maximum(var, 1e-8))
    zk = (Ck - mu) / sigma

    Sk = _sigmoid(alpha * (zk - z_th)).astype(np.float32)

    if freq_prior_gk is not None:
        freq_prior_gk = np.asarray(freq_prior_gk, dtype=np.float32)
        if freq_prior_gk.shape[0] != num_bins:
            raise ValueError(
                f"freq_prior_gk length mismatch: {freq_prior_gk.shape[0]} vs {num_bins}"
            )
        Sk = np.clip(Sk * freq_prior_gk, 0.0, 1.0)

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
        "Ck": Ck if num_bins <= 4000 else None,
        "Sk": Sk if num_bins <= 4000 else None,
        "zk": zk if num_bins <= 4000 else None,
    }
    return w1, aux
