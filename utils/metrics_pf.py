# metrics_pf.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np


@dataclass(frozen=True)
class MetricsCfg:
    """统一的指标配置（基函数层只读 cfg，不做 IO，不做打印）"""
    dt_s: float = 0.033          # 33ms 分箱
    min_count: int = 10          # 每个 bin 最少有效事件数（不足则该 bin 置 NaN）
    eps_speed: float = 1e-12     # 防止除 0 / 过滤零速度
    use_valid_only: bool = True  # 是否只用 valid 事件（对 ediff/dirvar 生效）


def _ensure_1d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={a.shape}")
    return a


def _normalize_valid(valid: np.ndarray, n: int) -> np.ndarray:
    valid = np.asarray(valid)
    if valid.dtype != np.bool_:
        valid = (valid.astype(np.int64) != 0)
    if valid.ndim != 1 or valid.shape[0] != n:
        raise ValueError(f"valid must be 1D of length {n}, got shape={valid.shape}")
    return valid


def _make_bins(t: np.ndarray, dt_s: float) -> Tuple[np.ndarray, int, float]:
    """
    将时间序列映射到 bin index。
    返回：
      bin_idx: (N,) int64
      K: bin 总数
      t0: 起始时间
    """
    if t.size == 0:
        return np.zeros((0,), dtype=np.int64), 0, 0.0
    if dt_s <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt_s}")

    t0 = float(t[0])
    # 数值稳健：减 t0 后再除 dt，取 floor
    bin_idx = np.floor((t - t0) / dt_s).astype(np.int64)

    # K = max(bin_idx) + 1
    K = int(bin_idx[-1] + 1)  # t 单调递增时，最后一个最大
    if K <= 0:
        K = 1
    return bin_idx, K, t0


def metric_coverage(
    t: np.ndarray,
    valid: np.ndarray,
    cfg: MetricsCfg,
) -> Tuple[float, np.ndarray]:
    """
    指标1：事件级光流覆盖率/有效比例（按 dt_s 分箱）
    物理意义：单位时间内“可用于光流”的事件占比（反映可用运动结构的覆盖程度）

    定义（推荐）：
      cov_bin = (#valid events in bin) / (#all events in bin)

    说明：
      - coverage 的分母用“该 bin 的全部事件数”，更符合“在该时间段内有多少事件可用”
      - cfg.min_count 对分母计数生效：bin 内事件太少则置 NaN
    """
    t = _ensure_1d(t, "t").astype(np.float64, copy=False)
    valid = _normalize_valid(valid, t.size)

    bin_idx, K, _ = _make_bins(t, cfg.dt_s)
    if K == 0:
        return 0.0, np.zeros((0,), dtype=np.float64)

    # 分母：bin 内事件总数
    cnt_total = np.bincount(bin_idx, minlength=K).astype(np.float64)
    # 分子：bin 内 valid 事件数
    cnt_valid = np.bincount(bin_idx, weights=valid.astype(np.float64), minlength=K).astype(np.float64)

    cov = np.full((K,), np.nan, dtype=np.float64)
    ok = cnt_total >= max(1, cfg.min_count)
    cov[ok] = cnt_valid[ok] / cnt_total[ok]

    mean_cov = float(np.nanmean(cov)) if np.any(ok) else float("nan")
    return mean_cov, cov


def metric_temporal_diff(
    t: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    cfg: MetricsCfg,
) -> Tuple[float, np.ndarray]:
    """
    指标2：时间一致性/速度差分抖动（按 dt_s 分箱）
    物理意义：同一时间段内相邻事件的速度变化幅度，越小表示越“稳”（更符合局部匀速）

    实现：
      - 默认只对 valid 事件计算（cfg.use_valid_only=True）
      - 每个 bin 内按时间顺序（t 已排序）计算相邻速度差分：
          d_i = || [u_i, v_i] - [u_{i-1}, v_{i-1}] ||_2
        bin 的值取 mean(d_i)
      - 若 bin 内有效事件数 < max(cfg.min_count, 2) 则该 bin = NaN
    """
    t = _ensure_1d(t, "t").astype(np.float64, copy=False)
    u = _ensure_1d(u, "u").astype(np.float64, copy=False)
    v = _ensure_1d(v, "v").astype(np.float64, copy=False)
    if not (t.size == u.size == v.size):
        raise ValueError("t/u/v length mismatch")
    valid = _normalize_valid(valid, t.size)

    if t.size == 0:
        return 0.0, np.zeros((0,), dtype=np.float64)

    # 过滤：只取 valid（建议）
    if cfg.use_valid_only:
        m = valid
        t2, u2, v2 = t[m], u[m], v[m]
    else:
        t2, u2, v2 = t, u, v

    if t2.size == 0:
        # 没有可用事件
        bin_idx, K, _ = _make_bins(t, cfg.dt_s)
        return float("nan"), np.full((K,), np.nan, dtype=np.float64)

    bin_idx, K, _ = _make_bins(t2, cfg.dt_s)
    ediff = np.full((K,), np.nan, dtype=np.float64)

    min_n = max(cfg.min_count, 2)
    # K 很小（2s/33ms≈61），直接循环最清晰
    for k in range(K):
        idx = np.where(bin_idx == k)[0]
        n = idx.size
        if n < min_n:
            continue
        uu = u2[idx]
        vv = v2[idx]
        du = np.diff(uu)
        dv = np.diff(vv)
        d = np.sqrt(du * du + dv * dv)
        # d 长度为 n-1，n>=2 才有意义
        if d.size == 0:
            continue
        ediff[k] = float(np.mean(d))

    mean_ediff = float(np.nanmean(ediff)) if np.any(np.isfinite(ediff)) else float("nan")
    return mean_ediff, ediff


def metric_direction_var(
    t: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    cfg: MetricsCfg,
) -> Tuple[float, np.ndarray]:
    """
    指标3：方向稳定性（角度的环形方差，按 dt_s 分箱）
    物理意义：同一时间段内速度方向是否一致；越小表示方向越稳定、运动一致性越强

    circular variance:
      theta = atan2(v, u)
      R = sqrt((mean cos theta)^2 + (mean sin theta)^2)
      var = 1 - R   (in [0, 1])

    实现细节：
      - 默认只对 valid 事件计算（cfg.use_valid_only=True）
      - 过滤零速度（hypot(u,v) <= eps_speed），避免方向无意义导致噪声
      - bin 内有效数量 < cfg.min_count -> NaN
    """
    t = _ensure_1d(t, "t").astype(np.float64, copy=False)
    u = _ensure_1d(u, "u").astype(np.float64, copy=False)
    v = _ensure_1d(v, "v").astype(np.float64, copy=False)
    if not (t.size == u.size == v.size):
        raise ValueError("t/u/v length mismatch")
    valid = _normalize_valid(valid, t.size)

    if t.size == 0:
        return 0.0, np.zeros((0,), dtype=np.float64)

    # 过滤：valid + 非零速度
    speed = np.hypot(u, v)
    m = (speed > cfg.eps_speed)
    if cfg.use_valid_only:
        m = m & valid

    t2, u2, v2 = t[m], u[m], v[m]
    if t2.size == 0:
        bin_idx, K, _ = _make_bins(t, cfg.dt_s)
        return float("nan"), np.full((K,), np.nan, dtype=np.float64)

    bin_idx, K, _ = _make_bins(t2, cfg.dt_s)
    var_bins = np.full((K,), np.nan, dtype=np.float64)

    for k in range(K):
        idx = np.where(bin_idx == k)[0]
        n = idx.size
        if n < cfg.min_count:
            continue

        uu = u2[idx]
        vv = v2[idx]
        theta = np.arctan2(vv, uu)
        c = np.cos(theta)
        s = np.sin(theta)
        mc = float(np.mean(c))
        ms = float(np.mean(s))
        R = float(np.sqrt(mc * mc + ms * ms))
        var_bins[k] = 1.0 - R

    mean_var = float(np.nanmean(var_bins)) if np.any(np.isfinite(var_bins)) else float("nan")
    return mean_var, var_bins


def compute_all_metrics(
    t: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    cfg: Optional[MetricsCfg] = None,
) -> Dict[str, object]:
    """
    统一入口（仍属于基函数层）：返回三个指标的均值和每-bin曲线。
    适配层可以只调这一个函数。
    """
    cfg = cfg or MetricsCfg()

    cov_mean, cov_bins = metric_coverage(t=t, valid=valid, cfg=cfg)
    ediff_mean, ediff_bins = metric_temporal_diff(t=t, u=u, v=v, valid=valid, cfg=cfg)
    var_mean, var_bins = metric_direction_var(t=t, u=u, v=v, valid=valid, cfg=cfg)

    return {
        "coverage_mean": float(cov_mean) if np.isfinite(cov_mean) else cov_mean,
        "coverage_bins": cov_bins,
        "ediff_mean": float(ediff_mean) if np.isfinite(ediff_mean) else ediff_mean,
        "ediff_bins": ediff_bins,
        "var_theta_mean": float(var_mean) if np.isfinite(var_mean) else var_mean,
        "var_theta_bins": var_bins,
    }
