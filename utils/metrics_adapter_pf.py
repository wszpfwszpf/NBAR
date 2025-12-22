# metrics_adapter_pf.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np

from utils.util import load_events_npz2, save_events_npz2  # 你现成的 IO，不重写
from utils.metrics_pf import MetricsCfg, compute_all_metrics  # 你刚才的基函数层


# -----------------------------
# 1) 字段提取（适配层的“映射规则”）
# -----------------------------
def _get_field(meta: Dict[str, Any], extra: Dict[str, np.ndarray], key: str) -> Any:
    """优先 extra(长通道)，否则 meta；不存在返回 None"""
    if key in extra:
        return extra[key]
    return meta.get(key, None)


def extract_pf_flow_channels(
    meta: Dict[str, Any],
    extra: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    从 npz 的 meta/extra 中提取 PF 输出通道：
      u, v, v_valid, pf_res, pf_c

    约定（按你当前 npz keys）：
      - u/v/v_valid/pf_res/pf_c 都是 per-event 长数组 -> 通常在 extra
      - 但为了兼容，meta 里也兜底取一次
    """
    u = _get_field(meta, extra, "u")
    v = _get_field(meta, extra, "v")
    v_valid = _get_field(meta, extra, "v_valid")

    pf_res = _get_field(meta, extra, "pf_res")
    pf_c = _get_field(meta, extra, "pf_c")

    if u is None or v is None or v_valid is None:
        raise KeyError("Missing PF flow channels in NPZ: require u, v, v_valid")

    u = np.asarray(u)
    v = np.asarray(v)
    v_valid = np.asarray(v_valid)

    if u.ndim != 1 or v.ndim != 1 or v_valid.ndim != 1:
        raise ValueError(f"u/v/v_valid must be 1D, got u={u.shape}, v={v.shape}, valid={v_valid.shape}")
    if not (u.size == v.size == v_valid.size):
        raise ValueError(f"u/v/v_valid length mismatch: {u.size}, {v.size}, {v_valid.size}")

    if pf_res is not None:
        pf_res = np.asarray(pf_res)
        if pf_res.ndim != 1 or pf_res.size != u.size:
            # 不强制报错，给个兜底：不用它就行
            pf_res = None

    if pf_c is not None:
        pf_c = np.asarray(pf_c)
        if pf_c.ndim != 1 or pf_c.size != u.size:
            pf_c = None

    return u, v, v_valid, pf_res, pf_c


# -----------------------------
# 2) 适配层核心接口：npz -> 指标
# -----------------------------
def run_pf_metrics_from_npz(
    npz_path: str,
    out_dir: str,
    cfg: Optional[MetricsCfg] = None,
    *,
    save_json: bool = True,
    save_npz_curves: bool = False,
) -> Dict[str, Any]:
    """
    适配层（流程控制 + 杂活）：
      - 读 npz（复用 load_events_npz2）
      - 提取 t/u/v/v_valid（以及可选 pf_res/pf_c）
      - 调用 compute_all_metrics（纯计算层）
      - 保存结果（json / 可选 npz 曲线）

    返回：
      metrics dict（含均值与每-bin曲线）
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or MetricsCfg()

    # 1) 读 npz（你现成的函数）
    t, x, y, p, meta, extra = load_events_npz2(npz_path)

    # 2) 抽取 PF 通道
    u, v, v_valid, pf_res, pf_c = extract_pf_flow_channels(meta, extra)

    # 3) 基本一致性检查（避免单位/格式坑）
    t = np.asarray(t).astype(np.float64, copy=False)
    if t.ndim != 1 or t.size != u.size:
        raise ValueError(f"t must be 1D and match u/v length: t={t.shape}, u={u.shape}")
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError("t not sorted (must be non-decreasing)")

    # v_valid 统一成 0/1 或 bool
    # 注意：你的 PF 里 v_valid 可能是 uint8(0/1)
    valid = (v_valid.astype(np.int64) != 0)

    # 4) 调指标基函数层（纯计算）
    metrics = compute_all_metrics(t=t, u=u, v=v, valid=valid, cfg=cfg)

    # 5) 组装保存用的 meta 信息（只保存轻量的）
    metrics_meta = {
        "src_npz": npz_path,
        "metrics_cfg": {
            "dt_s": float(cfg.dt_s),
            "min_count": int(cfg.min_count),
            "eps_speed": float(cfg.eps_speed),
            "use_valid_only": bool(cfg.use_valid_only),
        },
        "num_events": int(t.size),
        "valid_ratio": float(np.mean(valid)) if t.size else 0.0,
        # 可选：把 PF 的配置/统计也记录进来，便于追溯
        "pf_cfg": meta.get("pf_cfg", None),
        "pf_stats": meta.get("pf_stats", None),
    }

    # 6) 保存（默认 json；曲线可选另存 npz）
    base = os.path.splitext(os.path.basename(npz_path))[0]
    json_path = os.path.join(out_dir, f"metrics_{base}.json")

    if save_json:
        # json 不适合直接写 ndarray，做个转换
        payload = {
            **metrics_meta,
            "metrics": {
                "coverage_mean": float(metrics["coverage_mean"]),
                "ediff_mean": float(metrics["ediff_mean"]),
                "var_theta_mean": float(metrics["var_theta_mean"]),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    if save_npz_curves:
        curves_path = os.path.join(out_dir, f"metrics_curves_{base}.npz")
        np.savez_compressed(
            curves_path,
            coverage_bins=np.asarray(metrics["coverage_bins"]),
            ediff_bins=np.asarray(metrics["ediff_bins"]),
            var_theta_bins=np.asarray(metrics["var_theta_bins"]),
        )

    # 7) 返回（上层可用来打印/汇总成 csv）
    return {
        **metrics_meta,
        "metrics": metrics,
        "saved_json": json_path if save_json else None,
    }


# -----------------------------
# 3) 批处理壳（你以后多文件跑的时候用）
# -----------------------------
def batch_run_pf_metrics(
    npz_paths: list[str],
    out_dir: str,
    cfg: Optional[MetricsCfg] = None,
    *,
    save_json: bool = True,
    save_npz_curves: bool = False,
) -> list[Dict[str, Any]]:
    results = []
    for path in npz_paths:
        res = run_pf_metrics_from_npz(
            npz_path=path,
            out_dir=out_dir,
            cfg=cfg,
            save_json=save_json,
            save_npz_curves=save_npz_curves,
        )
        results.append(res)
    return results
