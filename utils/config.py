# config.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any


# ============ 全局常量 ============
TIME_UNIT: str = "seconds"
RESOLUTION: Tuple[int, int] = (346, 260)   # (W, H)
PIPELINE_NAME: str = "night_city_flow_s1s2_v1"


# ============ S1 配置 ============
@dataclass(frozen=True)
class S1Config:
    dt_s: float = 0.0005     # 0.5 ms
    Gx: int = 16
    Gy: int = 12
    rho: float = 0.02
    alpha: float = 4.0
    z_th: float = 1.0
    lam: float = 0.3
    normalize_t0: bool = True
    # 频率先验后续再接：freq_prior_gk: Any = None


# ============ S2 配置（占位，后续补全） ============
@dataclass(frozen=True)
class S2Config:
    # 先给占位，你后面确定窗口相关增强细节再补
    Tw_s: float = 0.033      # 33ms
    Ts_s: float = 0.010      # step
    r: int = 2               # spatial radius
    Tn_s: float = 0.010      # neighbor time range
    # 还可以加：kmin, mapping params, etc.


# 默认配置实例（全局单例）
S1 = S1Config()
S2 = S2Config()


def get_meta_base() -> Dict[str, Any]:
    """写npz时使用：极简meta（不参与逻辑，仅存档）"""
    W, H = RESOLUTION
    return {
        "pipeline": PIPELINE_NAME,
        "resolution": [W, H],
        "time_unit": TIME_UNIT,
    }


def get_s1_cfg_dict() -> Dict[str, Any]:
    """用于写meta存档（小字典）"""
    return asdict(S1)


# def get_s2_cfg_dict() -> Dict[str, Any]:
#     return asdict(S2)

def get_s2_cfg_dict():
    return {
        "normalize_t0": True,
        "r": 2,
        "Tn": 0.003,
        "tau": 0.001,
        "theta": 1,
        "beta": 2,
    }
