# -*- coding: utf-8 -*-
import os
import numpy as np


def compute_esr(t, x, y, p=None, width=346, height=260, M=100000, use_polarity=False):
    """
    计算 ESR 指标，严格对应 E-MLB 论文中的定义：
      NTSS := sum_i [ n_i (n_i - 1) / (N (N - 1)) ]
      LN   := K - sum_i (1 - M/N)^{n_i}
      ESR  := sqrt( NTSS * LN )

    参数
    ----
    t, x, y, p : 1D numpy arrays
        事件序列。p 可不传；ESR 原定义基于 IWE 像素上的事件计数 n_i。
    width, height : int
        分辨率。
    M : int
        论文中的“reference number of events used for interpolation”，评测时固定不变。
        你可以把 M 固定为某个常数（例如整个评测统一用 1e5 / 2e5 等），或用某个参考方法的 N。
    use_polarity : bool
        True 时按极性分别计算并求和（可选的工程扩展；默认 False 更贴近常规实现：忽略极性、按像素总计数）。

    返回
    ----
    esr : float
    details : dict
        包含 NTSS, LN, N, K 等便于 debug/记录。
    """
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    # 基本合法性
    if x.size == 0:
        return 0.0, {"NTSS": 0.0, "LN": 0.0, "N": 0, "K": width * height}

    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError("x/y 超出分辨率范围，请先检查事件数据。")

    K = width * height

    def _esr_from_counts(counts_1d, N, M_fixed):
        # NTSS
        if N < 2:
            NTSS = 0.0
        else:
            # sum n_i (n_i - 1) / (N (N - 1))
            # counts_1d 是每个像素的 n_i（展开）
            n = counts_1d.astype(np.float64)
            NTSS = np.sum(n * (n - 1.0)) / (float(N) * float(N - 1))

        # LN
        # LN := K - sum_i (1 - M/N)^{n_i}
        # 注意：若 N=0 已在外部排除；这里 N>=1
        base = 1.0 - (float(M_fixed) / float(N))
        # base 可能为负（当 M > N）——论文公式如此；工程上仍按实数计算
        # 为避免 nan：对 n_i 为0 的项，base**0 = 1 没问题
        n = counts_1d.astype(np.float64)
        LN = float(K) - float(np.sum(np.power(base, n)))

        # ESR
        ESR = float(np.sqrt(max(NTSS * LN, 0.0)))
        return ESR, NTSS, LN

    # 生成像素线性索引
    idx = y * width + x
    N_all = idx.size

    if (p is None) or (not use_polarity):
        counts = np.bincount(idx, minlength=K)
        esr, NTSS, LN = _esr_from_counts(counts, N_all, M)
        return esr, {"NTSS": float(NTSS), "LN": float(LN), "N": int(N_all), "K": int(K), "M": int(M)}

    # 可选：按极性分别计算（工程扩展，不是论文必须项）
    p = np.asarray(p)
    pos_mask = (p > 0)
    neg_mask = (p <= 0)

    esr_total = 0.0
    NTSS_total = 0.0
    LN_total = 0.0

    for mask in (pos_mask, neg_mask):
        idx_sub = idx[mask]
        N_sub = idx_sub.size
        if N_sub == 0:
            continue
        counts = np.bincount(idx_sub, minlength=K)
        esr_sub, NTSS_sub, LN_sub = _esr_from_counts(counts, N_sub, M)
        esr_total += esr_sub
        NTSS_total += NTSS_sub
        LN_total += LN_sub

    return float(esr_total), {
        "NTSS": float(NTSS_total), "LN": float(LN_total),
        "N": int(N_all), "K": int(K), "M": int(M),
        "use_polarity": True
    }


# -----------------------------
# AOCC (TCSVT 2025) 相关
# -----------------------------

def _sobel_grad_mag(binary_frame):
    """
    对二值事件帧做 Sobel（水平/垂直）并返回每像素梯度幅值 Gi = sqrt(Gx^2 + Gy^2)
    这里不依赖 cv2/scipy，纯 numpy 手写 3x3 卷积（够用且可控）。
    """
    # Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = kx.T

    I = binary_frame.astype(np.float32)

    # pad
    Ip = np.pad(I, ((1, 1), (1, 1)), mode="edge")

    # 3x3 卷积（矢量化）
    # Gx = sum_{u,v} kx[u,v] * Ip[y+u, x+v]
    Gx = (
        kx[0, 0] * Ip[:-2, :-2] + kx[0, 1] * Ip[:-2, 1:-1] + kx[0, 2] * Ip[:-2, 2:] +
        kx[1, 0] * Ip[1:-1, :-2] + kx[1, 1] * Ip[1:-1, 1:-1] + kx[1, 2] * Ip[1:-1, 2:] +
        kx[2, 0] * Ip[2:, :-2] + kx[2, 1] * Ip[2:, 1:-1] + kx[2, 2] * Ip[2:, 2:]
    )
    Gy = (
        ky[0, 0] * Ip[:-2, :-2] + ky[0, 1] * Ip[:-2, 1:-1] + ky[0, 2] * Ip[:-2, 2:] +
        ky[1, 0] * Ip[1:-1, :-2] + ky[1, 1] * Ip[1:-1, 1:-1] + ky[1, 2] * Ip[1:-1, 2:] +
        ky[2, 0] * Ip[2:, :-2] + ky[2, 1] * Ip[2:, 1:-1] + ky[2, 2] * Ip[2:, 2:]
    )

    Gi = np.sqrt(Gx * Gx + Gy * Gy)
    return Gi


def _contrast_of_binary_event_frame(binary_frame):
    """
    论文定义的对比度（contrast）：
      1) 先算每像素梯度幅值 Gi
      2) C = sqrt( 1/(N-1) * sum_i (Gi - mean(G))^2 )
    其中 N 是像素总数（width*height）。
    """
    Gi = _sobel_grad_mag(binary_frame)
    G = Gi.reshape(-1).astype(np.float64)
    N = G.size
    meanG = np.mean(G)
    # 方差的无偏估计形式（除以 N-1），与论文一致
    var = np.sum((G - meanG) ** 2) / max(N - 1, 1)
    return float(np.sqrt(var))


def _binary_event_frame_from_events(x, y, width, height):
    """
    AOCC 论文中的二值事件帧：
    若像素在该时间窗内出现过事件，则 I(x,y)=255，否则0。
    这里用 0/1 存储即可（对 Sobel 线性比例不影响结论）。
    """
    frame = np.zeros((height, width), dtype=np.uint8)
    if x.size == 0:
        return frame
    frame[y, x] = 1
    return frame


def _segment_events_by_time(t, x, y, t_start, t_end, dt):
    """
    把 [t_start, t_end) 以 dt 均匀切分，返回每个 segment 的 (x_seg, y_seg) 列表。
    t 单位：秒。
    """
    # 保证排序（AOCC 的时间分段默认按时间推进；不排序会导致分段错误）
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]

    segments = []
    cur = t_start
    i0 = 0
    n = t.size

    while cur + dt <= t_end + 1e-12:
        nxt = cur + dt
        # 找到 [cur, nxt) 的事件
        # i0 往前推进；用 searchsorted 加速
        left = np.searchsorted(t, cur, side="left")
        right = np.searchsorted(t, nxt, side="left")
        segments.append((x[left:right], y[left:right]))
        cur = nxt
        i0 = left

    return segments


def compute_aocc(t, x, y, width=346, height=260, dt_list_ms=None, t_start=None, t_end=None):
    """
    计算 AOCC，并同时返回 CCC 曲线（用于可视化/分析）。
    严格按论文流程：
      - 对给定时间窗 dt，把序列切成多个 dt 段；
      - 每段生成二值事件帧 -> Sobel -> 对比度 C；
      - 计算该 dt 下的平均对比度 Cavg(dt)；
      - AOCC = ∫ Cavg(dt) d(dt)（论文用离散求和近似，这里用梯形积分更稳定）

    参数
    ----
    dt_list_ms : list[float]
        多个候选 dt（毫秒），用于绘制 CCC。
        例如：np.arange(1, 41, 1) 表示 1ms~40ms。
    t_start, t_end : float
        若不提供，默认用 t 的最小最大值。

    返回
    ----
    aocc : float
    ccc_dt_ms : np.ndarray
    ccc_cavg : np.ndarray
    details : dict
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    if t.size == 0:
        return 0.0, np.array([]), np.array([]), {"T": 0.0, "num_dt": 0}

    if dt_list_ms is None:
        # 给一个默认范围（你可以在主程序里自己传）
        dt_list_ms = list(np.arange(1.0, 41.0, 1.0))  # 1ms~40ms

    if t_start is None:
        t_start = float(np.min(t))
    if t_end is None:
        t_end = float(np.max(t))

    T = float(t_end - t_start)
    if T <= 0:
        return 0.0, np.array([]), np.array([]), {"T": T, "num_dt": 0}

    ccc_dt_ms = np.asarray(dt_list_ms, dtype=np.float64)
    ccc_dt_s = ccc_dt_ms / 1000.0

    ccc_cavg = np.zeros_like(ccc_dt_s, dtype=np.float64)
    used_frames = []

    for i, dt in enumerate(ccc_dt_s):
        if dt <= 0:
            ccc_cavg[i] = 0.0
            used_frames.append(0)
            continue

        segments = _segment_events_by_time(t, x, y, t_start, t_end, dt)
        # 论文建议：如果最后一段不足 dt，则丢弃（我们这里天然只取满 dt 的段）
        if len(segments) == 0:
            ccc_cavg[i] = 0.0
            used_frames.append(0)
            continue

        contrasts = []
        for xs, ys in segments:
            frame = _binary_event_frame_from_events(xs, ys, width, height)
            c = _contrast_of_binary_event_frame(frame)
            contrasts.append(c)

        ccc_cavg[i] = float(np.mean(contrasts))
        used_frames.append(len(contrasts))

    # AOCC：对 Cavg(dt) 在 dt 轴上积分
    # 论文离散近似：sum C(ti)*Δti；这里用梯形积分等价且更数值稳定
    aocc = float(np.trapz(ccc_cavg, ccc_dt_s))

    details = {
        "T": T,
        "t_start": float(t_start),
        "t_end": float(t_end),
        "num_dt": int(ccc_dt_s.size),
        "dt_min_ms": float(np.min(ccc_dt_ms)) if ccc_dt_ms.size else None,
        "dt_max_ms": float(np.max(ccc_dt_ms)) if ccc_dt_ms.size else None,
        "used_frames_per_dt": used_frames,
    }

    return aocc, ccc_dt_ms, ccc_cavg, details


# -----------------------------
# 可视化（工程工具）
# -----------------------------
def plot_ccc(dt_ms, cavg, save_path=None, title="CCC (Cavg vs. Δt)"):
    """
    画 CCC 曲线：横轴 Δt(ms)，纵轴 Cavg(Δt)。
    save_path 不为空则保存图片（png）。
    """
    import matplotlib.pyplot as plt

    dt_ms = np.asarray(dt_ms, dtype=np.float64)
    cavg = np.asarray(cavg, dtype=np.float64)

    plt.figure()
    plt.plot(dt_ms, cavg)
    plt.xlabel("Δt (ms)")
    plt.ylabel("Cavg(Δt)")
    plt.title(title)
    plt.grid(True)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return plt.gcf()
