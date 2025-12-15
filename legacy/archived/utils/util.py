# util.py
# -*- coding: utf-8 -*-

import os
import numpy as np
import json

def assert_events_format(t, x, y, p, W=346, H=260):
    t = np.asarray(t); x = np.asarray(x); y = np.asarray(y); p = np.asarray(p)
    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError("t/x/y/p shape mismatch")
    if t.size and np.any(np.diff(t) < 0):
        raise ValueError("t not sorted")
    if not np.all(np.isin(p, [-1, 1])):
        raise ValueError("p must be in {-1, +1}")
    if np.any(x < 0) or np.any(x >= W) or np.any(y < 0) or np.any(y >= H):
        raise ValueError("x/y out of bounds")


# util.py (或 data_utils.py)





# def load_events_npz(npz_path):
#     """
#     Load events from a self-descriptive NPZ file.
#
#     Expected fields in NPZ:
#         - t : float64, timestamps (seconds)
#         - x : int32, pixel x
#         - y : int32, pixel y
#         - p : int8, polarity in {-1, +1}
#
#     Optional fields:
#         - resolution : (W, H)
#         - time_unit  : string
#
#     Returns
#     -------
#     t, x, y, p : np.ndarray
#         Event streams.
#     meta : dict
#         Metadata dictionary (resolution, time_unit, path).
#     """
#     data = np.load(npz_path, allow_pickle=True)
#
#     required_keys = {"t", "x", "y", "p"}
#     if not required_keys.issubset(data.files):
#         raise KeyError(
#             f"NPZ file {npz_path} must contain keys {required_keys}, "
#             f"but got {set(data.files)}"
#         )
#
#     t = np.asarray(data["t"], dtype=np.float64)
#     x = np.asarray(data["x"], dtype=np.int32)
#     y = np.asarray(data["y"], dtype=np.int32)
#     p = np.asarray(data["p"], dtype=np.int8)
#
#     # 基本一致性检查
#     if not (t.shape == x.shape == y.shape == p.shape):
#         raise ValueError("Loaded t, x, y, p have inconsistent shapes")
#
#     # polarity 检查（不强制修正，避免 silent bug）
#     if not np.all(np.isin(p, [-1, 1])):
#         raise ValueError("Polarity p must be in {-1, +1}")
#
#     meta = {
#         "path": npz_path,
#         "resolution": tuple(data["resolution"]) if "resolution" in data.files else None,
#         "time_unit": str(data["time_unit"]) if "time_unit" in data.files else None,
#         "num_events": int(t.size),
#         "t_start": float(t[0]) if t.size > 0 else None,
#         "t_end": float(t[-1]) if t.size > 0 else None,
#     }
#
#     return t, x, y, p, meta
#
# def save_events_npz(
#     save_path,
#     t,
#     x,
#     y,
#     p,
#     resolution=(346, 260),
#     time_unit="seconds"
# ):
#     """
#     Save events to a self-descriptive NPZ file.
#
#     Parameters
#     ----------
#     save_path : str
#         Output .npz file path.
#     t : array-like
#         Timestamps (float), unit specified by time_unit.
#     x, y : array-like
#         Pixel coordinates.
#     p : array-like
#         Polarity. Will be normalized to {-1, +1}.
#     resolution : tuple (W, H)
#         Sensor resolution.
#     time_unit : str
#         Usually "seconds".
#     """
#
#     t = np.asarray(t, dtype=np.float64)
#     x = np.asarray(x, dtype=np.int32)
#     y = np.asarray(y, dtype=np.int32)
#
#     # polarity normalization: {-1, +1}
#     p = np.asarray(p, dtype=np.int8)
#     p_pm = np.where(p > 0, 1, -1).astype(np.int8)
#
#     # basic sanity checks（只做最基础的，避免隐藏 bug）
#     if not (t.shape == x.shape == y.shape == p_pm.shape):
#         raise ValueError("t, x, y, p must have the same shape")
#
#     np.savez_compressed(
#         save_path,
#         t=t,
#         x=x,
#         y=y,
#         p=p_pm,
#         resolution=np.asarray(resolution, dtype=np.int32),
#         time_unit=time_unit
#     )


import os
from typing import Dict, Any, Optional, Tuple

from nbar.io import load_npz as load_events_npz_stream, save_npz as save_events_npz_stream
from nbar.types import EventStream


def load_events_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load events from a self-descriptive NPZ file.

    Delegates to :func:`nbar.io.load_npz` to preserve existing key handling
    while keeping downstream behavior unchanged.
    """
    events: EventStream = load_events_npz_stream(npz_path)
    return events.t, events.x, events.y, events.p, events.meta or {}


def save_events_npz(
        save_path: str,
        t: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        p: np.ndarray,
        meta: Optional[Dict[str, Any]] = None
):
    """
    Save events to a self-descriptive NPZ file with full metadata.

    Examples
    --------
    # 直接传递完整的meta字典
    save_events_npz("output.npz", t, x, y, p, meta=meta)
    """
    events = EventStream(
        t=np.asarray(t, dtype=np.float64),
        x=np.asarray(x, dtype=np.int32),
        y=np.asarray(y, dtype=np.int32),
        p=np.asarray(p, dtype=np.int8),
        meta=meta,
    )
    save_events_npz_stream(save_path, events)


def clip_events_npz(in_npz, out_npz, t_start, t_end, assume_sorted=True):
    data = np.load(in_npz)
    t = data["t"]
    x = data["x"]
    y = data["y"]
    p = data["p"]

    assert t_start < t_end, "t_start must be < t_end"
    assert t_start >= 0, "t_start must be >= 0"
    assert t_end <= float(t[-1]), f"t_end must be <= duration ({t[-1]:.3f}s)"

    # MVSEC 事件通常按时间排序；用 searchsorted 是最快且最省内存的切片方式
    if assume_sorted:
        i0 = int(np.searchsorted(t, t_start, side="left"))
        i1 = int(np.searchsorted(t, t_end, side="right"))
    else:
        mask = (t >= t_start) & (t <= t_end)
        i0, i1 = np.where(mask)[0][[0, -1]]
        i1 = i1 + 1

    t_clip = t[i0:i1].copy()
    x_clip = x[i0:i1].copy()
    y_clip = y[i0:i1].copy()
    p_clip = p[i0:i1].copy()

    # 标准化时间戳（从 0 开始）
    t_clip -= t_clip[0]

    np.savez(
        out_npz,
        t=t_clip, x=x_clip, y=y_clip, p=p_clip,
        resolution=np.array([346, 260]),
        time_unit="seconds",
        src=in_npz,
        t_range=np.array([t_start, t_end], dtype=np.float64),
        count=np.array([len(t_clip)], dtype=np.int64),
    )

    print(f"Saved: {out_npz}")
    print(f"  Range: [{t_start:.3f}, {t_end:.3f}] s  ->  count={len(t_clip):,}")
    print(f"  Duration after norm: {t_clip[-1]:.6f} s")


import os
import matplotlib.pyplot as plt


def render_event_frames_from_npz(
    npz_path: str,
    out_dir: str | None = None,
    win_ms: float = 33.0,               # 累积时间：毫秒（默认33ms）
    resolution: tuple[int, int] = (346, 260),  # (W, H)，若 npz meta 有 resolution 可覆盖
    max_frames: int | None = None,
    point_size: float = 1.0,
    verbose: bool = True,
):
    """
    Render event frames from an NPZ event stream by fixed accumulation time.

    Input NPZ format:
        - t : float64 (seconds), sorted & normalized from 0
        - x : int32
        - y : int32
        - p : int8 in {-1, +1}

    Output:
        - If out_dir is None:
            Create a folder beside npz file:
                <stem>_vis_<win_ms_int>
            Example:
                a.npz -> a_vis_33/
        - Save PNG frames as frame_00000.png, ...

    Notes:
        - t is in seconds, win_ms is milliseconds -> converted internally.
        - Uses white background, pos=red, neg=blue.
        - Flips y for image-like coordinate system (origin at top-left).
    """

    # ----------------------------
    # 1) Load (use your standard loader)
    # ----------------------------
    t, x, y, p, meta = load_events_npz(npz_path)

    if t.size == 0:
        raise ValueError("Empty event stream")

    # resolution: meta preferred
    W, H = resolution
    if meta.get("resolution") is not None:
        W, H = int(meta["resolution"][0]), int(meta["resolution"][1])

    # basic checks
    if t[0] < 0:
        raise ValueError("t should be normalized >= 0 (seconds)")
    if not (np.all((x >= 0) & (x < W)) and np.all((y >= 0) & (y < H))):
        raise ValueError("x/y out of range for given resolution")
    if not np.all(np.isin(p, [-1, 1])):
        raise ValueError("p must be in {-1, +1}")

    # ----------------------------
    # 2) Resolve output directory
    # ----------------------------
    win_ms_int = int(round(float(win_ms)))
    if out_dir is None:
        parent = os.path.dirname(os.path.abspath(npz_path))
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = os.path.join(parent, f"{stem}_vis_{win_ms_int}")

    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # 3) Convert accumulation window
    # ----------------------------
    win_s = float(win_ms) * 1e-3  # ms -> seconds

    duration = float(t[-1])  # seconds
    n_frames = int(np.ceil(duration / win_s))
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    if verbose:
        print(
            f"[VIS] npz={os.path.basename(npz_path)} | N={t.size} | "
            f"duration≈{duration:.6f}s | win={win_ms:.2f}ms | frames={n_frames} | out={out_dir}"
        )

    # ----------------------------
    # 4) Render frames using searchsorted (fast & low-memory)
    # ----------------------------
    for k in range(n_frames):
        t0 = k * win_s
        t1 = (k + 1) * win_s

        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="left"))

        xk = x[i0:i1]
        yk = y[i0:i1]
        pk = p[i0:i1]

        # white canvas
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = plt.axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_axis_off()
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        if pk.size > 0:
            pos = (pk == 1)
            neg = ~pos

            # flip y to match image coords (origin top-left)
            if np.any(pos):
                ax.scatter(
                    xk[pos], (H - 1 - yk[pos]),
                    s=point_size, c="red", marker=".", linewidths=0
                )
            if np.any(neg):
                ax.scatter(
                    xk[neg], (H - 1 - yk[neg]),
                    s=point_size, c="blue", marker=".", linewidths=0
                )

        out_path = os.path.join(out_dir, f"frame_{k:05d}.png")
        fig.savefig(out_path, dpi=100, facecolor="white")
        plt.close(fig)

        if verbose and ((k + 1) % 50 == 0 or k == n_frames - 1):
            print(f"[VIS] saved {k + 1}/{n_frames}")

    if verbose:
        print(f"[VIS] Done. Output dir: {out_dir}")

    return out_dir


import os
import numpy as np
# import matplotlib.pyplot as plt

# from util import load_events_npz2  # v2: (t,x,y,p,meta,extra)

def render_event_frames_from_npz2(
    npz_path: str,
    out_dir: str | None = None,
    win_ms: float = 33.0,
    resolution: tuple[int, int] = (346, 260),
    max_frames: int | None = None,
    point_size: float = 1.0,
    verbose: bool = True,
    *,
    stage: str = "s1",         # raw / s1 / s2（注意：raw 时仍会画 raw/soft/hard，但 soft/hard 等价于 raw）
    eta: float = 0.9,          # hard 阈值
    alpha_min: float = 0.08,    # soft 最低透明度
    gamma: float = 3.0,         # soft 曲线
):
    """
    每帧输出一张图：三个子图 [RAW | SOFT | HARD]
    - stage="s1": soft/hard 使用 w1
    - stage="s2": soft/hard 使用 w2
    - stage="raw": soft/hard 退化为 raw（方便对照模板不变）

    输出目录命名：<stem>_vis_<win_ms_int>_<stage>/
    """

    assert stage in ("raw", "s1", "s2"), f"stage must be one of raw/s1/s2, got {stage}"

    # ----------------------------
    # 1) Load
    # ----------------------------
    t, x, y, p, meta, extra = load_events_npz2(npz_path)
    if t.size == 0:
        raise ValueError("Empty event stream")

    # resolution: meta preferred
    W, H = resolution
    if meta is not None and meta.get("resolution") is not None:
        W, H = int(meta["resolution"][0]), int(meta["resolution"][1])

    # polarity: allow {-1,+1} or {0,1}
    if np.all(np.isin(p, [-1, 1])):
        p_bin = (p == 1)
    elif np.all(np.isin(p, [0, 1])):
        p_bin = (p == 1)
    else:
        raise ValueError("p must be in {-1,+1} or {0,1}")

    # weights
    w = None
    if stage == "s1":
        if extra is None or "w1" not in extra:
            raise ValueError("stage='s1' requires per-event weight 'w1' in extra.")
        w = np.asarray(extra["w1"], dtype=np.float32)
    elif stage == "s2":
        if extra is None or "w2" not in extra:
            raise ValueError("stage='s2' requires per-event weight 'w2' in extra.")
        w = np.asarray(extra["w2"], dtype=np.float32)

    if w is not None and w.shape[0] != t.shape[0]:
        raise ValueError(f"weight length mismatch: {w.shape[0]} vs {t.shape[0]}")

    # ----------------------------
    # 2) Resolve output directory (append stage)
    # ----------------------------
    win_ms_int = int(round(float(win_ms)))
    if out_dir is None:
        parent = os.path.dirname(os.path.abspath(npz_path))
        stem = os.path.splitext(os.path.basename(npz_path))[0]
        out_dir = os.path.join(parent, f"{stem}_vis_{win_ms_int}_{stage}")
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------------
    # 3) Convert accumulation window
    # ----------------------------
    win_s = float(win_ms) * 1e-3
    duration = float(t[-1])
    n_frames = int(np.ceil(duration / win_s))
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    if verbose:
        print(
            f"[VIS] npz={os.path.basename(npz_path)} | N={t.size} | "
            f"duration≈{duration:.6f}s | win={win_ms:.2f}ms | frames={n_frames} | stage={stage} | eta={eta}"
        )

    # soft alpha mapping（简化版：整幅图用一个 alpha，不做逐点 alpha）
    def soft_alpha_from_w(wk: np.ndarray) -> float:
        if wk.size == 0:
            return 1.0
        wk = np.clip(wk, 0.0, 1.0)
        w_min = float(np.min(wk))
        denom = max(1e-6, 1.0 - w_min)
        u = (wk - w_min) / denom
        a = np.clip(u ** float(gamma), 0.0, 1.0)
        a = np.clip(a, float(alpha_min), 1.0)
        return float(np.mean(a))  # 简化：均值透明度

    # ----------------------------
    # 4) Render frames
    # ----------------------------
    for k in range(n_frames):
        t0 = k * win_s
        t1 = (k + 1) * win_s
        i0 = int(np.searchsorted(t, t0, side="left"))
        i1 = int(np.searchsorted(t, t1, side="left"))

        xk = x[i0:i1]
        yk = y[i0:i1]
        pk = p_bin[i0:i1]
        wk = None if w is None else w[i0:i1]

        yy = (H - 1 - yk)  # flip y

        # --- prepare masks / alpha ---
        # raw: 全画
        m_raw = np.ones(pk.shape[0], dtype=bool)

        # soft: 全画，但降低透明度（stage=raw 时 alpha=1）
        a_soft = 1.0 if wk is None else soft_alpha_from_w(wk)
        m_soft = np.ones(pk.shape[0], dtype=bool)

        # hard: 按阈值过滤（stage=raw 时等价 raw）
        m_hard = m_raw if wk is None else (wk >= float(eta))
        # removed: 被 hard 去掉的事件
        m_removed = np.zeros_like(m_raw) if wk is None else (wk < float(eta))

        # --- one figure with 3 subplots ---
        fig, axes = plt.subplots(1, 3, figsize=(3 * W / 100, H / 100), dpi=100)
        # titles = ["RAW", "SOFT", "HARD"]
        # titles = ["RAW", "SOFT", rf"HARD ($\eta={eta:.2f}$)"]
        fig, axes = plt.subplots(1, 4, figsize=(4 * W / 100, H / 100), dpi=100)
        titles = [
            "RAW",
            "SOFT",
            rf"HARD ($\eta={eta:.2f}$)",
            r"REMOVED ($w<\eta$)"
        ]

        # for ax, title in zip(axes, titles):
        #     ax.set_xlim(0, W)
        #     ax.set_ylim(0, H)
        #     ax.set_axis_off()
        #     ax.set_facecolor("black")
        #     # ax.set_edgecolor("black")

        for ax, title in zip(axes, titles):
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_facecolor("white")
            ax.set_title(title, fontsize=10)

            # === 黑色边框 ===
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.2)
                spine.set_edgecolor("black")

            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        fig.patch.set_facecolor("white")

        def draw(ax, mask: np.ndarray, alpha_val: float = 1.0):
            if pk.size == 0 or not np.any(mask):
                return
            pos = pk & mask
            neg = (~pk) & mask

            if np.any(pos):
                ax.scatter(xk[pos], yy[pos], s=point_size, c="red", marker=".", linewidths=0, alpha=alpha_val)
            if np.any(neg):
                ax.scatter(xk[neg], yy[neg], s=point_size, c="blue", marker=".", linewidths=0, alpha=alpha_val)

        def draw_removed(ax, mask: np.ndarray):
            if pk.size == 0 or not np.any(mask):
                return
            ax.scatter(
                xk[mask], yy[mask],
                s=point_size * 0.8,
                c="green",
                marker=".",
                linewidths=0,
                alpha=1.0
            )

        draw(axes[0], m_raw, 1.0)
        draw(axes[1], m_soft, a_soft)
        draw(axes[2], m_hard, 1.0)
        draw_removed(axes[3], m_removed)

        # 可选：标题（你如果嫌占地方可以注释掉）
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=10)


        plt.tight_layout(pad=0.6, w_pad=0.9)

        out_path = os.path.join(out_dir, f"frame_{k:05d}.png")
        fig.savefig(out_path, dpi=100, facecolor="white")
        plt.close(fig)
        plt.close("all")

        if verbose and ((k + 1) % 50 == 0 or k == n_frames - 1):
            print(f"[VIS] saved {k + 1}/{n_frames}")

    if verbose:
        print(f"[VIS] Done. Output dir: {out_dir}")

    return out_dir




def run_frequency_diagnosis_npz(
    npz_path: str,
    bin_ms: float = 1.0,
    target_freqs=(50.0, 60.0),
    max_plot_freq: float = 200.0,
    verbose: bool = True,
):
    """
    Frequency diagnosis for event streams.

    This function analyzes the global event-rate sequence r[k]
    and performs FFT-based spectral analysis.

    Input:
        - npz_path: path to event npz
        - bin_ms: temporal bin size in milliseconds (default: 1 ms)
        - target_freqs: frequencies of interest (e.g., 50Hz, 60Hz)
        - max_plot_freq: upper bound for visualization (Hz)

    Output:
        A folder named 'freq_dia' will be created beside npz file,
        containing:
            - freq_spectrum.png
            - freq_scores.json
    """

    # ------------------------------------------------------------
    # 1. Load events (standard loader)
    # ------------------------------------------------------------
    t, x, y, p, meta = load_events_npz(npz_path)
    if t.size == 0:
        raise ValueError("Empty event stream")

    # ------------------------------------------------------------
    # 2. Prepare output directory
    # ------------------------------------------------------------
    parent = os.path.dirname(os.path.abspath(npz_path))
    out_dir = os.path.join(parent, "freq_dia")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 3. Build global rate sequence r[k]
    # ------------------------------------------------------------
    bin_s = float(bin_ms) * 1e-3  # ms -> seconds
    duration = float(t[-1])
    n_bins = int(np.ceil(duration / bin_s))

    k = np.floor(t / bin_s).astype(np.int64)
    k = np.clip(k, 0, n_bins - 1)

    r = np.bincount(k, minlength=n_bins).astype(np.float64)

    # remove DC for better spectral contrast
    r_dc = r - np.mean(r)

    # ------------------------------------------------------------
    # 4. FFT
    # ------------------------------------------------------------
    fft_vals = np.fft.rfft(r_dc)
    freqs = np.fft.rfftfreq(r_dc.size, d=bin_s)
    mag = np.abs(fft_vals)

    # ------------------------------------------------------------
    # 5. Frequency scores (energy at target freqs)
    # ------------------------------------------------------------
    freq_scores = {}
    for f0 in target_freqs:
        idx = np.argmin(np.abs(freqs - f0))
        freq_scores[f"{f0:.1f}Hz"] = {
            "frequency": float(freqs[idx]),
            "magnitude": float(mag[idx]),
        }

    # ------------------------------------------------------------
    # 6. Save numeric results
    # ------------------------------------------------------------
    json_path = os.path.join(out_dir, "freq_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "npz": os.path.basename(npz_path),
                "bin_ms": float(bin_ms),
                "duration_s": duration,
                "num_events": int(t.size),
                "scores": freq_scores,
            },
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # 7. Visualization
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    mask = freqs <= max_plot_freq
    ax.plot(freqs[mask], mag[mask], lw=1.0)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Event Rate Spectrum")

    for f0 in target_freqs:
        ax.axvline(f0, color="r", linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "freq_spectrum.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    if verbose:
        print(f"[FreqDiag] bin={bin_ms} ms | duration≈{duration:.3f}s")
        for k, v in freq_scores.items():
            print(f"[FreqDiag] {k}: freq≈{v['frequency']:.2f}Hz, mag={v['magnitude']:.3f}")
        print(f"[FreqDiag] Results saved to: {out_dir}")

    return {
        "out_dir": out_dir,
        "bin_ms": float(bin_ms),
        "scores": freq_scores,
    }


# ------------------------------------------------------------
# Lightweight Global Sync Burst Detection (plug-and-play)
# ------------------------------------------------------------
def detect_global_sync_burst(
    t_s: np.ndarray,
    p_pm: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    bin_ms: float = 1.0,
    # --- rate abnormality threshold (robust) ---
    rate_q: float = 0.85,         # quantile threshold on r[k]
    rate_min: int = 0,            # absolute min rate to be considered
    # --- polarity balance threshold ---
    balance_th: float = 0.35,     # |(n+ - n-)/(n+ + n-)| <= balance_th
    min_events_bin: int = 50,     # bins with too few events are ignored
    # --- optional spatial uniformity gate (coarse) ---
    use_spatial_gate: bool = False,
    grid_w: int = 12,             # coarse grid in x
    grid_h: int = 9,              # coarse grid in y
    uniformity_ent_th: float = 0.70,  # normalized entropy threshold [0,1]
    # --- optional frequency refinement (very light, no FFT) ---
    use_freq_refine: bool = False,
    refine_freqs_hz: tuple[float, ...] = (100.0, 120.0),
    refine_win_s: float = 2.0,     # analysis window length for refinement
    refine_score_th: float = 6.0,  # larger => stricter (peak/background-like)
    # --- hysteresis / smoothing ---
    smooth_len: int = 1,          # >=1, apply 1D smoothing on decision mask
):
    """
    Detect global synchronous burst bins in an event stream.

    Inputs
    ------
    t_s   : (N,) float64 timestamps in seconds, sorted, normalized recommended.
    p_pm  : (N,) int8 polarity in {-1, +1}.
    x,y   : (N,) int32 pixel coordinates (optional, only for spatial gate).

    Returns
    -------
    mask_bins : (M,) bool, whether each time bin is considered a sync-burst bin.
    stats     : dict, diagnostic statistics (rates, balances, thresholds, etc.)

    Notes
    -----
    - This is a *task-oriented nuisance detector*, not a "noise detector".
    - It targets globally synchronous, polarity-balanced, high-rate bursts.
    - Optional refine uses Goertzel energy around given frequencies on r[k]
      within a sliding window, far cheaper than full FFT, and can be disabled.
    """

    # -----------------------
    # basic checks
    # -----------------------
    t_s = np.asarray(t_s, dtype=np.float64)
    p_pm = np.asarray(p_pm, dtype=np.int8)

    if t_s.ndim != 1 or p_pm.ndim != 1 or t_s.size != p_pm.size:
        raise ValueError("t_s and p_pm must be 1D arrays of same length")
    if t_s.size == 0:
        return np.zeros((0,), dtype=bool), {"reason": "empty"}
    if not np.all(np.isin(p_pm, [-1, 1])):
        raise ValueError("p_pm must be in {-1, +1}")

    dt_s = float(bin_ms) * 1e-3
    T = float(t_s[-1]) - float(t_s[0])
    if T <= 0:
        # degenerate
        return np.zeros((1,), dtype=bool), {"reason": "degenerate_time"}

    # number of bins
    t0 = float(t_s[0])
    t1 = float(t_s[-1])
    M = int(np.floor((t1 - t0) / dt_s)) + 1

    # bin index for each event
    k = np.floor((t_s - t0) / dt_s).astype(np.int32)
    k = np.clip(k, 0, M - 1)

    # -----------------------
    # r[k], n_pos[k], n_neg[k]
    # -----------------------
    r = np.bincount(k, minlength=M).astype(np.int32)

    pos = (p_pm > 0)
    n_pos = np.bincount(k[pos], minlength=M).astype(np.int32)
    n_neg = (r - n_pos).astype(np.int32)

    # polarity balance metric in [-1,1], safe for r=0
    denom = np.maximum(r, 1)
    bal = (n_pos - n_neg).astype(np.float64) / denom.astype(np.float64)
    bal_abs = np.abs(bal)

    # -----------------------
    # rate abnormality gate
    # -----------------------
    # robust threshold by quantile
    thr_rate = float(np.quantile(r, rate_q))
    thr_rate = max(thr_rate, float(rate_min))

    gate_rate = (r >= thr_rate) & (r >= min_events_bin)
    gate_bal = (bal_abs <= float(balance_th)) & (r >= min_events_bin)

    gate = gate_rate & gate_bal

    # -----------------------
    # optional spatial uniformity gate (coarse entropy)
    # -----------------------
    ent_norm = None
    if use_spatial_gate:
        if x is None or y is None:
            raise ValueError("use_spatial_gate=True requires x and y")
        x = np.asarray(x, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)
        if x.size != t_s.size or y.size != t_s.size:
            raise ValueError("x,y must have same length as t_s")

        # coarse cell id (grid)
        # note: we do not assume resolution here; use observed max+1
        W = int(np.max(x)) + 1
        H = int(np.max(y)) + 1
        gx = np.clip((x.astype(np.float64) / max(W, 1) * grid_w).astype(np.int32), 0, grid_w - 1)
        gy = np.clip((y.astype(np.float64) / max(H, 1) * grid_h).astype(np.int32), 0, grid_h - 1)
        cell = gy * grid_w + gx
        G = grid_w * grid_h

        # compute normalized entropy per bin (only for bins that already pass basic gate)
        ent_norm = np.zeros((M,), dtype=np.float64)

        # iterate bins that are candidates (this loop is cheap because candidates are few)
        cand_bins = np.where(gate)[0]
        for kk in cand_bins:
            # get indices of events in this bin via range search on k
            # we avoid heavy structures; use boolean mask slice only on candidate bins
            idx = (k == kk)
            if not np.any(idx):
                continue
            c = cell[idx]
            hist = np.bincount(c, minlength=G).astype(np.float64)
            s = hist.sum()
            if s <= 0:
                continue
            p = hist / s
            # entropy
            eps = 1e-12
            Hc = -np.sum(p * np.log(p + eps))
            # normalize by log(G)
            ent_norm[kk] = float(Hc / np.log(G + eps))

        gate_uni = (ent_norm >= float(uniformity_ent_th))
        gate = gate & gate_uni

    # -----------------------
    # optional frequency refinement (Goertzel on r[k])
    # -----------------------
    refine_score = None
    refine_best_hz = None
    if use_freq_refine:
        fs = 1.0 / dt_s  # sampling frequency of r[k]
        win_bins = max(8, int(round(refine_win_s / dt_s)))
        win_bins = min(win_bins, M)

        refine_score = np.zeros((M,), dtype=np.float64)
        refine_best_hz = np.zeros((M,), dtype=np.float64)

        # compute a simple "peak/background" like score per bin using a centered window
        # (only evaluate on bins already passing gate to keep it cheap)
        cand_bins = np.where(gate)[0]
        for kk in cand_bins:
            half = win_bins // 2
            a = max(0, kk - half)
            b = min(M, kk + half)
            sig = r[a:b].astype(np.float64)
            if sig.size < 8:
                continue
            # detrend to reduce DC dominance
            sig = sig - np.median(sig)

            # compute energy for each candidate frequency
            best = 0.0
            best_f = 0.0
            bg = np.median(sig * sig) + 1e-12  # cheap background proxy
            for f0 in refine_freqs_hz:
                e = _goertzel_energy(sig, fs=fs, f0=float(f0))
                score = e / bg
                if score > best:
                    best = score
                    best_f = float(f0)
            refine_score[kk] = best
            refine_best_hz[kk] = best_f

        gate_ref = (refine_score >= float(refine_score_th))
        gate = gate & gate_ref

    # -----------------------
    # optional smoothing (hysteresis-like)
    # -----------------------
    mask = gate.astype(bool)
    if smooth_len >= 2:
        L = int(smooth_len)
        kernel = np.ones((L,), dtype=np.float64) / float(L)
        # simple moving average on {0,1}, then threshold at 0.5
        sm = np.convolve(mask.astype(np.float64), kernel, mode="same")
        mask = (sm >= 0.5)

    stats = {
        "bin_ms": float(bin_ms),
        "dt_s": float(dt_s),
        "num_bins": int(M),
        "t0": float(t0),
        "t1": float(t1),
        "rate_q": float(rate_q),
        "thr_rate": float(thr_rate),
        "balance_th": float(balance_th),
        "min_events_bin": int(min_events_bin),
        "num_trigger_bins": int(np.sum(mask)),
        "trigger_ratio": float(np.mean(mask)),
        "r": r,
        "bal_abs": bal_abs,
    }
    if use_spatial_gate:
        stats["use_spatial_gate"] = True
        stats["uniformity_ent_th"] = float(uniformity_ent_th)
        stats["ent_norm"] = ent_norm
    if use_freq_refine:
        stats["use_freq_refine"] = True
        stats["refine_freqs_hz"] = tuple(float(z) for z in refine_freqs_hz)
        stats["refine_win_s"] = float(refine_win_s)
        stats["refine_score_th"] = float(refine_score_th)
        stats["refine_score"] = refine_score
        stats["refine_best_hz"] = refine_best_hz

    return mask, stats


def _goertzel_energy(x: np.ndarray, *, fs: float, f0: float) -> float:
    """
    Goertzel algorithm: compute energy at frequency f0 for real signal x.
    This is much cheaper than FFT if you only need a few frequencies.
    """
    x = np.asarray(x, dtype=np.float64)
    N = int(x.size)
    if N <= 0 or fs <= 0 or f0 <= 0:
        return 0.0

    # Clamp f0 to Nyquist
    f0 = min(f0, 0.5 * fs - 1e-6)
    w = 2.0 * np.pi * (f0 / fs)
    cosw = np.cos(w)
    coeff = 2.0 * cosw

    s_prev = 0.0
    s_prev2 = 0.0
    for n in range(N):
        s = x[n] + coeff * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s

    # power at f0
    power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return float(power)


import numpy as np

def detect_global_sync_burst_lite(
    t_s: np.ndarray,
    p_pm: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    *,
    bin_ms: float = 1.0,
    # rate gate
    rate_q: float = 0.85,        # r[k] >= quantile(r, rate_q)
    rate_min: int = 0,           # absolute minimal rate
    min_events_bin: int = 50,    # ignore small bins
    # polarity balance gate
    balance_th: float = 0.35,    # |(n+ - n-)/(n+ + n-)| <= balance_th
    # optional spatial gate (OFF by default)
    use_spatial_gate: bool = False,
    grid_w: int = 12,
    grid_h: int = 9,
    uniformity_ent_th: float = 0.70,  # normalized entropy threshold [0,1]
    # optional smoothing on decision mask
    smooth_len: int = 1,
):
    """
    Lightweight detection of globally synchronous burst bins.

    Targets bins that are:
      - high event-rate (global burst)
      - polarity-balanced (illumination-induced nuisance typical)
      - (optional) spatially uniform on a coarse grid

    Parameters are meant to be set offline (on MVSEC night etc.) and then fixed.

    Returns
    -------
    mask_bins : (M,) bool
        Whether each time bin is classified as global sync burst.
    stats : dict
        Diagnostics: thresholds, r[k], balance, trigger ratio, etc.
    """
    t_s = np.asarray(t_s, dtype=np.float64)
    p_pm = np.asarray(p_pm, dtype=np.int8)

    if t_s.ndim != 1 or p_pm.ndim != 1 or t_s.size != p_pm.size:
        raise ValueError("t_s and p_pm must be 1D arrays of same length")
    if t_s.size == 0:
        return np.zeros((0,), dtype=bool), {"reason": "empty"}
    if not np.all(np.isin(p_pm, [-1, 1])):
        raise ValueError("p_pm must be in {-1, +1}")

    dt_s = float(bin_ms) * 1e-3
    t0 = float(t_s[0])
    t1 = float(t_s[-1])
    if t1 <= t0:
        return np.zeros((1,), dtype=bool), {"reason": "degenerate_time"}

    M = int(np.floor((t1 - t0) / dt_s)) + 1
    k = np.floor((t_s - t0) / dt_s).astype(np.int32)
    k = np.clip(k, 0, M - 1)

    # global rate r[k]
    r = np.bincount(k, minlength=M).astype(np.int32)

    # polarity counts
    pos = (p_pm > 0)
    n_pos = np.bincount(k[pos], minlength=M).astype(np.int32)
    n_neg = (r - n_pos).astype(np.int32)

    denom = np.maximum(r, 1)
    bal = (n_pos - n_neg).astype(np.float64) / denom.astype(np.float64)
    bal_abs = np.abs(bal)

    # rate threshold
    thr_rate = float(np.quantile(r, rate_q))
    thr_rate = max(thr_rate, float(rate_min))

    gate_rate = (r >= thr_rate) & (r >= min_events_bin)
    gate_bal  = (bal_abs <= float(balance_th)) & (r >= min_events_bin)

    mask = (gate_rate & gate_bal)

    # optional spatial uniformity gate
    ent_norm = None
    if use_spatial_gate:
        if x is None or y is None:
            raise ValueError("use_spatial_gate=True requires x and y")
        x = np.asarray(x, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)
        if x.size != t_s.size or y.size != t_s.size:
            raise ValueError("x,y must have same length as t_s")

        W = int(np.max(x)) + 1
        H = int(np.max(y)) + 1
        gx = np.clip((x.astype(np.float64) / max(W, 1) * grid_w).astype(np.int32), 0, grid_w - 1)
        gy = np.clip((y.astype(np.float64) / max(H, 1) * grid_h).astype(np.int32), 0, grid_h - 1)
        cell = gy * grid_w + gx
        G = grid_w * grid_h

        ent_norm = np.zeros((M,), dtype=np.float64)
        cand = np.where(mask)[0]
        eps = 1e-12
        for kk in cand:
            idx = (k == kk)
            if not np.any(idx):
                continue
            c = cell[idx]
            hist = np.bincount(c, minlength=G).astype(np.float64)
            s = hist.sum()
            if s <= 0:
                continue
            p = hist / s
            Hc = -np.sum(p * np.log(p + eps))
            ent_norm[kk] = float(Hc / np.log(G + eps))

        mask = mask & (ent_norm >= float(uniformity_ent_th))

    # optional smoothing
    if smooth_len >= 2:
        L = int(smooth_len)
        kernel = np.ones((L,), dtype=np.float64) / float(L)
        sm = np.convolve(mask.astype(np.float64), kernel, mode="same")
        mask = (sm >= 0.5)

    stats = {
        "bin_ms": float(bin_ms),
        "dt_s": float(dt_s),
        "num_bins": int(M),
        "t0": float(t0),
        "t1": float(t1),
        "rate_q": float(rate_q),
        "thr_rate": float(thr_rate),
        "balance_th": float(balance_th),
        "min_events_bin": int(min_events_bin),
        "num_trigger_bins": int(np.sum(mask)),
        "trigger_ratio": float(np.mean(mask)),
        "r": r,
        "bal_abs": bal_abs,
    }
    if use_spatial_gate:
        stats["use_spatial_gate"] = True
        stats["uniformity_ent_th"] = float(uniformity_ent_th)
        stats["ent_norm"] = ent_norm

    return mask, stats



# ------------------------------
# v2: meta(轻量) + extra(长通道)
# ------------------------------
from typing import Dict, Any, Optional, Tuple

def _is_small_meta_array(arr: np.ndarray, max_elems: int = 4096) -> bool:
    """用于判定一个ndarray是否可以放进meta（小数组/常量）。"""
    if arr.ndim == 0:
        return True
    return arr.size <= max_elems

def load_events_npz2(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                             Dict[str, Any], Dict[str, np.ndarray]]:
    """
    返回 meta(轻量) + extra(per-event长数组通道，如 w1/w2).
    规则：
      - t/x/y/p 固定为事件主通道
      - 其余字段：
          * 若是 ndarray 且 shape[0]==N 且 N>0 -> extra
          * 否则 -> meta（标量/小数组/字符串/字典等）
    """
    data = np.load(npz_path, allow_pickle=True)

    required = {"t", "x", "y", "p"}
    if not required.issubset(data.files):
        raise KeyError(f"NPZ must contain {required}, got {set(data.files)}")

    t = np.asarray(data["t"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.int32)
    y = np.asarray(data["y"], dtype=np.int32)
    p = np.asarray(data["p"], dtype=np.int8)

    if not (t.shape == x.shape == y.shape == p.shape):
        raise ValueError("Loaded t/x/y/p have inconsistent shapes")
    if not np.all(np.isin(p, [-1, 1])):
        raise ValueError("p must be in {-1, +1}")

    N = int(t.size)

    meta: Dict[str, Any] = {
        "path": npz_path,
        "num_events": N,
        "t_start": float(t[0]) if N > 0 else None,
        "t_end": float(t[-1]) if N > 0 else None,
    }
    extra: Dict[str, np.ndarray] = {}

    # 强制meta键（你认为应当属于常量/配置）
    META_KEYS = {"resolution", "time_unit", "src", "t_range", "count"}

    for key in data.files:
        if key in ("t", "x", "y", "p"):
            continue
        val = data[key]

        # numpy标量 -> python标量
        if isinstance(val, np.ndarray) and val.ndim == 0:
            meta[key] = val.item()
            continue

        # per-event 长通道：shape[0]==N
        if N > 0 and isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == N and key not in META_KEYS:
            extra[key] = val
            continue

        # 其余都进入meta（包括小数组）
        meta[key] = val

    # 统一 resolution/time_unit 的存在性与类型
    meta.setdefault("resolution", None)
    meta.setdefault("time_unit", None)
    if meta["resolution"] is not None:
        meta["resolution"] = np.asarray(meta["resolution"], dtype=np.int32).tolist()

    return t, x, y, p, meta, extra


def save_events_npz2(save_path: str,
                     t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray,
                     meta: Optional[Dict[str, Any]] = None,
                     extra: Optional[Dict[str, np.ndarray]] = None,
                     *,
                     default_resolution=(346, 260),
                     default_time_unit="seconds"):
    """
    meta: 轻量配置；extra: per-event长数组通道（w1/w2等）
    写入NPZ时：t/x/y/p + meta字段 + extra字段（都在NPZ顶层）
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p, dtype=np.int8)
    p_pm = np.where(p > 0, 1, -1).astype(np.int8)

    if not (t.shape == x.shape == y.shape == p_pm.shape):
        raise ValueError("t, x, y, p must have the same shape")

    save_dict: Dict[str, Any] = {"t": t, "x": x, "y": y, "p": p_pm}

    if meta is not None:
        for k, v in meta.items():
            if k in ("t", "x", "y", "p"):
                continue
            if k == "resolution" and v is not None:
                save_dict[k] = np.asarray(v, dtype=np.int32)
            else:
                save_dict[k] = v

    if extra is not None:
        N = int(t.size)
        for k, v in extra.items():
            arr = np.asarray(v)
            if arr.ndim == 0:
                raise ValueError(f"extra[{k}] must be array-like, got scalar")
            if arr.shape[0] != N:
                raise ValueError(f"extra[{k}] length mismatch: {arr.shape[0]} vs N={N}")
            save_dict[k] = arr

    # 兜底常量
    if "resolution" not in save_dict:
        save_dict["resolution"] = np.asarray(default_resolution, dtype=np.int32)
    if "time_unit" not in save_dict:
        save_dict["time_unit"] = default_time_unit

    np.savez_compressed(save_path, **save_dict)
    print(f"Saved {len(t)} events to {save_path}")

