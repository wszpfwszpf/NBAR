import os
import numpy as np
import matplotlib.pyplot as plt



# =========================================================
# User Configuration (PyCharm-friendly)
# =========================================================

NPZ_PATH = r"mvsec_clip_2s.npz"   # ← 改成你的 npz 路径

# Event camera resolution
W = 346
H = 260

# Output directory
OUT_DIR = "out_analysis"

# -------- Stage-1 analysis hyper-parameters --------
# BA analysis
BA_DT_MS = 10.0        # time bin (ms) for spatio-temporal support

# Flicker analysis
FLICKER_BIN_MS = 1.0   # time bin (ms) for event-rate signal

# Time normalization
NORMALIZE_TIME = True



# ---------------- Stage-1 Global Flicker Thinning ----------------
STAGE1_ENABLE = True
STAGE1_MODE = "global"          # 仅用于你自己标记

STAGE1_BIN_MS = 1.0             # 建议 1.0ms（与你的 flicker 分析一致）
STAGE1_BALANCE_TH = 0.35        # 先用 0.25；如果误杀少可降到 0.20
STAGE1_HIGH_Q = 80            # 80 分位作为“高事件率bin”
STAGE1_ALPHA_MIN = 0.35         # 最低保留比例（越大越保守）
STAGE1_RNG_SEED = 0
STAGE1_SAVE_NAME = "stage1_flicker_global_thinning.npz"

# -----------------------------
# 0) Robust loader for common event npz layouts
# -----------------------------

def _sum_neighbors_at_events(counts_flat, x, y, W, H, r=1):
    """
    counts_flat: shape (H*W,) uint32
    x,y: shape (M,)
    return: sum over (2r+1)x(2r+1) neighborhood for each event point
    """
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    M = x.shape[0]
    out = np.zeros(M, dtype=np.uint32)

    for dy in range(-r, r + 1):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < H)
        if not np.any(valid_y):
            continue

        for dx in range(-r, r + 1):
            xx = x + dx
            valid = valid_y & (xx >= 0) & (xx < W)
            if not np.any(valid):
                continue

            idx = (yy[valid] * W + xx[valid]).astype(np.int32)
            out[valid] += counts_flat[idx]

    return out


def filter_flicker_stage1_streaming(
    t, x, y, p01,  # p01 ∈ {0,1}; 1=pos, 0=neg
    W=346, H=260,
    dt_ms=2.0, r=1,
    n_th=18, bal_th=0.20, min_pol=3,
    verbose=True
):
    """
    Stage-1 Flicker coarse filter:
      Remove event e if:
        n = n_pos + n_neg >= n_th
        b = |n_pos - n_neg| / (n + eps) <= bal_th
        and min(n_pos, n_neg) >= min_pol
    Neighborhood defined on:
      - spatial: (2r+1)x(2r+1)
      - temporal: current bin only（保守版）
    Streaming per time bin to keep memory low.

    Returns:
      keep_mask (bool, N), stats dict
    """
    eps = 1e-6
    N = len(t)
    if N == 0:
        return np.zeros(0, dtype=bool), {"removed": 0, "kept": 0, "remove_ratio": 0.0}

    dt = dt_ms / 1000.0
    Tbins = int(np.ceil((t[-1] - t[0] + 1e-12) / dt))
    Tbins = max(1, Tbins)

    tb = np.minimum((t / dt).astype(np.int32), Tbins - 1)

    # 由于 t 已经按时间排序，tb 也基本非降序；做一个 bin -> [start,end) 索引表
    bin_starts = np.full(Tbins + 1, -1, dtype=np.int64)
    # 找每个 bin 的起点
    cur = 0
    for b in range(Tbins):
        # 前进到 tb==b 的第一个位置
        while cur < N and tb[cur] < b:
            cur += 1
        bin_starts[b] = cur
        # 前进到 tb>b
        while cur < N and tb[cur] == b:
            cur += 1
    bin_starts[Tbins] = N

    keep = np.ones(N, dtype=bool)

    HW = W * H
    removed = 0

    for b in range(Tbins):
        s = int(bin_starts[b])
        e = int(bin_starts[b + 1])
        if e <= s:
            continue

        xb = x[s:e]
        yb = y[s:e]
        pb = p01[s:e]

        pix = (yb * W + xb).astype(np.int32)

        # 当前 bin 的 pos/neg 像素计数
        counts_pos = np.bincount(pix[pb == 1], minlength=HW).astype(np.uint32)
        counts_neg = np.bincount(pix[pb == 0], minlength=HW).astype(np.uint32)

        # 在“事件点”上取邻域和（避免全图卷积，速度更稳）
        npos = _sum_neighbors_at_events(counts_pos, xb, yb, W, H, r=r).astype(np.float32)
        nneg = _sum_neighbors_at_events(counts_neg, xb, yb, W, H, r=r).astype(np.float32)

        n = npos + nneg
        bscore = np.abs(npos - nneg) / (n + eps)
        minpn = np.minimum(npos, nneg)

        # Flicker removal mask within this bin
        rm_local = (n >= n_th) & (bscore <= bal_th) & (minpn >= float(min_pol))

        keep[s:e] = ~rm_local
        removed += int(np.sum(rm_local))

        if verbose and (b % max(1, Tbins // 10) == 0):
            print(f"[Stage1] bin {b}/{Tbins}: events={e-s}, removed={int(np.sum(rm_local))}")

    kept = int(np.sum(keep))
    stats = {
        "dt_ms": float(dt_ms),
        "r": int(r),
        "n_th": int(n_th),
        "bal_th": float(bal_th),
        "min_pol": int(min_pol),
        "removed": int(removed),
        "kept": int(kept),
        "remove_ratio": float(removed) / float(N + 1e-12),
    }
    return keep, stats


def save_events_txpy_npz(save_path, t, x, y, p):
    """
    Save events in a clean, self-descriptive NPZ format.

    Fields:
        t : float64, seconds, shape (N,)
        x : int32, pixel x-coordinate
        y : int32, pixel y-coordinate
        p : int8, polarity in {-1, +1}
        resolution : [W, H]
        time_unit : string
    """
    t = t.astype(np.float64)
    x = x.astype(np.int32)
    y = y.astype(np.int32)

    # polarity: ensure {-1, +1}
    p_pm = np.where(p.astype(np.int8) > 0, 1, -1).astype(np.int8)

    np.savez_compressed(
        save_path,
        t=t,
        x=x,
        y=y,
        p=p_pm,
        resolution=np.array([346, 260], dtype=np.int32),
        time_unit="seconds"
    )



def load_events_from_npz(npz_path: str, W: int = 346, H: int = 260, normalize_time: bool = True):
    """
    Load events from .npz assuming txpy order by default:
      events: Nx4 with columns [t, x, p, y]
    - t is in seconds (float)
    - p is in {-1, +1}
    - resolution is W x H (default 346 x 260)

    Returns:
      t (float64, seconds, sorted), x (int32), y (int32), p01 (int8 in {0,1})
    """
    data = np.load(npz_path, allow_pickle=True)

    # Prefer explicit arrays if present
    keys = set(data.files)
    if {"t", "x", "y", "p"}.issubset(keys):
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.int32)
        y = np.asarray(data["y"], dtype=np.int32)
        p = np.asarray(data["p"])
        return normalize_events_txpy(t, x, y, p, W=W, H=H, normalize_time=normalize_time)

    # Otherwise read events matrix (txpy)
    if "events" not in keys:
        raise ValueError(f"Cannot find 'events' in {npz_path}. Available keys: {sorted(list(keys))}")

    ev = np.asarray(data["events"])
    if ev.ndim != 2 or ev.shape[1] < 4:
        raise ValueError(f"'events' must be Nx4 or Nx>=4, got shape={ev.shape}")

    ev = ev[:, :4]
    t = ev[:, 0].astype(np.float64)  # seconds
    x = ev[:, 1].astype(np.int32)
    p = ev[:, 2]                     # {-1, +1}
    y = ev[:, 3].astype(np.int32)

    return normalize_events_txpy(t, x, y, p, W=W, H=H, normalize_time=normalize_time)


def normalize_events_txpy(t, x, y, p, W: int, H: int, normalize_time: bool = True):
    """
    Normalize and validate:
      - sort by t
      - optionally shift t so that min(t)=0
      - p: {-1,+1} -> {0,1} (p01 = 1 if p==+1 else 0)
      - validate x,y are within [0,W-1]/[0,H-1] (clip is NOT applied; we raise)
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)
    p = np.asarray(p)

    # basic validity checks
    if t.ndim != 1 or x.ndim != 1 or y.ndim != 1 or p.ndim != 1:
        raise ValueError("t,x,y,p must be 1D arrays.")
    if not (len(t) == len(x) == len(y) == len(p)):
        raise ValueError(f"Length mismatch: len(t)={len(t)}, len(x)={len(x)}, len(y)={len(y)}, len(p)={len(p)}")

    # coordinate range check (strict)
    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())
    if xmin < 0 or xmax >= W or ymin < 0 or ymax >= H:
        raise ValueError(
            f"(x,y) out of bounds for resolution {W}x{H}. "
            f"x range [{xmin},{xmax}], y range [{ymin},{ymax}]"
        )

    # polarity: {-1,+1} -> {0,1}
    # if data accidentally contains {0,1}, we still handle it.
    uniq = np.unique(p[: min(len(p), 200000)])
    if set(uniq.tolist()).issubset({-1, 1}):
        p01 = (p > 0).astype(np.int8)
    else:
        # fallback: treat positive as 1, else 0
        p01 = (p > 0).astype(np.int8)

    # sort by time
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    y = y[order]
    p01 = p01[order]

    if normalize_time and len(t) > 0:
        t = t - float(t[0])

    return t, x, y, p01


# -----------------------------
# 1) Hot pixel dominance analysis
# -----------------------------
def analyze_hot_pixels(x, y, W, H, out_dir):
    """
    Compute:
      - per-pixel event counts
      - top-1% contribution ratio
      - Gini coefficient (inequality)
    Save heatmap.
    """
    N = x.shape[0]
    pix = y * W + x
    counts = np.bincount(pix, minlength=W * H).astype(np.int64)

    # top-1% contribution
    k = max(1, int(0.01 * counts.size))
    sorted_counts = np.sort(counts)
    top_sum = sorted_counts[-k:].sum()
    total = counts.sum()
    top1_ratio = float(top_sum) / float(total + 1e-12)

    # Gini coefficient
    # gini = sum_i sum_j |xi-xj| / (2*n*sum(x))
    # Efficient: using sorted x
    x_sorted = sorted_counts.astype(np.float64)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    if total <= 0:
        gini = 0.0
    else:
        gini = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

    # heatmap
    heat = counts.reshape(H, W)
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.imshow(heat, aspect="auto")
    plt.title("Per-pixel event count heatmap")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hot_pixel_heatmap.png"), dpi=200)
    plt.close()

    # simple dominance verdict
    # You can tune thresholds; these are practical defaults for night clips.
    hot_dominant = (top1_ratio > 0.25) or (gini > 0.85)

    return {
        "top1_ratio": top1_ratio,
        "gini": float(gini),
        "hot_dominant": bool(hot_dominant),
        "counts": counts,  # could be large; keep for further use if needed
    }


# -----------------------------
# 2) BA strength via spatio-temporal neighborhood support
# -----------------------------
def compute_spatiotemporal_support(t, x, y, W, H, dt_ms=10.0):
    """
    Approximate local support using voxelized counts:
      - time bins: dt_ms
      - voxel count per (bin, y, x)
      - compute 3x3 spatial neighborhood sum for each bin
      - support per event = sum over (bin-1, bin, bin+1) of 3x3 sums at (y,x) minus 1 (self)

    Returns:
      support_per_event (int32)
      stats dict
    Memory: (Tbins * H * W) uint16/uint32; dt_ms=10ms for 2s => 200 bins -> manageable
    """
    dt = dt_ms / 1000.0
    Tbins = int(np.ceil((t[-1] - t[0] + 1e-12) / dt))
    Tbins = max(1, min(Tbins, 10000))

    N = t.shape[0]
    tb = np.minimum((t / dt).astype(np.int32), Tbins - 1)

    # Build counts per bin as (Tbins, H, W) in a memory-friendly way
    # Use uint16 if safe; fall back to uint32 if too dense.
    max_events_per_pixel_bin = 0
    counts_tb = np.zeros((Tbins, H * W), dtype=np.uint16)

    pix = (y * W + x).astype(np.int32)

    # Fill per time bin
    for b in range(Tbins):
        mask = (tb == b)
        if not np.any(mask):
            continue
        c = np.bincount(pix[mask], minlength=H * W)
        m = int(c.max())
        if m > 65535:
            # upgrade dtype
            counts_tb = counts_tb.astype(np.uint32)
        counts_tb[b, :] = c.astype(counts_tb.dtype)
        max_events_per_pixel_bin = max(max_events_per_pixel_bin, m)

    # spatial 3x3 sum per bin using shifts (no scipy)
    counts_3x3 = np.zeros_like(counts_tb, dtype=np.uint32)

    # reshape view: (H,W)
    for b in range(Tbins):
        frame = counts_tb[b].reshape(H, W).astype(np.uint32)
        s = np.zeros_like(frame, dtype=np.uint32)
        # sum of 3x3 via 9 shifts
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys = slice(max(0, dy), H + min(0, dy))
                xs = slice(max(0, dx), W + min(0, dx))
                yt = slice(max(0, -dy), H - max(0, dy))
                xt = slice(max(0, -dx), W - max(0, dx))
                s[yt, xt] += frame[ys, xs]
        counts_3x3[b] = s.reshape(-1)

    # event support: sum across adjacent time bins at same (x,y) neighborhood
    idx = pix
    support = counts_3x3[tb, idx].astype(np.int32)

    # add tb-1 and tb+1
    tbm1 = np.clip(tb - 1, 0, Tbins - 1)
    tbp1 = np.clip(tb + 1, 0, Tbins - 1)
    support += counts_3x3[tbm1, idx].astype(np.int32)
    support += counts_3x3[tbp1, idx].astype(np.int32)

    # remove self-count roughly (self is included in the center bin 3x3 sum)
    support = np.maximum(support - 1, 0)

    return support, {
        "dt_ms": float(dt_ms),
        "Tbins": int(Tbins),
        "max_events_per_pixel_bin": int(max_events_per_pixel_bin),
    }


def analyze_ba(t, x, y, W, H, out_dir, dt_ms=10.0):
    """
    BA tends to create many low-support events (isolated in local spatio-temporal neighborhood).
    We compute support_per_event and report fractions below thresholds.
    """
    os.makedirs(out_dir, exist_ok=True)
    support, st = compute_spatiotemporal_support(t, x, y, W, H, dt_ms=dt_ms)

    # fractions of "isolated" events
    frac_le0 = float(np.mean(support <= 0))
    frac_le1 = float(np.mean(support <= 1))
    frac_le2 = float(np.mean(support <= 2))

    # histogram
    plt.figure()
    vmax = int(np.percentile(support, 99))
    plt.hist(np.clip(support, 0, vmax), bins=80)
    plt.title(f"Local spatio-temporal support histogram (clipped at p99={vmax})")
    plt.xlabel("support count")
    plt.ylabel("events")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ba_support_hist.png"), dpi=200)
    plt.close()

    # heuristic BA-strong verdict: too many isolated events
    # tuneable; for night scenes, >40% support<=1 usually indicates strong BA/isolated noise
    ba_strong = (frac_le1 > 0.40)

    return {
        "dt_ms": st["dt_ms"],
        "frac_support_le0": frac_le0,
        "frac_support_le1": frac_le1,
        "frac_support_le2": frac_le2,
        "ba_strong": bool(ba_strong),
    }


# -----------------------------
# 3) Flicker / unstable illumination analysis
# -----------------------------
def flicker_band_score(rate, fs, f_lo=40.0, f_hi=80.0, band_hz=6.0):
    """
    自适应主峰：在[f_lo,f_hi]内找最大功率频点f0，
    返回: band_power(f0±band_hz) / median_power(背景)
    """
    rate = rate.astype(np.float64)
    rate = rate - np.mean(rate)
    n = len(rate)
    if n < 64:
        return 0.0, 0.0

    fft = np.fft.rfft(rate)
    power = (np.abs(fft) ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # search band
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        return 0.0, 0.0

    # dominant peak in band
    idx_band = np.where(band)[0]
    k0 = idx_band[np.argmax(power[idx_band])]
    f0 = float(freqs[k0])

    # integrate neighborhood around f0
    nb = (freqs >= (f0 - band_hz)) & (freqs <= (f0 + band_hz))
    band_power = float(np.sum(power[nb]))

    # background median (exclude DC and very low freq)
    bg = power[(freqs >= 1.0)]
    bg_med = float(np.median(bg)) if bg.size > 0 else 1e-12

    score = band_power / (bg_med + 1e-12)
    return score, f0



def dominant_flicker_score(rate, fs, target_hz):
    """
    rate: 1D time series
    fs: sampling frequency (Hz)
    target_hz: e.g., 50 or 60
    Returns: ratio = power(target +/- 1Hz) / median(power)
    """
    rate = rate.astype(np.float64)
    rate = rate - np.mean(rate)
    n = len(rate)
    if n < 16:
        return 0.0

    # FFT
    fft = np.fft.rfft(rate)
    power = (np.abs(fft) ** 2)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # band around target
    band = (freqs >= (target_hz - 1.0)) & (freqs <= (target_hz + 1.0))
    peak = float(np.max(power[band])) if np.any(band) else 0.0
    med = float(np.median(power[1:])) if len(power) > 2 else 1e-12
    return peak / (med + 1e-12)


def dominant_flicker_score_improved(rate, fs, target_hz, bandwidth=1.0, min_len=16, eps=1e-12):
    """
    更稳健的频谱比值估计（纯 numpy 实现）：
      - 对 rate 去直流并乘 Hann 窗
      - 零填充到接近 1Hz 分辨率（当需要）
      - 计算 rfft 并用 power = |FFT|^2
      - band: target_hz +/- bandwidth
      - 背景使用 power 在 [1Hz, nyquist] 的中位数（排除直流）
    返回: ratio (float)
    """
    rate = np.asarray(rate, dtype=np.float64)
    n = len(rate)
    if n < min_len:
        return 0.0

    # 去直流
    rate = rate - np.mean(rate)

    # 窗函数
    win = np.hanning(n)
    rate_win = rate * win
    # 等效能量归一因子（用于 PSD 比较，可省略但更严谨）
    win_corr = np.sum(win**2)

    # 期望的频率分辨率，尝试使 df <= 0.5Hz for stable band selection
    desired_df = min(0.5, fs / max(1, n))
    nfft = int(2 ** int(np.ceil(np.log2(max(n, int(fs / desired_df))))))
    # 但不要小于 n
    nfft = max(nfft, n)

    fft = np.fft.rfft(rate_win, n=nfft)
    power = (np.abs(fft) ** 2) / (win_corr + eps)  # 相对 PSD（比例常数不影响比值）

    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    # 带宽选择
    band_mask = (freqs >= (target_hz - bandwidth)) & (freqs <= (target_hz + bandwidth))
    peak = float(np.max(power[band_mask])) if np.any(band_mask) else 0.0

    # 背景：排除直流（freq==0）和非常低频（例如 < 1Hz）
    bg_mask = (freqs >= 1.0)  # 可改为更合适的下限
    if np.sum(bg_mask) < 1:
        med = float(np.median(power[1:])) if len(power) > 2 else eps
    else:
        med = float(np.median(power[bg_mask]))

    return peak / (med + eps)


def analyze_flicker(t, p, out_dir, bin_ms=1.0):
    """
    Flicker often appears as periodic modulation in global event rate,
    with peaks near 50/60 Hz (or harmonics).
    We compute event rate time series and frequency scores at 50/60Hz.
    Also compute separately for polarity to see symmetric/asymmetric flicker.
    """
    os.makedirs(out_dir, exist_ok=True)

    dt = bin_ms / 1000.0
    Tbins = int(np.ceil((t[-1] - t[0] + 1e-12) / dt))
    Tbins = max(1, min(Tbins, 200000))  # 2s @1ms => 2000

    tb = np.minimum((t / dt).astype(np.int32), Tbins - 1)

    rate_all = np.bincount(tb, minlength=Tbins).astype(np.float64) / dt
    rate_pos = np.bincount(tb[p == 1], minlength=Tbins).astype(np.float64) / dt
    rate_neg = np.bincount(tb[p == 0], minlength=Tbins).astype(np.float64) / dt

    fs = 1.0 / dt

    s50_all = dominant_flicker_score(rate_all, fs, 50.0)
    s60_all = dominant_flicker_score(rate_all, fs, 60.0)
    s50_pos = dominant_flicker_score(rate_pos, fs, 50.0)
    s60_pos = dominant_flicker_score(rate_pos, fs, 60.0)
    s50_neg = dominant_flicker_score(rate_neg, fs, 50.0)
    s60_neg = dominant_flicker_score(rate_neg, fs, 60.0)

    score_all, f0_all = flicker_band_score(rate_all, fs, 40, 80, band_hz=6)
    score_pos, f0_pos = flicker_band_score(rate_pos, fs, 40, 80, band_hz=6)
    score_neg, f0_neg = flicker_band_score(rate_neg, fs, 40, 80, band_hz=6)

    print('score_all''f0_all', score_all, f0_all)
    print('score_pos''f0_pos', score_pos, f0_pos)
    print('score_neg''f0_neg', score_neg, f0_neg)



    # Plot rate curves (first 2s)
    time_axis = np.arange(Tbins) * dt
    plt.figure()
    plt.plot(time_axis, rate_all, label="all")
    plt.plot(time_axis, rate_pos, label="p=1")
    plt.plot(time_axis, rate_neg, label="p=0")
    plt.title(f"Event rate over time (bin={bin_ms}ms)")
    plt.xlabel("time (s)")
    plt.ylabel("events/s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "flicker_event_rate.png"), dpi=200)
    plt.close()

    # Heuristic verdict: if 50/60 peak >> median background, flicker likely
    # This threshold is empirical; adjust after you see real clips.
    flicker_score = max(s50_all, s60_all, s50_pos, s60_pos, s50_neg, s60_neg)
    flicker_strong = flicker_score > 8.0

    return {
        "bin_ms": float(bin_ms),
        "score50_all": float(s50_all),
        "score60_all": float(s60_all),
        "score50_pos": float(s50_pos),
        "score60_pos": float(s60_pos),
        "score50_neg": float(s50_neg),
        "score60_neg": float(s60_neg),
        "flicker_score": float(flicker_score),
        "flicker_strong": bool(flicker_strong),
    }

# python
def filter_flicker_stage1_global_thinning(
    t, x, y, p01,
    bin_ms=1.0,
    balance_th=0.25,     # 极性平衡阈值，越小越严格（建议 0.20~0.35）
    high_q=80,           # “高事件率bin”阈值：用分位数（建议 75~90）
    alpha_min=0.35,      # 最低保留比例，越大越保守（建议 0.30~0.60）
    rng_seed=0,
    verbose=True
):
    """
    Global flicker suppression by per-bin thinning:
      If bin has high rate AND polarity-balanced -> randomly keep with alpha_b.

    Returns:
      keep_mask (bool, N)
      stats dict
    """
    eps = 1e-6
    N = len(t)
    if N == 0:
        return np.zeros(0, dtype=bool), {}

    dt = bin_ms / 1000.0
    Tbins = int(np.ceil((t[-1] - t[0] + 1e-12) / dt))
    Tbins = max(1, Tbins)

    tb = np.minimum((t / dt).astype(np.int32), Tbins - 1)

    # per-bin counts
    Nb = np.bincount(tb, minlength=Tbins).astype(np.int32)
    Npos = np.bincount(tb[p01 == 1], minlength=Tbins).astype(np.int32)
    Nneg = Nb - Npos
    bal = np.abs(Npos - Nneg) / (Nb + eps)

    # baseline and high-rate threshold
    Nbase = float(np.median(Nb[Nb > 0])) if np.any(Nb > 0) else 0.0
    thr = float(np.percentile(Nb, high_q))

    # bins to suppress
    suppress_bins = (Nb >= thr) & (bal <= balance_th) & (Nb > 0)

    # keep probability per bin
    alpha = np.ones(Tbins, dtype=np.float32)
    # only suppress bins have alpha < 1
    idx = np.where(suppress_bins)[0]
    if len(idx) > 0 and Nbase > 0:
        alpha[idx] = np.clip(Nbase / (Nb[idx].astype(np.float32) + eps), alpha_min, 1.0)
    else:
        alpha[idx] = 1.0

    # sample keep mask
    rng = np.random.default_rng(rng_seed)
    u = rng.random(N).astype(np.float32)
    keep = u < alpha[tb]

    removed = int(np.sum(~keep))
    kept = int(np.sum(keep))

    if verbose:
        print("\n[Stage1-Global] baseline(median bin count) =", Nbase)
        print("[Stage1-Global] high-rate threshold (percentile) =", high_q, "=>", thr)
        print("[Stage1-Global] suppress_bins =", int(np.sum(suppress_bins)), "/", Tbins)
        if len(idx) > 0:
            print("[Stage1-Global] alpha (min/median/max on suppressed bins):",
                  float(np.min(alpha[idx])), float(np.median(alpha[idx])), float(np.max(alpha[idx])))
        print("[Stage1-Global] removed =", removed, " kept =", kept, " remove_ratio =", removed / (N + 1e-12))

    stats = {
        "bin_ms": float(bin_ms),
        "balance_th": float(balance_th),
        "high_q": int(high_q),
        "alpha_min": float(alpha_min),
        "baseline_median_bin": float(Nbase),
        "threshold_bin_count": float(thr),
        "suppressed_bins": int(np.sum(suppress_bins)),
        "removed": removed,
        "kept": kept,
        "remove_ratio": float(removed) / float(N + 1e-12),
    }
    return keep, stats


# -----------------------------
# Main
# -----------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load events (txpy, t in seconds, p in {-1,+1})
    t, x, y, p = load_events_from_npz(
        NPZ_PATH,
        W=W,
        H=H,
        normalize_time=NORMALIZE_TIME
    )

    # -----------------------------
    # Basic sanity stats
    # -----------------------------
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    print(f"[INFO] Loaded events: N={len(t):,}, duration≈{duration:.6f}s")
    print(f"[INFO] x range: [{x.min()}, {x.max()}], y range: [{y.min()}, {y.max()}]")
    print(f"[INFO] polarity unique values (after map): {np.unique(p)}")

    # -----------------------------
    # 1) Hot pixel dominance
    # -----------------------------
    hot = analyze_hot_pixels(x, y, W, H, OUT_DIR)
    print("\n=== Hot Pixel Dominance ===")
    print(f"Top-1% pixels contribution ratio: {hot['top1_ratio']:.4f}")
    print(f"Gini coefficient: {hot['gini']:.4f}")
    print(f"Verdict: hot_dominant = {hot['hot_dominant']}")

    # -----------------------------
    # 2) Background Activity (BA)
    # -----------------------------
    ba = analyze_ba(
        t, x, y,
        W, H,
        OUT_DIR,
        dt_ms=BA_DT_MS
    )
    print("\n=== Background Activity (BA) ===")
    print(f"Support dt: {ba['dt_ms']:.2f} ms")
    print(f"Fraction support<=0 : {ba['frac_support_le0']:.4f}")
    print(f"Fraction support<=1 : {ba['frac_support_le1']:.4f}")
    print(f"Fraction support<=2 : {ba['frac_support_le2']:.4f}")
    print(f"Verdict: ba_strong = {ba['ba_strong']}")

    # -----------------------------
    # 3) Flicker / unstable illumination
    # -----------------------------
    flick = analyze_flicker(
        t, p,
        OUT_DIR,
        bin_ms=FLICKER_BIN_MS
    )
    print("\n=== Flicker / Unstable Illumination ===")
    print(f"Rate bin: {flick['bin_ms']:.2f} ms")
    print(f"50Hz score (all/pos/neg): "
          f"{flick['score50_all']:.2f}, {flick['score50_pos']:.2f}, {flick['score50_neg']:.2f}")
    print(f"60Hz score (all/pos/neg): "
          f"{flick['score60_all']:.2f}, {flick['score60_pos']:.2f}, {flick['score60_neg']:.2f}")
    print(f"Overall flicker_score: {flick['flicker_score']:.2f}")
    print(f"Verdict: flicker_strong = {flick['flicker_strong']}")

    # -----------------------------
    # Stage-1 priority suggestion
    # -----------------------------
    print("\n=== Stage-1 Priority Suggestion ===")
    if hot["hot_dominant"]:
        print("→ 优先：Hot pixel 抑制（高置信、低成本、收益通常最大）")
    elif ba["ba_strong"]:
        print("→ 优先：时空邻域支持（抑制 BA/孤立事件，阈值从保守开始）")
    elif flick["flicker_strong"]:
        print("→ 优先：Flicker / 极性一致性约束（注意误杀，先做保守版本）")
    else:
        print("→ 未发现单一主导噪声：建议 Hot + 轻度邻域支持组合")

    print(f"\n[DONE] Results saved to: {os.path.abspath(OUT_DIR)}")


    # -----------------------------
    # Stage-1 Flicker coarse filter (optional)
    # -----------------------------
    if STAGE1_ENABLE:
        keep_mask, st1 = filter_flicker_stage1_global_thinning(
            t, x, y, p,
            bin_ms=STAGE1_BIN_MS,
            balance_th=STAGE1_BALANCE_TH,
            high_q=STAGE1_HIGH_Q,
            alpha_min=STAGE1_ALPHA_MIN,
            rng_seed=STAGE1_RNG_SEED,
            verbose=True
        )

        print("\n=== Stage-1 Global Flicker Thinning Stats ===")
        for k, v in st1.items():
            print(f"{k}: {v}")

        t1, x1, y1, p1 = t[keep_mask], x[keep_mask], y[keep_mask], p[keep_mask]

        save_path = os.path.join(OUT_DIR, STAGE1_SAVE_NAME)
        from utils.util import save_events_npz

        save_events_npz(
            save_path,
            t=t1,
            x=x1,
            y=y1,
            p=p1,
            resolution=(346, 260),
            time_unit="seconds"
        )

        # save_events_txpy_npz(save_path, t1, x1, y1, p1)
        print(f"[Stage1] Saved filtered events to: {save_path}")

        # 立刻闭环：看 60Hz score 是否下降
        flick1 = analyze_flicker(t1, p1, OUT_DIR, bin_ms=FLICKER_BIN_MS)
        print("\n=== Flicker After Stage-1 (Global) ===")
        print(
            f"50Hz score (all/pos/neg): {flick1['score50_all']:.2f}, {flick1['score50_pos']:.2f}, {flick1['score50_neg']:.2f}")
        print(
            f"60Hz score (all/pos/neg): {flick1['score60_all']:.2f}, {flick1['score60_pos']:.2f}, {flick1['score60_neg']:.2f}")
        print(f"Overall flicker_score: {flick1['flicker_score']:.2f}")
        print(f"Verdict: flicker_strong = {flick1['flicker_strong']}")


if __name__ == "__main__":
    main()






