# sanity_check_s1.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.util import load_events_npz2
from utils.s1_global_sync import s1_compute_w1
from utils.config import RESOLUTION, S1 as S1_DEFAULT


# =========================
# PyCharm 配置区：只改这里
# =========================
STAGE = "s1"
INPUT_NPZ = rf"outputs/{STAGE}/mvsec_clip_2s_{STAGE}.npz"
OUT_DIR = r"outputs/check_s1_output"

MAX_SECONDS = 0.5       # None = 全部
PREVIEW_ETA = 0.85          # 仅预览 harden 保留率
OVERRIDE = None             # 例如 {"lam": 0.4, "alpha": 6.0}


def _fft_peak_near(freqs, amp, target_hz=120.0, tol_hz=5.0):
    m = (freqs >= target_hz - tol_hz) & (freqs <= target_hz + tol_hz)
    if not np.any(m):
        return None
    idx = np.argmax(amp[m])
    return float(freqs[m][idx]), float(amp[m][idx])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t, x, y, p, meta, extra = load_events_npz2(INPUT_NPZ)
    if t.size == 0:
        print("Empty event stream.")
        return

    # clip
    if MAX_SECONDS is not None:
        t0 = float(t[0])
        m = (t - t0) <= float(MAX_SECONDS)
        t, x, y, p = t[m], x[m], y[m], p[m]
        if "w1" in extra:
            extra["w1"] = np.asarray(extra["w1"])[m]

    # 从 config 取默认参数 + override
    params = {
        "dt_s": S1_DEFAULT.dt_s,
        "Gx": S1_DEFAULT.Gx,
        "Gy": S1_DEFAULT.Gy,
        "rho": S1_DEFAULT.rho,
        "alpha": S1_DEFAULT.alpha,
        "z_th": S1_DEFAULT.z_th,
        "lam": S1_DEFAULT.lam,
        "normalize_t0": S1_DEFAULT.normalize_t0,
    }
    if OVERRIDE:
        params.update(OVERRIDE)

    W, H = RESOLUTION

    # 复算（保证一致）
    w1_re, aux = s1_compute_w1(
        t, x, y,
        W=W, H=H,
        dt_s=params["dt_s"],
        Gx=params["Gx"], Gy=params["Gy"],
        rho=params["rho"],
        alpha=params["alpha"],
        z_th=params["z_th"],
        lam=params["lam"],
        freq_prior_gk=None,
        normalize_t0=params["normalize_t0"],
    )

    # 文件内 w1 对齐检查
    w1_file = extra.get("w1", None)
    if w1_file is not None:
        w1_file = np.asarray(w1_file, dtype=np.float32)
        if w1_file.shape == w1_re.shape:
            mae = float(np.mean(np.abs(w1_file - w1_re)))
            print(f"[Check] w1(file) vs w1(recompute) MAE = {mae:.6f}")
        else:
            print(f"[Warn] w1(file) shape {w1_file.shape} != recompute {w1_re.shape}")

    # bin级序列：这里直接用 aux（短 clip 通常 num_bins 不大；若你想更稳，可以强制 clip 更短）
    Ck = aux.get("Ck", None)
    Sk = aux.get("Sk", None)
    zk = aux.get("zk", None)

    print("\n==== S1 Sanity Report ====")
    print(f"File: {INPUT_NPZ}")
    print(f"Events used: {t.size}")
    print("Params:", params)
    print(f"w1 mean/std/min/p5/p50/p95/max: "
          f"{w1_re.mean():.4f}/{w1_re.std():.4f}/{w1_re.min():.4f}/"
          f"{np.quantile(w1_re,0.05):.4f}/{np.quantile(w1_re,0.50):.4f}/{np.quantile(w1_re,0.95):.4f}/{w1_re.max():.4f}")

    keep_ratio = float(np.mean(w1_re >= PREVIEW_ETA))
    print(f"[Preview harden] eta={PREVIEW_ETA:.3f} keep_ratio={keep_ratio*100:.2f}%")

    # w1 直方图
    plt.figure(figsize=(8, 4))
    plt.hist(w1_re, bins=60, alpha=0.8)
    plt.xlabel("w1")
    plt.ylabel("count")
    plt.title("S1: w1 distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "w1_hist.png"), dpi=150)
    plt.close()

    # bin曲线 + FFT（如果有）
    if Ck is not None and Sk is not None and zk is not None:
        Ck = np.asarray(Ck, dtype=np.float32)
        Sk = np.asarray(Sk, dtype=np.float32)
        zk = np.asarray(zk, dtype=np.float32)

        tb = (np.arange(Ck.shape[0], dtype=np.float32) + 0.5) * params["dt_s"]

        plt.figure(figsize=(12, 7))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(tb, Ck, linewidth=1)
        ax1.set_ylabel("Ck")
        ax1.set_title("S1: Ck / zk / Sk over time (bin-level)")
        ax1.grid(alpha=0.3)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(tb, zk, linewidth=1)
        ax2.axhline(y=params["z_th"], linestyle="--", linewidth=1)
        ax2.set_ylabel("zk")
        ax2.grid(alpha=0.3)

        ax3 = plt.subplot(3, 1, 3, sharex=ax1)
        ax3.plot(tb, Sk, linewidth=1)
        ax3.set_xlabel("time (s)")
        ax3.set_ylabel("Sk")
        ax3.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ck_zk_sk_timeseries.png"), dpi=150)
        plt.close()

        ck = Ck - Ck.mean()
        win = np.hanning(len(ck)).astype(np.float32)
        amp = np.abs(np.fft.rfft(ck * win))
        freqs = np.fft.rfftfreq(len(ck), d=params["dt_s"])

        m = freqs <= 300.0
        peak = _fft_peak_near(freqs[m], amp[m], 120.0, 5.0)
        if peak is not None:
            pf, pa = peak
            print(f"[FFT] peak near 120Hz: f={pf:.2f}Hz, amp={pa:.4e}")
        else:
            print("[FFT] no peak near 120Hz (maybe clip too short or no flicker segment)")

        plt.figure(figsize=(10, 4))
        plt.plot(freqs[m], amp[m], linewidth=1)
        plt.xlabel("frequency (Hz)")
        plt.ylabel("|FFT(Ck)|")
        plt.title("S1: FFT of Ck (0~300Hz)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "ck_fft_0_300Hz.png"), dpi=150)
        plt.close()
    else:
        print("[Info] aux does not contain Ck/Sk/zk. 建议：把 MAX_SECONDS 设小一点（例如 2~5s）再跑。")

    print(f"\nSaved sanity-check figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
