# compare_stage1.py
# -*- coding: utf-8 -*-

import os
import numpy as np

from utils.util import compute_esr, compute_aocc, plot_ccc, load_events_npz



def main():
    # ============================
    # 超参（只改这里）
    # ============================
    STAGE = "s1"
    RAW_NPZ = "data/mvsec_clip_2s.npz"
    STAGE1_NPZ = f"outputs/{STAGE}/mvsec_clip_2s_{STAGE}.npz"

    OUT_DIR = os.path.join("outputs", "compare")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 传感器分辨率
    W, H = 346, 260

    # ESR 的 M：必须固定（所有方法同一个 M），否则会被挑毛病
    # 你可以选一个常数，例如 1e5 / 2e5。建议在整篇实验里固定不变。
    ESR_M = 100000

    # AOCC 的 Δt 范围（论文 CCC 曲线横轴），单位 ms
    # 你也可以改成 1~50 或 1~33 等，但要所有方法一致
    DT_LIST_MS = list(np.arange(1, 41, 1))  # 1ms~40ms

    # AOCC 计算时是否只取共同时间区间（更公平）
    # True：用 raw 和 stage1 的 [max(t0), min(t1)] 作为共同区间
    USE_COMMON_TIME_RANGE = True

    # ============================
    # Load
    # ============================
    t0, x0, y0, p0, meta0= load_events_npz(RAW_NPZ)
    t1, x1, y1, p1, meta1= load_events_npz(STAGE1_NPZ)

    print(f"[INFO] RAW    : N={len(t0):,}, t=[{t0.min():.6f}, {t0.max():.6f}]")
    print(f"[INFO] STAGE1 : N={len(t1):,}, t=[{t1.min():.6f}, {t1.max():.6f}]")

    # ============================
    # ESR
    # ============================
    esr_raw, esr_info_raw = compute_esr(t0, x0, y0, p=p0, width=W, height=H, M=ESR_M)
    esr_s1,  esr_info_s1  = compute_esr(t1, x1, y1, p=p1, width=W, height=H, M=ESR_M)

    # ============================
    # AOCC
    # ============================
    if USE_COMMON_TIME_RANGE:
        t_start = max(float(t0.min()), float(t1.min()))
        t_end   = min(float(t0.max()), float(t1.max()))
    else:
        t_start, t_end = None, None

    aocc_raw, dt_ms_raw, cavg_raw, info_raw = compute_aocc(
        t0, x0, y0, width=W, height=H, dt_list_ms=DT_LIST_MS, t_start=t_start, t_end=t_end
    )
    aocc_s1, dt_ms_s1, cavg_s1, info_s1 = compute_aocc(
        t1, x1, y1, width=W, height=H, dt_list_ms=DT_LIST_MS, t_start=t_start, t_end=t_end
    )

    # ============================
    # Save CCC plots
    # ============================
    fig1 = plot_ccc(dt_ms_raw, cavg_raw, save_path=os.path.join(OUT_DIR, "ccc_raw.png"),
                    title=f"CCC RAW | AOCC={aocc_raw:.4f}")
    fig2 = plot_ccc(dt_ms_s1, cavg_s1, save_path=os.path.join(OUT_DIR, "ccc_stage1.png"),
                    title=f"CCC Stage-1 | AOCC={aocc_s1:.4f}")

    # 对比图（同一张图上）
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(dt_ms_raw, cavg_raw, label="RAW")
    plt.plot(dt_ms_s1,  cavg_s1,  label="Stage-1")
    plt.xlabel("Δt (ms)")
    plt.ylabel("Cavg(Δt)")
    plt.title("CCC Comparison (RAW vs Stage-1)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUT_DIR, "ccc_compare.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # ============================
    # Print & Save metrics
    # ============================
    lines = []
    lines.append("=== Settings ===")
    lines.append(f"W,H = {W},{H}")
    lines.append(f"ESR_M = {ESR_M}")
    lines.append(f"DT_LIST_MS = [{DT_LIST_MS[0]}..{DT_LIST_MS[-1]}], step={DT_LIST_MS[1]-DT_LIST_MS[0]}ms")
    lines.append(f"USE_COMMON_TIME_RANGE = {USE_COMMON_TIME_RANGE}")
    if USE_COMMON_TIME_RANGE:
        lines.append(f"COMMON_RANGE = [{t_start:.6f}, {t_end:.6f}] (s)")
    lines.append("")

    lines.append("=== ESR (E-MLB) ===")
    lines.append(f"RAW    ESR = {esr_raw:.6f} | NTSS={esr_info_raw['NTSS']:.6f} | LN={esr_info_raw['LN']:.6f} | N={esr_info_raw['N']}")
    lines.append(f"Stage1 ESR = {esr_s1 :.6f} | NTSS={esr_info_s1 ['NTSS']:.6f} | LN={esr_info_s1 ['LN']:.6f} | N={esr_info_s1 ['N']}")
    lines.append("")

    lines.append("=== AOCC (paper) ===")
    lines.append(f"RAW    AOCC = {aocc_raw:.6f} | T={info_raw['T']:.6f}s")
    lines.append(f"Stage1 AOCC = {aocc_s1 :.6f} | T={info_s1 ['T']:.6f}s")
    lines.append("")

    text = "\n".join(lines)
    print("\n" + text)

    with open(os.path.join(OUT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[DONE] Saved to: {OUT_DIR}")
    print("       - ccc_raw.png")
    print("       - ccc_stage1.png")
    print("       - ccc_compare.png")
    print("       - metrics.txt")


if __name__ == "__main__":
    main()

