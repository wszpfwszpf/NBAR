import numpy as np
import os
from utils.util import load_events_npz, save_events_npz
def dwf_denoise_npz(npz_path: str, L: int = 350, sigma: int = 3):
    """
    Double Window Filter (DWF) - 简化版本

    参数（根据论文6.2节）：
    - L: 窗口长度（论文推荐350）
    - sigma: 距离阈值（论文推荐3像素）

    返回：
    降噪后的事件保存路径
    """

    # ============================================================
    # 1. 加载事件（使用规范化函数）
    # ============================================================
    t, x, y, p, original_meta = load_events_npz(npz_path)

    N = t.size
    if N == 0:
        raise ValueError("Empty event stream")

    print(f"[DWF] Loading {N} events...")

    # ============================================================
    # 2. 初始化DWF窗口（双窗口：信号窗口 + 噪声窗口）
    # ============================================================
    L_half = L // 2

    # 信号窗口（存储预测为信号的事件）
    signal_window_x = np.zeros(L_half, dtype=np.uint16)
    signal_window_y = np.zeros(L_half, dtype=np.uint16)
    signal_window_ptr = 0  # 当前写入位置

    # 噪声窗口（存储预测为噪声的事件）
    noise_window_x = np.zeros(L_half, dtype=np.uint16)
    noise_window_y = np.zeros(L_half, dtype=np.uint16)
    noise_window_ptr = 0  # 当前写入位置

    # 用于存储事件是否通过DWF过滤
    keep_mask = np.zeros(N, dtype=bool)

    # ============================================================
    # 3. 应用DWF过滤（高效实现）
    # ============================================================
    for i in range(N):
        xi, yi = x[i], y[i]

        # 检查信号窗口
        min_dist_signal = float('inf')
        if signal_window_ptr > 0:  # 信号窗口非空时才检查
            # 曼哈顿距离计算
            dists = np.abs(xi - signal_window_x[:signal_window_ptr]) + \
                    np.abs(yi - signal_window_y[:signal_window_ptr])
            min_dist_signal = np.min(dists) if len(dists) > 0 else float('inf')

        # 检查噪声窗口
        min_dist_noise = float('inf')
        if noise_window_ptr > 0:  # 噪声窗口非空时才检查
            # 曼哈顿距离计算
            dists = np.abs(xi - noise_window_x[:noise_window_ptr]) + \
                    np.abs(yi - noise_window_y[:noise_window_ptr])
            min_dist_noise = np.min(dists) if len(dists) > 0 else float('inf')

        # DWF决策规则：曼哈顿距离 < sigma
        min_dist = min(min_dist_signal, min_dist_noise)
        is_signal = min_dist < sigma

        keep_mask[i] = is_signal

        # 根据分类结果更新相应窗口
        if is_signal:
            # 事件被分类为信号，更新信号窗口
            signal_window_x[signal_window_ptr] = xi
            signal_window_y[signal_window_ptr] = yi
            signal_window_ptr = (signal_window_ptr + 1) % L_half
        else:
            # 事件被分类为噪声，更新噪声窗口
            noise_window_x[noise_window_ptr] = xi
            noise_window_y[noise_window_ptr] = yi
            noise_window_ptr = (noise_window_ptr + 1) % L_half

        # 进度显示
        if i % 100000 == 0 and i > 0:
            print(f"[DWF] Processed {i}/{N} events...")

    # ============================================================
    # 4. 统计信息
    # ============================================================
    kept = int(np.sum(keep_mask))
    removed = N - kept
    remove_ratio = removed / N

    print(f"[DWF] total={N}, kept={kept}, removed={removed}, remove_ratio={remove_ratio:.4f}")

    # ============================================================
    # 5. 应用掩码获取降噪后事件
    # ============================================================
    t_denoised = t[keep_mask]
    x_denoised = x[keep_mask]
    y_denoised = y[keep_mask]
    p_denoised = p[keep_mask]

    # ============================================================
    # 6. 创建新元数据并保存结果
    # ============================================================
    base, _ = os.path.splitext(npz_path)
    out_path = base + f"_DWF_denoised.npz"

    # 创建新的元数据：复制原始元数据并添加DWF信息
    new_meta = original_meta.copy()

    # 添加DWF处理信息
    new_meta["denoised_by"] = "DWF"
    new_meta["DWF_parameters"] = {
        "window_size": L,
        "sigma_distance": sigma,
        "signal_window_size": L_half,
        "noise_window_size": L_half
    }
    new_meta["processing_stats"] = {
        "original_total_events": N,
        "denoised_total_events": kept,
        "removed_events": removed,
        "remove_ratio": remove_ratio,
        "keep_ratio": kept / N if N > 0 else 0
    }

    # 保存文件（使用规范化函数）
    save_events_npz(out_path, t_denoised, x_denoised, y_denoised, p_denoised, meta=new_meta)

    print(f"[DWF] Saved to {out_path}")

    return out_path