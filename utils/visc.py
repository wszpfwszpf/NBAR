import os
import numpy as np
import matplotlib.pyplot as plt

from utils.util import load_events_npz2  # 你现有的 load_events_npz2


def _infer_resolution(meta: dict):
    """优先用 meta['resolution']，否则尝试从 config.RESOLUTION 读取。"""
    res = meta.get("resolution", None) if meta else None
    if res is not None:
        W, H = int(res[0]), int(res[1])
        return W, H
    try:
        from utils.config import RESOLUTION
        W, H = int(RESOLUTION[0]), int(RESOLUTION[1])
        return W, H
    except Exception as e:
        raise ValueError("Cannot infer resolution. Please provide meta['resolution'] or utils.config.RESOLUTION") from e


def _events_to_rgb_image(x, y, p, W, H):
    """
    把事件栅格化成 RGB 图（白底）。
    正极性：红；负极性：蓝。
    注意：同一像素多次命中时“后写覆盖先写”，用于可视化足够。
    """
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    if x.size == 0:
        return img

    x = x.astype(np.int32, copy=False)
    y = y.astype(np.int32, copy=False)
    p = p.astype(np.int8, copy=False)

    m = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    if not np.all(m):
        x, y, p = x[m], y[m], p[m]

    pos = (p > 0)
    neg = (p < 0)

    # red for positive
    img[y[pos], x[pos], 0] = 255
    img[y[pos], x[pos], 1] = 0
    img[y[pos], x[pos], 2] = 0

    # blue for negative
    img[y[neg], x[neg], 0] = 0
    img[y[neg], x[neg], 1] = 0
    img[y[neg], x[neg], 2] = 255

    return img


def vis_s2_c2_grid(
    npz_path: str,
    *,
    c2_key: str = "c2",
    thresholds=(0, 1, 2, 3),
    win_s: float = 0.033,     # 33ms 分帧
    dpi: int = 160,
    max_frames: int | None = None,  # 可选：限制输出帧数，None 表示全输出
):
    """
    针对 S2 处理后的数据（extra 内含 c2），按 33ms 分帧可视化，每帧输出一张 3x3 大图：

      [0] RAW(该帧内全部事件)
      [1] KEEP(k=0)  [2] REMOV(k=0)
      [3] KEEP(k=1)  [4] REMOV(k=1)
      [5] KEEP(k=2)  [6] REMOV(k=2)
      [7] KEEP(k=3)  [8] REMOV(k=3)

    其中：
      keep:   c2 > k
      remov:  c2 <= k
      retain = keep_mask.mean()  (在该帧内统计)

    仅保存合成大图，不保存子图单图。

    输出目录：
      输入文件同级目录 / <输入文件名>_vis_c/
      - grid_3x3_f0000.png, grid_3x3_f0001.png, ...
    """

    # ---------- 1) Load ----------
    t, x, y, p, meta, extra = load_events_npz2(npz_path)
    if c2_key not in extra:
        raise KeyError(f"'{c2_key}' not found in extra channels. extra keys={list(extra.keys())}")

    c2 = np.asarray(extra[c2_key]).reshape(-1)
    if c2.shape[0] != x.shape[0]:
        raise ValueError(f"c2 length mismatch: len(c2)={c2.shape[0]} vs N={x.shape[0]}")

    W, H = _infer_resolution(meta)

    # ---------- 2) Output dir ----------
    in_dir = os.path.dirname(os.path.abspath(npz_path))
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    out_dir = os.path.join(in_dir, f"{stem}_vis_c")
    os.makedirs(out_dir, exist_ok=True)

    if t.size == 0:
        raise ValueError("Empty event stream.")

    # ---------- 3) Frame slicing ----------
    t0 = float(t[0])
    t1 = float(t[-1])
    total_s = max(t1 - t0, 0.0)

    # 至少输出 1 帧
    n_frames = int(np.ceil(total_s / float(win_s))) if total_s > 0 else 1
    if max_frames is not None:
        n_frames = min(n_frames, int(max_frames))

    grid_paths = []

    # 预先把 t 转 float64，避免反复转换
    t = t.astype(np.float64, copy=False)

    for fi in range(n_frames):
        start = t0 + fi * win_s
        end = start + win_s

        frame_mask = (t >= start) & (t < end)
        xf = x[frame_mask]
        yf = y[frame_mask]
        pf = p[frame_mask]
        c2f = c2[frame_mask]

        # ---------- 4) Build 9 panels (title, image) ----------
        panels = []

        # raw
        img_raw = _events_to_rgb_image(xf, yf, pf, W, H)
        panels.append((f"RAW  [{start - t0:.3f}s,{end - t0:.3f}s)\nN={xf.size}", img_raw))

        # thresholds
        for k in thresholds:
            keep_mask = (c2f > k)
            rem_mask = ~keep_mask

            retain = float(keep_mask.mean()) if c2f.size > 0 else 0.0

            img_keep = _events_to_rgb_image(xf[keep_mask], yf[keep_mask], pf[keep_mask], W, H)
            img_rem = _events_to_rgb_image(xf[rem_mask], yf[rem_mask], pf[rem_mask], W, H)

            panels.append((f"KEEP  (c2 > {k})\nretain={retain:.3f}", img_keep))
            panels.append((f"REMOV (c2 <= {k})\nretain={retain:.3f}", img_rem))

        # ---------- 5) Draw 3x3 grid ----------
        n = len(panels)
        cols = 3
        rows = int(np.ceil(n / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.3), dpi=dpi)
        axes = np.array(axes).reshape(rows, cols)

        for i in range(rows * cols):
            rr = i // cols
            cc = i % cols
            ax = axes[rr, cc]
            ax.axis("off")

            if i < n:
                title, img = panels[i]
                ax.imshow(img)
                ax.set_title(title, fontsize=10)

                # 黑色边框
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.0)
            else:
                ax.set_visible(False)

        # 子图间隙
        plt.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.94,
            bottom=0.06,
            wspace=0.12,
            hspace=0.18,
        )

        grid_path = os.path.join(out_dir, f"grid_3x3_f{fi:04d}.png")
        plt.savefig(grid_path, dpi=dpi)
        plt.close(fig)

        grid_paths.append(grid_path)

    return out_dir, grid_paths
