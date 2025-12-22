import os
import time
import numpy as np

from utils.util import load_events_npz2, save_events_npz2
from utils.config import get_meta_base



def select_events_by_tag(
    t, x, y, p,
    extra,          # npz 里 extra 字段的 dict（含 w1 / c2）
    tag: str,
    sel_cfg: dict | None = None,
):
    """
    根据 tag + sel_cfg 生成事件筛选 mask
    不改原数组，只返回筛选后的视图和统计信息
    """
    N = t.shape[0]
    sel_cfg = sel_cfg or {}

    # 默认：全保留（raw）
    mask = np.ones(N, dtype=bool)

    stats = {
        "tag": tag,
        "N_in": int(N),
        "N_out": int(N),
    }

    if tag == "raw":
        return t, x, y, p, mask, stats

    if tag in ("s1", "fuseand", "fuseor"):
        if "w1" not in extra:
            raise KeyError("tag requires field 'w1', but not found in npz")
        eta = sel_cfg.get("eta", None)
        if eta is None:
            raise ValueError("sel_cfg['eta'] must be provided for s1/fuse")
        w1 = extra["w1"]
        m1 = (w1 >= eta)          # ★关键：w1 越大越可信
    else:
        m1 = None

    if tag in ("s2", "fuseand", "fuseor"):
        if "c2" not in extra:
            raise KeyError("tag requires field 'c2', but not found in npz")
        k = sel_cfg.get("k", None)
        if k is None:
            raise ValueError("sel_cfg['k'] must be provided for s2/fuse")
        c2 = extra["c2"]
        m2 = (c2 >= k)
    else:
        m2 = None

    if tag == "s1":
        mask = m1
    elif tag == "s2":
        mask = m2
    elif tag == "fuseand":
        mask = m1 & m2
    elif tag == "fuseor":
        mask = m1 | m2
    else:
        raise ValueError(f"Unknown tag: {tag}")

    stats["N_out"] = int(mask.sum())
    stats["keep_ratio"] = float(mask.mean())

    return t[mask], x[mask], y[mask], p[mask], mask, stats


# =========================
# PF 核心计算（占位）
# =========================


def pf_compute_flow(t, x, y, p, cfg):
    """
    事件级 PF 光流（SVD 平面拟合）
    - 因果邻域：只用过去事件
    - 输出与事件一一对齐

    Args:
        t: float array, shape (N,)  (单位：秒，且已标准化从 0 开始；最好已排序)
        x,y: int arrays, shape (N,)
        p: polarity (本实现不使用，可保留接口)
        cfg: dict with keys:
            r (int): 空间半径
            Tn_ms (float): 时间窗，毫秒
            Nmin (int): 最小邻域事件数（建议 4）
            eps_c (float): |c| 稳定性阈值（建议 1e-3）
    Returns:
        u, v: float32 (N,)
        v_valid: uint8 (N,)
        pf_res: float32 (N,)   # 最小奇异值
        pf_c: float32 (N,)
    """
    r = int(cfg.get("r", 2))
    Tn_ms = float(cfg.get("Tn_ms", 3.0))
    Nmin = int(cfg.get("Nmin", 4))
    eps_c = float(cfg.get("eps_c", 1e-3))

    Tn = Tn_ms / 1000.0  # seconds

    N = len(t)
    u = np.zeros(N, dtype=np.float32)
    v = np.zeros(N, dtype=np.float32)
    v_valid = np.zeros(N, dtype=np.uint8)
    pf_res = np.full(N, np.nan, dtype=np.float32)
    pf_c = np.full(N, np.nan, dtype=np.float32)

    if N == 0:
        return u, v, v_valid, pf_res, pf_c

    cnt_total = 0
    cnt_nmin_fail = 0
    cnt_c_fail = 0

    # --------- 保守检查：确保按时间排序（不强制，但建议）---------
    # 若你已保证输入有序，可以注释掉这一段
    if not np.all(t[1:] >= t[:-1]):
        order = np.argsort(t)
        t = t[order]
        x = x[order]
        y = y[order]
        p = p[order]
        # 注意：如果你在外层需要保持原顺序，请不要在这里排序；
        # 更推荐在 load 阶段就保证排序。

    # --------- 双指针：维护过去 Tn 秒的时间窗口 [i0, i) ---------
    i0 = 0

    # 为了减少重复创建临时数组，这里不做更深优化，先跑通
    for i in range(N):

        cnt_total += 1
        ti = t[i]

        # 移动左指针，保证 t[i0] > ti - Tn（即窗口内都是过去 Tn 的事件）
        t_min = ti - Tn
        while i0 < i and t[i0] < t_min:
            i0 += 1

        # 过去窗口事件索引范围 [i0, i)
        # 若窗口太小，直接跳过（先判断数量，再拟合）
        # 这里“数量”还没做空间过滤，所以只作为快速剪枝
        if (i - i0) < Nmin:
            continue

        # 空间过滤：只保留在 (xi, yi) 的 r 邻域内的事件
        xi = x[i]
        yi = y[i]

        idx = np.arange(i0, i, dtype=np.int32)
        dx = x[idx] - xi
        dy = y[idx] - yi
        msk = (dx >= -r) & (dx <= r) & (dy >= -r) & (dy <= r)
        idx = idx[msk]

        # 关键：空间过滤后再判断邻域数量
        if idx.size < Nmin:
            cnt_nmin_fail += 1
            continue

        # 组装点云：P = [x, y, t]，为数值稳定建议中心化
        X = x[idx].astype(np.float64)
        Y = y[idx].astype(np.float64)
        TT = t[idx].astype(np.float64)

        P = np.stack([X, Y, TT], axis=1)  # (M, 3)
        mu = P.mean(axis=0, keepdims=True)
        Pc = P - mu

        # SVD：Pc = U S Vt，最小奇异向量对应法向量
        # full_matrices=False 更快
        try:
            _, S, Vt = np.linalg.svd(Pc, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        n = Vt[-1, :]  # (a, b, c) up to scale
        a, b, c = n[0], n[1], n[2]

        pf_res[i] = float(S[-1])
        pf_c[i] = float(c)

        # 有效性判定：|c| 足够大，避免 u,v 爆炸
        if abs(c) <= eps_c:
            cnt_c_fail += 1
            continue

        # 光流解析解
        u[i] = float(-a / c)
        v[i] = float(-b / c)
        v_valid[i] = 1

    stats = {
        "cnt_total": cnt_total,
        "cnt_nmin_fail": cnt_nmin_fail,
        "cnt_c_fail": cnt_c_fail,
    }
    return u, v, v_valid, pf_res, pf_c, stats




# =========================
# PF 主入口（像 s1_global_sync）
# =========================
# def pf_run_npz(
#     in_npz: str,
#     out_dir: str,
#     tag: str = "",
#     cfg: dict | None = None,
# ):
#     os.makedirs(out_dir, exist_ok=True)
#
#     t0 = time.time()
#     t, x, y, p, meta, extra = load_events_npz2(in_npz)
#
#     if cfg is None:
#         cfg = {
#             "r": 2,
#             "Tn_ms": 3.0,
#             "Nmin": 4,
#             "eps_c": 1e-3,
#         }
#
#     u, v, v_valid, pf_res, pf_c, pf_dbg = pf_compute_flow(t, x, y, p, cfg)
#
#     # -------- meta --------
#     meta_out = get_meta_base()
#     meta_out.update(meta)   # 继承原 meta
#     meta_out["pf_cfg"] = cfg
#     meta_out["pf_stats"] = {
#         "valid_ratio": float(np.mean(v_valid)),
#         "mean_residual": (
#             float(np.nanmean(pf_res[v_valid > 0]))
#             if np.any(v_valid) else None
#         ),
#         "runtime_ms": 1000.0 * (time.time() - t0),
#     }
#     meta_out["pf_stats"].update({
#         "cnt_total": pf_dbg["cnt_total"],
#         "cnt_nmin_fail": pf_dbg["cnt_nmin_fail"],
#         "cnt_c_fail": pf_dbg["cnt_c_fail"],
#     })
#     print(f"  fail(Nmin): {pf_dbg['cnt_nmin_fail']} / {pf_dbg['cnt_total']}")
#     print(f"  fail(|c|):  {pf_dbg['cnt_c_fail']} / {pf_dbg['cnt_total']}")
#
#     # -------- extra --------
#     extra_out = dict(extra) if extra is not None else {}
#     extra_out.update({
#         "u": u,
#         "v": v,
#         "v_valid": v_valid,
#         "pf_res": pf_res,
#         "pf_c": pf_c,
#     })
#
#     # -------- filename --------
#     stem = os.path.splitext(os.path.basename(in_npz))[0]
#     tag_str = f"_{tag}" if tag else ""
#     out_npz = os.path.join(out_dir, f"flow_{stem}{tag_str}.npz")
#
#     # -------- debug print (meta summary) --------
#     pf_stats = meta_out.get("pf_stats", {})
#     pf_cfg = meta_out.get("pf_cfg", {})
#
#     print("\n[PF META SUMMARY]")
#     print(f"  cfg: r={pf_cfg.get('r')}, Tn_ms={pf_cfg.get('Tn_ms')}, "
#           f"Nmin={pf_cfg.get('Nmin')}, eps_c={pf_cfg.get('eps_c')}")
#     print(f"  valid_ratio (per-event): {pf_stats.get('valid_ratio'):.4f}")
#     print(f"  mean_residual: {pf_stats.get('mean_residual')}")
#     print(f"  runtime_ms: {pf_stats.get('runtime_ms'):.2f}")
#     print("----------------------------------\n")
#
#     save_events_npz2(
#         out_npz,
#         t=t, x=x, y=y, p=p,
#         meta=meta_out,
#         extra=extra_out,
#     )
#
#     print(f"[PF] saved: {out_npz}")


def pf_run_npz(
    in_npz: str,
    out_dir: str,
    tag: str = "",
    cfg: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    t, x, y, p, meta, extra = load_events_npz2(in_npz)

    if cfg is None:
        cfg = {
            "r": 2,
            "Tn_ms": 3.0,
            "Nmin": 4,
            "eps_c": 1e-3,
            # "sel": {"eta": 0.9, "k": 1}  # 可选：由外部传入
        }

    # =========================
    # 0) 先根据 tag 做事件筛选
    # =========================
    sel_cfg = cfg.get("sel", None)  # 约定：筛选参数放在 cfg["sel"] 里
    extra_dict = dict(extra) if extra is not None else {}

    t_sel, x_sel, y_sel, p_sel, mask, sel_stats = select_events_by_tag(
        t, x, y, p,
        extra=extra_dict,
        tag=tag if tag else "raw",
        sel_cfg=sel_cfg,
    )

    # 同步裁剪 extra 里的逐事件通道，避免长度不一致
    # 注意：save_events_npz2 要求 extra 里的数组长度必须等于 t 的长度
    extra_sel = {}
    if extra_dict:
        N_in = int(t.shape[0])
        for k, v in extra_dict.items():
            arr = np.asarray(v)
            # 只裁剪“逐事件长数组”（shape[0] == N_in），其它非逐事件字段直接丢弃更安全
            if arr.ndim >= 1 and arr.shape[0] == N_in:
                extra_sel[k] = arr[mask]

    # =========================
    # 1) PF 光流计算（对筛选后的事件）
    # =========================
    u, v, v_valid, pf_res, pf_c, pf_dbg = pf_compute_flow(t_sel, x_sel, y_sel, p_sel, cfg)

    # =========================
    # 2) meta（继承 + 记录配置/统计）
    # =========================
    meta_out = get_meta_base()
    meta_out.update(meta)  # 继承原 meta（例如 pipeline / resolution / time_unit / src 等）

    # 记录：筛选配置与统计（便于复现实验）
    meta_out["pf_tag"] = (tag if tag else "raw")
    meta_out["pf_sel_cfg"] = sel_cfg
    meta_out["pf_sel_stats"] = sel_stats

    meta_out["pf_cfg"] = cfg
    meta_out["pf_stats"] = {
        "valid_ratio": float(np.mean(v_valid)) if v_valid.size > 0 else 0.0,
        "mean_residual": (
            float(np.nanmean(pf_res[v_valid > 0]))
            if np.any(v_valid) else None
        ),
        "runtime_ms": 1000.0 * (time.time() - t0),
    }
    meta_out["pf_stats"].update({
        "cnt_total": pf_dbg["cnt_total"],
        "cnt_nmin_fail": pf_dbg["cnt_nmin_fail"],
        "cnt_c_fail": pf_dbg["cnt_c_fail"],
    })

    print(f"  fail(Nmin): {pf_dbg['cnt_nmin_fail']} / {pf_dbg['cnt_total']}")
    print(f"  fail(|c|):  {pf_dbg['cnt_c_fail']} / {pf_dbg['cnt_total']}")

    # =========================
    # 3) extra（逐事件通道 + PF 输出）
    # =========================
    extra_out = dict(extra_sel)
    extra_out.update({
        "u": u,
        "v": v,
        "v_valid": v_valid,
        "pf_res": pf_res,
        "pf_c": pf_c,
    })

    # =========================
    # 4) filename
    # =========================
    stem = os.path.splitext(os.path.basename(in_npz))[0]
    tag_str = f"_{tag}" if tag else ""
    out_npz = os.path.join(out_dir, f"flow_{stem}{tag_str}.npz")

    # =========================
    # 5) debug print
    # =========================
    pf_stats = meta_out.get("pf_stats", {})
    pf_cfg = meta_out.get("pf_cfg", {})
    pf_sel = meta_out.get("pf_sel_stats", {})

    print("\n[PF META SUMMARY]")
    print(f"  tag: {meta_out.get('pf_tag')}")
    if pf_sel:
        print(f"  select: N_in={pf_sel.get('N_in')}  N_out={pf_sel.get('N_out')}  "
              f"keep_ratio={pf_sel.get('keep_ratio', None)}")
    print(f"  cfg: r={pf_cfg.get('r')}, Tn_ms={pf_cfg.get('Tn_ms')}, "
          f"Nmin={pf_cfg.get('Nmin')}, eps_c={pf_cfg.get('eps_c')}")
    print(f"  valid_ratio (per-event): {pf_stats.get('valid_ratio'):.4f}")
    print(f"  mean_residual: {pf_stats.get('mean_residual')}")
    print(f"  runtime_ms: {pf_stats.get('runtime_ms'):.2f}")
    print("----------------------------------\n")

    # =========================
    # 6) save（保存的是“筛选后的事件流 + 对齐的逐事件通道”）
    # =========================
    save_events_npz2(
        out_npz,
        t=t_sel, x=x_sel, y=y_sel, p=p_sel,
        meta=meta_out,
        extra=extra_out,
    )

    print(f"[PF] saved: {out_npz}")


