# -*- coding: utf-8 -*-
from utils.s1_global_sync import s1_compute_w1, s1_run_npz

if __name__ == "__main__":
    IN = r"mvsec_clip_2s.npz"
    # out_npz=None -> 自动保存到输入目录，文件名加 _s1
    out = s1_run_npz(IN, out_npz=None, override=None)
    print("Saved:", out)

