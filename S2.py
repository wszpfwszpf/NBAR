# -*- coding: utf-8 -*-
from utils.s2_global_sync import s2_run_npz, s2_run_npz_c2

if __name__ == "__main__":
    IN = r"mvsec_clip_2s_s1.npz"
    # out_npz=None -> 自动保存到输入目录，文件名加 _s1
    out = s2_run_npz_c2(IN, out_npz=None, override=None)
    print("Saved:", out)