# -*- coding: utf-8 -*-
from utils.util import render_event_frames_from_npz, run_frequency_diagnosis_npz, render_event_frames_from_npz2
from utils.visc import vis_s2_c2_grid
# render_event_frames_from_npz('mvsec_clip_2s_stage1.npz')
# render_event_frames_from_npz('mvsec_clip_2s_DWF_denoised.npz')
# render_event_frames_from_npz2('mvsec_clip_2s_s1.npz')
# render_event_frames_from_npz2('mvsec_clip_2s_s2.npz', stage='s2')
# run_frequency_diagnosis_npz('mvsec_clip_2s.npz')
# run_frequency_diagnosis_npz('mvsec_clip_2s_DWF_denoised.npz')
# vis_s2_c2_grid('mvsec_clip_2s_s2c2.npz')

from utils.visc import vis_s2_c2_grid

if __name__ == "__main__":
    out_dir, grids = vis_s2_c2_grid(r"mvsec_clip_2s_s2c2.npz", win_s=0.033, thresholds=(0,1,2,3))
    print("Saved to:", out_dir)
    print("First grid:", grids[0] if grids else None)
