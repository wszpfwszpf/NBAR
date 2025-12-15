from baselines.DWF import dwf_denoise_npz
from utils.util import load_events_npz
# dwf_denoise_npz('mvsec_clip_2s.npz')
# 1. 使用DWF降噪
out_path = dwf_denoise_npz('mvsec_clip_2s.npz', L=350, sigma=3)

# 2. 检查保存的元数据
t, x, y, p, meta = load_events_npz(out_path)
print("Resolution:", meta['resolution'])
print("Original file path:", meta['path'])  # 这个是原始文件路径
print("Denoised by:", meta['denoised_by'])
print("DWF parameters:", meta['DWF_parameters'])
print("Processing stats:", meta['processing_stats'])