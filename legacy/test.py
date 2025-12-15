# -*- coding: utf-8 -*-
import numpy as np

# # 方法1：查看所有键名
# with np.load('mvsec_clip_2s_s1.npz') as data:
#     print("Keys:", list(data.keys()))
#     # 或更详细：
#     for key in data.keys():
#         print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")

# # 方法2：不解压直接看
data = np.load('mvsec_clip_2s_s1.npz', allow_pickle=True)
print("Keys:", list(data.files))