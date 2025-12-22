# # -*- coding: utf-8 -*-
import numpy as np
#
# # # 方法1：查看所有键名
# # with np.load('mvsec_clip_2s_s1.npz') as data:
# #     print("Keys:", list(data.keys()))
# #     # 或更详细：
# #     for key in data.keys():
# #         print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
#
# 方法2：不解压直接看
path = 'mvsec_clip_2s_s1_s2c2.npz'
data = np.load(path, allow_pickle=True)
print("Keys:", list(data.files))

# # check_w2.py
# from utils.inspect import inspect_w2_distribution
#
# if __name__ == "__main__":
#     NPZ = r"mvsec_clip_2s_s2.npz"  # 改成你的输出文件
#     inspect_w2_distribution(
#         NPZ,
#         key="w2",
#         bins=60,
#         clip_range=(0.0, 1.0),
#     )
# test_metrics_pf.py
# -*- coding: utf-8 -*-

# import traceback
#
# from utils.metrics_adapter_pf import run_pf_metrics_from_npz
# from utils.metrics_pf import MetricsCfg
#
#
# def main():
#     # 1) 指定一个你已经跑过 PF 的 npz
#     npz_path = r"output\flow_mvsec_clip_2s_raw.npz"
#     out_dir = r"output\metrics_test"
#
#     # 2) 指标配置（随便给，反正现在不算）
#     cfg = MetricsCfg(
#         dt_s=0.033,
#         min_count=10,
#         use_valid_only=True,
#     )
#
#     print("[TEST] start metrics pipeline")
#     print("  npz:", npz_path)
#     print("  out:", out_dir)
#     print("  cfg:", cfg)
#
#     try:
#         res = run_pf_metrics_from_npz(
#             npz_path=npz_path,
#             out_dir=out_dir,
#             cfg=cfg,
#             save_json=False,      # 现在不用存
#             save_npz_curves=False
#         )
#
#         # 如果能跑到这里，说明：
#         # - npz 成功读取
#         # - PF 通道成功提取
#         # - 已进入指标计算入口
#         print("[TEST] pipeline reached metrics stage successfully")
#         print("Returned keys:", res.keys())
#
#     except NotImplementedError as e:
#         # 这是“预期中的失败”
#         print("[TEST] metrics not implemented yet (expected)")
#         print("  message:", str(e))
#
#     except Exception as e:
#         print("[TEST] unexpected error!")
#         traceback.print_exc()
#
#
# if __name__ == "__main__":
#     main()
