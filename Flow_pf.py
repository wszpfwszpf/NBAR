

from utils.flow_pf import pf_run_npz

# pf_run_npz(
#     in_npz="mvsec_clip_2s_s1_s2c2.npz",
#     out_dir="output",
#     tag="raw",
#     cfg={
#         "r": 2,
#         "Tn_ms": 3.0,
#         "Nmin": 4,
#         "eps_c": 1e-3,
#     }
# )

# pf_run_npz(
#     in_npz="mvsec_clip_2s.npz",
#     out_dir="output",
#     tag="s1",
#     cfg={
#         "r": 2,
#         "Tn_ms": 3.0,
#         "Nmin": 4,
#         "eps_c": 1e-3,
#         "sel": {
#             "eta": 0.7,   # w1 >= eta 才保留
#         }
#     }
# )
#
# pf_run_npz(
#     in_npz="mvsec_clip_2s.npz",
#     out_dir="output",
#     tag="s2",
#     cfg={
#         "r": 2,
#         "Tn_ms": 3.0,
#         "Nmin": 4,
#         "eps_c": 1e-3,
#         "sel": {
#             "k": 1,       # c2 >= k 才保留
#         }
#     }
# )
#
# pf_run_npz(
#     in_npz="mvsec_clip_2s.npz",
#     out_dir="output",
#     tag="fuseand",
#     cfg={
#         "r": 2,
#         "Tn_ms": 3.0,
#         "Nmin": 4,
#         "eps_c": 1e-3,
#         "sel": {
#             "eta": 0.7,   # w1 >= eta
#             "k": 1,       # c2 >= k
#         }
#     }
# )
#
# pf_run_npz(
#     in_npz="mvsec_clip_2s.npz",
#     out_dir="output",
#     tag="fuseor",
#     cfg={
#         "r": 2,
#         "Tn_ms": 3.0,
#         "Nmin": 4,
#         "eps_c": 1e-3,
#         "sel": {
#             "eta": 0.7,
#             "k": 1,
#         }
#     }
# )
#
base_cfg = {
    "r": 2,
    "Tn_ms": 3.0,
    "Nmin": 4,
    "eps_c": 1e-3,
    "sel": {
        "eta": 0.7,
        "k": 1,
    }
}

for tag in ["raw", "s1", "s2", "fuseand", "fuseor"]:
    pf_run_npz(
        in_npz="mvsec_clip_2s_s1_s2c2.npz",
        out_dir="output",
        tag=tag,
        cfg=base_cfg,
    )
