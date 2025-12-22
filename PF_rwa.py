# run_metrics_raw.py
# -*- coding: utf-8 -*-

from utils.metrics_adapter_pf import run_pf_metrics_from_npz
from utils.metrics_pf import MetricsCfg


def main():
    # ===== 输入 =====
    npz_path = r"output\flow_mvsec_clip_2s_raw.npz"
    out_dir = r"output\metrics_raw"

    # ===== 指标配置 =====
    cfg = MetricsCfg(
        dt_s=0.033,        # 33 ms
        min_count=5,       # 先给小一点，观察趋势
        eps_speed=1e-12,
        use_valid_only=True,
    )

    print("[RUN METRICS | RAW]")
    print("  input:", npz_path)
    print("  output:", out_dir)
    print("  cfg:", cfg)

    res = run_pf_metrics_from_npz(
        npz_path=npz_path,
        out_dir=out_dir,
        cfg=cfg,
        save_json=True,        # 建议打开，方便你看
        save_npz_curves=True   # bin 曲线存下来，后面画图用
    )

    # print("\n[RESULT SUMMARY]")
    # for k, v in res.items():
    #     if isinstance(v, float):
    #         print(f"  {k}: {v:.6f}")
    #     else:
    #         print(f"  {k}: shape={getattr(v, 'shape', None)}")

    print("\n[RESULT SUMMARY]")
    print("  valid_ratio:", res.get("valid_ratio", None))

    metrics = res.get("metrics", {})
    if metrics:
        print("\n[METRICS]")
        print(f"  coverage_mean:     {metrics.get('coverage_mean')}")
        print(f"  ediff_mean:        {metrics.get('ediff_mean')}")
        print(f"  var_theta_mean:    {metrics.get('var_theta_mean')}")
    else:
        print("  metrics: <empty>")


if __name__ == "__main__":
    main()
