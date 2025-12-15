import h5py
import numpy as np

mvsec_data_path = "outdoor_night1_data-002.hdf5"
save_path = "mvsec_events.npy"

with h5py.File(mvsec_data_path, "r") as f:
    events = f["davis"]["left"]["events"][:]  # shape: (N, 4)

# events format: [x, y, t, p]
x = events[:, 0].astype(np.int16)
y = events[:, 1].astype(np.int16)
t = events[:, 2].astype(np.float64)  # seconds
p = events[:, 3].astype(np.int8)

# 时间排序（强烈建议，保险）
order = np.argsort(t)
x, y, t, p = x[order], y[order], t[order], p[order]

# 时间戳标准化：从 0 开始（秒）
t = t - t[0]

# 统一成你后面用的顺序 (t, x, y, p)
events_txyp = np.stack([t, x, y, p], axis=1)


from utils.util import save_events_npz

save_events_npz(
    save_path="mvsec_events.npz",
    t=t,
    x=x,
    y=y,
    p=p,
    resolution=(346, 260),
    time_unit="seconds"
)


print(f"Saved events: {events_txyp.shape} -> {save_path}")
