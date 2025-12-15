from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import EventStream


def load_npz(path: str) -> EventStream:
    """Load an event stream from an NPZ file.

    The expected keys match existing utilities: ``t``, ``x``, ``y``, ``p``
    with optional metadata such as ``resolution`` and ``time_unit`` left
    unchanged.
    """

    data = np.load(path, allow_pickle=True)

    required_keys = {"t", "x", "y", "p"}
    if not required_keys.issubset(data.files):
        raise KeyError(
            f"NPZ file {path} must contain keys {required_keys}, but got {set(data.files)}"
        )

    t = np.asarray(data["t"], dtype=np.float64)
    x = np.asarray(data["x"], dtype=np.int32)
    y = np.asarray(data["y"], dtype=np.int32)
    p = np.asarray(data["p"], dtype=np.int8)

    events = EventStream(t=t, x=x, y=y, p=p, meta=None)
    events.validate(check_monotonic=False)

    meta: Dict[str, Any] = {
        "path": path,
        "num_events": int(t.size),
        "t_start": float(t[0]) if t.size > 0 else None,
        "t_end": float(t[-1]) if t.size > 0 else None,
    }

    for key in data.files:
        if key in ("t", "x", "y", "p"):
            continue
        value = data[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                meta[key] = value.item()
            else:
                meta[key] = value
        else:
            meta[key] = value

    meta.setdefault("resolution", None)
    meta.setdefault("time_unit", None)
    if meta["resolution"] is not None and isinstance(meta["resolution"], tuple):
        meta["resolution"] = list(meta["resolution"])

    events.meta = meta
    return events


def save_npz(path: str, events: EventStream) -> None:
    """Save an :class:`EventStream` to an NPZ file.

    Keys remain identical to existing outputs (t, x, y, p plus metadata).
    """

    # normalize polarity to {-1, +1}
    p_pm = np.where(events.p > 0, 1, -1).astype(np.int8)

    events = EventStream(t=np.asarray(events.t, dtype=np.float64),
                         x=np.asarray(events.x, dtype=np.int32),
                         y=np.asarray(events.y, dtype=np.int32),
                         p=p_pm,
                         meta=events.meta)
    events.validate(check_monotonic=False)

    save_dict: Dict[str, Any] = {
        "t": events.t,
        "x": events.x,
        "y": events.y,
        "p": events.p,
    }

    if events.meta is not None:
        for key, value in events.meta.items():
            if key in ("t", "x", "y", "p"):
                continue
            if key == "resolution" and value is not None:
                save_dict[key] = np.asarray(value, dtype=np.int32)
            else:
                save_dict[key] = value
    else:
        max_x, max_y = np.max(events.x), np.max(events.y)
        save_dict["resolution"] = np.asarray([max_y + 1, max_x + 1]
                                             if max_y < max_x else
                                             [max_x + 1, max_y + 1],
                                             dtype=np.int32)
        save_dict["time_unit"] = "seconds"

    if "resolution" not in save_dict:
        max_x, max_y = np.max(events.x), np.max(events.y)
        save_dict["resolution"] = np.asarray([max_y + 1, max_x + 1]
                                             if max_y < max_x else
                                             [max_x + 1, max_y + 1],
                                             dtype=np.int32)

    if "time_unit" not in save_dict:
        save_dict["time_unit"] = "seconds"

    np.savez_compressed(path, **save_dict)
    print(f"Saved {len(events.t)} events to {path}")
