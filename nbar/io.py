from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .types import EventStream


_MAIN_KEYS = {"t", "x", "y", "p"}


def _extract_txyp(data: Any, *, path: str) -> EventStream:
    files = set(getattr(data, "files", []))

    if _MAIN_KEYS.issubset(files):
        t = np.asarray(data["t"], dtype=np.float64)
        x = np.asarray(data["x"], dtype=np.int32)
        y = np.asarray(data["y"], dtype=np.int32)
        p = np.asarray(data["p"], dtype=np.int8)
    elif "events" in files:
        ev = np.asarray(data["events"])
        if ev.ndim != 2 or ev.shape[1] < 4:
            raise ValueError(f"'events' must be Nx4+, got shape={ev.shape} in {path}")
        t = np.asarray(ev[:, 0], dtype=np.float64)
        x = np.asarray(ev[:, 1], dtype=np.int32)
        y = np.asarray(ev[:, 2], dtype=np.int32)
        p = np.asarray(ev[:, 3], dtype=np.int8)
    else:
        raise KeyError(f"NPZ file {path} must contain t/x/y/p or an 'events' matrix")

    p = np.where(p > 0, 1, -1).astype(np.int8)

    events = EventStream(t=t, x=x, y=y, p=p, meta=None)
    events.validate(check_monotonic=False)
    return events


def _collect_meta(data: Any, *, path: str, events: EventStream) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "path": path,
        "num_events": int(events.t.size),
        "t_start": float(events.t[0]) if events.t.size > 0 else None,
        "t_end": float(events.t[-1]) if events.t.size > 0 else None,
    }

    per_event_len = int(events.t.size)
    for key in getattr(data, "files", []):
        if key in _MAIN_KEYS or key == "events":
            continue
        value = data[key]
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                meta[key] = value.item()
            elif per_event_len and value.shape[0] == per_event_len:
                meta[key] = value
            else:
                meta[key] = value
        else:
            meta[key] = value

    meta.setdefault("resolution", None)
    meta.setdefault("time_unit", None)
    if isinstance(meta["resolution"], tuple):
        meta["resolution"] = list(meta["resolution"])

    return meta


def load_npz(path: str) -> EventStream:
    """Load an event stream from an NPZ file.

    Supports both ``t/x/y/p`` arrays and ``events`` matrices (Nx4, columns
    ``t, x, y, p``). Per-event extras such as ``w1``/``w2`` are preserved in
    ``meta``.
    """

    data = np.load(path, allow_pickle=True)

    events = _extract_txyp(data, path=path)
    events.meta = _collect_meta(data, path=path, events=events)
    return events


def save_npz(path: str, events: EventStream) -> None:
    """Save an :class:`EventStream` to an NPZ file using the unified format."""

    events = EventStream(
        t=np.asarray(events.t, dtype=np.float64),
        x=np.asarray(events.x, dtype=np.int32),
        y=np.asarray(events.y, dtype=np.int32),
        p=np.where(np.asarray(events.p) > 0, 1, -1).astype(np.int8),
        meta=events.meta,
    )
    events.validate(check_monotonic=False)

    save_dict: Dict[str, Any] = {"t": events.t, "x": events.x, "y": events.y, "p": events.p}

    if events.meta:
        for key, value in events.meta.items():
            if key in _MAIN_KEYS or key == "events":
                continue
            if key == "resolution" and value is not None:
                save_dict[key] = np.asarray(value, dtype=np.int32)
            else:
                save_dict[key] = value

    if "resolution" not in save_dict:
        max_x, max_y = np.max(events.x), np.max(events.y)
        save_dict["resolution"] = np.asarray([max_y + 1, max_x + 1]
                                             if max_y < max_x else
                                             [max_x + 1, max_y + 1],
                                             dtype=np.int32)

    save_dict.setdefault("time_unit", "seconds")

    np.savez_compressed(path, **save_dict)
    print(f"Saved {len(events.t)} events to {path}")
