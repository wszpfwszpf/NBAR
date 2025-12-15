from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from nbar.types import EventStream
from utils.config import RESOLUTION, get_meta_base, get_s1_cfg_dict
from utils.s1_global_sync import s1_compute_w1


_FORCED_META_KEYS = {"resolution", "time_unit", "src", "t_range", "count"}


def _extract_extra_channels(meta: Optional[Dict[str, Any]], num_events: int) -> Dict[str, np.ndarray]:
    """Recover per-event extra channels from metadata if present."""

    extra: Dict[str, np.ndarray] = {}
    if not meta:
        return extra

    for key, value in meta.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim >= 1 and value.shape[0] == num_events and key not in _FORCED_META_KEYS:
            extra[key] = value
    return extra


def run(
    events: EventStream,
    *,
    cfg=None,
    override: Optional[Dict[str, Any]] = None,
    store_bin_arrays: bool = True,
) -> EventStream:
    """Compute S1 weights for an event stream.

    This function performs only the core computation: it expects an
    :class:`EventStream` and returns a new :class:`EventStream` containing
    the original events with an added ``w1`` channel and S1 metadata. File
    I/O and CLI parsing are intentionally excluded.
    """

    params = get_s1_cfg_dict()
    if override:
        params.update(override)

    W, H = RESOLUTION

    w1, aux = s1_compute_w1(
        events.t,
        events.x,
        events.y,
        W=W,
        H=H,
        dt_s=params["dt_s"],
        Gx=params["Gx"],
        Gy=params["Gy"],
        rho=params["rho"],
        alpha=params["alpha"],
        z_th=params["z_th"],
        lam=params["lam"],
        freq_prior_gk=None,
        normalize_t0=params["normalize_t0"],
    )

    meta_out: Dict[str, Any] = get_meta_base()
    meta_out["s1_cfg"] = params
    meta_out["s1_stats"] = {k: v for k, v in aux.items() if k not in ("Ck", "Sk", "zk")}

    if store_bin_arrays and aux["num_bins"] <= 4000:
        meta_out["s1_debug"] = {"Ck": aux["Ck"], "Sk": aux["Sk"], "zk": aux["zk"]}

    extra_in = _extract_extra_channels(events.meta, events.t.size)
    combined_meta: Dict[str, Any] = dict(meta_out)
    combined_meta.update(extra_in)
    combined_meta["w1"] = w1

    return EventStream(t=events.t, x=events.x, y=events.y, p=events.p, meta=combined_meta)
