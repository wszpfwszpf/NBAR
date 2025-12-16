# -*- coding: utf-8 -*-
"""
Quick visualization helper for event streams.

Run directly in PyCharm to render frames from the latest S1 output.
"""

from pathlib import Path

from utils.util import render_event_frames_from_npz, render_event_frames_from_npz2

# =========================================================
# User configuration (PyCharm-friendly)
# =========================================================
INPUT_NAME = "mvsec_clip_2s"   # stem of the input file (without extension)
OUT_ROOT = Path("outputs")
STAGE = "s1"

# Visualization parameters
WIN_MS = 33.0
POINT_SIZE = 1.0
MAX_FRAMES = None


def main():
    s1_npz = OUT_ROOT / STAGE / f"{INPUT_NAME}_{STAGE}.npz"
    render_event_frames_from_npz2(
        str(s1_npz),
        win_ms=WIN_MS,
        point_size=POINT_SIZE,
        max_frames=MAX_FRAMES,
    )


if __name__ == "__main__":
    main()
