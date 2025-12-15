"""Lightweight event data utilities for NBAR."""

from .types import EventStream
from .io import load_npz, save_npz

__all__ = ["EventStream", "load_npz", "save_npz"]
