"""Optional ctypes bridge to the engine's C ABI.

The engine library (libgranite4.so / .dylib) is optional in Phase 0. The
bridge never raises for a missing library - the /metrics endpoint treats
`present: false` as a normal state. If a built library is present but its
ABI drifts, `status()` raises and the caller surfaces it as
`status: "unavailable"` with the probe error.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

_LIB_NAMES = ("libgranite4.so", "libgranite4.dylib")


def _candidate_paths() -> list[Path]:
    """Build directories this repo is known to produce, then the CWD."""
    here = Path(__file__).resolve().parent
    repo = here.parent
    paths = [
        *[repo / "build" / "engine" / name for name in _LIB_NAMES],
        *[repo / "build" / name for name in _LIB_NAMES],
        *[Path.cwd() / "build" / "engine" / name for name in _LIB_NAMES],
    ]
    env = os.environ.get("GRANITE4_LIB")
    if env:
        paths.insert(0, Path(env))
    return [p for p in paths if p.is_file()]


def load() -> ctypes.CDLL | None:
    """Return the library handle, or None if no build artifact exists."""
    for p in _candidate_paths():
        try:
            return ctypes.CDLL(str(p))
        except OSError:
            continue
    return None


class _StatusInfo(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int),
        ("cuda_device_present", ctypes.c_int),
        ("status_text", ctypes.c_char_p),
        ("message", ctypes.c_char_p),
    ]


def status(lib: ctypes.CDLL) -> dict[str, Any]:
    """Call granite4_status and decode the result. Raises on ABI drift."""
    fn = lib.granite4_status
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(_StatusInfo)]
    info = _StatusInfo()
    fn(ctypes.byref(info))
    names = {0: "ok", 1: "unimplemented", 2: "no-device", 3: "unavailable"}
    out = {
        "present": True,
        "status": names.get(info.status, f"unknown({info.status})"),
        "cuda_device_present": bool(info.cuda_device_present),
    }
    if info.status_text:
        out["status_text"] = info.status_text.decode("utf-8", "replace")
    if info.message:
        out["message"] = info.message.decode("utf-8", "replace")
    return out
