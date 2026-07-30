"""ctypes bindings for the compiled Mojo order-processing kernels."""

from __future__ import annotations

import ctypes
import os
from numbers import Integral

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_ZIPLINE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-zipline.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mzl_check_triggers": ([I] * 10, None),
    "mzl_process_orders": (
        [I] * 19 + [F, F, I, F, F],
        None,
    ),
    "mzl_process_order_buffers": (
        [I] * 19 + [F, F, I, F, F],
        None,
    ),
}

_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _library = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_library, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous NumPy arrays")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("native buffers must have a non-null address")
    return address


def i64(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.size and raw.dtype.kind in "fc":
        raise TypeError("integer buffers cannot contain floating-point values")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise OverflowError("integer value does not fit in int64")
    if raw.dtype.kind == "O":
        flat = raw.ravel()
        if any(not isinstance(value, Integral) for value in flat):
            raise TypeError("integer buffers must contain integers")
        limit = np.iinfo(np.int64)
        if any(value < limit.min or value > limit.max for value in flat):
            raise OverflowError("integer value does not fit in int64")
    try:
        return np.ascontiguousarray(raw, dtype=np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("integer buffers must contain int64-compatible values") from exc


def f64(values) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "c":
        raise TypeError("floating-point buffers cannot contain complex values")
    try:
        return np.ascontiguousarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TypeError("floating-point buffers must contain numeric values") from exc
