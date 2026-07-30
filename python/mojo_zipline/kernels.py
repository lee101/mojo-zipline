"""Array APIs for the Mojo kernels, useful outside the stateful blotter."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib


def check_order_triggers(
    amounts,
    prices,
    *,
    stops=None,
    limits=None,
    stop_reached=None,
    limit_reached=None,
):
    """Evaluate Zipline market/limit/stop/stop-limit triggers in one Mojo call.

    Returns ``(triggered, stop_reached, limit_reached, stop_active)``. A false
    ``stop_active`` marks a stop-limit order that has converted into a limit
    order.
    """
    amounts = i64(amounts)
    prices = f64(prices)
    if amounts.ndim != 1 or prices.ndim != 1:
        raise ValueError("prices and amounts must be one-dimensional")
    n = len(amounts)
    if prices.shape != (n,):
        raise ValueError("prices and amounts must be one-dimensional and equal length")

    stops = f64(np.zeros(n) if stops is None else stops)
    limits = f64(np.zeros(n) if limits is None else limits)
    stop_reached = i64(
        np.zeros(n, dtype=np.int64) if stop_reached is None else stop_reached
    )
    limit_reached = i64(
        np.zeros(n, dtype=np.int64) if limit_reached is None else limit_reached
    )
    named = {
        "stops": stops,
        "limits": limits,
        "stop_reached": stop_reached,
        "limit_reached": limit_reached,
    }
    for name, value in named.items():
        if value.shape != (n,):
            raise ValueError(f"{name} must be one-dimensional with length {n}")
    if np.isinf(stops).any() or np.isinf(limits).any():
        raise ValueError("stop and limit prices must be finite or NaN")
    stop_active = np.isfinite(stops).astype(np.int64)
    limit_active = np.isfinite(limits).astype(np.int64)
    stops = np.nan_to_num(stops, nan=0.0)
    limits = np.nan_to_num(limits, nan=0.0)
    triggered = np.empty(n, dtype=np.int64)
    if n == 0:
        return (
            triggered.astype(bool),
            stop_reached.astype(bool),
            limit_reached.astype(bool),
            stop_active.astype(bool),
        )
    lib().mzl_check_triggers(
        addr(amounts),
        addr(stops),
        addr(limits),
        addr(stop_active),
        addr(limit_active),
        addr(stop_reached),
        addr(limit_reached),
        addr(prices),
        addr(triggered),
        n,
    )
    return (
        triggered.astype(bool),
        stop_reached.astype(bool),
        limit_reached.astype(bool),
        stop_active.astype(bool),
    )
