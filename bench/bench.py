"""Benchmarks against zipline-reloaded on identical order books."""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojo_zipline as mz  # noqa: E402
from zipline.assets import Equity as ZEquity  # noqa: E402
from zipline.assets.exchange_info import ExchangeInfo  # noqa: E402
from zipline.finance.order import Order as ZOrder  # noqa: E402
from zipline.finance.slippage import VolumeShareSlippage as ZVolumeShare  # noqa: E402

EXCHANGE = ExchangeInfo("TEST", "TEST", "US")


class FakeData:
    def __init__(self, close, volume):
        self.close = close
        self.volume = volume
        self.current_dt = 1

    def current(self, _asset, field):
        return {"close": self.close, "volume": self.volume}[field]


def best_time(setup, repeat=3):
    best = math.inf
    result = None
    for _ in range(repeat):
        fn = setup()
        started = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - started)
    return best, result


def mixed_trigger_ours(n=250_000):
    rng = np.random.default_rng(4)
    amounts = rng.choice(np.array([-100, 100], dtype=np.int64), n)
    prices = rng.uniform(90, 110, n)
    kinds = rng.integers(0, 4, n)
    stops = np.where((kinds == 1) | (kinds == 3), 100.0, np.nan)
    limits = np.where((kinds == 2) | (kinds == 3), 100.0, np.nan)
    return lambda: mz.check_order_triggers(
        amounts, prices, stops=stops, limits=limits
    )[0].sum()


def mixed_trigger_upstream(n=250_000):
    rng = np.random.default_rng(4)
    amounts = rng.choice(np.array([-100, 100], dtype=np.int64), n)
    prices = rng.uniform(90, 110, n)
    kinds = rng.integers(0, 4, n)
    stops = np.where((kinds == 1) | (kinds == 3), 100.0, np.nan)
    limits = np.where((kinds == 2) | (kinds == 3), 100.0, np.nan)
    asset = ZEquity(1, EXCHANGE, symbol="A")
    orders = [
        ZOrder(
            0,
            asset,
            int(amounts[i]),
            stop=None if np.isnan(stops[i]) else stops[i],
            limit=None if np.isnan(limits[i]) else limits[i],
        )
        for i in range(n)
    ]

    def run():
        for i, order in enumerate(orders):
            order.check_triggers(prices[i], 1)
        return sum(order.triggered for order in orders)

    return run


def limit_book_ours(n=50_000):
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.VolumeShareSlippage(volume_limit=1.0),
        equity_commission=mz.NoCommission(),
    )
    blotter.current_dt = 0
    for _ in range(n):
        blotter.order(asset, 1, mz.LimitOrder(90))
    data = FakeData(100.0, 1_000_000.0)
    return lambda: len(blotter.get_transactions(data)[0])


def limit_book_upstream(n=50_000):
    asset = ZEquity(1, EXCHANGE, symbol="A")
    orders = [ZOrder(0, asset, 1, limit=90.0) for _ in range(n)]
    model = ZVolumeShare(volume_limit=1.0)
    data = FakeData(100.0, 1_000_000.0)
    return lambda: len(list(model.simulate(data, asset, orders)))


def filling_book_ours(n=20_000):
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.VolumeShareSlippage(
            volume_limit=1.0, price_impact=0.1
        ),
        equity_commission=mz.PerShare(cost=0.001),
    )
    blotter.current_dt = 0
    for _ in range(n):
        blotter.order(asset, 1, mz.MarketOrder())
    data = FakeData(100.0, 1_000_000.0)
    return lambda: len(blotter.get_transactions(data)[0])


def filling_book_upstream(n=20_000):
    from zipline.finance.commission import PerShare

    asset = ZEquity(1, EXCHANGE, symbol="A")
    orders = [ZOrder(0, asset, 1) for _ in range(n)]
    model = ZVolumeShare(volume_limit=1.0, price_impact=0.1)
    commission = PerShare(cost=0.001)
    data = FakeData(100.0, 1_000_000.0)

    def run():
        count = 0
        for order, txn in model.simulate(data, asset, orders):
            cost = commission.calculate(order, txn)
            order.filled += txn.amount
            order.commission += cost
            count += 1
        return count

    return run


CASES = [
    ("mixed order triggers (250k)", mixed_trigger_ours, mixed_trigger_upstream),
    ("non-triggering limit book (50k)", limit_book_ours, limit_book_upstream),
    ("volume fills + commission (20k)", filling_book_ours, filling_book_upstream),
]


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    mz.check_order_triggers([1], [1.0])
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print(f"Python {platform.python_version()}; Zipline {__import__('zipline').__version__}")
    print()
    print("| case | mojo-zipline | zipline | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours_setup, upstream_setup in CASES:
        ours, ours_result = best_time(ours_setup)
        upstream, upstream_result = best_time(upstream_setup)
        if ours_result != upstream_result:
            raise RuntimeError(
                f"benchmark result mismatch for {name}: "
                f"{ours_result} != {upstream_result}"
            )
        ratio = upstream / ours
        label = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(
            f"| {name} | {ours * 1e3:.2f} ms | "
            f"{upstream * 1e3:.2f} ms | {label} |"
        )


if __name__ == "__main__":
    main()
