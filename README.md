# mojo-zipline

`mojo-zipline` is a focused port of Zipline's backtest event loop and
simulation blotter. It keeps strategy callbacks and mutable portfolio state in
Python, while a compiled Mojo kernel evaluates order triggers, allocates
bar-limited volume, applies price impact, and calculates commissions in one
batch.

The Python API exposes a deliberately small subset using familiar
`zipline.finance` names. This project does not shadow an installed Zipline,
which lets its tests compare both implementations in the same process.

## Tested coverage

Every item below has an automated parity or behavior test.

| Area | Tested behavior |
| --- | --- |
| Orders | Market, limit, stop, and stop-limit trigger state; cancellation, rejection, hold, partial fills, splits |
| Execution styles | `MarketOrder`, `LimitOrder`, `StopOrder`, `StopLimitOrder`, including asymmetric tick rounding |
| Slippage | `VolumeShareSlippage`, `FixedBasisPointsSlippage`, `FixedSlippage`, `NoSlippage`; shared per-asset bar volume and impacted-limit rejection |
| Commission | `PerShare` with minimum trade cost, `PerTrade`, `PerDollar`, `NoCommission`; correct incremental charges across partial fills |
| Blotter | Ordering, trigger/fill processing, pruning, open-order lookup, status changes, buffer compaction, and serial/batched/host-worker paths |
| Event loop | In-memory bars, fill-before-`handle_data` ordering, `initialize`, portfolio cash/positions/mark-to-market |
| Algorithm data/API | `order`, `record`, `BarData.current`, and `BarData.history` |

This is not a port of Zipline's data ingestion, bundles, exchange calendars,
asset database, Pipeline API, corporate-action position accounting,
specialized futures behavior, risk/metrics stack, CLI, or live trading. The
covered `run_algorithm` consumes in-memory bars and returns a lightweight
`BacktestResult`, not Zipline's pandas performance frame.

The package contains additional small compatibility helpers, but they are not
claimed as upstream-compatible until a parity test covers them. Only Linux
`x86_64` is configured, and the Mojo shared library must be built locally.

Parity tests run against the `zipline-reloaded` package pinned in
`pixi.toml`.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/capi.mojo` as one shared-library compilation
unit and writes `dist/libmojo-zipline.so`.

## Usage

This example buys ten shares on the second bar. As in Zipline, an order placed
inside `handle_data` is not eligible to fill until the next bar.

```python
from mojo_zipline import (
    Equity,
    NoCommission,
    NoSlippage,
    SimulationBlotter,
    run_algorithm,
)
from mojo_zipline.api import order, record

asset = Equity(1, "AAPL")
bars = [
    ("2026-01-02", {asset: {"close": 100.0, "volume": 10_000}}),
    ("2026-01-05", {asset: {"close": 101.0, "volume": 12_000}}),
]

def initialize(context):
    context.asset = asset

def handle_data(context, data):
    if data.current_dt == "2026-01-02":
        order(context.asset, 10)
    record(price=data.current(context.asset, "price"))

result = run_algorithm(
    bars=bars,
    initialize=initialize,
    handle_data=handle_data,
    capital_base=10_000,
    blotter=SimulationBlotter(
        equity_slippage=NoSlippage(),
        equity_commission=NoCommission(),
    ),
)

assert result.transactions[0].price == 101.0
assert result.portfolio.positions[asset].amount == 10
```

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic, Python 3.13.14, and Zipline 3.1.1. Each row processes
the same prebuilt order book; setup is outside the timed region. Times are the
best of three runs.

| case | mojo-zipline | zipline | result |
| --- | ---: | ---: | ---: |
| mixed order triggers (250k) | 16.27 ms | 212.98 ms | 13.09x faster |
| non-triggering limit book (50k) | 1.00 ms | 36.39 ms | 36.28x faster |
| volume fills + commission (20k) | 34.13 ms | 67.90 ms | 1.99x faster |

The stateful blotter now builds persistent contiguous order buffers as orders
arrive, so a bar no longer repacks every Python object. Native code marks only
rows whose trigger or fill state changed, avoiding a full Python object scan
on a cold book. Filled amounts, paid commission, and native status now stay in
those buffers after the kernel call, and the market-fill collection path avoids
generic trigger-state and order-status work. The filling case still creates
Python transactions, orders, and commission records, so it retains more Python
overhead than the trigger-only cases.

There is no GPU path. Order processing is branch-heavy and moves substantially
more buffer data than its small amount of arithmetic requires, remaining below
the roughly two-flops-per-byte threshold. Shared per-asset volume also makes
fills order-dependent, so GPU dispatch and transfers would not be justified.

## How it works

Python owns orders, transactions, callbacks, and portfolio state. Each asset
has persistent typed structure-of-arrays buffers that are extended when an
order is created. NumPy exposes zero-copy views over those buffers, and
`ctypes` passes only `Int` addresses, lengths, and scalar model parameters
across the C ABI.

The Mojo side reconstructs each address as
`UnsafePointer[..., AnyOrigin[mut=True]]`. Orders use structure-of-arrays
storage: signed `int64` amounts and fill counts, `float64` prices and
commissions, and `int64` state flags. Each asset retains its insertion-order
fill priority and shared bar-volume budget. Output reset uses native-width
SIMD stores with a scalar remainder. Large independent batches may use host
workers; smaller work remains serial. Mojo allocates nothing, and inactive
buffers are compacted or released.

## License

MIT
