from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import mojo_zipline as mz
from mojo_zipline.finance.execution import asymmetric_round_price

zipline = pytest.importorskip("zipline")
from zipline.assets import Equity as ZEquity
from zipline.assets.exchange_info import ExchangeInfo
from zipline.finance import commission as zcommission
from zipline.finance import execution as zexecution
from zipline.finance import slippage as zslippage
from zipline.finance.order import Order as ZOrder

EXCHANGE = ExchangeInfo("TEST", "TEST", "US")


def zasset(sid=1, symbol="A"):
    return ZEquity(sid, EXCHANGE, symbol=symbol)


class FakeData:
    def __init__(self, close, volume, dt=1):
        self.close = close
        self.volume = volume
        self.current_dt = dt

    def current(self, _asset, field):
        if isinstance(field, (list, tuple)):
            return {name: self.current(_asset, name) for name in field}
        return {"close": self.close, "volume": self.volume}[field]


@pytest.mark.parametrize(
    "style,args",
    [
        ("limit", (10.005,)),
        ("limit", (10.0095,)),
        ("stop", (10.005,)),
        ("stop", (10.0095,)),
        ("stop_limit", (10.005, 10.015)),
    ],
)
@pytest.mark.parametrize("is_buy", [False, True])
def test_execution_style_price_parity(style, args, is_buy):
    ours_type = {
        "limit": mz.LimitOrder,
        "stop": mz.StopOrder,
        "stop_limit": mz.StopLimitOrder,
    }[style]
    theirs_type = {
        "limit": zexecution.LimitOrder,
        "stop": zexecution.StopOrder,
        "stop_limit": zexecution.StopLimitOrder,
    }[style]
    ours = ours_type(*args)
    theirs = theirs_type(*args)
    assert ours.get_limit_price(is_buy) == theirs.get_limit_price(is_buy)
    assert ours.get_stop_price(is_buy) == theirs.get_stop_price(is_buy)


@pytest.mark.parametrize(
    "amount,stop,limit,prices",
    [
        (10, None, None, [90, 100]),
        (10, 100, None, [99, 100]),
        (-10, 100, None, [101, 100]),
        (10, None, 100, [101, 100]),
        (-10, None, 100, [99, 100]),
        (10, 100, 99, [98, 101, 99]),
        (-10, 100, 101, [102, 99, 101]),
    ],
)
def test_order_trigger_state_machine_parity(amount, stop, limit, prices):
    ours = mz.Order(0, mz.Equity(1, "A"), amount, stop=stop, limit=limit)
    theirs = ZOrder(0, zasset(), amount, stop=stop, limit=limit)
    for dt, price in enumerate(prices, 1):
        ours.check_triggers(price, dt)
        theirs.check_triggers(price, dt)
        assert ours.triggered == theirs.triggered
        assert ours.stop_reached == theirs.stop_reached
        assert ours.limit_reached == theirs.limit_reached
        assert ours.stop == theirs.stop


def test_batched_trigger_kernel_matches_upstream():
    rng = np.random.default_rng(42)
    n = 10_000
    amounts = rng.choice([-100, 100], n)
    prices = rng.uniform(90, 110, n)
    kinds = rng.integers(0, 4, n)
    stops = np.where((kinds == 1) | (kinds == 3), 100.0, np.nan)
    limits = np.where((kinds == 2) | (kinds == 3), 100.0, np.nan)
    triggered, sr, lr, active = mz.check_order_triggers(
        amounts, prices, stops=stops, limits=limits
    )
    for i in range(n):
        order = ZOrder(
            0,
            zasset(i + 1),
            int(amounts[i]),
            stop=None if np.isnan(stops[i]) else stops[i],
            limit=None if np.isnan(limits[i]) else limits[i],
        )
        order.check_triggers(prices[i], 1)
        assert triggered[i] == order.triggered
        assert sr[i] == order.stop_reached
        assert lr[i] == order.limit_reached
        assert active[i] == (order.stop is not None)


def test_trigger_kernel_rejects_unsafe_buffer_inputs():
    with pytest.raises(ValueError, match="length"):
        mz.check_order_triggers([1, 2], [100.0, 101.0], stops=[99.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        mz.check_order_triggers([[1, 2]], [[100.0, 101.0]])
    with pytest.raises(TypeError, match="floating-point"):
        mz.check_order_triggers([1.5], [100.0])
    with pytest.raises(OverflowError, match="int64"):
        mz.check_order_triggers([2**63], [100.0])
    with pytest.raises(ValueError, match="finite or NaN"):
        mz.check_order_triggers([1], [100.0], limits=[np.inf])


def test_trigger_kernel_accepts_empty_input_without_entering_ffi():
    result = mz.check_order_triggers([], [])
    assert all(array.shape == (0,) for array in result)


@pytest.mark.parametrize(
    "ours_model,theirs_model",
    [
        (mz.PerShare(cost=0.01), zcommission.PerShare(cost=0.01)),
        (
            mz.PerShare(cost=0.01, min_trade_cost=1.0),
            zcommission.PerShare(cost=0.01, min_trade_cost=1.0),
        ),
        (mz.PerTrade(cost=4.95), zcommission.PerTrade(cost=4.95)),
        (mz.PerDollar(cost=0.0015), zcommission.PerDollar(cost=0.0015)),
        (mz.NoCommission(), zcommission.NoCommission()),
    ],
)
def test_incremental_commission_parity(ours_model, theirs_model):
    ours = mz.Order(0, mz.Equity(1, "A"), 250)
    theirs = ZOrder(0, zasset(), 250)
    for amount, price in [(40, 10.0), (60, 11.0), (150, 9.0)]:
        ours_txn = mz.Transaction(ours.asset, amount, 1, price, ours.id)
        theirs_txn = SimpleNamespace(amount=amount, price=price)
        ours_cost = ours_model.calculate(ours, ours_txn)
        theirs_cost = theirs_model.calculate(theirs, theirs_txn)
        assert ours_cost == pytest.approx(theirs_cost)
        ours.filled += amount
        theirs.filled += amount
        ours.commission += ours_cost
        theirs.commission += theirs_cost


def upstream_volume_fills(orders, close, volume, model):
    data = FakeData(close, volume)
    fills = []
    for order, txn in model.simulate(data, orders[0].asset, orders):
        fills.append((txn.amount, txn.price))
        order.filled += txn.amount
    return fills


@pytest.mark.parametrize(
    "model,upstream",
    [
        (
            mz.VolumeShareSlippage(volume_limit=0.1, price_impact=0.2),
            zslippage.VolumeShareSlippage(volume_limit=0.1, price_impact=0.2),
        ),
        (
            mz.FixedBasisPointsSlippage(basis_points=8, volume_limit=0.2),
            zslippage.FixedBasisPointsSlippage(basis_points=8, volume_limit=0.2),
        ),
        (mz.FixedSlippage(spread=0.04), zslippage.FixedSlippage(spread=0.04)),
    ],
)
def test_blotter_fill_parity(model, upstream):
    asset = mz.Equity(1, "A")
    z_asset = zasset()
    blotter = mz.SimulationBlotter(
        equity_slippage=model, equity_commission=mz.NoCommission()
    )
    blotter.current_dt = 0
    for amount in (70, 90, -40):
        blotter.order(asset, amount, mz.MarketOrder())
    z_orders = [ZOrder(0, z_asset, amount) for amount in (70, 90, -40)]
    expected = upstream_volume_fills(z_orders, 100.0, 1000.0, upstream)
    txns, _, _ = blotter.get_transactions(FakeData(100.0, 1000.0))
    actual = [(txn.amount, txn.price) for txn in txns]
    assert [item[0] for item in actual] == [item[0] for item in expected]
    assert np.allclose(
        [item[1] for item in actual],
        [item[1] for item in expected],
    )


def test_volume_share_impacted_limit_rejection_parity():
    ours_asset = mz.Equity(1, "A")
    theirs_asset = zasset()
    ours = mz.SimulationBlotter(
        equity_slippage=mz.VolumeShareSlippage(volume_limit=0.1, price_impact=1.0),
        equity_commission=mz.NoCommission(),
    )
    ours.current_dt = 0
    ours.order(ours_asset, 100, mz.LimitOrder(100.05))
    their_order = ZOrder(0, theirs_asset, 100, limit=100.05)
    their_fills = upstream_volume_fills(
        [their_order],
        100.0,
        1000.0,
        zslippage.VolumeShareSlippage(volume_limit=0.1, price_impact=1.0),
    )
    our_fills, _, _ = ours.get_transactions(FakeData(100.0, 1000.0))
    assert our_fills == []
    assert their_fills == []


def test_partial_fill_commission_and_pruning():
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.VolumeShareSlippage(volume_limit=0.1, price_impact=0),
        equity_commission=mz.PerShare(cost=0.01, min_trade_cost=1.0),
    )
    blotter.current_dt = 0
    order_id = blotter.order(asset, 150, mz.MarketOrder())
    txns, commissions, closed = blotter.get_transactions(FakeData(10, 1000, 1))
    assert txns[0].amount == 100
    assert commissions[0]["cost"] == 1.0
    assert closed == []
    txns, commissions, closed = blotter.get_transactions(FakeData(10, 1000, 2))
    assert txns[0].amount == 50
    assert commissions[0]["cost"] == 0.5
    assert closed == [blotter.orders[order_id]]
    blotter.prune_orders(closed)
    assert blotter.get_open_orders() == {}


def test_blotter_rejects_fractional_amount_instead_of_narrowing():
    blotter = mz.SimulationBlotter()
    with pytest.raises(TypeError, match="integer"):
        blotter.order(mz.Equity(1, "A"), 1.5)


def test_simd_output_reset_handles_scalar_tail():
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.NoSlippage(),
        equity_commission=mz.NoCommission(),
    )
    blotter.current_dt = 0
    for _ in range(7):
        blotter.order(asset, 1, mz.MarketOrder())
    first, _, _ = blotter.get_transactions(FakeData(100, 1000, 1))
    second, _, _ = blotter.get_transactions(FakeData(101, 1000, 2))
    assert len(first) == 7
    assert second == []


def test_small_independent_asset_batch_stays_correct():
    assets = [mz.Equity(i, str(i)) for i in range(2)]
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.NoSlippage(),
        equity_commission=mz.NoCommission(),
    )
    blotter.current_dt = 0
    for asset in assets:
        blotter.order(asset, 1, mz.MarketOrder())
    txns, _, _ = blotter.get_transactions(FakeData(100, 1000, 1))
    assert [(txn.asset, txn.amount) for txn in txns] == [
        (asset, 1) for asset in assets
    ]


@pytest.mark.parametrize("close,volume", [(np.inf, 100), (100, np.nan), (100, -1)])
def test_blotter_rejects_values_unsafe_for_native_conversion(close, volume):
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.VolumeShareSlippage(),
        equity_commission=mz.NoCommission(),
    )
    blotter.order(asset, 1)
    with pytest.raises(ValueError):
        blotter.get_transactions(FakeData(close, volume))


def test_large_independent_asset_batch_parallel_threshold():
    assets = [mz.Equity(i, str(i)) for i in range(8)]
    blotter = mz.SimulationBlotter(
        equity_slippage=mz.NoSlippage(),
        equity_commission=mz.NoCommission(),
    )
    blotter.current_dt = 0
    for asset in assets:
        for _ in range(8191):
            blotter.order(asset, 1, mz.LimitOrder(90))
        blotter.order(asset, 1, mz.LimitOrder(110))
    txns, _, _ = blotter.get_transactions(FakeData(100, 1_000_000, 1))
    assert [(txn.asset, txn.amount, txn.price) for txn in txns] == [
        (asset, 1, 100.0) for asset in assets
    ]


def test_cancel_reject_hold_and_split():
    asset = mz.Equity(1, "A")
    blotter = mz.SimulationBlotter()
    blotter.current_dt = 0
    cancel_id = blotter.order(asset, 100, mz.MarketOrder())
    blotter.cancel(cancel_id)
    assert blotter.orders[cancel_id].status == mz.ORDER_STATUS.CANCELLED
    reject_id = blotter.order(asset, 100, mz.MarketOrder())
    blotter.reject(reject_id, "test")
    assert blotter.orders[reject_id].status == mz.ORDER_STATUS.REJECTED
    held_id = blotter.order(asset, 100, mz.LimitOrder(20))
    blotter.hold(held_id, "review")
    assert blotter.orders[held_id].status == mz.ORDER_STATUS.HELD
    blotter.process_splits([(asset, 2)])
    assert blotter.orders[held_id].amount == 50
    assert blotter.orders[held_id].limit == 40


def test_event_loop_fills_on_next_bar_and_marks_portfolio():
    asset = mz.Equity(1, "A")
    seen = []

    def initialize(context):
        context.asset = asset

    def handle_data(context, data):
        seen.append((data.current_dt, context.portfolio.positions.get(asset)))
        if data.current_dt == 0:
            context.order(asset, 10)

    bars = [
        (0, {asset: {"close": 10.0, "volume": 1000}}),
        (1, {asset: {"close": 11.0, "volume": 1000}}),
        (2, {asset: {"close": 12.0, "volume": 1000}}),
    ]
    result = mz.run_algorithm(
        bars=bars,
        initialize=initialize,
        handle_data=handle_data,
        capital_base=1000,
        blotter=mz.SimulationBlotter(
            equity_slippage=mz.NoSlippage(),
            equity_commission=mz.NoCommission(),
        ),
    )
    assert seen[0][1] is None
    assert seen[1][1].amount == 10
    assert result.transactions[0].dt == 1
    assert result.portfolio.cash == 890
    assert result.portfolio.portfolio_value == 1010
    assert result.portfolio.returns == pytest.approx(0.01)


def test_event_loop_stop_limit_persists_across_bars():
    asset = mz.Equity(1, "A")

    def initialize(context):
        context.asset = asset
        context.order(asset, 10, mz.StopLimitOrder(99, 100))

    bars = [
        (0, {asset: {"close": 101.0, "volume": 1000}}),
        (1, {asset: {"close": 99.0, "volume": 1000}}),
    ]
    result = mz.run_algorithm(
        bars=bars,
        initialize=initialize,
        blotter=mz.SimulationBlotter(
            equity_slippage=mz.NoSlippage(),
            equity_commission=mz.NoCommission(),
        ),
    )
    assert len(result.transactions) == 1
    assert result.transactions[0].dt == 1
    assert result.transactions[0].price == 99


def test_history_and_record_api():
    from mojo_zipline.api import record

    asset = mz.Equity(1, "A")

    def handle_data(context, data):
        values = data.history(asset, "price", 2, "1d")
        record(last=float(values[-1]), count=len(values))

    result = mz.run_algorithm(
        bars=[
            (0, {asset: {"close": 10.0, "volume": 1}}),
            (1, {asset: {"close": 11.0, "volume": 1}}),
        ],
        handle_data=handle_data,
    )
    assert result.performance[-1]["recorded_vars"] == {"last": 11.0, "count": 2}
