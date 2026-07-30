from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from .api import _active_context
from .assets import Equity
from .finance.blotter import SimulationBlotter
from .protocol import Bar, BarData, Portfolio, Position


@dataclass(slots=True)
class BacktestResult:
    performance: list
    transactions: list
    orders: dict
    portfolio: Portfolio

    def __len__(self):
        return len(self.performance)


class AlgorithmContext(SimpleNamespace):
    def order(self, asset, amount, style=None):
        return self.blotter.order(asset, amount, style)

    def order_target(self, asset, target, style=None):
        position = self.portfolio.positions.get(asset)
        current = 0 if position is None else position.amount
        return self.order(asset, int(target) - current, style)

    def record(self, **kwargs):
        self.recorded_vars.update(kwargs)


class SimulationEventLoop:
    """Small event loop preserving Zipline's fill-before-callback sequencing."""

    def __init__(
        self,
        bars,
        initialize=None,
        handle_data=None,
        before_trading_start=None,
        capital_base=100000.0,
        blotter=None,
    ):
        self.bars = bars
        self.initialize = initialize
        self.handle_data = handle_data
        self.before_trading_start = before_trading_start
        self.blotter = blotter or SimulationBlotter()
        self.context = AlgorithmContext(
            blotter=self.blotter,
            portfolio=Portfolio(float(capital_base), float(capital_base)),
            recorded_vars={},
            assets={},
            data=None,
        )
        self.transactions = []
        self.performance = []

    @staticmethod
    def _coerce_bar(value):
        if isinstance(value, Bar):
            return value
        dt, data = value
        return Bar(dt, data)

    def _register_assets(self, values):
        for key in values:
            if hasattr(key, "symbol"):
                self.context.assets[key.symbol] = key
            elif isinstance(key, str) and key not in self.context.assets:
                self.context.assets[key] = Equity(len(self.context.assets), key)

    def _apply_transaction(self, txn, commission):
        portfolio = self.context.portfolio
        position = portfolio.positions.setdefault(txn.asset, Position(txn.asset))
        old_amount = position.amount
        new_amount = old_amount + txn.amount
        if old_amount == 0 or (old_amount > 0) == (txn.amount > 0):
            total = abs(old_amount) + abs(txn.amount)
            position.cost_basis = (
                abs(old_amount) * position.cost_basis
                + abs(txn.amount) * txn.price
            ) / total
        elif new_amount == 0:
            position.cost_basis = 0.0
        elif (old_amount > 0) != (new_amount > 0):
            position.cost_basis = txn.price
        position.amount = new_amount
        position.last_sale_price = txn.price
        portfolio.cash -= txn.amount * txn.price + commission

    def _mark_to_market(self, data):
        portfolio = self.context.portfolio
        value = portfolio.cash
        for asset, position in portfolio.positions.items():
            if position.amount == 0:
                continue
            try:
                price = float(data.current(asset, "close"))
            except KeyError:
                price = position.last_sale_price
            position.last_sale_price = price
            value += position.amount * price
        portfolio.portfolio_value = value
        portfolio.returns = value / portfolio.starting_cash - 1.0

    def run(self):
        token = _active_context.set(self.context)
        history = []
        try:
            if self.initialize is not None:
                self.initialize(self.context)
            for raw_bar in self.bars:
                bar = self._coerce_bar(raw_bar)
                self._register_assets(bar.data)
                data = BarData(bar.dt, bar.data, history + [(bar.dt, bar.data)])
                self.context.data = data
                self.blotter.current_dt = bar.dt
                self.context.recorded_vars = {}
                if self.before_trading_start is not None:
                    self.before_trading_start(self.context, data)

                txns, commissions, closed = self.blotter.get_transactions(data)
                costs_by_order = {
                    item["order"].id: item["cost"] for item in commissions
                }
                for txn in txns:
                    self._apply_transaction(txn, costs_by_order.get(txn.order_id, 0.0))
                self.transactions.extend(txns)
                self.blotter.prune_orders(closed)
                self._mark_to_market(data)

                if self.handle_data is not None:
                    self.handle_data(self.context, data)
                self._mark_to_market(data)
                self.performance.append(
                    {
                        "dt": bar.dt,
                        "portfolio_value": self.context.portfolio.portfolio_value,
                        "cash": self.context.portfolio.cash,
                        "returns": self.context.portfolio.returns,
                        "transactions": list(txns),
                        "recorded_vars": dict(self.context.recorded_vars),
                    }
                )
                self.blotter.new_orders = []
                history.append((bar.dt, bar.data))
        finally:
            _active_context.reset(token)
        return BacktestResult(
            performance=self.performance,
            transactions=self.transactions,
            orders=dict(self.blotter.orders),
            portfolio=self.context.portfolio,
        )


def run_algorithm(
    start=None,
    end=None,
    initialize=None,
    capital_base=100000.0,
    handle_data=None,
    before_trading_start=None,
    data_frequency="daily",
    data=None,
    bars=None,
    blotter=None,
    **kwargs,
):
    """Run the covered event-loop subset.

    ``bars`` (or ``data``) is an iterable of ``(datetime, asset->OHLCV mapping)``.
    Other data-ingestion and metrics arguments from full Zipline are outside
    this port and are rejected.
    """
    del start, end, data_frequency
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"unsupported full-Zipline arguments: {names}")
    source = bars if bars is not None else data
    if source is None:
        raise ValueError("run_algorithm requires bars= (or data=)")
    return SimulationEventLoop(
        source,
        initialize=initialize,
        handle_data=handle_data,
        before_trading_start=before_trading_start,
        capital_base=capital_base,
        blotter=blotter,
    ).run()
