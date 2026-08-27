from __future__ import annotations

import os
import operator
from array import array
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .._lib import addr, lib
from ..assets import Equity, Future
from .commission import NoCommission, PerDollar, PerShare, PerTrade
from .execution import MarketOrder
from .order import ORDER_STATUS, Order
from .slippage import (
    FixedBasisPointsSlippage,
    FixedSlippage,
    NoSlippage,
    VolumeShareSlippage,
)
from .transaction import Transaction


class _OrderBuffer:
    _INT_FIELDS = (
        "amount",
        "filled",
        "stop_active",
        "limit_active",
        "stop_reached",
        "limit_reached",
        "status",
        "txn_amount",
        "changed",
    )
    _FLOAT_FIELDS = (
        "stop",
        "limit",
        "paid",
        "txn_price",
        "txn_commission",
    )

    def __init__(self):
        self.orders = []
        for name in self._INT_FIELDS:
            setattr(self, name, array("q"))
        for name in self._FLOAT_FIELDS:
            setattr(self, name, array("d"))
        self.group_starts = np.array([0, 0], dtype=np.int64)
        self.close = np.empty(1, dtype=np.float64)
        self.volume = np.empty(1, dtype=np.float64)

    def append(self, order):
        index = len(self.orders)
        self.orders.append(order)
        self.amount.append(order.amount)
        self.filled.append(order.filled)
        self.stop_active.append(order.stop is not None)
        self.limit_active.append(order.limit is not None)
        self.stop.append(0.0 if order.stop is None else order.stop)
        self.limit.append(0.0 if order.limit is None else order.limit)
        self.stop_reached.append(order.stop_reached)
        self.limit_reached.append(order.limit_reached)
        self.status.append(int(order.status))
        self.paid.append(order.commission)
        self.txn_amount.append(0)
        self.txn_price.append(0.0)
        self.txn_commission.append(0.0)
        self.changed.append(0)
        self.group_starts[1] = index + 1
        return index

    def views(self):
        result = {
            name: np.frombuffer(getattr(self, name), dtype=np.int64)
            for name in self._INT_FIELDS
        }
        result.update(
            {
                name: np.frombuffer(getattr(self, name), dtype=np.float64)
                for name in self._FLOAT_FIELDS
            }
        )
        return result


def _slippage_parameters(model):
    if isinstance(model, VolumeShareSlippage):
        return 0, model.price_impact, model.volume_limit
    if isinstance(model, FixedBasisPointsSlippage):
        return 1, model.percentage, model.volume_limit
    if isinstance(model, FixedSlippage):
        return 2, model.spread, 1.0
    if isinstance(model, NoSlippage):
        return 3, 0.0, 1.0
    raise TypeError(f"unsupported slippage model: {type(model).__name__}")


def _commission_parameters(model):
    if isinstance(model, NoCommission):
        return 0, 0.0, 0.0
    if isinstance(model, PerShare):
        return 1, model.cost_per_share, model.min_trade_cost
    if isinstance(model, PerTrade):
        return 2, model.cost, 0.0
    if isinstance(model, PerDollar):
        return 3, model.cost_per_dollar, 0.0
    raise TypeError(f"unsupported commission model: {type(model).__name__}")


class SimulationBlotter:
    """Stateful Zipline-compatible blotter backed by one batched Mojo call."""

    def __init__(
        self,
        equity_slippage=None,
        future_slippage=None,
        equity_commission=None,
        future_commission=None,
        cancel_policy=None,
    ):
        self.open_orders = defaultdict(list)
        self.orders = {}
        self.new_orders = []
        self.current_dt = None
        self.max_shares = int(1e11)
        self.cancel_policy = cancel_policy
        self.equity_slippage = equity_slippage or FixedBasisPointsSlippage()
        self.future_slippage = future_slippage or VolumeShareSlippage(
            volume_limit=0.05
        )
        self.equity_commission = equity_commission or PerShare()
        self.future_commission = future_commission or PerShare(cost=0.85)
        self.slippage_models = {
            Equity: self.equity_slippage,
            Future: self.future_slippage,
        }
        self.commission_models = {
            Equity: self.equity_commission,
            Future: self.future_commission,
        }
        self._buffers = {}
        self._order_slots = {}

    def _models(self, asset):
        if isinstance(asset, Future):
            return self.slippage_models[Future], self.commission_models[Future]
        return self.slippage_models[Equity], self.commission_models[Equity]

    def order(self, asset, amount, style=None, order_id=None):
        try:
            amount = operator.index(amount)
        except TypeError as exc:
            raise TypeError("order amount must be an integer") from exc
        if amount == 0:
            return None
        if abs(amount) > self.max_shares:
            raise OverflowError(f"Can't order more than {self.max_shares} shares")
        style = MarketOrder() if style is None else style
        is_buy = amount > 0
        order = Order(
            dt=self.current_dt,
            asset=asset,
            amount=amount,
            stop=style.get_stop_price(is_buy),
            limit=style.get_limit_price(is_buy),
            id=order_id,
        )
        self.open_orders[asset].append(order)
        self.orders[order.id] = order
        self.new_orders.append(order)
        buffer = self._buffers.setdefault(asset, _OrderBuffer())
        self._order_slots[order.id] = (buffer, buffer.append(order))
        return order.id

    def _sync_status(self, order):
        slot = self._order_slots.get(order.id)
        if slot is not None:
            buffer, index = slot
            buffer.status[index] = int(order.status)

    def _compact_asset_buffer(self, asset):
        buffer = self._buffers.get(asset)
        if buffer is None:
            return
        live = self.open_orders.get(asset, ())
        if live and (
            len(buffer.orders) < 64
            or len(live) * 2 > len(buffer.orders)
        ):
            return
        for old_order in buffer.orders:
            self._order_slots.pop(old_order.id, None)
        if not live:
            self._buffers.pop(asset, None)
            return
        replacement = _OrderBuffer()
        self._buffers[asset] = replacement
        for live_order in live:
            index = replacement.append(live_order)
            self._order_slots[live_order.id] = (replacement, index)

    def cancel(self, order_id, relay_status=True):
        order = self.orders.get(order_id)
        if order is None or not order.open:
            return
        if order in self.open_orders[order.asset]:
            self.open_orders[order.asset].remove(order)
        if order in self.new_orders:
            self.new_orders.remove(order)
        order.cancel()
        self._sync_status(order)
        order.dt = self.current_dt
        if relay_status:
            self.new_orders.append(order)
        self._compact_asset_buffer(order.asset)

    def cancel_all_orders_for_asset(self, asset, warn=False, relay_status=True):
        del warn
        for order in self.open_orders.get(asset, ())[:]:
            self.cancel(order.id, relay_status)
        self.open_orders.pop(asset, None)

    def reject(self, order_id, reason=""):
        order = self.orders.get(order_id)
        if order is None:
            return
        if order in self.open_orders[order.asset]:
            self.open_orders[order.asset].remove(order)
        if order in self.new_orders:
            self.new_orders.remove(order)
        order.reject(reason)
        self._sync_status(order)
        order.dt = self.current_dt
        self.new_orders.append(order)
        self._compact_asset_buffer(order.asset)

    def hold(self, order_id, reason=""):
        order = self.orders.get(order_id)
        if order is None or not order.open:
            return
        if order in self.new_orders:
            self.new_orders.remove(order)
        order.hold(reason)
        self._sync_status(order)
        order.dt = self.current_dt
        self.new_orders.append(order)

    def process_splits(self, splits):
        for asset, ratio in splits:
            for order in self.open_orders.get(asset, ()):
                order.handle_split(ratio)
                slot = self._order_slots.get(order.id)
                if slot is not None:
                    buffer, index = slot
                    buffer.amount[index] = order.amount
                    buffer.stop[index] = (
                        0.0 if order.stop is None else order.stop
                    )
                    buffer.limit[index] = (
                        0.0 if order.limit is None else order.limit
                    )
                    buffer.stop_active[index] = order.stop is not None
                    buffer.limit_active[index] = order.limit is not None

    def execute_cancel_policy(self, event):
        if self.cancel_policy is not None and self.cancel_policy.should_cancel(event):
            for asset in list(self.open_orders):
                self.cancel_all_orders_for_asset(asset, relay_status=False)

    execute_daily_cancel_policy = execute_cancel_policy

    def get_transactions(self, bar_data):
        batches = defaultdict(list)
        for asset, orders in self.open_orders.items():
            if orders:
                batches[self._models(asset)].append(
                    (asset, self._buffers[asset])
                )

        transactions = []
        commissions = []
        closed_orders = []
        for (slippage, commission), groups in batches.items():
            total_orders = sum(
                len(buffer.orders) for _, buffer in groups
            )
            if len(groups) >= 8 and total_orders >= 65536:
                lib()
                workers = min(len(groups), os.cpu_count() or 1)
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    results = executor.map(
                        lambda group: self._process_buffer(
                            bar_data,
                            group[0],
                            group[1],
                            slippage,
                            commission,
                        ),
                        groups,
                    )
                    for txns, costs, closed in results:
                        transactions.extend(txns)
                        commissions.extend(costs)
                        closed_orders.extend(closed)
                continue
            if len(groups) > 1:
                result = self._process_buffers(
                    bar_data, groups, slippage, commission
                )
                txns, costs, closed = result
                transactions.extend(txns)
                commissions.extend(costs)
                closed_orders.extend(closed)
                continue
            for asset, buffer in groups:
                result = self._process_buffer(
                    bar_data, asset, buffer, slippage, commission
                )
                txns, costs, closed = result
                transactions.extend(txns)
                commissions.extend(costs)
                closed_orders.extend(closed)
        return transactions, commissions, closed_orders

    def _process_buffer(self, data, asset, buffer, slippage, commission):
        arrays = buffer.views()
        close, volume = self._bar_values(data, asset)
        buffer.close[0] = close
        buffer.volume[0] = volume
        sl_kind, sl_param, volume_limit = _slippage_parameters(slippage)
        com_kind, com_cost, com_minimum = _commission_parameters(commission)

        lib().mzl_process_orders(
            addr(buffer.group_starts),
            addr(buffer.close),
            addr(buffer.volume),
            addr(arrays["amount"]),
            addr(arrays["filled"]),
            addr(arrays["stop"]),
            addr(arrays["limit"]),
            addr(arrays["stop_active"]),
            addr(arrays["limit_active"]),
            addr(arrays["stop_reached"]),
            addr(arrays["limit_reached"]),
            addr(arrays["status"]),
            addr(arrays["paid"]),
            addr(arrays["txn_amount"]),
            addr(arrays["txn_price"]),
            addr(arrays["txn_commission"]),
            addr(arrays["changed"]),
            1,
            sl_kind,
            sl_param,
            volume_limit,
            com_kind,
            com_cost,
            com_minimum,
        )
        return self._collect_buffer_results(data.current_dt, buffer, arrays)

    def _process_buffers(self, data, groups, slippage, commission):
        arrays_by_group = [buffer.views() for _, buffer in groups]
        count = len(groups)
        bar_values = [self._bar_values(data, asset) for asset, _ in groups]
        lengths = np.fromiter(
            (len(buffer.orders) for _, buffer in groups),
            dtype=np.int64,
            count=count,
        )
        close = np.fromiter(
            (values[0] for values in bar_values),
            dtype=np.float64,
            count=count,
        )
        volume = np.fromiter(
            (values[1] for values in bar_values),
            dtype=np.float64,
            count=count,
        )
        pointer_names = (
            "amount",
            "filled",
            "stop",
            "limit",
            "stop_active",
            "limit_active",
            "stop_reached",
            "limit_reached",
            "status",
            "paid",
            "txn_amount",
            "txn_price",
            "txn_commission",
            "changed",
        )
        pointers = {
            name: np.fromiter(
                (addr(arrays[name]) for arrays in arrays_by_group),
                dtype=np.int64,
                count=count,
            )
            for name in pointer_names
        }
        sl_kind, sl_param, volume_limit = _slippage_parameters(slippage)
        com_kind, com_cost, com_minimum = _commission_parameters(commission)
        lib().mzl_process_order_buffers(
            addr(lengths),
            addr(close),
            addr(volume),
            addr(pointers["amount"]),
            addr(pointers["filled"]),
            addr(pointers["stop"]),
            addr(pointers["limit"]),
            addr(pointers["stop_active"]),
            addr(pointers["limit_active"]),
            addr(pointers["stop_reached"]),
            addr(pointers["limit_reached"]),
            addr(pointers["status"]),
            addr(pointers["paid"]),
            addr(pointers["txn_amount"]),
            addr(pointers["txn_price"]),
            addr(pointers["txn_commission"]),
            addr(pointers["changed"]),
            count,
            sl_kind,
            sl_param,
            volume_limit,
            com_kind,
            com_cost,
            com_minimum,
        )

        transactions = []
        commissions = []
        closed = []
        for (_, buffer), arrays in zip(groups, arrays_by_group):
            txns, costs, done = self._collect_buffer_results(
                data.current_dt, buffer, arrays
            )
            transactions.extend(txns)
            commissions.extend(costs)
            closed.extend(done)
        return transactions, commissions, closed

    @staticmethod
    def _bar_values(data, asset):
        close = float(data.current(asset, "close"))
        volume = float(data.current(asset, "volume"))
        if not (np.isfinite(close) or np.isnan(close)):
            raise ValueError(f"close for {asset!r} must be finite or NaN")
        if not np.isfinite(volume) or volume < 0:
            raise ValueError(f"volume for {asset!r} must be finite and nonnegative")
        return close, volume

    @staticmethod
    def _collect_buffer_results(current_dt, buffer, arrays):
        transactions = []
        commissions = []
        closed = []
        orders = buffer.orders
        stop_active = buffer.stop_active
        stop_reached = buffer.stop_reached
        limit_reached = buffer.limit_reached
        txn_amount = buffer.txn_amount
        txn_price = buffer.txn_price
        txn_commission = buffer.txn_commission
        filled = buffer.filled
        paid = buffer.paid
        status = buffer.status
        append_transaction = transactions.append
        append_commission = commissions.append
        append_closed = closed.append
        for i in np.flatnonzero(arrays["changed"]):
            order = orders[i]
            if order.stop is not None or order.limit is not None:
                old_stop = order.stop
                old_stop_reached = order.stop_reached
                old_limit_reached = order.limit_reached
                if old_stop is not None and not stop_active[i]:
                    order.stop = None
                order.stop_reached = bool(stop_reached[i])
                order.limit_reached = bool(limit_reached[i])
                if (
                    old_stop != order.stop
                    or old_stop_reached != order.stop_reached
                    or old_limit_reached != order.limit_reached
                ):
                    order.dt = current_dt
            amount = int(txn_amount[i])
            if amount == 0:
                continue
            txn = Transaction(
                order.asset, amount, current_dt, txn_price[i], order.id
            )
            cost = txn_commission[i]
            if cost > 0:
                append_commission(
                    {"asset": order.asset, "order": order, "cost": cost}
                )
            order.filled = filled[i]
            order.commission = paid[i]
            order.dt = current_dt
            if status[i] == ORDER_STATUS.FILLED:
                append_closed(order)
            append_transaction(txn)
        return transactions, commissions, closed

    def prune_orders(self, closed_orders):
        touched_assets = set()
        for order in closed_orders:
            orders = self.open_orders.get(order.asset)
            if orders and order in orders:
                orders.remove(order)
                touched_assets.add(order.asset)
        for asset in touched_assets:
            self._compact_asset_buffer(asset)
        for asset in list(self.open_orders):
            if not self.open_orders[asset]:
                del self.open_orders[asset]

    def get_open_orders(self, asset=None):
        if asset is not None:
            return list(self.open_orders.get(asset, ()))
        return {
            key: list(value)
            for key, value in self.open_orders.items()
            if value
        }
