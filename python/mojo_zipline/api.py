from __future__ import annotations

from contextvars import ContextVar

from .assets import Equity
from .finance.execution import MarketOrder

_active_context = ContextVar("mojo_zipline_context", default=None)
_symbols = {}


def _context():
    context = _active_context.get()
    if context is None:
        raise RuntimeError("Zipline API functions must be called from an algorithm callback")
    return context


def symbol(symbol_str):
    context = _active_context.get()
    if context is not None and symbol_str in context.assets:
        return context.assets[symbol_str]
    if symbol_str not in _symbols:
        _symbols[symbol_str] = Equity(len(_symbols), symbol_str)
    return _symbols[symbol_str]


def order(asset, amount, limit_price=None, stop_price=None, style=None):
    if style is None and (limit_price is not None or stop_price is not None):
        from .finance.execution import LimitOrder, StopLimitOrder, StopOrder

        if limit_price is not None and stop_price is not None:
            style = StopLimitOrder(limit_price, stop_price, asset=asset)
        elif limit_price is not None:
            style = LimitOrder(limit_price, asset=asset)
        else:
            style = StopOrder(stop_price, asset=asset)
    return _context().order(asset, amount, style or MarketOrder())


def order_target(asset, target, limit_price=None, stop_price=None, style=None):
    position = _context().portfolio.positions.get(asset)
    current = 0 if position is None else position.amount
    return order(asset, int(target) - current, limit_price, stop_price, style)


def order_value(asset, value, limit_price=None, stop_price=None, style=None):
    price = _context().data.current(asset, "price")
    return order(asset, int(value / price), limit_price, stop_price, style)


def order_target_value(asset, target, limit_price=None, stop_price=None, style=None):
    price = _context().data.current(asset, "price")
    return order_target(asset, int(target / price), limit_price, stop_price, style)


def order_percent(asset, percent, limit_price=None, stop_price=None, style=None):
    return order_value(
        asset,
        _context().portfolio.portfolio_value * percent,
        limit_price,
        stop_price,
        style,
    )


def order_target_percent(asset, target, limit_price=None, stop_price=None, style=None):
    return order_target_value(
        asset,
        _context().portfolio.portfolio_value * target,
        limit_price,
        stop_price,
        style,
    )


def cancel_order(order_param):
    order_id = getattr(order_param, "id", order_param)
    return _context().blotter.cancel(order_id)


def get_open_orders(asset=None):
    return _context().blotter.get_open_orders(asset)


def record(*args, **kwargs):
    if args:
        if len(args) != 2:
            raise TypeError("record positional arguments must be a name/value pair")
        kwargs[args[0]] = args[1]
    _context().recorded_vars.update(kwargs)
