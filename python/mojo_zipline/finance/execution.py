from __future__ import annotations

import math


def _validate_price(price, label):
    try:
        price = float(price)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label} price: {price!r}") from exc
    if not math.isfinite(price) or price < 0:
        raise ValueError(f"invalid {label} price: {price!r}")
    return price


def _round_half_away(value):
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def asymmetric_round_price(price, prefer_round_down, tick_size=0.01, diff=0.95):
    precision = max(0, -int(math.floor(math.log10(tick_size))))
    multiplier = int(round(tick_size * 10**precision))
    offset = (diff - 0.5) * 10**-precision * multiplier
    offset -= 10 * float.fromhex("0x1.0000000000000p-52")
    shifted = price - (offset if prefer_round_down else -offset)
    rounded = tick_size * _round_half_away(shifted / tick_size)
    return 0.0 if math.isclose(rounded, 0.0, abs_tol=1e-15) else rounded


class ExecutionStyle:
    _exchange = None

    @property
    def exchange(self):
        return self._exchange


class MarketOrder(ExecutionStyle):
    def __init__(self, exchange=None):
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, _is_buy):
        return None


class LimitOrder(ExecutionStyle):
    def __init__(self, limit_price, asset=None, exchange=None):
        self.limit_price = _validate_price(limit_price, "limit")
        self.asset = asset
        self._exchange = exchange

    def get_limit_price(self, is_buy):
        tick = 0.01 if self.asset is None else self.asset.tick_size
        return asymmetric_round_price(self.limit_price, is_buy, tick)

    def get_stop_price(self, _is_buy):
        return None


class StopOrder(ExecutionStyle):
    def __init__(self, stop_price, asset=None, exchange=None):
        self.stop_price = _validate_price(stop_price, "stop")
        self.asset = asset
        self._exchange = exchange

    def get_limit_price(self, _is_buy):
        return None

    def get_stop_price(self, is_buy):
        tick = 0.01 if self.asset is None else self.asset.tick_size
        return asymmetric_round_price(self.stop_price, not is_buy, tick)


class StopLimitOrder(ExecutionStyle):
    def __init__(self, limit_price, stop_price, asset=None, exchange=None):
        self.limit_price = _validate_price(limit_price, "limit")
        self.stop_price = _validate_price(stop_price, "stop")
        self.asset = asset
        self._exchange = exchange

    def get_limit_price(self, is_buy):
        tick = 0.01 if self.asset is None else self.asset.tick_size
        return asymmetric_round_price(self.limit_price, is_buy, tick)

    def get_stop_price(self, is_buy):
        tick = 0.01 if self.asset is None else self.asset.tick_size
        return asymmetric_round_price(self.stop_price, not is_buy, tick)
