from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _lookup(mapping, asset):
    if asset in mapping:
        return mapping[asset]
    symbol = getattr(asset, "symbol", asset)
    if symbol in mapping:
        return mapping[symbol]
    raise KeyError(asset)


class BarData:
    def __init__(self, current_dt, values, history=()):
        self.current_dt = current_dt
        self._values = values
        self._history = history

    def current(self, assets, fields):
        if isinstance(assets, (list, tuple)):
            return {
                asset: self.current(asset, fields)
                for asset in assets
            }
        row = _lookup(self._values, assets)
        if isinstance(fields, (list, tuple)):
            return {field: self.current(assets, field) for field in fields}
        if fields == "price":
            fields = "close"
        if isinstance(row, dict):
            return row[fields]
        if fields == "close":
            return row
        raise KeyError(fields)

    def history(self, assets, fields, bar_count, frequency="1d"):
        del frequency
        rows = list(self._history)[-int(bar_count) :]
        if not isinstance(assets, (list, tuple)):
            return np.asarray(
                [BarData(dt, values).current(assets, fields) for dt, values in rows],
                dtype=np.float64,
            )
        return {
            asset: np.asarray(
                [BarData(dt, values).current(asset, fields) for dt, values in rows],
                dtype=np.float64,
            )
            for asset in assets
        }

    def can_trade(self, asset):
        try:
            close = float(self.current(asset, "close"))
            volume = float(self.current(asset, "volume"))
        except (KeyError, TypeError, ValueError):
            return False
        return np.isfinite(close) and volume > 0


@dataclass(slots=True)
class Position:
    asset: object
    amount: int = 0
    cost_basis: float = 0.0
    last_sale_price: float = 0.0


@dataclass(slots=True)
class Portfolio:
    starting_cash: float
    cash: float
    positions: dict = field(default_factory=dict)
    portfolio_value: float = 0.0
    returns: float = 0.0

    def __post_init__(self):
        self.portfolio_value = self.cash


@dataclass(slots=True)
class Bar:
    dt: object
    data: dict
