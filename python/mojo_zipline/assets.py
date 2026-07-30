from __future__ import annotations


class Asset:
    __slots__ = ("sid", "symbol", "tick_size", "auto_close_date")

    def __init__(
        self,
        sid: int,
        symbol: str | None = None,
        tick_size: float = 0.01,
        auto_close_date=None,
    ):
        self.sid = int(sid)
        self.symbol = symbol or str(sid)
        self.tick_size = float(tick_size)
        self.auto_close_date = auto_close_date

    def __hash__(self):
        return hash((type(self), self.sid))

    def __eq__(self, other):
        return type(self) is type(other) and self.sid == other.sid

    def __repr__(self):
        return f"{type(self).__name__}({self.sid} [{self.symbol}])"


class Equity(Asset):
    pass


class Future(Asset):
    __slots__ = ("root_symbol",)

    def __init__(self, sid: int, symbol: str | None = None, root_symbol=None, **kwargs):
        super().__init__(sid, symbol, **kwargs)
        self.root_symbol = root_symbol or self.symbol
