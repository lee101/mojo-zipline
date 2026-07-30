from __future__ import annotations

import math
import uuid
from enum import IntEnum


class ORDER_STATUS(IntEnum):
    OPEN = 0
    FILLED = 1
    CANCELLED = 2
    REJECTED = 3
    HELD = 4


class Order:
    __slots__ = (
        "id",
        "dt",
        "reason",
        "created",
        "asset",
        "amount",
        "filled",
        "commission",
        "_status",
        "stop",
        "limit",
        "stop_reached",
        "limit_reached",
        "direction",
        "broker_order_id",
    )

    def __init__(
        self,
        dt,
        asset,
        amount,
        stop=None,
        limit=None,
        filled=0,
        commission=0,
        id=None,
    ):
        self.id = uuid.uuid4().hex if id is None else id
        self.dt = dt
        self.reason = None
        self.created = dt
        self.asset = asset
        self.amount = int(amount)
        self.filled = int(filled)
        self.commission = float(commission)
        self._status = ORDER_STATUS.OPEN
        self.stop = stop
        self.limit = limit
        self.stop_reached = False
        self.limit_reached = False
        self.direction = math.copysign(1, self.amount)
        self.broker_order_id = None

    @property
    def sid(self):
        return self.asset

    @property
    def status(self):
        if not self.open_amount:
            return ORDER_STATUS.FILLED
        if self._status == ORDER_STATUS.HELD and self.filled:
            return ORDER_STATUS.OPEN
        return self._status

    @status.setter
    def status(self, value):
        self._status = ORDER_STATUS(value)

    @property
    def open(self):
        return self.status in (ORDER_STATUS.OPEN, ORDER_STATUS.HELD)

    @property
    def open_amount(self):
        return self.amount - self.filled

    @property
    def triggered(self):
        return (
            (self.stop is None or self.stop_reached)
            and (self.limit is None or self.limit_reached)
        )

    def check_order_triggers(self, current_price):
        if self.triggered:
            return self.stop_reached, self.limit_reached, False
        stop_reached = limit_reached = sl_stop_reached = False
        buy = self.amount > 0
        if self.stop is not None and self.limit is not None:
            if (buy and current_price >= self.stop) or (
                not buy and current_price <= self.stop
            ):
                sl_stop_reached = True
                limit_reached = (buy and current_price <= self.limit) or (
                    not buy and current_price >= self.limit
                )
        elif self.stop is not None:
            stop_reached = (buy and current_price >= self.stop) or (
                not buy and current_price <= self.stop
            )
        elif self.limit is not None:
            limit_reached = (buy and current_price <= self.limit) or (
                not buy and current_price >= self.limit
            )
        return stop_reached, limit_reached, sl_stop_reached

    def check_triggers(self, price, dt):
        stop, limit, stop_limit = self.check_order_triggers(price)
        if (stop, limit) != (self.stop_reached, self.limit_reached):
            self.dt = dt
        self.stop_reached = stop
        self.limit_reached = limit
        if stop_limit:
            self.stop = None

    def handle_split(self, ratio):
        self.amount = int(self.amount / ratio)
        if self.limit is not None:
            self.limit = round(self.limit * ratio, 2)
        if self.stop is not None:
            self.stop = round(self.stop * ratio, 2)

    def cancel(self):
        self.status = ORDER_STATUS.CANCELLED

    def reject(self, reason=""):
        self.status = ORDER_STATUS.REJECTED
        self.reason = reason

    def hold(self, reason=""):
        self.status = ORDER_STATUS.HELD
        self.reason = reason

    def to_dict(self):
        result = {
            key: getattr(self, key)
            for key in self.__slots__
            if key not in {"_status", "asset"}
        }
        if self.broker_order_id is None:
            result.pop("broker_order_id")
        result.update(sid=self.asset, status=self.status)
        return result

    def __repr__(self):
        return f"Order({self.to_dict()!r})"
