"""C ABI for the Python bindings; buffers cross as integer addresses."""

from kernels import (
    FPtr,
    IPtr,
    check_triggers,
    process_order_buffers,
    process_orders,
)


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("mzl_check_triggers")
def mzl_check_triggers(
    amount: Int,
    stop: Int,
    limit: Int,
    stop_active: Int,
    limit_active: Int,
    stop_reached: Int,
    limit_reached: Int,
    prices: Int,
    triggered: Int,
    n: Int,
) abi("C"):
    check_triggers(
        ip(amount),
        fp(stop),
        fp(limit),
        ip(stop_active),
        ip(limit_active),
        ip(stop_reached),
        ip(limit_reached),
        fp(prices),
        ip(triggered),
        n,
    )


@export("mzl_process_orders")
def mzl_process_orders(
    group_starts: Int,
    close: Int,
    volume: Int,
    amount: Int,
    filled: Int,
    stop: Int,
    limit: Int,
    stop_active: Int,
    limit_active: Int,
    stop_reached: Int,
    limit_reached: Int,
    status: Int,
    paid_commission: Int,
    txn_amount: Int,
    txn_price: Int,
    txn_commission: Int,
    changed: Int,
    groups: Int,
    slippage_kind: Int,
    slippage_param: Float64,
    volume_limit: Float64,
    commission_kind: Int,
    commission_cost: Float64,
    commission_minimum: Float64,
) abi("C"):
    process_orders(
        ip(group_starts),
        fp(close),
        fp(volume),
        ip(amount),
        ip(filled),
        fp(stop),
        fp(limit),
        ip(stop_active),
        ip(limit_active),
        ip(stop_reached),
        ip(limit_reached),
        ip(status),
        fp(paid_commission),
        ip(txn_amount),
        fp(txn_price),
        fp(txn_commission),
        ip(changed),
        groups,
        slippage_kind,
        slippage_param,
        volume_limit,
        commission_kind,
        commission_cost,
        commission_minimum,
    )


@export("mzl_process_order_buffers")
def mzl_process_order_buffers(
    lengths: Int,
    close: Int,
    volume: Int,
    amount_addrs: Int,
    filled_addrs: Int,
    stop_addrs: Int,
    limit_addrs: Int,
    stop_active_addrs: Int,
    limit_active_addrs: Int,
    stop_reached_addrs: Int,
    limit_reached_addrs: Int,
    status_addrs: Int,
    paid_addrs: Int,
    txn_amount_addrs: Int,
    txn_price_addrs: Int,
    txn_commission_addrs: Int,
    changed_addrs: Int,
    groups: Int,
    slippage_kind: Int,
    slippage_param: Float64,
    volume_limit: Float64,
    commission_kind: Int,
    commission_cost: Float64,
    commission_minimum: Float64,
) abi("C"):
    process_order_buffers(
        ip(lengths),
        fp(close),
        fp(volume),
        ip(amount_addrs),
        ip(filled_addrs),
        ip(stop_addrs),
        ip(limit_addrs),
        ip(stop_active_addrs),
        ip(limit_active_addrs),
        ip(stop_reached_addrs),
        ip(limit_reached_addrs),
        ip(status_addrs),
        ip(paid_addrs),
        ip(txn_amount_addrs),
        ip(txn_price_addrs),
        ip(txn_commission_addrs),
        ip(changed_addrs),
        groups,
        slippage_kind,
        slippage_param,
        volume_limit,
        commission_kind,
        commission_cost,
        commission_minimum,
    )
