"""Batched Zipline order triggering, fills, price impact, and commissions."""

from std.math import isnan
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def check_one(
    amount: IPtr,
    stop: FPtr,
    limit: FPtr,
    stop_active: IPtr,
    limit_active: IPtr,
    stop_reached: IPtr,
    limit_reached: IPtr,
    price: Float64,
    i: Int,
) -> Bool:
    var has_stop = stop_active[i] != 0
    var has_limit = limit_active[i] != 0
    if (not has_stop or stop_reached[i] != 0) and (
        not has_limit or limit_reached[i] != 0
    ):
        return True

    var buy = amount[i] > 0
    if has_stop and has_limit:
        var reached = (buy and price >= stop[i]) or (
            not buy and price <= stop[i]
        )
        if reached:
            if (buy and price <= limit[i]) or (not buy and price >= limit[i]):
                limit_reached[i] = 1
            stop_active[i] = 0
            has_stop = False
    elif has_stop:
        if (buy and price >= stop[i]) or (not buy and price <= stop[i]):
            stop_reached[i] = 1
    elif has_limit:
        if (buy and price <= limit[i]) or (not buy and price >= limit[i]):
            limit_reached[i] = 1

    return (stop_active[i] == 0 or stop_reached[i] != 0) and (
        not has_limit or limit_reached[i] != 0
    )


def check_triggers(
    amount: IPtr,
    stop: FPtr,
    limit: FPtr,
    stop_active: IPtr,
    limit_active: IPtr,
    stop_reached: IPtr,
    limit_reached: IPtr,
    prices: FPtr,
    triggered: IPtr,
    n: Int,
):
    for i in range(n):
        triggered[i] = 1 if check_one(
            amount,
            stop,
            limit,
            stop_active,
            limit_active,
            stop_reached,
            limit_reached,
            prices[i],
            i,
        ) else 0


def commission_for_fill(
    kind: Int,
    txn_amount: Int64,
    txn_price: Float64,
    filled: Int64,
    paid: Float64,
    cost: Float64,
    minimum: Float64,
) -> Float64:
    if kind == 1:
        var additional = abs(Float64(txn_amount) * cost)
        if paid == 0.0:
            return max(minimum, additional)
        var total = abs(Float64(filled) * cost) + additional
        return 0.0 if total < minimum else total - paid
    if kind == 2:
        return cost if paid == 0.0 else 0.0
    if kind == 3:
        return abs(Float64(txn_amount)) * txn_price * cost
    return 0.0


def clear_outputs(
    txn_amount: IPtr,
    txn_price: FPtr,
    txn_commission: FPtr,
    changed: IPtr,
    n: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = 0
    while i + W <= n:
        txn_amount.store(i, SIMD[DType.int64, W](0))
        txn_price.store(i, SIMD[DType.float64, W](0.0))
        txn_commission.store(i, SIMD[DType.float64, W](0.0))
        changed.store(i, SIMD[DType.int64, W](0))
        i += W
    while i < n:
        txn_amount[i] = 0
        txn_price[i] = 0.0
        txn_commission[i] = 0.0
        changed[i] = 0
        i += 1


def process_order_range(
    amount: IPtr,
    filled: IPtr,
    stop: FPtr,
    limit: FPtr,
    stop_active: IPtr,
    limit_active: IPtr,
    stop_reached: IPtr,
    limit_reached: IPtr,
    status: IPtr,
    paid_commission: FPtr,
    txn_amount: IPtr,
    txn_price: FPtr,
    txn_commission: FPtr,
    changed: IPtr,
    start: Int,
    end: Int,
    bar_price: Float64,
    bar_volume: Float64,
    slippage_kind: Int,
    slippage_param: Float64,
    volume_limit: Float64,
    commission_kind: Int,
    commission_cost: Float64,
    commission_minimum: Float64,
):
    if bar_volume == 0.0 or isnan(bar_price):
        return
    var used = Int64(0)
    for i in range(start, end):
        if status[i] != 0 and status[i] != 4:
            continue
        var open_amount = amount[i] - filled[i]
        if open_amount == 0:
            continue
        var old_stop_active = stop_active[i]
        var old_stop_reached = stop_reached[i]
        var old_limit_reached = limit_reached[i]
        if not check_one(
            amount,
            stop,
            limit,
            stop_active,
            limit_active,
            stop_reached,
            limit_reached,
            bar_price,
            i,
        ):
            if (
                old_stop_active != stop_active[i]
                or (old_stop_reached != stop_reached[i])
                or old_limit_reached != limit_reached[i]
            ):
                changed[i] = 1
            continue
        if (
            old_stop_active != stop_active[i]
            or (old_stop_reached != stop_reached[i])
            or old_limit_reached != limit_reached[i]
        ):
            changed[i] = 1

        var direction = Int64(1) if open_amount > 0 else Int64(-1)
        var wanted = abs(open_amount)
        var shares: Int64
        var execution_price = bar_price

        if slippage_kind == 0:
            var remaining = volume_limit * bar_volume - Float64(used)
            if remaining < 1.0:
                break
            shares = min(Int64(remaining), wanted)
            if shares < 1:
                continue
            var total = used + shares
            var share = min(Float64(total) / bar_volume, volume_limit)
            execution_price = bar_price * (
                1.0 + Float64(direction) * slippage_param * share * share
            )
            if limit_active[i] != 0 and (
                (direction > 0 and execution_price > limit[i])
                or (direction < 0 and execution_price < limit[i])
            ):
                continue
        elif slippage_kind == 1:
            var max_volume = Int64(volume_limit * bar_volume)
            shares = min(wanted, max_volume - used)
            if shares == 0:
                break
            execution_price = bar_price * (
                1.0 + Float64(direction) * slippage_param
            )
        elif slippage_kind == 2:
            shares = wanted
            execution_price = bar_price + (
                Float64(direction) * slippage_param * 0.5
            )
        else:
            shares = wanted

        var signed_shares = direction * shares
        txn_amount[i] = signed_shares
        txn_price[i] = execution_price
        txn_commission[i] = commission_for_fill(
            commission_kind,
            signed_shares,
            execution_price,
            filled[i],
            paid_commission[i],
            commission_cost,
            commission_minimum,
        )
        changed[i] = 1
        used += shares


def process_orders(
    group_starts: IPtr,
    close: FPtr,
    volume: FPtr,
    amount: IPtr,
    filled: IPtr,
    stop: FPtr,
    limit: FPtr,
    stop_active: IPtr,
    limit_active: IPtr,
    stop_reached: IPtr,
    limit_reached: IPtr,
    status: IPtr,
    paid_commission: FPtr,
    txn_amount: IPtr,
    txn_price: FPtr,
    txn_commission: FPtr,
    changed: IPtr,
    groups: Int,
    slippage_kind: Int,
    slippage_param: Float64,
    volume_limit: Float64,
    commission_kind: Int,
    commission_cost: Float64,
    commission_minimum: Float64,
):
    clear_outputs(
        txn_amount,
        txn_price,
        txn_commission,
        changed,
        Int(group_starts[groups]),
    )
    for g in range(groups):
        process_order_range(
            amount,
            filled,
            stop,
            limit,
            stop_active,
            limit_active,
            stop_reached,
            limit_reached,
            status,
            paid_commission,
            txn_amount,
            txn_price,
            txn_commission,
            changed,
            Int(group_starts[g]),
            Int(group_starts[g + 1]),
            close[g],
            volume[g],
            slippage_kind,
            slippage_param,
            volume_limit,
            commission_kind,
            commission_cost,
            commission_minimum,
        )


def process_order_buffers(
    lengths: IPtr,
    close: FPtr,
    volume: FPtr,
    amount_addrs: IPtr,
    filled_addrs: IPtr,
    stop_addrs: IPtr,
    limit_addrs: IPtr,
    stop_active_addrs: IPtr,
    limit_active_addrs: IPtr,
    stop_reached_addrs: IPtr,
    limit_reached_addrs: IPtr,
    status_addrs: IPtr,
    paid_addrs: IPtr,
    txn_amount_addrs: IPtr,
    txn_price_addrs: IPtr,
    txn_commission_addrs: IPtr,
    changed_addrs: IPtr,
    groups: Int,
    slippage_kind: Int,
    slippage_param: Float64,
    volume_limit: Float64,
    commission_kind: Int,
    commission_cost: Float64,
    commission_minimum: Float64,
):
    def work(g: Int) {imm}:
        var n = Int(lengths[g])
        var txn_amount = IPtr(unsafe_from_address=Int(txn_amount_addrs[g]))
        var txn_price = FPtr(unsafe_from_address=Int(txn_price_addrs[g]))
        var txn_commission = FPtr(
            unsafe_from_address=Int(txn_commission_addrs[g])
        )
        var changed = IPtr(unsafe_from_address=Int(changed_addrs[g]))
        clear_outputs(txn_amount, txn_price, txn_commission, changed, n)
        process_order_range(
            IPtr(unsafe_from_address=Int(amount_addrs[g])),
            IPtr(unsafe_from_address=Int(filled_addrs[g])),
            FPtr(unsafe_from_address=Int(stop_addrs[g])),
            FPtr(unsafe_from_address=Int(limit_addrs[g])),
            IPtr(unsafe_from_address=Int(stop_active_addrs[g])),
            IPtr(unsafe_from_address=Int(limit_active_addrs[g])),
            IPtr(unsafe_from_address=Int(stop_reached_addrs[g])),
            IPtr(unsafe_from_address=Int(limit_reached_addrs[g])),
            IPtr(unsafe_from_address=Int(status_addrs[g])),
            FPtr(unsafe_from_address=Int(paid_addrs[g])),
            txn_amount,
            txn_price,
            txn_commission,
            changed,
            0,
            n,
            close[g],
            volume[g],
            slippage_kind,
            slippage_param,
            volume_limit,
            commission_kind,
            commission_cost,
            commission_minimum,
        )

    for g in range(groups):
        work(g)
