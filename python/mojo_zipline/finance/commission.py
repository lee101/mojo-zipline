DEFAULT_PER_SHARE_COST = 0.001
DEFAULT_PER_DOLLAR_COST = 0.0015


class CommissionModel:
    pass


class NoCommission(CommissionModel):
    @staticmethod
    def calculate(order, transaction):
        return 0.0


class PerShare(CommissionModel):
    def __init__(self, cost=DEFAULT_PER_SHARE_COST, min_trade_cost=0.0):
        self.cost_per_share = float(cost)
        self.min_trade_cost = float(min_trade_cost or 0)

    def calculate(self, order, transaction):
        additional = abs(transaction.amount * self.cost_per_share)
        if order.commission == 0:
            return max(self.min_trade_cost, additional)
        total = (
            abs(order.filled * self.cost_per_share)
            + additional
        )
        return 0.0 if total < self.min_trade_cost else total - order.commission


class PerTrade(CommissionModel):
    def __init__(self, cost=0.0):
        self.cost = float(cost)

    def calculate(self, order, transaction):
        return self.cost if order.commission == 0 else 0.0


class PerDollar(CommissionModel):
    def __init__(self, cost=DEFAULT_PER_DOLLAR_COST):
        self.cost_per_dollar = float(cost)

    def calculate(self, order, transaction):
        return abs(transaction.amount) * transaction.price * self.cost_per_dollar
