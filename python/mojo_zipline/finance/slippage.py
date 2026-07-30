DEFAULT_EQUITY_VOLUME_SLIPPAGE_BAR_LIMIT = 0.025


class SlippageModel:
    def __init__(self):
        self._volume_for_bar = 0

    @property
    def volume_for_bar(self):
        return self._volume_for_bar


class NoSlippage(SlippageModel):
    pass


class VolumeShareSlippage(SlippageModel):
    def __init__(
        self,
        volume_limit=DEFAULT_EQUITY_VOLUME_SLIPPAGE_BAR_LIMIT,
        price_impact=0.1,
    ):
        super().__init__()
        self.volume_limit = float(volume_limit)
        self.price_impact = float(price_impact)


class FixedSlippage(SlippageModel):
    def __init__(self, spread=0.0):
        super().__init__()
        self.spread = float(spread)


class FixedBasisPointsSlippage(SlippageModel):
    def __init__(self, basis_points=5.0, volume_limit=0.1):
        super().__init__()
        if basis_points < 0 or volume_limit <= 0:
            raise ValueError("basis_points must be nonnegative and volume_limit positive")
        self.basis_points = float(basis_points)
        self.percentage = self.basis_points / 10000.0
        self.volume_limit = float(volume_limit)
