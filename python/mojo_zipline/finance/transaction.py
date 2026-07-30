from dataclasses import dataclass


@dataclass(slots=True)
class Transaction:
    asset: object
    amount: int
    dt: object
    price: float
    order_id: str

    @property
    def sid(self):
        return self.asset
