"""
Common types for broker adapters.

Every broker adapter normalises its holdings into `Holding` objects so the
aggregator and the agent never see broker-specific payloads.
"""
from dataclasses import dataclass, asdict, field
from typing import Optional, Protocol, List


@dataclass
class Holding:
    ticker: str                 # exchange symbol, e.g. "AAPL"
    quantity: float
    avg_price: float            # average buy price per unit, in `currency`
    current_price: float        # last price per unit, in `currency`
    invested_value: float
    current_value: float
    pnl: float
    pnl_pct: float
    currency: str = "USD"
    broker: str = ""
    name: str = ""
    as_of: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class BrokerAdapter(Protocol):
    """Interface each broker must implement."""

    name: str
    supported_markets: List[str]

    def is_connected(self) -> bool: ...

    def connect(self) -> dict: ...

    def fetch_holdings(self, market: str = "US") -> List[Holding]: ...

    def fetch_cash(self, market: str = "US") -> Optional[float]: ...
