"""
Broker registry. Adding a broker = one adapter class + one entry here.
"""
from brokers.indmoney.adapter import IndmoneyAdapter

BROKERS = {
    "indmoney": IndmoneyAdapter,
}


def available_brokers():
    return list(BROKERS.keys())


def get_adapters(broker="all"):
    """
    Return instantiated adapters. `broker` is a registry name or "all".
    Raises ValueError for an unknown name.
    """
    if broker in (None, "all", "*"):
        return [cls() for cls in BROKERS.values()]

    if broker not in BROKERS:
        raise ValueError(
            f"Unknown broker '{broker}'. Available: {', '.join(available_brokers())}"
        )

    return [BROKERS[broker]()]
