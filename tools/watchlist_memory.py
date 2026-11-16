from agent.memory_engine import load_memory, save_memory

WATCHLIST_FILE = "watchlist.json"


def get_local_watchlist():
    """Local extras only (memory/watchlist.json). Does not call the broker."""
    watchlist = load_memory(WATCHLIST_FILE)
    if not isinstance(watchlist, dict):
        watchlist = {}
    if "tickers" not in watchlist:
        watchlist["tickers"] = []
        save_memory(WATCHLIST_FILE, watchlist)
    return watchlist


def get_watchlist(broker="indmoney", market="US"):
    """
    Prefer the live INDmoney US watchlist. Falls back to local JSON if the
    broker is not connected.
    """
    print(f"[tools] get_watchlist called with broker={broker}, market={market}")
    from tools.holdings_aggregator import fetch_watchlist

    return fetch_watchlist(broker=broker, market=market)


def add_to_watchlist(ticker):
    """Append to the local extras file (INDmoney watchlist is read-only via MCP)."""
    print(f"[tools] add_to_watchlist called with ticker={ticker}")

    watchlist = get_local_watchlist()
    ticker = str(ticker).upper()

    if ticker not in watchlist["tickers"]:
        watchlist["tickers"].append(ticker)

    save_memory(WATCHLIST_FILE, watchlist)
    return watchlist
