"""
Generic holdings aggregator across broker adapters (US stocks only for now).

Replaces the hand-maintained memory/portfolio.json: holdings are fetched live
from connected brokers, normalised, merged per ticker and cached briefly in
memory/holdings_cache.json.
"""
import json
import os
import time
from datetime import datetime, timezone

from brokers.registry import get_adapters, available_brokers

CACHE_FILE = os.path.join("memory", "holdings_cache.json")
CACHE_TTL_MINUTES = 15

RISK_SETTINGS_FILE = os.path.join("memory", "risk_settings.json")

DEFAULT_MARKET = "US"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _read_cache():
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(data):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _cache_key(adapters):
    return ",".join(sorted(a.name for a in adapters))


def _cache_is_fresh(cache, key, market):
    if not cache:
        return False
    if cache.get("cache_key") != key or cache.get("market") != market:
        return False
    age_min = (time.time() - cache.get("fetched_at_epoch", 0)) / 60
    return age_min < CACHE_TTL_MINUTES


def _cash_fallback():
    try:
        with open(RISK_SETTINGS_FILE, "r", encoding="utf-8") as f:
            return float(json.load(f).get("cash_available_fallback", 0.0))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0.0


# ---------------------------------------------------------------------------
# Broker connection
# ---------------------------------------------------------------------------

def connect_broker(broker="indmoney"):
    """
    Run the broker's login flow (opens the browser the first time). Tokens are
    cached locally; later calls are silent.
    """
    print(f"[tools] connect_broker called with broker={broker}")

    try:
        adapters = get_adapters(broker)
    except ValueError as e:
        return {"error": str(e), "available_brokers": available_brokers()}

    results = {}
    for adapter in adapters:
        try:
            info = adapter.connect()
            results[adapter.name] = {"connected": True, **{k: v for k, v in info.items() if k != "tools"}}
        except Exception as e:
            results[adapter.name] = {"connected": False, "error": str(e)}

    return {"brokers": results}


def broker_status():
    return {a.name: {"connected": a.is_connected(), "markets": a.supported_markets} for a in get_adapters("all")}


# ---------------------------------------------------------------------------
# Holdings
# ---------------------------------------------------------------------------

def fetch_holdings(broker="all", market=DEFAULT_MARKET, refresh=False):
    """
    Fetch and merge holdings from the requested broker(s).
    Returns {holdings: [...], cash_available, cash_source, brokers, as_of, cache_hit, warnings}
    """
    print(f"[tools] fetch_holdings called with broker={broker}, market={market}, refresh={refresh}")

    try:
        adapters = get_adapters(broker)
    except ValueError as e:
        return {"error": str(e), "available_brokers": available_brokers()}

    key = _cache_key(adapters)
    cache = _read_cache()
    if not refresh and _cache_is_fresh(cache, key, market):
        cache = dict(cache)
        cache["cache_hit"] = True
        return cache

    merged = {}
    warnings = []
    brokers_used = []
    cash_total = None
    cash_sources = []

    for adapter in adapters:
        if market not in adapter.supported_markets:
            warnings.append(f"{adapter.name} does not support market {market}; skipped")
            continue
        if not adapter.is_connected():
            warnings.append(f"{adapter.name} not connected; call connect_broker('{adapter.name}')")
            continue

        try:
            rows = adapter.fetch_holdings(market)
        except Exception as e:
            warnings.append(f"{adapter.name} holdings fetch failed: {e}")
            continue

        brokers_used.append(adapter.name)

        for h in rows:
            d = h.to_dict()
            ticker = d["ticker"]
            if ticker in merged:
                m = merged[ticker]
                m["quantity"] += d["quantity"]
                m["invested_value"] += d["invested_value"]
                m["current_value"] += d["current_value"]
                m["brokers"] = sorted(set(m["brokers"] + [d["broker"]]))
            else:
                d["brokers"] = [d.pop("broker")]
                merged[ticker] = d

        try:
            cash = adapter.fetch_cash(market)
        except Exception as e:
            warnings.append(f"{adapter.name} cash fetch failed: {e}")
            cash = None

        if cash is not None:
            cash_total = (cash_total or 0.0) + cash
            cash_sources.append(adapter.name)

    if not brokers_used:
        return {
            "error": "no_broker_connected",
            "message": "No connected broker returned holdings. Call connect_broker first.",
            "warnings": warnings,
            "available_brokers": available_brokers(),
        }

    holdings = []
    for d in merged.values():
        q = d["quantity"]
        d["avg_price"] = round(d["invested_value"] / q, 4) if q else 0.0
        d["current_price"] = round(d["current_value"] / q, 4) if q else 0.0
        d["pnl"] = round(d["current_value"] - d["invested_value"], 2)
        d["pnl_pct"] = round(d["pnl"] / d["invested_value"] * 100, 2) if d["invested_value"] else 0.0
        d["invested_value"] = round(d["invested_value"], 2)
        d["current_value"] = round(d["current_value"], 2)
        d["quantity"] = round(q, 6)
        holdings.append(d)

    holdings.sort(key=lambda x: x["current_value"], reverse=True)

    if cash_total is None:
        cash_total = _cash_fallback()
        cash_source = "configured_fallback"
        warnings.append("No broker exposed a cash balance; using cash_available_fallback from risk settings")
    else:
        cash_source = "+".join(cash_sources)

    result = {
        "broker": broker,
        "cache_key": key,
        "market": market,
        "currency": "USD",
        "holdings": holdings,
        "cash_available": round(cash_total, 2),
        "cash_source": cash_source,
        "brokers": brokers_used,
        "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_at_epoch": time.time(),
        "cache_hit": False,
        "warnings": warnings,
    }

    _write_cache(result)
    return result


# ---------------------------------------------------------------------------
# Snapshot (what the agent should call)
# ---------------------------------------------------------------------------

def get_portfolio_snapshot(broker="all", market=DEFAULT_MARKET, refresh=False, max_position_pct=None):
    """
    Live portfolio summary for position sizing: holdings with weights, totals,
    cash, portfolio_size and concentration flags. US stocks only.
    """
    print(f"[tools] get_portfolio_snapshot called with broker={broker}, market={market}, refresh={refresh}")

    data = fetch_holdings(broker=broker, market=market, refresh=refresh)
    if "error" in data:
        return data

    holdings = data["holdings"]
    invested = sum(h["invested_value"] for h in holdings)
    current = sum(h["current_value"] for h in holdings)
    cash = data["cash_available"]
    portfolio_size = current + cash

    if max_position_pct is None:
        try:
            with open(RISK_SETTINGS_FILE, "r", encoding="utf-8") as f:
                max_position_pct = float(json.load(f).get("max_position_pct", 0.20))
        except (OSError, ValueError, json.JSONDecodeError):
            max_position_pct = 0.20

    for h in holdings:
        h["weight_pct"] = round(h["current_value"] / portfolio_size * 100, 2) if portfolio_size else 0.0

    top = sorted(holdings, key=lambda h: h["weight_pct"], reverse=True)[:3]
    over = [h["ticker"] for h in holdings if h["weight_pct"] > max_position_pct * 100]

    snapshot = {
        "market": market,
        "currency": "USD",
        "as_of": data["as_of"],
        "brokers": data["brokers"],
        "cache_hit": data.get("cache_hit", False),
        "holdings": [
            {
                "ticker": h["ticker"],
                "name": h.get("name", ""),
                "quantity": h["quantity"],
                "avg_price": h["avg_price"],
                "current_price": h["current_price"],
                "invested_value": h["invested_value"],
                "current_value": h["current_value"],
                "pnl": h["pnl"],
                "pnl_pct": h["pnl_pct"],
                "weight_pct": h["weight_pct"],
                "brokers": h["brokers"],
            }
            for h in holdings
        ],
        "totals": {
            "positions": len(holdings),
            "invested_value": round(invested, 2),
            "current_value": round(current, 2),
            "unrealised_pnl": round(current - invested, 2),
            "unrealised_pnl_pct": round((current - invested) / invested * 100, 2) if invested else 0.0,
            "cash_available": round(cash, 2),
            "cash_source": data["cash_source"],
            "portfolio_size": round(portfolio_size, 2),
            "cash_pct": round(cash / portfolio_size * 100, 2) if portfolio_size else 0.0,
        },
        "concentration": {
            "max_position_pct": max_position_pct * 100,
            "top_positions": [{"ticker": h["ticker"], "weight_pct": h["weight_pct"]} for h in top],
            "over_limit": over,
        },
        "warnings": data.get("warnings", []),
    }

    print(
        f"[tools] get_portfolio_snapshot: {len(holdings)} positions, value=${current:,.2f}, "
        f"cash=${cash:,.2f} ({data['cash_source']}), portfolio_size=${portfolio_size:,.2f}"
    )
    return snapshot


def get_holding(ticker, broker="all", market=DEFAULT_MARKET):
    """Convenience: a single ticker's merged holding or None."""
    data = fetch_holdings(broker=broker, market=market)
    if "error" in data:
        return None
    t = ticker.upper()
    return next((h for h in data["holdings"] if h["ticker"] == t), None)


def fetch_watchlist(broker="indmoney", market=DEFAULT_MARKET):
    """
    Live watchlist from the broker (INDmoney user_watchlist, US only).
    Falls back to local memory/watchlist.json when no broker is connected.
    """
    print(f"[tools] fetch_watchlist called with broker={broker}, market={market}")

    from tools.watchlist_memory import get_local_watchlist

    try:
        adapters = get_adapters(broker)
    except ValueError as e:
        local = get_local_watchlist()
        return {
            "tickers": local.get("tickers", []),
            "source": "local",
            "lists": [],
            "warnings": [str(e)],
        }

    tickers = []
    lists = []
    brokers_used = []
    warnings = []
    seen = set()

    for adapter in adapters:
        if market not in adapter.supported_markets:
            continue
        if not adapter.is_connected():
            warnings.append(f"{adapter.name} not connected")
            continue
        if not hasattr(adapter, "fetch_watchlist"):
            warnings.append(f"{adapter.name} has no fetch_watchlist")
            continue
        try:
            data = adapter.fetch_watchlist(market)
        except Exception as e:
            warnings.append(f"{adapter.name} watchlist fetch failed: {e}")
            continue

        brokers_used.append(adapter.name)
        for t in data.get("tickers") or []:
            t = str(t).upper()
            if t and t not in seen:
                seen.add(t)
                tickers.append(t)
        for lst in data.get("lists") or []:
            lists.append({"broker": adapter.name, **lst})

    if not brokers_used:
        local = get_local_watchlist()
        return {
            "tickers": [str(t).upper() for t in local.get("tickers", [])],
            "source": "local",
            "lists": [],
            "warnings": warnings or ["no_broker_connected; using local watchlist.json"],
        }

    # Merge any extra local tickers the user added in-app
    local = get_local_watchlist()
    for t in local.get("tickers") or []:
        t = str(t).upper()
        if t and t not in seen:
            seen.add(t)
            tickers.append(t)

    print(f"[tools] fetch_watchlist: {len(tickers)} tickers from {brokers_used}")
    return {
        "tickers": tickers,
        "source": "+".join(brokers_used),
        "lists": lists,
        "brokers": brokers_used,
        "warnings": warnings,
    }
