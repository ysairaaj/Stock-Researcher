from tools.market_data_engine import get_market_data
from tools.risk_engine import calculate_position_risk


def evaluate_position(ticker, entry_price=None, shares_owned=None, portfolio_size=None, risk_percent=None):
    """
    High-level evaluation of a single position.

    entry_price, shares_owned and portfolio_size are optional: when omitted
    they are read from the live holdings snapshot (avg_price, quantity and
    holdings value + cash respectively).
    """
    from tools.trade_planner import resolve_risk_percent
    risk_percent = resolve_risk_percent(risk_percent)

    print(
        f"[tools] evaluate_position called with "
        f"ticker={ticker}, entry_price={entry_price}, "
        f"shares_owned={shares_owned}, portfolio_size={portfolio_size}, "
        f"risk_percent={risk_percent}"
    )

    ticker = ticker.upper()
    holding = None

    if entry_price is None or shares_owned is None or portfolio_size is None:
        from tools.holdings_aggregator import get_portfolio_snapshot

        snap = get_portfolio_snapshot()
        if "error" in snap:
            return {"error": "position details not provided and no broker connected", "detail": snap}

        holding = next((h for h in snap["holdings"] if h["ticker"] == ticker), None)

        if portfolio_size is None:
            portfolio_size = snap["totals"]["portfolio_size"]

        if holding is None and (entry_price is None or shares_owned is None):
            return {
                "error": f"{ticker} is not in the live holdings; pass entry_price and shares_owned explicitly",
                "held_tickers": [h["ticker"] for h in snap["holdings"]],
            }

        if entry_price is None:
            entry_price = holding["avg_price"]
        if shares_owned is None:
            shares_owned = holding["quantity"]

    market = get_market_data(ticker)

    if "error" in market:
        return market

    risk = calculate_position_risk(
        entry_price,
        market["current_price"],
        market["weekly_atr"],
        shares_owned,
        portfolio_size,
        risk_percent,
        ticker=ticker,
    )

    return {
        "ticker": ticker,
        "holding": holding,
        "entry_price": entry_price,
        "shares_owned": shares_owned,
        "portfolio_size": portfolio_size,
        "market": market,
        "risk": risk
    }
