def calculate_position_risk(
    entry_price,
    current_price,
    atr,
    shares_owned=None,
    portfolio_size=None,
    risk_percent=None,
    ticker=None,
):
    """
    Risk metrics for an existing or planned position.

    shares_owned and portfolio_size are optional: when omitted they are read
    from the live holdings snapshot (shares_owned requires `ticker`).
    """
    from tools.trade_planner import resolve_risk_percent
    risk_percent = resolve_risk_percent(risk_percent)

    print(
        f"[tools] calculate_position_risk called with "
        f"entry_price={entry_price}, current_price={current_price}, atr={atr}, "
        f"shares_owned={shares_owned}, portfolio_size={portfolio_size}, "
        f"risk_percent={risk_percent}, ticker={ticker}"
    )

    sources = {}
    if shares_owned is None or portfolio_size is None:
        from tools.holdings_aggregator import get_portfolio_snapshot

        snap = get_portfolio_snapshot()
        if "error" in snap:
            return {"error": "shares_owned/portfolio_size not provided and no broker connected", "detail": snap}

        if portfolio_size is None:
            portfolio_size = snap["totals"]["portfolio_size"]
            sources["portfolio_size"] = "holdings_snapshot"

        if shares_owned is None:
            if not ticker:
                return {"error": "shares_owned omitted: pass ticker so it can be read from holdings"}
            h = next((x for x in snap["holdings"] if x["ticker"] == ticker.upper()), None)
            shares_owned = h["quantity"] if h else 0.0
            sources["shares_owned"] = "holdings_snapshot" if h else "not_held"

    atr_stop = current_price - (1.5 * atr)

    risk_per_share = entry_price - atr_stop
    total_risk = shares_owned * risk_per_share

    portfolio_risk_percent = (total_risk / portfolio_size) * 100 if portfolio_size > 0 else 0

    max_risk_dollars = portfolio_size * risk_percent

    fresh_risk_per_share = current_price - atr_stop

    optimal_shares = max_risk_dollars / fresh_risk_per_share if fresh_risk_per_share > 0 else 0

    return {
        "shares_owned": float(shares_owned),
        "portfolio_size": float(portfolio_size),
        "sources": sources,
        "risk_per_share": float(risk_per_share),
        "total_risk": float(total_risk),
        "portfolio_risk_percent": float(portfolio_risk_percent),
        "max_risk_dollars": float(max_risk_dollars),
        "optimal_shares": float(optimal_shares),
        "atr_stop": float(atr_stop)
    }
