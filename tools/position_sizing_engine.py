def _snapshot_portfolio_size():
    """Live portfolio_size (holdings + cash) from connected brokers, or None."""
    from tools.holdings_aggregator import get_portfolio_snapshot

    snap = get_portfolio_snapshot()
    if "error" in snap:
        return None, snap
    return snap["totals"]["portfolio_size"], snap


def calculate_position_size(
    entry_price,
    stop_loss,
    portfolio_size=None,
    risk_percent=None
):
    """
    Risk-based share count. portfolio_size is optional: when omitted it is
    taken from the live holdings snapshot (holdings value + cash).
    For a full plan with cash / concentration / heat caps use plan_trade.
    """
    from tools.trade_planner import resolve_risk_percent
    risk_percent = resolve_risk_percent(risk_percent)

    print(
        f"[tools] calculate_position_size called with "
        f"portfolio_size={portfolio_size}, entry_price={entry_price}, "
        f"stop_loss={stop_loss}, risk_percent={risk_percent}"
    )

    portfolio_source = "provided"
    if portfolio_size is None:
        portfolio_size, snap = _snapshot_portfolio_size()
        if portfolio_size is None:
            return {
                "error": "portfolio_size not provided and no broker connected",
                "detail": snap,
            }
        portfolio_source = "holdings_snapshot"

    risk_amount = portfolio_size * risk_percent

    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share == 0:
        return {"error": "invalid stop loss"}

    shares = risk_amount / risk_per_share

    position_value = shares * entry_price

    return {
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "risk_percent": risk_percent,
        "portfolio_size": portfolio_size,
        "portfolio_source": portfolio_source,
        "risk_amount": risk_amount,
        "shares": round(shares, 4),
        "position_value": position_value
    }
