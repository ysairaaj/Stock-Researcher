"""
Risk/reward trade planner driven by live holdings.

plan_trade() answers "how much should I buy?" using:
- portfolio_size and cash from get_portfolio_snapshot (never from the user)
- risk settings in memory/risk_settings.json
- the proposed entry, stop and target (defaults from the risk settings)
"""
import json
import os

from tools.holdings_aggregator import get_portfolio_snapshot, RISK_SETTINGS_FILE

DEFAULT_RISK_SETTINGS = {
    "risk_percent": 0.02,            # max loss per trade as a fraction of portfolio_size
    "max_position_pct": 0.20,        # max single-position weight after the trade
    "max_portfolio_heat": 0.20,      # max total open risk across all positions
    "min_reward_risk": 2.0,
    "default_stop_pct": 15.0,
    "default_target_pct": 30.0,
    "cash_available_fallback": 0.0,  # used only when no broker exposes cash
    "allow_fractional_shares": True, # INDmoney US supports fractional shares
}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _load_settings():
    settings = dict(DEFAULT_RISK_SETTINGS)
    if os.path.exists(RISK_SETTINGS_FILE):
        try:
            with open(RISK_SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
    else:
        _save_settings(settings)
    return settings


def _save_settings(settings):
    os.makedirs(os.path.dirname(RISK_SETTINGS_FILE), exist_ok=True)
    with open(RISK_SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def resolve_risk_percent(risk_percent):
    """None uses the saved risk_percent. Values above 1 are treated as percents."""
    if risk_percent is None:
        risk_percent = float(_load_settings()["risk_percent"])
    else:
        risk_percent = float(risk_percent)
    if risk_percent > 1:
        risk_percent /= 100.0
    return risk_percent


def get_risk_settings():
    print("[tools] get_risk_settings called")
    return _load_settings()


def set_risk_settings(**kwargs):
    """
    Update one or more risk settings. Percent-style inputs are accepted as
    fractions (0.02) or percents (2) for risk_percent, max_position_pct,
    max_portfolio_heat.
    """
    print(f"[tools] set_risk_settings called with {kwargs}")
    settings = _load_settings()
    changed = {}

    for key, value in kwargs.items():
        if key not in DEFAULT_RISK_SETTINGS:
            return {"error": f"unknown setting '{key}'", "allowed": list(DEFAULT_RISK_SETTINGS)}
        if key in ("risk_percent", "max_position_pct", "max_portfolio_heat"):
            value = float(value)
            if value > 1:
                value = value / 100.0
        elif key == "allow_fractional_shares":
            value = bool(value)
        else:
            value = float(value)
        settings[key] = value
        changed[key] = value

    _save_settings(settings)
    return {"status": "updated", "changed": changed, "settings": settings}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

def plan_trade(
    ticker,
    entry_price,
    stop_loss=None,
    target_price=None,
    risk_percent=None,
    broker="all",
    refresh=False,
):
    print(
        f"[tools] plan_trade called with ticker={ticker}, entry_price={entry_price}, "
        f"stop_loss={stop_loss}, target_price={target_price}, risk_percent={risk_percent}"
    )

    settings = _load_settings()
    ticker = ticker.upper()
    entry_price = float(entry_price)

    if entry_price <= 0:
        return {"error": "entry_price must be positive"}

    snapshot = get_portfolio_snapshot(broker=broker, refresh=refresh)
    if "error" in snapshot:
        return snapshot

    totals = snapshot["totals"]
    portfolio_size = totals["portfolio_size"]
    cash = totals["cash_available"]

    if portfolio_size <= 0:
        return {"error": "portfolio_size is zero; nothing to size against", "snapshot_totals": totals}

    risk_percent = resolve_risk_percent(risk_percent)

    stop_loss = float(stop_loss) if stop_loss is not None else entry_price * (1 - settings["default_stop_pct"] / 100)
    target_price = float(target_price) if target_price is not None else entry_price * (1 + settings["default_target_pct"] / 100)

    risk_per_share = entry_price - stop_loss
    if risk_per_share <= 0:
        return {"error": "stop_loss must be below entry_price for a long trade"}

    reward_per_share = target_price - entry_price
    reward_risk = reward_per_share / risk_per_share if risk_per_share else 0.0

    existing = next((h for h in snapshot["holdings"] if h["ticker"] == ticker), None)
    existing_value = existing["current_value"] if existing else 0.0

    # Open risk of current holdings, assuming each is protected at default_stop_pct
    heat_existing_amount = totals["current_value"] * (settings["default_stop_pct"] / 100)

    constraints = []
    reasons = []

    # 1. Risk-based size
    risk_amount_allowed = portfolio_size * risk_percent
    shares = risk_amount_allowed / risk_per_share
    reasons.append(
        f"risk budget ${risk_amount_allowed:,.2f} ({risk_percent:.1%} of ${portfolio_size:,.2f}) / "
        f"${risk_per_share:.2f} risk per share -> {shares:.4f} shares"
    )
    risk_based_shares = shares

    # 2. Cash cap
    if shares * entry_price > cash:
        shares = cash / entry_price
        constraints.append("cash")
        reasons.append(f"capped by available cash ${cash:,.2f} -> {shares:.4f} shares")

    # 3. Concentration cap
    max_value = portfolio_size * settings["max_position_pct"]
    room = max_value - existing_value
    if room <= 0:
        shares = 0.0
        constraints.append("concentration")
        reasons.append(
            f"{ticker} already ${existing_value:,.2f} >= max position ${max_value:,.2f} "
            f"({settings['max_position_pct']:.0%}); no room"
        )
    elif shares * entry_price > room:
        shares = room / entry_price
        constraints.append("concentration")
        reasons.append(f"capped by max position {settings['max_position_pct']:.0%} -> {shares:.4f} shares")

    # 4. Portfolio heat cap
    heat_limit_amount = portfolio_size * settings["max_portfolio_heat"]
    heat_room = heat_limit_amount - heat_existing_amount
    if heat_room <= 0:
        shares = 0.0
        constraints.append("portfolio_heat")
        reasons.append(
            f"existing open risk ${heat_existing_amount:,.2f} >= heat limit ${heat_limit_amount:,.2f} "
            f"({settings['max_portfolio_heat']:.0%}); no room"
        )
    elif shares * risk_per_share > heat_room:
        shares = heat_room / risk_per_share
        constraints.append("portfolio_heat")
        reasons.append(f"capped by portfolio heat {settings['max_portfolio_heat']:.0%} -> {shares:.4f} shares")

    if not settings.get("allow_fractional_shares", True):
        shares = float(int(shares))

    shares = max(0.0, round(shares, 4))
    position_value = shares * entry_price
    risk_amount = shares * risk_per_share
    reward_amount = shares * reward_per_share

    # Decision
    if reward_risk < settings["min_reward_risk"]:
        decision = "reject"
        reasons.append(
            f"reward:risk {reward_risk:.2f} below minimum {settings['min_reward_risk']:.1f}"
        )
    elif shares <= 0:
        decision = "reject"
        reasons.append("no size available after constraints")
    elif risk_based_shares > 0 and shares < 0.5 * risk_based_shares:
        decision = "reduce"
        reasons.append("constraints cut the risk-based size by more than half")
    else:
        decision = "fit"

    new_position_value = existing_value + position_value
    result = {
        "ticker": ticker,
        "decision": decision,
        "entry_price": round(entry_price, 4),
        "stop_loss": round(stop_loss, 4),
        "target_price": round(target_price, 4),
        "stop_pct": round(risk_per_share / entry_price * 100, 2),
        "target_pct": round(reward_per_share / entry_price * 100, 2),
        "reward_risk": round(reward_risk, 2),
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_pct_of_portfolio": round(risk_amount / portfolio_size * 100, 2),
        "reward_amount": round(reward_amount, 2),
        "cash_before": round(cash, 2),
        "cash_after": round(cash - position_value, 2),
        "position_weight_after_pct": round(new_position_value / portfolio_size * 100, 2),
        "portfolio_heat_after_pct": round((heat_existing_amount + risk_amount) / portfolio_size * 100, 2),
        "existing_position": existing,
        "portfolio_size": round(portfolio_size, 2),
        "constraints_applied": constraints,
        "reasons": reasons,
        "settings_used": {
            "risk_percent": risk_percent,
            "max_position_pct": settings["max_position_pct"],
            "max_portfolio_heat": settings["max_portfolio_heat"],
            "min_reward_risk": settings["min_reward_risk"],
        },
        "snapshot_as_of": snapshot["as_of"],
    }

    print(
        f"[tools] plan_trade: {ticker} decision={decision} shares={shares} "
        f"value=${position_value:,.2f} risk=${risk_amount:,.2f} R:R={reward_risk:.2f} "
        f"constraints={constraints}"
    )
    return result
