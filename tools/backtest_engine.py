from tools.ath_tools import compute_ath_stats
from tools.price_history import get_weekly_history, LOOKBACK_YEARS, INTERVAL


STRATEGY_PRIORITY = ["trend_pullback", "breakout", "mean_reversion"]

# All bars are WEEKLY over a 7-year window (about 364 bars):
#   ma20 / ma50            -> 20-week / 50-week moving averages
#   prior_20w_high         -> highest high of the previous 20 weeks
#   prior_10w_low          -> lowest low of the previous 10 weeks

# Breakout "current state" check is intentionally looser than the backtest
# entry rule (strict close above prior 20-week high), which is rarely true on
# any single bar.
BREAKOUT_PROXIMITY = 0.98


def _check_suitability(strategy, state):
    """
    Decide whether the stock's latest bar matches the entry conditions of a
    strategy. Returns (suitable: bool, reason: str).
    """
    close = state["close"]
    ma20 = state["ma20"]
    ma50 = state["ma50"]
    prior_20w_high = state["prior_20w_high"]
    prior_10w_low = state["prior_10w_low"]
    ath_zone = (state.get("ath_stats") or {}).get("zone")

    if strategy == "trend_pullback":
        ok = close > ma50 and close < ma20
        reason = (
            f"close {close:.2f} {'>' if close > ma50 else '<='} ma50 {ma50:.2f} "
            f"and {'<' if close < ma20 else '>='} ma20 {ma20:.2f}"
        )
        return ok, reason

    if strategy == "breakout":
        near_high = close >= prior_20w_high * BREAKOUT_PROXIMITY
        near_ath = ath_zone == "near_ath"
        ok = near_high or near_ath
        reason = (
            f"close {close:.2f} vs prior 20-week high {prior_20w_high:.2f} "
            f"(within {int((1 - BREAKOUT_PROXIMITY) * 100)}%: {near_high}), "
            f"ath_zone={ath_zone} (near_ath: {near_ath})"
        )
        return ok, reason

    if strategy == "mean_reversion":
        ok = close < prior_10w_low
        reason = (
            f"close {close:.2f} {'<' if ok else '>='} prior 10-week low {prior_10w_low:.2f}"
        )
        return ok, reason

    return False, f"unknown strategy '{strategy}'"


def backtest_strategy(
    ticker,
    strategy=None,
    initial_capital=10000,
    target_pct=20,
    stop_pct=10
):
    logs = []

    def log(msg):
        line = f"[tools] backtest_strategy: {msg}"
        print(line)
        logs.append(msg)

    target_pct = float(target_pct)
    stop_pct = float(stop_pct)

    log(
        f"params received ticker={ticker}, strategy={strategy or 'auto'}, "
        f"initial_capital={initial_capital}, target_pct={target_pct}, stop_pct={stop_pct}"
    )

    if target_pct <= 0 or stop_pct <= 0:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "error": "target_pct and stop_pct must both be greater than zero",
            "logs": logs,
        }

    if strategy is not None and strategy not in STRATEGY_PRIORITY:
        log(f"unknown strategy '{strategy}', ignoring and using auto-selection")
        strategy = None

    target_multiplier = 1 + (target_pct / 100)
    stop_multiplier = 1 - (stop_pct / 100)

    df = get_weekly_history(ticker)
    if df is None or df.empty or len(df) < 51:
        return {
            "ticker": ticker,
            "strategy": strategy,
            "error": "No or insufficient historical market data found",
            "logs": logs,
        }

    df = df.rename(columns=str.lower)

    log(f"loaded {len(df)} {INTERVAL} bars over {LOOKBACK_YEARS} years")

    df["ma50"] = df["close"].rolling(50).mean()
    df["ma20"] = df["close"].rolling(20).mean()

    # -------- Current state (latest weekly bar) --------

    current_state = {
        "interval": INTERVAL,
        "lookback_years": LOOKBACK_YEARS,
        "bars": int(len(df)),
        "close": float(df["close"].iloc[-1]),
        "ma20": float(df["ma20"].iloc[-1]),
        "ma50": float(df["ma50"].iloc[-1]),
        "prior_20w_high": float(df["high"].iloc[-21:-1].max()),
        "prior_10w_low": float(df["low"].iloc[-11:-1].min()),
        "ath_stats": compute_ath_stats(df),
    }

    log(
        f"current state close={current_state['close']:.2f} "
        f"ma20w={current_state['ma20']:.2f} ma50w={current_state['ma50']:.2f} "
        f"prior_20w_high={current_state['prior_20w_high']:.2f} "
        f"prior_10w_low={current_state['prior_10w_low']:.2f} "
        f"ath_zone={(current_state['ath_stats'] or {}).get('zone')}"
    )

    # -------- Strategy selection --------

    requested_strategy = strategy
    order = []
    if requested_strategy:
        order.append(requested_strategy)
    order += [s for s in STRATEGY_PRIORITY if s not in order]

    suitability = {}
    selected_strategy = None

    for candidate in order:
        ok, reason = _check_suitability(candidate, current_state)
        suitability[candidate] = {"suitable": ok, "reason": reason}
        log(f"suitability {candidate}: {'OK' if ok else 'NOT suitable'} ({reason})")
        if ok:
            selected_strategy = candidate
            break

    strategy_overridden = bool(
        requested_strategy
        and selected_strategy
        and selected_strategy != requested_strategy
    )

    if strategy_overridden:
        log(
            f"requested {requested_strategy} not suitable -> overridden to {selected_strategy}"
        )

    if selected_strategy is None:
        log("decision: no_trade (no strategy conditions match current state)")
        return {
            "ticker": ticker,
            "decision": "no_trade",
            "reason": (
                "No strategy conditions match the current price state; "
                "do not take a trade based on backtest"
            ),
            "requested_strategy": requested_strategy,
            "selected_strategy": None,
            "strategy_overridden": False,
            "suitability": suitability,
            "current_state": current_state,
            "target_pct": target_pct,
            "stop_pct": stop_pct,
            "logs": logs,
        }

    log(f"decision: trade, running backtest with strategy={selected_strategy}")

    # -------- Backtest loop --------

    trades = []

    capital = initial_capital
    position = None

    for i in range(50, len(df)):

        price = df["close"].iloc[i]

        # -------- Trend Pullback --------

        if position is None and selected_strategy == "trend_pullback":

            if price > df["ma50"].iloc[i] and price < df["ma20"].iloc[i]:

                entry = price
                stop = entry * stop_multiplier
                target = entry * target_multiplier

                position = (entry, stop, target)

        # -------- Breakout --------

        if position is None and selected_strategy == "breakout":

            prev_high = df["high"].iloc[i-20:i].max()

            if price > prev_high:

                entry = price
                stop = entry * stop_multiplier
                target = entry * target_multiplier

                position = (entry, stop, target)

        # -------- Mean Reversion --------

        if position is None and selected_strategy == "mean_reversion":

            prev_low = df["low"].iloc[i-10:i].min()

            if price < prev_low:

                entry = price
                stop = entry * stop_multiplier
                target = entry * target_multiplier

                position = (entry, stop, target)

        # -------- Trade management --------

        if position:

            entry, stop, target = position

            if price <= stop:

                capital *= stop_multiplier
                trades.append("loss")
                position = None

            elif price >= target:

                capital *= target_multiplier
                trades.append("win")
                position = None

    wins = trades.count("win")
    losses = trades.count("loss")

    total = wins + losses

    win_rate = wins / total if total > 0 else 0

    log(
        f"result trades={total} wins={wins} losses={losses} "
        f"win_rate={win_rate:.2%} final_capital={capital:.2f}"
    )

    return {
        "ticker": ticker,
        "decision": "trade",
        "strategy": selected_strategy,
        "requested_strategy": requested_strategy,
        "selected_strategy": selected_strategy,
        "strategy_overridden": strategy_overridden,
        "suitability": suitability,
        "current_state": current_state,
        "target_pct": target_pct,
        "stop_pct": stop_pct,
        "reward_risk_ratio": target_pct / stop_pct,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "final_capital": capital,
        "return_pct": (capital - initial_capital) / initial_capital,
        "logs": logs,
    }
