"""
Distance-from-all-time-high analysis on WEEKLY bars over a 7-year window.

Hybrid algorithm:
- drawdown_from_ath_pct: how far the current close is below the period high.
- range_percentile: where the current close sits inside the period low..high
  range (0 = at the low, 100 = at the high). This catches low-volatility stocks
  whose drawdown looks small but are actually near their period low.

Zones:
- near_ath   : drawdown <= ATH_THRESHOLD_PCT
- near_low   : range_percentile <= LOW_PERCENTILE
- mid_range  : everything else
"""
from tools.price_history import get_weekly_history, LOOKBACK_YEARS, INTERVAL

ATH_THRESHOLD_PCT = 5
LOW_PERCENTILE = 10


def compute_ath_stats(df):
    """
    Compute ATH statistics from a yfinance history frame.
    Accepts either capitalised (High/Low/Close) or lower-case column names.
    Returns None if the frame is empty.
    """
    if df is None or df.empty:
        return None

    cols = {c.lower(): c for c in df.columns}
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")

    if not all([high_col, low_col, close_col]):
        return None

    ath = float(df[high_col].max())
    ath_date = str(df[high_col].idxmax().date())

    period_low = float(df[low_col].min())
    period_low_date = str(df[low_col].idxmin().date())

    close = float(df[close_col].iloc[-1])

    drawdown_from_ath_pct = (ath - close) / ath * 100 if ath > 0 else 0.0

    price_range = ath - period_low
    range_percentile = (
        (close - period_low) / price_range * 100 if price_range > 0 else 100.0
    )

    if drawdown_from_ath_pct <= ATH_THRESHOLD_PCT:
        zone = "near_ath"
    elif range_percentile <= LOW_PERCENTILE:
        zone = "near_low"
    else:
        zone = "mid_range"

    return {
        "current_price": close,
        "ath": ath,
        "ath_date": ath_date,
        "period_low": period_low,
        "period_low_date": period_low_date,
        "drawdown_from_ath_pct": round(drawdown_from_ath_pct, 2),
        "range_percentile": round(range_percentile, 2),
        "zone": zone,
        "ath_threshold_pct": ATH_THRESHOLD_PCT,
        "low_percentile_threshold": LOW_PERCENTILE,
        "lookback_years": LOOKBACK_YEARS,
        "interval": INTERVAL,
        "bars": int(len(df)),
    }


def check_distance_from_ath(ticker):
    """
    Agent-facing tool: where is the current price relative to the 7-year
    weekly all-time high and low?
    """
    print(f"[tools] check_distance_from_ath called with ticker={ticker}")

    df = get_weekly_history(ticker)

    stats = compute_ath_stats(df)

    if stats is None:
        print(f"[tools] check_distance_from_ath: no data for {ticker}")
        return {"ticker": ticker, "error": "No historical market data found"}

    print(
        f"[tools] check_distance_from_ath: zone={stats['zone']} "
        f"drawdown={stats['drawdown_from_ath_pct']}% "
        f"percentile={stats['range_percentile']}% "
        f"({stats['bars']} weekly bars)"
    )

    return {"ticker": ticker, **stats}
