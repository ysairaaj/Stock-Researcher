"""
Compact multi-ticker scanner for portfolio / watchlist research.

Returns short rows only (no full support/resistance lists) so the agent
context is not flooded.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from tools.price_history import get_weekly_history
from tools.ath_tools import compute_ath_stats
from tools.signal_engine import analyze_structure

MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)

NEAR_SUPPORT_PCT = 5.0
NEAR_RESISTANCE_PCT = 3.0


def _classify_momentum(df):
    """20-week return on weekly bars: strong / weak / negative / unknown."""
    if df is None or df.empty or "Close" not in df.columns or len(df) < 21:
        return "unknown", None
    recent = df["Close"].iloc[-21:]
    ret = float((recent.iloc[-1] / recent.iloc[0]) - 1)
    if ret > 0.10:
        label = "strong"
    elif ret < -0.05:
        label = "negative"
    else:
        label = "weak"
    return label, round(ret * 100, 2)


def _trend_from_weekly(df, ath_zone):
    """
    Trend from weekly MAs (and ATH zone as a tie-break), not from a 7-year
    swing range that marks almost everything as sideways.
    """
    if df is None or df.empty or len(df) < 50 or "Close" not in df.columns:
        return "unknown"

    close = float(df["Close"].iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1])

    if close > ma20 > ma50:
        return "bullish"
    if close < ma20 < ma50:
        return "bearish"
    if ath_zone == "near_ath" and close > ma50:
        return "bullish"
    if ath_zone == "near_low" and close < ma50:
        return "bearish"
    return "sideways"


def _structure_label(support_distance, resistance_distance):
    if support_distance is not None and support_distance <= NEAR_SUPPORT_PCT:
        return "near_support"
    if resistance_distance is not None and resistance_distance <= NEAR_RESISTANCE_PCT:
        return "near_resistance"
    if (
        support_distance is not None
        and resistance_distance is not None
        and support_distance > 15
        and resistance_distance > 10
    ):
        return "mid_range_extended"
    return "mid_range"


def analyze_ticker(ticker, source=None, weight_pct=None):
    """
    Compact analysis for one ticker. Optional source/weight_pct are passed
    through for portfolio context.
    """
    print(f"[tools] analyze_ticker called with ticker={ticker}")

    try:
        df = get_weekly_history(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            print(f"[tools] analyze_ticker: skipping {ticker} due to no weekly data")
            return None

        price = float(df["Close"].iloc[-1])
        ath = compute_ath_stats(df) or {}
        ath_zone = ath.get("zone")
        trend = _trend_from_weekly(df, ath_zone)
        momentum, momentum_ret_20w = _classify_momentum(df)

        structure = analyze_structure(ticker)
        if not isinstance(structure, dict) or "error" in structure:
            print(
                f"[tools] analyze_ticker: skipping {ticker} "
                f"due to structure error={structure.get('error') if isinstance(structure, dict) else structure}"
            )
            return None

        nearest_support = structure.get("nearest_support")
        nearest_resistance = structure.get("nearest_resistance")

        support_distance = None
        if nearest_support is not None:
            support_distance = round(abs(price - nearest_support) / price * 100, 2)

        resistance_distance = None
        if nearest_resistance is not None:
            resistance_distance = round(abs(nearest_resistance - price) / price * 100, 2)

        weekly_atr = structure.get("weekly_atr")
        weekly_atr_pct = structure.get("weekly_atr_pct")
        label = _structure_label(support_distance, resistance_distance)

        row = {
            "ticker": ticker.upper(),
            "price": round(price, 4),
            "trend": trend,
            "label": label,
            "support_distance": support_distance,
            "nearest_support": nearest_support,
            "resistance_distance": resistance_distance,
            "nearest_resistance": nearest_resistance,
            "momentum": momentum,
            "momentum_ret_20w_pct": momentum_ret_20w,
            "weekly_atr": weekly_atr,
            "weekly_atr_pct": weekly_atr_pct,
            "ath_zone": ath_zone,
            "drawdown_from_ath_pct": ath.get("drawdown_from_ath_pct"),
            "range_percentile": ath.get("range_percentile"),
        }
        if source is not None:
            row["source"] = source
        if weight_pct is not None:
            row["weight_pct"] = weight_pct
        return row

    except Exception as e:
        print(f"[tools] analyze_ticker: exception for {ticker}: {e}")
        return None


def scan_stocks(tickers, meta=None):
    """
    Scan many tickers. meta is an optional dict ticker -> {source, weight_pct}.
    Returns {results: [compact rows...]}.
    """
    print(f"[tools] scan_stocks called with tickers={tickers}")

    if not tickers:
        return {"results": []}

    tickers = [str(t).upper() for t in list(tickers)[:100]]
    meta = meta or {}
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for ticker in tickers:
            info = meta.get(ticker, {})
            futures[
                executor.submit(
                    analyze_ticker,
                    ticker,
                    info.get("source"),
                    info.get("weight_pct"),
                )
            ] = ticker

        for future in as_completed(futures):
            try:
                result = future.result(timeout=60)
            except Exception as e:
                print(f"[tools] scan_stocks: worker error: {e}")
                continue
            if result:
                results.append(result)

    results.sort(key=lambda r: r.get("ticker", ""))
    return {"results": results}
