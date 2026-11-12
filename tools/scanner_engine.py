from concurrent.futures import ThreadPoolExecutor, as_completed

from tools.market_data_engine import get_market_data
from tools.signal_engine import analyze_structure
import os
import yfinance as yf

MAX_WORKERS = min(32, os.cpu_count() * 2)


def _classify_momentum(ticker):

    """
    Simple momentum classification based on recent returns.

    - strong  : 20‑day return > +10%
    - weak    : between -5% and +10%
    - negative: < -5%
    """

    try:
        df = yf.Ticker(ticker).history(period="6mo")

        if df is None or df.empty or "Close" not in df.columns:
            return "unknown"

        if len(df) < 21:
            return "unknown"

        recent = df["Close"].iloc[-21:]
        ret_20d = (recent.iloc[-1] / recent.iloc[0]) - 1

        if ret_20d > 0.10:
            return "strong"
        if ret_20d < -0.05:
            return "negative"
        return "weak"

    except Exception as e:
        print(f"[tools] _classify_momentum: exception for {ticker}: {e}")
        return "unknown"


def analyze_ticker(ticker):

    print(f"[tools] analyze_ticker called with ticker={ticker}")

    try:
        market = get_market_data(ticker)

        # Skip tickers where we can't get usable market data
        if not isinstance(market, dict) or "error" in market:
            print(f"[tools] analyze_ticker: skipping {ticker} due to bad market data")
            return None

        structure = analyze_structure(ticker)

        # Skip tickers where structure analysis failed
        if not isinstance(structure, dict) or "error" in structure:
            print(
                f"[tools] analyze_ticker: skipping {ticker} "
                f"due to structure error={structure.get('error')}"
            )
            return None

        price = market.get("current_price")

        # analyze_structure returns supports/resistance sorted nearest-first
        # and exposes the nearest level directly.
        nearest_support = structure.get("nearest_support")
        nearest_resistance = structure.get("nearest_resistance")

        support_distance = None
        if price is not None and nearest_support is not None:
            support_distance = abs(price - nearest_support) / price * 100

        resistance_distance = None
        if price is not None and nearest_resistance is not None:
            resistance_distance = abs(nearest_resistance - price) / price * 100

        # Very simple trend proxy based on price vs swing range
        recent_high = structure.get("recent_high", price)
        recent_low = structure.get("recent_low", price)

        if price is not None and recent_high is not None and recent_low is not None:
            if price > recent_high:
                trend = "bullish"
            elif price < recent_low:
                trend = "bearish"
            else:
                trend = "sideways"
        else:
            trend = "unknown"

        momentum = _classify_momentum(ticker)

        return {
            "ticker": ticker,
            "trend": trend,
            "support_distance": support_distance,
            "nearest_support": nearest_support,
            "resistance_distance": resistance_distance,
            "nearest_resistance": nearest_resistance,
            "momentum": momentum,
        }

    except Exception as e:
        print(f"[tools] analyze_ticker: exception for {ticker}: {e}")
        return None


def scan_stocks(tickers):

    print(f"[tools] scan_stocks called with tickers={tickers}")

    if not tickers:
        return {"results": []}

    # Hard cap to avoid huge scans causing very long runs
    tickers = list(tickers)[:100]

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {
            executor.submit(analyze_ticker, ticker): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
            except Exception as e:
                print(f"[tools] scan_stocks: worker error: {e}")
                continue

            if result:
                results.append(result)

    return {"results": results}