import numpy as np

from tools.price_history import get_weekly_history, LOOKBACK_YEARS, INTERVAL


# Weekly returns are roughly sqrt(5) times more volatile than daily returns,
# so the old daily threshold of 3% is rescaled for weekly bars.
HIGH_VOL_THRESHOLD = 0.065


def detect_market_regime(symbol="SPY"):

    print(f"[tools] detect_market_regime called with symbol={symbol}")

    # Weekly bars over a 7-year window (about 364 bars).
    df = get_weekly_history(symbol)

    if df is None or df.empty or "Close" not in df.columns:
        print(f"[tools] detect_market_regime: no data for {symbol}, defaulting")
        return {
            "symbol": symbol,
            "trend": "sideways",
            "volatility": "normal_volatility",
            "volatility_value": 0.0,
            "interval": INTERVAL,
            "lookback_years": LOOKBACK_YEARS,
        }

    df = df.copy()
    df["ma50"] = df["Close"].rolling(50).mean()
    df["ma200"] = df["Close"].rolling(200).mean()

    df["returns"] = df["Close"].pct_change()

    # Require at least 200 weekly bars so the long MA is meaningful
    if len(df) < 200:
        print(f"[tools] detect_market_regime: insufficient history for {symbol}")
        return {
            "symbol": symbol,
            "trend": "sideways",
            "volatility": "normal_volatility",
            "volatility_value": 0.0,
            "interval": INTERVAL,
            "lookback_years": LOOKBACK_YEARS,
        }

    volatility = df["returns"].rolling(20).std().iloc[-1]

    price = df["Close"].iloc[-1]
    ma50 = df["ma50"].iloc[-1]
    ma200 = df["ma200"].iloc[-1]

    # ----- Trend detection (weekly MAs) -----

    if price > ma50 > ma200:
        trend = "bull"
    elif price < ma50 < ma200:
        trend = "bear"
    else:
        trend = "sideways"

    # ----- Volatility detection -----

    if volatility > HIGH_VOL_THRESHOLD:
        volatility_regime = "high_volatility"
    else:
        volatility_regime = "normal_volatility"

    print(
        f"[tools] detect_market_regime: trend={trend} price={price:.2f} "
        f"ma50w={ma50:.2f} ma200w={ma200:.2f} vol={volatility:.4f} "
        f"({len(df)} weekly bars)"
    )

    return {
        "symbol": symbol,
        "trend": trend,
        "volatility": volatility_regime,
        "volatility_value": float(volatility),
        "price": float(price),
        "ma50_weekly": float(ma50),
        "ma200_weekly": float(ma200),
        "interval": INTERVAL,
        "lookback_years": LOOKBACK_YEARS,
        "bars": int(len(df)),
    }
