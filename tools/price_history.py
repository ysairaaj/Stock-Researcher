"""
Shared weekly price history loader.

yfinance only accepts fixed period strings (1y, 2y, 5y, 10y, max...), so a
7-year window is requested by start date instead of period.
"""
from datetime import date, timedelta

import yfinance as yf

LOOKBACK_YEARS = 7
INTERVAL = "1wk"


def get_weekly_history(ticker, years=LOOKBACK_YEARS):
    """
    Return weekly OHLCV bars for the last `years` years (about 52 bars/year).
    Returns an empty DataFrame when no data is available.
    """
    start = date.today() - timedelta(days=int(years * 365.25))
    df = yf.Ticker(ticker).history(start=start.isoformat(), interval=INTERVAL)

    if df is None or df.empty:
        return df

    # Drop a possibly incomplete trailing bar for the current week? Keep it:
    # the user wants to evaluate the current state, which includes this week.
    return df
