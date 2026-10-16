import yfinance as yf
import pandas as pd

def get_market_data(ticker):

    print(f"[tools] get_market_data called with ticker={ticker}")

    stock = yf.Ticker(ticker)

    df_daily = stock.history(period="2y")
    df_weekly = stock.history(period="2y", interval="1wk")

    if df_daily.empty or df_weekly.empty:
        return {"error": "No data found"}

    current_price = float(df_daily["Close"].iloc[-1])

    df_weekly["H-L"] = df_weekly["High"] - df_weekly["Low"]
    df_weekly["H-PC"] = abs(df_weekly["High"] - df_weekly["Close"].shift(1))
    df_weekly["L-PC"] = abs(df_weekly["Low"] - df_weekly["Close"].shift(1))

    df_weekly["TR"] = df_weekly[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df_weekly["ATR_14"] = df_weekly["TR"].rolling(14).mean()

    weekly_atr = float(df_weekly["ATR_14"].iloc[-1])
    print(f"[tools] weekly_atr={weekly_atr}")
    return {
        "ticker": ticker,
        "current_price": current_price,
        "weekly_atr": weekly_atr
    }

if __name__ == "__main__":
    print("Running in main")