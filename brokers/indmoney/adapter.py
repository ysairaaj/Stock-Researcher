"""
INDmoney adapter: US stocks only.

Field mapping is taken from the live `networth_holdings(asset_type="US_STOCK")`
payload observed during discovery:

    investment_code      ticker, e.g. "AMZN"
    investment           company name
    total_units          quantity (fractional)
    unit_price           last price in INR
    invested_amount      invested in INR
    market_value         current value in INR
    invested_value_usd   invested in USD
    current_value_usd    current value in USD
    pnl_per              P&L percent
    holding_percent      weight within the US sleeve
    broker               custodian, e.g. "Drivewealth"

US cash comes from `networth_snapshot` -> investments[].asset_type ==
"US_STOCK_WALLET" (INR). It is converted to USD with the implied FX rate
derived from the holdings (market_value / current_value_usd).
"""
from datetime import datetime, timezone

from brokers.base import Holding
from brokers.indmoney import client


US_HOLDINGS_ASSET_TYPE = "US_STOCK"
US_WALLET_ASSET_TYPE = "US_STOCK_WALLET"


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class IndmoneyAdapter:
    name = "indmoney"
    supported_markets = ["US"]

    def __init__(self):
        self._last_fx = None  # INR per USD implied by the last holdings fetch

    # ----- connection -----

    def is_connected(self):
        return client.is_connected()

    def connect(self):
        return client.connect()

    # ----- holdings -----

    def fetch_holdings(self, market="US"):
        if market != "US":
            raise ValueError(f"{self.name} adapter supports US only, got {market}")

        payload = client.call_tool("networth_holdings", {"asset_type": US_HOLDINGS_ASSET_TYPE})
        rows = payload.get("holdings", []) if isinstance(payload, dict) else (payload or [])

        as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
        holdings = []
        fx_samples = []

        for r in rows:
            if r.get("asset_type") not in (None, US_HOLDINGS_ASSET_TYPE):
                continue

            qty = _num(r.get("total_units"))
            if qty <= 0:
                continue

            invested_usd = _num(r.get("invested_value_usd"))
            current_usd = _num(r.get("current_value_usd"))
            inr_value = _num(r.get("market_value"))

            if current_usd > 0 and inr_value > 0:
                fx_samples.append(inr_value / current_usd)

            avg_price = invested_usd / qty if qty else 0.0
            cur_price = current_usd / qty if qty else 0.0
            pnl = current_usd - invested_usd
            pnl_pct = _num(r.get("pnl_per"), (pnl / invested_usd * 100) if invested_usd else 0.0)

            holdings.append(Holding(
                ticker=str(r.get("investment_code", "")).upper(),
                name=r.get("investment", ""),
                quantity=qty,
                avg_price=round(avg_price, 4),
                current_price=round(cur_price, 4),
                invested_value=round(invested_usd, 2),
                current_value=round(current_usd, 2),
                pnl=round(pnl, 2),
                pnl_pct=round(pnl_pct, 2),
                currency="USD",
                broker=self.name,
                as_of=as_of,
                extra={
                    "custodian": r.get("broker"),
                    "market_cap": r.get("market_cap"),
                    "holding_percent": _num(r.get("holding_percent")),
                    "one_day_change_usd": _num(r.get("one_day_change_usd")),
                    "one_day_change_pct": _num(r.get("one_day_change_percentage")),
                    "market_value_inr": inr_value,
                },
            ))

        if fx_samples:
            self._last_fx = sum(fx_samples) / len(fx_samples)

        print(f"[broker:{self.name}] fetched {len(holdings)} US holdings (fx~{self._last_fx and round(self._last_fx, 2)})")
        return holdings

    # ----- cash -----

    def fetch_cash(self, market="US"):
        """US wallet balance in USD, or None if not exposed."""
        if market != "US":
            return None

        snap = client.call_tool("networth_snapshot", {})
        if not isinstance(snap, dict):
            return None

        wallet_inr = None
        for inv in snap.get("investments", []) or []:
            if inv.get("asset_type") == US_WALLET_ASSET_TYPE:
                wallet_inr = _num(inv.get("current_value"))
                break

        if wallet_inr is None:
            return None

        fx = self._last_fx or self._fx_from_snapshot(snap)
        if not fx:
            print(f"[broker:{self.name}] US wallet found ({wallet_inr:.0f} INR) but no FX rate; returning None")
            return None

        cash_usd = wallet_inr / fx
        print(f"[broker:{self.name}] US wallet {wallet_inr:.0f} INR -> ${cash_usd:.2f} (fx {fx:.2f})")
        return round(cash_usd, 2)

    def _fx_from_snapshot(self, snap):
        """Fallback: derive FX by fetching holdings once if not cached."""
        try:
            self.fetch_holdings("US")
        except Exception as e:
            print(f"[broker:{self.name}] could not derive FX: {e}")
        return self._last_fx

    # ----- watchlist -----

    def fetch_watchlist(self, market="US"):
        """
        Live INDmoney watchlist. market US -> MCP type "us" (US stocks only).
        Returns {tickers, lists, broker, market}.
        """
        if market != "US":
            raise ValueError(f"{self.name} adapter supports US watchlist only, got {market}")

        payload = client.call_tool("user_watchlist", {"type": "us"})
        lists = []
        tickers = []
        seen = set()

        rows = []
        if isinstance(payload, dict):
            rows = payload.get("watchlists") or []
        elif isinstance(payload, list):
            rows = payload

        for wl in rows:
            name = wl.get("name") or f"list_{wl.get('watchlist_id', '')}"
            stocks = wl.get("stocks") or []
            list_tickers = []
            for s in stocks:
                t = str(s.get("ticker") or s.get("ind_key") or "").upper().strip()
                if not t:
                    continue
                list_tickers.append(t)
                if t not in seen:
                    seen.add(t)
                    tickers.append(t)
            lists.append({
                "name": name,
                "watchlist_id": wl.get("watchlist_id"),
                "tickers": list_tickers,
            })

        print(f"[broker:{self.name}] fetched US watchlist: {len(tickers)} tickers across {len(lists)} lists")
        return {
            "broker": self.name,
            "market": "US",
            "tickers": tickers,
            "lists": lists,
        }
