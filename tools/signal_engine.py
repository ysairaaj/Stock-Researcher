"""
Signal engine: swing points, support / resistance levels, and structure analysis.

- All structure calculations use WEEKLY bars over a 7-year window
  (see tools/price_history.py).
- Swing high/low: local extremum over a window of (2*SWING_LOOKBACK+1) bars.
- Support and resistance zones are built with one shared clustering function:
    * swing lows  -> support candidates
    * swing highs -> resistance candidates
  Zones are then split by position relative to the current price (polarity
  flip: a broken support above price becomes resistance and vice versa),
  merged where a low-cluster and a high-cluster overlap (confluence), filtered,
  and sorted nearest-first.
- Fibonacci retracement uses the most recent confirmed swing high and the
  swing low that occurred before it.
"""
import numpy as np

from tools.price_history import get_weekly_history, LOOKBACK_YEARS, INTERVAL


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

SWING_LOOKBACK = 3            # bars on each side for a confirmed swing
BASE_TOLERANCE = 0.02         # minimum cluster width (2%)
ATR_TOLERANCE_FACTOR = 0.5    # tolerance = max(BASE, factor * ATR / close)
RECENT_WEEKS_STRONG = 26      # last touch within this -> +2 recency
RECENT_WEEKS_MEDIUM = 52      # last touch within this -> +1 recency
MAX_LEVELS = 5                # per list (supports / resistance)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def _add_indicators(df):
    """Add Volume_MA20 and weekly ATR14 columns in place (copy returned)."""
    df = df.copy()

    if "Volume" in df.columns:
        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    else:
        df["Volume"] = 0.0
        df["Volume_MA20"] = 0.0

    prev_close = df["Close"].shift(1)
    tr = np.maximum.reduce([
        (df["High"] - df["Low"]).values,
        (df["High"] - prev_close).abs().values,
        (df["Low"] - prev_close).abs().values,
    ])
    df["TR"] = tr
    df["ATR14"] = df["TR"].rolling(14).mean()

    return df


def _tolerance_for(df, close):
    """Volatility-scaled cluster tolerance."""
    atr = df["ATR14"].iloc[-1] if "ATR14" in df.columns else np.nan
    if atr is None or np.isnan(atr) or close <= 0:
        return BASE_TOLERANCE
    return max(BASE_TOLERANCE, ATR_TOLERANCE_FACTOR * float(atr) / close)


# ---------------------------------------------------------------------------
# Swing points
# ---------------------------------------------------------------------------

def detect_swing_points(df, lookback=SWING_LOOKBACK):
    """
    Find swing highs and swing lows: bars whose High (Low) is the max (min)
    over [i-lookback, i+lookback]. Returns two chronological lists of
    (timestamp, price, volume).
    """
    print(f"[tools] detect_swing_points called with lookback={lookback}")

    if df is None or len(df) == 0:
        return [], []

    has_volume = "Volume" in df.columns

    swing_highs = []
    swing_lows = []

    highs = df["High"].values
    lows = df["Low"].values

    for i in range(lookback, len(df) - lookback):
        ts = df.index[i]
        high = float(highs[i])
        low = float(lows[i])
        vol = float(df["Volume"].iloc[i]) if has_volume else 0.0

        window_high = highs[i - lookback: i + lookback + 1]
        window_low = lows[i - lookback: i + lookback + 1]

        if high == window_high.max():
            swing_highs.append((ts, high, vol))
        if low == window_low.min():
            swing_lows.append((ts, low, vol))

    return swing_highs, swing_lows


# ---------------------------------------------------------------------------
# Clustering and scoring
# ---------------------------------------------------------------------------

def _strength_for(score):
    if score >= 4:
        return "STRONG"
    if score >= 2:
        return "MEDIUM"
    return "WEAK"


def _cluster_levels(points, df, tolerance, kind):
    """
    Cluster swing points into price levels and score them.

    points    : chronological list of (timestamp, price, volume)
    df        : weekly frame with Volume_MA20 column
    tolerance : fractional cluster width (e.g. 0.02)
    kind      : "support" (swing lows) or "resistance" (swing highs)

    Matching uses the zone's anchor (first touch) so zones cannot drift.
    Zone price is the volume-weighted mean of all touches.
    """
    if not points:
        return []

    if "Volume_MA20" not in df.columns:
        df = _add_indicators(df)

    latest_ts = points[-1][0]

    zones = []
    for ts, price, vol in points:
        for zone in zones:
            if abs(price - zone["anchor"]) / zone["anchor"] < tolerance:
                zone["prices"].append(price)
                zone["volumes"].append(vol)
                zone["dates"].append(ts)
                break
        else:
            zones.append({
                "anchor": price,
                "prices": [price],
                "volumes": [vol],
                "dates": [ts],
            })

    last_idx = len(df) - 1
    source = "swing_low" if kind == "support" else "swing_high"

    out = []
    for z in zones:
        prices = np.asarray(z["prices"], dtype=float)
        vols = np.asarray(z["volumes"], dtype=float)

        if vols.sum() > 0:
            level = float(np.average(prices, weights=vols))
        else:
            level = float(prices.mean())

        touches = int(len(prices))

        ratios = []
        for ts, v in zip(z["dates"], z["volumes"]):
            ma = df["Volume_MA20"].loc[ts] if ts in df.index else np.nan
            if ma is not None and not np.isnan(ma) and ma > 0:
                ratios.append(v / ma)
        avg_volume_ratio = float(np.mean(ratios)) if ratios else 0.0

        last_touch = max(z["dates"])
        weeks_since = int(last_idx - df.index.get_loc(last_touch))

        score = 0
        if touches >= 3:
            score += 2
        elif touches == 2:
            score += 1

        if avg_volume_ratio > 1.0:
            score += 1

        if weeks_since <= RECENT_WEEKS_STRONG:
            score += 2
        elif weeks_since <= RECENT_WEEKS_MEDIUM:
            score += 1

        out.append({
            "price": level,
            "kind": kind,
            "strength": _strength_for(score),
            "score": int(score),
            "touches": touches,
            "last_touch_date": str(last_touch.date()) if hasattr(last_touch, "date") else str(last_touch),
            "weeks_since_touch": weeks_since,
            "avg_volume_ratio": round(avg_volume_ratio, 2),
            "source": source,
            "is_latest_swing": latest_ts in z["dates"],
        })

    return out


def detect_support_levels(df, lookback=SWING_LOOKBACK, tolerance=BASE_TOLERANCE):
    """
    Legacy wrapper kept for backwards compatibility. Returns support zones in
    the old shape ({price, strength, touches}) sorted by price descending.
    Prefer analyze_structure() for the full support / resistance output.
    """
    print(
        f"[tools] detect_support_levels called with "
        f"lookback={lookback}, tolerance={tolerance}"
    )

    if df is None or len(df) == 0 or "Volume" not in df.columns:
        return []

    df = _add_indicators(df)
    _, swing_lows = detect_swing_points(df, lookback)
    zones = _cluster_levels(swing_lows, df, tolerance, kind="support")

    legacy = [
        {"price": z["price"], "strength": z["strength"], "touches": z["touches"]}
        for z in zones
    ]
    return sorted(legacy, key=lambda x: x["price"], reverse=True)


# ---------------------------------------------------------------------------
# Split by price, polarity flip, confluence
# ---------------------------------------------------------------------------

def _merge_confluence(zones, tolerance):
    """
    Merge pairs of zones of different origin (one swing-low cluster, one
    swing-high cluster) whose prices are within tolerance. Merged zone gets
    +1 score and source="confluence".
    """
    merged = []
    used = [False] * len(zones)

    for i, a in enumerate(zones):
        if used[i]:
            continue
        partner = None
        for j in range(i + 1, len(zones)):
            b = zones[j]
            if used[j] or a["kind"] == b["kind"]:
                continue
            if abs(a["price"] - b["price"]) / a["price"] < tolerance:
                partner = j
                break

        if partner is None:
            merged.append(a)
            used[i] = True
            continue

        b = zones[partner]
        used[i] = used[partner] = True

        total_touches = a["touches"] + b["touches"]
        price = (a["price"] * a["touches"] + b["price"] * b["touches"]) / total_touches
        score = max(a["score"], b["score"]) + 1

        merged.append({
            "price": float(price),
            "kind": a["kind"],
            "strength": _strength_for(score),
            "score": int(score),
            "touches": int(total_touches),
            "last_touch_date": max(a["last_touch_date"], b["last_touch_date"]),
            "weeks_since_touch": min(a["weeks_since_touch"], b["weeks_since_touch"]),
            "avg_volume_ratio": round((a["avg_volume_ratio"] + b["avg_volume_ratio"]) / 2, 2),
            "source": "confluence",
            "is_latest_swing": a["is_latest_swing"] or b["is_latest_swing"],
            "confluence": True,
        })

    return merged


def _split_by_price(low_zones, high_zones, close, tolerance):
    """
    Assign every zone to the support or resistance side based on the current
    price, tagging polarity flips, then merge confluence and add distance_pct.
    """
    supports = []
    resistance = []

    for z in low_zones:
        z = dict(z)
        if z["price"] < close:
            supports.append(z)
        else:
            z["source"] = "flipped_support"
            resistance.append(z)

    for z in high_zones:
        z = dict(z)
        if z["price"] >= close:
            resistance.append(z)
        else:
            z["source"] = "flipped_resistance"
            supports.append(z)

    supports = _merge_confluence(supports, tolerance)
    resistance = _merge_confluence(resistance, tolerance)

    for z in supports + resistance:
        z["distance_pct"] = round(abs(close - z["price"]) / close * 100, 2)
        z.setdefault("confluence", False)

    return supports, resistance


def _finalise(zones):
    """Filter single-touch noise, sort nearest-first, cap, drop internal keys."""
    kept = [z for z in zones if z["touches"] >= 2 or z["is_latest_swing"]]
    kept.sort(key=lambda z: (z["distance_pct"], -z["score"]))
    kept = kept[:MAX_LEVELS]

    for z in kept:
        z.pop("kind", None)

    return kept


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

def analyze_structure(ticker, interval="1wk"):
    """
    Analyze price structure (swing range, Fib levels, support / resistance)
    for a ticker on WEEKLY bars over a 7-year window. The interval argument is
    kept for API compatibility but is ignored.
    """
    interval = INTERVAL  # enforce weekly only
    print(f"[tools] analyze_structure called with ticker={ticker}, interval={interval}")

    df = get_weekly_history(ticker)

    if df is None or df.empty:
        print(f"[tools] analyze_structure: no price data for {ticker}, skipping")
        return {"ticker": ticker, "error": "no_data"}

    if not {"High", "Low", "Close", "Volume"}.issubset(df.columns):
        print(f"[tools] analyze_structure: missing columns for {ticker}, skipping")
        return {"ticker": ticker, "error": "bad_schema"}

    df = _add_indicators(df)
    close = float(df["Close"].iloc[-1])
    tolerance = _tolerance_for(df, close)

    atr_raw = df["ATR14"].iloc[-1]
    weekly_atr = float(atr_raw) if atr_raw is not None and not np.isnan(atr_raw) else None
    weekly_atr_pct = round(weekly_atr / close * 100, 2) if weekly_atr and close > 0 else None

    print(
        f"[tools] analyze_structure: loaded {len(df)} {interval} bars over "
        f"{LOOKBACK_YEARS}y, close={close:.2f}, tolerance={tolerance * 100:.2f}% "
        f"(swing_lookback={SWING_LOOKBACK})"
    )

    swing_highs, swing_lows = detect_swing_points(df, SWING_LOOKBACK)

    if not swing_highs or not swing_lows:
        print(f"[tools] analyze_structure: insufficient swing points for {ticker}")
        return {"ticker": ticker, "error": "insufficient_history"}

    # ----- Fibonacci: most recent swing high and the lowest swing low before it -----

    pivot_high_ts, pivot_high_price, _ = swing_highs[-1]

    lows_before_high = [(ts, p) for ts, p, _ in swing_lows if ts < pivot_high_ts]
    if lows_before_high:
        pivot_low_ts, pivot_low_price = min(lows_before_high, key=lambda x: x[1])
    else:
        pivot_low_ts, pivot_low_price, _ = swing_lows[-1]

    recent_high = max(pivot_high_price, pivot_low_price)
    recent_low = min(pivot_high_price, pivot_low_price)

    print(
        f"[tools] analyze_structure pivots for {ticker}: "
        f"high={pivot_high_price:.2f} on {pivot_high_ts.date()}, "
        f"low={pivot_low_price:.2f} on {pivot_low_ts.date()}"
    )

    diff = recent_high - recent_low

    if diff == 0:
        print(f"[tools] analyze_structure: zero range for {ticker}")
        return {"ticker": ticker, "error": "flat_price"}

    fib_levels = {
        "0.236": recent_high - 0.236 * diff,
        "0.382": recent_high - 0.382 * diff,
        "0.5": recent_high - 0.5 * diff,
        "0.618": recent_high - 0.618 * diff,
    }

    # ----- Support / resistance -----

    low_zones = _cluster_levels(swing_lows, df, tolerance, kind="support")
    high_zones = _cluster_levels(swing_highs, df, tolerance, kind="resistance")

    print(
        f"[tools] analyze_structure: raw clusters lows={len(low_zones)} "
        f"highs={len(high_zones)}"
    )

    supports, resistance = _split_by_price(low_zones, high_zones, close, tolerance)
    supports = _finalise(supports)
    resistance = _finalise(resistance)

    nearest_support = supports[0]["price"] if supports else None
    nearest_resistance = resistance[0]["price"] if resistance else None

    def _fmt(zones):
        if not zones:
            return "None"
        return f"{zones[0]['price']:.2f} ({zones[0]['distance_pct']}% away)"

    print(
        f"[tools] analyze_structure: after filtering supports={len(supports)} "
        f"resistance={len(resistance)}; nearest_support={_fmt(supports)}, "
        f"nearest_resistance={_fmt(resistance)}"
    )

    # ----- Provisional (unconfirmed) extremes in the last SWING_LOOKBACK bars -----

    tail = df.iloc[-SWING_LOOKBACK:]
    provisional_low = {
        "price": float(tail["Low"].min()),
        "date": str(tail["Low"].idxmin().date()),
    }
    provisional_high = {
        "price": float(tail["High"].max()),
        "date": str(tail["High"].idxmax().date()),
    }

    return {
        "ticker": ticker,
        "interval": interval,
        "lookback_years": LOOKBACK_YEARS,
        "bars": int(len(df)),
        "current_price": close,
        "weekly_atr": weekly_atr,
        "weekly_atr_pct": weekly_atr_pct,
        "tolerance_pct": round(tolerance * 100, 2),
        "recent_high": float(recent_high),
        "recent_low": float(recent_low),
        "fib_levels": fib_levels,
        "supports": supports,
        "resistance": resistance,
        "provisional_low": provisional_low,
        "provisional_high": provisional_high,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
    }
