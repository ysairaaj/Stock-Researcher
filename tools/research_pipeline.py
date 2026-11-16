"""
Code-owned portfolio + watchlist research pipeline.

Runs a fixed checklist, writes progress to memory/reports/{run_id}/todos.json,
stores full per-ticker detail under tickers/, and keeps report.json compact.
"""
import json
import os
import time
import uuid
from datetime import datetime, timezone

from tools.holdings_aggregator import get_portfolio_snapshot, fetch_watchlist
from tools.regime_engine import detect_market_regime
from tools.scanner_engine import scan_stocks, analyze_ticker
from tools.signal_engine import analyze_structure
from tools.ath_tools import check_distance_from_ath
from tools.backtest_engine import backtest_strategy
from tools.trade_planner import plan_trade, _load_settings

REPORTS_DIR = os.path.join("memory", "reports")
LATEST_POINTER = os.path.join(REPORTS_DIR, "latest.json")


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _run_dir(run_id):
    return os.path.join(REPORTS_DIR, run_id)


def _ensure_dirs(run_id):
    base = _run_dir(run_id)
    os.makedirs(os.path.join(base, "tickers"), exist_ok=True)
    os.makedirs(os.path.join(base, "narrative"), exist_ok=True)
    return base


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _set_latest(run_id):
    _write_json(LATEST_POINTER, {"run_id": run_id, "updated_at": _now()})


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_run_id(run_id=None):
    if run_id:
        return run_id
    latest = _read_json(LATEST_POINTER)
    if latest and latest.get("run_id"):
        return latest["run_id"]
    return None


# ---------------------------------------------------------------------------
# Todos
# ---------------------------------------------------------------------------

def _init_todos(run_id, deep_tickers_planned=None):
    steps = [
        {"id": "fetch_universe", "status": "pending", "detail": ""},
        {"id": "market_regime", "status": "pending", "detail": ""},
        {"id": "quick_scan", "status": "pending", "detail": ""},
        {"id": "rank", "status": "pending", "detail": ""},
        {"id": "deep_dive", "status": "pending", "detail": "", "tickers": []},
        {"id": "finalize_report", "status": "pending", "detail": ""},
    ]
    data = {"run_id": run_id, "steps": steps, "updated_at": _now()}
    _write_json(os.path.join(_run_dir(run_id), "todos.json"), data)
    return data


def _update_todo(run_id, step_id, status, detail="", tickers=None):
    path = os.path.join(_run_dir(run_id), "todos.json")
    data = _read_json(path) or _init_todos(run_id)
    for step in data["steps"]:
        if step["id"] == step_id:
            step["status"] = status
            step["detail"] = detail
            if tickers is not None:
                step["tickers"] = tickers
            break
    data["updated_at"] = _now()
    _write_json(path, data)
    # Mirror pointer for older agent_todos.json readers
    _write_json(os.path.join("memory", "agent_todos.json"), {
        "active_run_id": run_id,
        "updated_at": data["updated_at"],
        "steps": data["steps"],
    })
    return data


def _mark_deep_ticker(run_id, ticker, status, detail=""):
    path = os.path.join(_run_dir(run_id), "todos.json")
    data = _read_json(path) or {}
    for step in data.get("steps", []):
        if step["id"] != "deep_dive":
            continue
        found = False
        for t in step.get("tickers", []):
            if t.get("ticker") == ticker:
                t["status"] = status
                t["detail"] = detail
                found = True
                break
        if not found:
            step.setdefault("tickers", []).append(
                {"ticker": ticker, "status": status, "detail": detail}
            )
        break
    data["updated_at"] = _now()
    _write_json(path, data)
    return data


# ---------------------------------------------------------------------------
# Scoring / ranking
# ---------------------------------------------------------------------------

def score_row(row):
    """
    Higher = stronger setup for longs / hold quality.
    Explicit code rules; no LLM scoring.
    """
    score = 50.0
    trend = row.get("trend")
    if trend == "bullish":
        score += 15
    elif trend == "bearish":
        score -= 15

    mom = row.get("momentum")
    if mom == "strong":
        score += 10
    elif mom == "negative":
        score -= 10

    label = row.get("label")
    if label == "near_support":
        score += 12
    elif label == "near_resistance":
        score -= 8
    elif label == "mid_range_extended":
        score -= 5

    zone = row.get("ath_zone")
    if zone == "near_ath":
        score += 5
    elif zone == "near_low":
        score -= 5

    # Prefer room to resistance for upside
    rd = row.get("resistance_distance")
    if rd is not None:
        if rd >= 15:
            score += 8
        elif rd <= 3:
            score -= 6

    sd = row.get("support_distance")
    if sd is not None and sd <= 5:
        score += 5

    # Concentration penalty for oversized holdings
    w = row.get("weight_pct")
    if w is not None and w > 20:
        score -= 10

    return round(score, 2)


def _pick_deep_tickers(quick_scan, over_limit, top_n, bottom_n, watch_n):
    holdings = [r for r in quick_scan if r.get("source") == "holding"]
    watch = [r for r in quick_scan if r.get("source") == "watchlist"]

    holdings_sorted = sorted(holdings, key=lambda r: r.get("score", 0), reverse=True)
    watch_sorted = sorted(watch, key=lambda r: r.get("score", 0), reverse=True)

    strong = [r["ticker"] for r in holdings_sorted[:top_n]]
    weak = [r["ticker"] for r in holdings_sorted[-bottom_n:]] if holdings_sorted else []
    weak = [t for t in weak if t not in strong]
    buys = [r["ticker"] for r in watch_sorted[:watch_n]]

    over = [t.upper() for t in (over_limit or [])]
    selected = []
    for t in strong + weak + buys + over:
        if t not in selected:
            selected.append(t)

    return {
        "holdings_strong": strong,
        "holdings_weak": weak,
        "watchlist_buys": buys,
        "over_limit": over,
        "deep_tickers": selected,
    }


def _suggest_action(row, rankings, plan=None, backtest=None):
    ticker = row["ticker"]
    if row.get("source") == "watchlist":
        if plan and plan.get("decision") == "reject":
            return "wait", "plan_trade rejected the size/setup"
        if plan and plan.get("decision") in ("fit", "reduce"):
            return "buy_candidate", f"plan_trade={plan.get('decision')}, shares={plan.get('shares')}"
        if ticker in rankings.get("watchlist_buys", []):
            return "buy_candidate", "ranked among top watchlist setups"
        return "wait", "watchlist name not selected for a buy"

    if ticker in (rankings.get("over_limit") or []):
        return "trim", "above max position weight"
    if ticker in rankings.get("holdings_weak", []):
        if backtest and backtest.get("decision") == "no_trade":
            return "wait", "weak structure and no suitable backtest strategy"
        return "wait", "ranked among weakest holdings"
    if ticker in rankings.get("holdings_strong", []):
        return "hold", "ranked among strongest holdings"
    return "hold", "mid-pack holding"


# ---------------------------------------------------------------------------
# Deep dive (one ticker)
# ---------------------------------------------------------------------------

def _compact_structure(structure):
    if not isinstance(structure, dict) or "error" in structure:
        return structure
    return {
        "current_price": structure.get("current_price"),
        "weekly_atr": structure.get("weekly_atr"),
        "weekly_atr_pct": structure.get("weekly_atr_pct"),
        "nearest_support": structure.get("nearest_support"),
        "nearest_resistance": structure.get("nearest_resistance"),
        "fib_levels": structure.get("fib_levels"),
        "supports_top": (structure.get("supports") or [])[:2],
        "resistance_top": (structure.get("resistance") or [])[:2],
    }


def _compact_backtest(bt):
    if not isinstance(bt, dict):
        return bt
    keys = [
        "decision", "selected_strategy", "strategy_overridden", "reason",
        "win_rate", "trades", "return_pct", "target_pct", "stop_pct",
        "reward_risk_ratio", "suitability",
    ]
    return {k: bt.get(k) for k in keys if k in bt}


def _compact_plan(plan):
    if not isinstance(plan, dict):
        return plan
    keys = [
        "decision", "shares", "position_value", "risk_amount", "reward_risk",
        "entry_price", "stop_loss", "target_price", "stop_pct", "target_pct",
        "constraints_applied", "reasons", "cash_before", "cash_after",
        "position_weight_after_pct", "portfolio_heat_after_pct", "stop_check",
    ]
    return {k: plan.get(k) for k in keys if k in plan}


def deep_dive_ticker(ticker, source, entry_price=None):
    ticker = ticker.upper()
    print(f"[tools] research deep_dive {ticker} (source={source})")

    structure = analyze_structure(ticker)
    ath = check_distance_from_ath(ticker)
    settings = _load_settings()
    bt = backtest_strategy(
        ticker,
        strategy=None,
        target_pct=settings.get("default_target_pct", 30),
        stop_pct=settings.get("default_stop_pct", 15),
    )

    plan = None
    if source == "watchlist" or source == "holding_add":
        price = entry_price
        if price is None and isinstance(structure, dict):
            price = structure.get("current_price")
        if price:
            stop = None
            target = None
            if isinstance(structure, dict) and structure.get("nearest_support"):
                stop = float(structure["nearest_support"])
            plan = plan_trade(ticker, float(price), stop_loss=stop, target_price=target)
            if isinstance(plan, dict) and "error" not in plan:
                from agent.agent_controller import stop_check
                check = stop_check(plan, structure if isinstance(structure, dict) else {})
                if check:
                    plan = dict(plan)
                    plan["stop_check"] = check

    summary = {
        "ticker": ticker,
        "source": source,
        "price": (structure or {}).get("current_price") if isinstance(structure, dict) else None,
        "trend_note": None,
        "ath_zone": ath.get("zone") if isinstance(ath, dict) else None,
        "drawdown_from_ath_pct": ath.get("drawdown_from_ath_pct") if isinstance(ath, dict) else None,
        "nearest_support": (structure or {}).get("nearest_support") if isinstance(structure, dict) else None,
        "nearest_resistance": (structure or {}).get("nearest_resistance") if isinstance(structure, dict) else None,
        "weekly_atr_pct": (structure or {}).get("weekly_atr_pct") if isinstance(structure, dict) else None,
        "backtest_decision": bt.get("decision") if isinstance(bt, dict) else None,
        "selected_strategy": bt.get("selected_strategy") if isinstance(bt, dict) else None,
        "plan_decision": plan.get("decision") if isinstance(plan, dict) else None,
        "plan_shares": plan.get("shares") if isinstance(plan, dict) else None,
        "stop_check_describe": (plan or {}).get("stop_check", {}).get("describe_as") if isinstance(plan, dict) else None,
    }

    return {
        "ticker": ticker,
        "source": source,
        "as_of": _now(),
        "summary": summary,
        "structure": _compact_structure(structure) if isinstance(structure, dict) else structure,
        "structure_full_error": structure.get("error") if isinstance(structure, dict) and "error" in structure else None,
        "ath": {k: ath.get(k) for k in (
            "zone", "drawdown_from_ath_pct", "range_percentile", "ath", "current_price"
        )} if isinstance(ath, dict) else ath,
        "backtest": _compact_backtest(bt),
        "plan_trade": _compact_plan(plan) if plan else None,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def research_portfolio(
    include_watchlist=True,
    refresh=False,
    deep_top_n=3,
    deep_bottom_n=3,
    deep_watchlist_n=3,
    broker="all",
):
    """
    Full code-owned research run. Returns the compact report.json contents
    (also written under memory/reports/{run_id}/).
    """
    print(
        f"[tools] research_portfolio called include_watchlist={include_watchlist}, "
        f"refresh={refresh}, deep_top_n={deep_top_n}, deep_bottom_n={deep_bottom_n}, "
        f"deep_watchlist_n={deep_watchlist_n}"
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    _ensure_dirs(run_id)
    _set_latest(run_id)
    _init_todos(run_id)
    t0 = time.time()

    # 1. Universe
    _update_todo(run_id, "fetch_universe", "running")
    snapshot = get_portfolio_snapshot(broker=broker, refresh=refresh)
    if "error" in snapshot:
        _update_todo(run_id, "fetch_universe", "error", snapshot.get("error", "snapshot failed"))
        return {"error": snapshot.get("error"), "detail": snapshot, "run_id": run_id}

    holdings = snapshot.get("holdings", [])
    watch = fetch_watchlist(broker=broker if broker != "all" else "indmoney") if include_watchlist else {
        "tickers": [], "source": "skipped",
    }
    watch_tickers = [str(t).upper() for t in (watch.get("tickers") or [])]

    held_set = {h["ticker"].upper() for h in holdings}
    watch_only = [t for t in watch_tickers if t not in held_set]

    _update_todo(
        run_id, "fetch_universe", "done",
        f"{len(holdings)} holdings, {len(watch_tickers)} watchlist "
        f"({len(watch_only)} not already held, source={watch.get('source')})",
    )

    # 2. Regime
    _update_todo(run_id, "market_regime", "running")
    regime = detect_market_regime("SPY")
    _update_todo(run_id, "market_regime", "done", regime.get("trend", ""))

    # 3. Quick scan
    _update_todo(run_id, "quick_scan", "running")
    meta = {}
    tickers = []
    for h in holdings:
        t = h["ticker"].upper()
        tickers.append(t)
        meta[t] = {"source": "holding", "weight_pct": h.get("weight_pct")}
    for t in watch_only:
        tickers.append(t)
        meta[t] = {"source": "watchlist"}

    scan = scan_stocks(tickers, meta=meta)
    quick = scan.get("results", [])
    for row in quick:
        row["score"] = score_row(row)
    quick.sort(key=lambda r: r.get("score", 0), reverse=True)
    _write_json(os.path.join(_run_dir(run_id), "quick_scan.json"), quick)
    _update_todo(run_id, "quick_scan", "done", f"{len(quick)} tickers")

    # 4. Rank
    _update_todo(run_id, "rank", "running")
    over_limit = (snapshot.get("concentration") or {}).get("over_limit") or []
    rankings = _pick_deep_tickers(
        quick, over_limit, deep_top_n, deep_bottom_n, deep_watchlist_n
    )
    _update_todo(
        run_id, "rank", "done",
        f"deep={rankings['deep_tickers']}",
    )

    # 5. Deep dive sequentially
    deep_list = []
    for t in rankings["deep_tickers"]:
        deep_list.append({"ticker": t, "status": "pending", "detail": ""})
    _update_todo(run_id, "deep_dive", "running", tickers=deep_list)

    by_ticker = {r["ticker"]: r for r in quick}
    deep_summaries = []

    for t in rankings["deep_tickers"]:
        _mark_deep_ticker(run_id, t, "running")
        row = by_ticker.get(t, {"ticker": t, "source": "holding"})
        source = row.get("source", "holding")
        # For strong holdings, still allow plan_trade as "add" if user cash exists;
        # watchlist always sizes; weak holdings skip buy sizing unless watchlist.
        dive_source = "watchlist" if source == "watchlist" else "holding"
        if source == "holding" and t in rankings["holdings_strong"]:
            dive_source = "holding_add"
        try:
            detail = deep_dive_ticker(t, dive_source, entry_price=row.get("price"))
            action, reason = _suggest_action(row, rankings, detail.get("plan_trade"), detail.get("backtest"))
            detail["summary"]["action"] = action
            detail["summary"]["action_reason"] = reason
            detail["action"] = action
            path = os.path.join("tickers", f"{t}.json")
            _write_json(os.path.join(_run_dir(run_id), path), detail)
            deep_summaries.append({
                "ticker": t,
                "source": source,
                "action": action,
                "action_reason": reason,
                "score": row.get("score"),
                "summary": detail["summary"],
                "file": path,
            })
            _mark_deep_ticker(run_id, t, "done", action)
        except Exception as e:
            print(f"[tools] research deep_dive failed for {t}: {e}")
            _mark_deep_ticker(run_id, t, "error", str(e))
            deep_summaries.append({
                "ticker": t,
                "source": source,
                "action": "error",
                "action_reason": str(e),
                "summary": {"ticker": t, "error": str(e)},
                "file": None,
            })

    _update_todo(run_id, "deep_dive", "done", f"{len(deep_summaries)} tickers")

    # 6. Finalize
    _update_todo(run_id, "finalize_report", "running")
    suggestions = []
    for d in deep_summaries:
        suggestions.append({
            "ticker": d["ticker"],
            "action": d["action"],
            "reason": d.get("action_reason", ""),
        })

    report = {
        "run_id": run_id,
        "as_of": _now(),
        "elapsed_sec": round(time.time() - t0, 1),
        "regime": {
            "trend": regime.get("trend"),
            "volatility": regime.get("volatility"),
            "price": regime.get("price"),
            "ma50_weekly": regime.get("ma50_weekly"),
            "ma200_weekly": regime.get("ma200_weekly"),
        },
        "portfolio": {
            "portfolio_size": (snapshot.get("totals") or {}).get("portfolio_size"),
            "cash_available": (snapshot.get("totals") or {}).get("cash_available"),
            "cash_source": (snapshot.get("totals") or {}).get("cash_source"),
            "current_value": (snapshot.get("totals") or {}).get("current_value"),
            "unrealised_pnl": (snapshot.get("totals") or {}).get("unrealised_pnl"),
            "unrealised_pnl_pct": (snapshot.get("totals") or {}).get("unrealised_pnl_pct"),
            "positions": (snapshot.get("totals") or {}).get("positions"),
            "over_limit": over_limit,
            "top_positions": (snapshot.get("concentration") or {}).get("top_positions"),
        },
        "watchlist": watch_tickers,
        "watchlist_source": watch.get("source"),
        "watchlist_not_held": watch_only,
        "watchlist_lists": [
            {"name": lst.get("name"), "count": len(lst.get("tickers") or [])}
            for lst in (watch.get("lists") or [])
        ],
        "quick_scan": [
            {
                "ticker": r["ticker"],
                "source": r.get("source"),
                "score": r.get("score"),
                "label": r.get("label"),
                "trend": r.get("trend"),
                "price": r.get("price"),
                "weight_pct": r.get("weight_pct"),
                "support_distance": r.get("support_distance"),
                "resistance_distance": r.get("resistance_distance"),
                "momentum": r.get("momentum"),
                "ath_zone": r.get("ath_zone"),
                "drawdown_from_ath_pct": r.get("drawdown_from_ath_pct"),
                "weekly_atr_pct": r.get("weekly_atr_pct"),
            }
            for r in quick
        ],
        "rankings": {
            "holdings_strong": rankings["holdings_strong"],
            "holdings_weak": rankings["holdings_weak"],
            "watchlist_buys": rankings["watchlist_buys"],
            "over_limit": rankings["over_limit"],
            "deep_tickers": rankings["deep_tickers"],
        },
        "deep_dives": deep_summaries,
        "suggestions": suggestions,
    }
    _write_json(os.path.join(_run_dir(run_id), "report.json"), report)
    _update_todo(run_id, "finalize_report", "done", f"elapsed={report['elapsed_sec']}s")

    print(
        f"[tools] research_portfolio done run_id={run_id} "
        f"deep={len(deep_summaries)} elapsed={report['elapsed_sec']}s"
    )
    return report


# ---------------------------------------------------------------------------
# Agent-facing readers
# ---------------------------------------------------------------------------

def get_research_status(run_id=None):
    print(f"[tools] get_research_status called with run_id={run_id}")
    run_id = _resolve_run_id(run_id)
    if not run_id:
        return {"error": "no_research_run"}
    todos = _read_json(os.path.join(_run_dir(run_id), "todos.json"))
    if not todos:
        return {"error": "todos_missing", "run_id": run_id}
    steps = todos.get("steps", [])
    done = sum(1 for s in steps if s.get("status") == "done")
    return {
        "run_id": run_id,
        "updated_at": todos.get("updated_at"),
        "progress": f"{done}/{len(steps)}",
        "complete": all(s.get("status") == "done" for s in steps),
        "steps": steps,
    }


def get_research_ticker(ticker, run_id=None, full=False):
    print(f"[tools] get_research_ticker called with ticker={ticker}, run_id={run_id}, full={full}")
    run_id = _resolve_run_id(run_id)
    if not run_id:
        return {"error": "no_research_run"}
    ticker = ticker.upper()
    path = os.path.join(_run_dir(run_id), "tickers", f"{ticker}.json")
    data = _read_json(path)
    if not data:
        # Fall back to quick_scan row
        report = _read_json(os.path.join(_run_dir(run_id), "report.json")) or {}
        row = next((r for r in report.get("quick_scan", []) if r.get("ticker") == ticker), None)
        if row:
            return {"run_id": run_id, "ticker": ticker, "summary": row, "deep": False}
        return {"error": "ticker_not_in_report", "run_id": run_id, "ticker": ticker}
    if full:
        return {"run_id": run_id, **data}
    return {
        "run_id": run_id,
        "ticker": ticker,
        "action": data.get("action"),
        "summary": data.get("summary"),
        "file": f"tickers/{ticker}.json",
    }


def get_research_report(run_id=None):
    print(f"[tools] get_research_report called with run_id={run_id}")
    run_id = _resolve_run_id(run_id)
    if not run_id:
        return {"error": "no_research_run"}
    report = _read_json(os.path.join(_run_dir(run_id), "report.json"))
    if not report:
        return {"error": "report_missing", "run_id": run_id}
    return report


def save_research_narrative(ticker, text, run_id=None):
    """Persist a short LLM note for one ticker during the narration pass."""
    run_id = _resolve_run_id(run_id)
    if not run_id:
        return {"error": "no_research_run"}
    ticker = ticker.upper()
    path = os.path.join(_run_dir(run_id), "narrative", f"{ticker}.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return {"run_id": run_id, "ticker": ticker, "file": f"narrative/{ticker}.md"}


def load_research_narratives(run_id=None):
    run_id = _resolve_run_id(run_id)
    if not run_id:
        return {}
    folder = os.path.join(_run_dir(run_id), "narrative")
    if not os.path.isdir(folder):
        return {}
    out = {}
    for name in os.listdir(folder):
        if name.endswith(".md"):
            with open(os.path.join(folder, name), "r", encoding="utf-8") as f:
                out[name[:-3]] = f.read().strip()
    return out
