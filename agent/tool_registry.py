from tools.market_data_engine import get_market_data
from tools.signal_engine import analyze_structure
from tools.risk_engine import calculate_position_risk
from tools.scanner_engine import scan_stocks
from tools.portfolio_manager import evaluate_position
from tools.universe_tools import get_sp500_list, get_nifty50_list, get_major_etfs
from tools.web_tools import web_search
from tools.ranking_engine import rank_trades
from tools.watchlist_memory import get_watchlist, add_to_watchlist
from agent.strategy_engine import set_strategy, get_strategy
from tools.regime_engine import detect_market_regime
from tools.position_sizing_engine import calculate_position_size
from tools.backtest_engine import backtest_strategy
from tools.ath_tools import check_distance_from_ath
from tools.holdings_aggregator import connect_broker, fetch_holdings, get_portfolio_snapshot
from tools.trade_planner import plan_trade, get_risk_settings, set_risk_settings
from tools.research_pipeline import (
    research_portfolio,
    get_research_status,
    get_research_ticker,
    get_research_report,
)

TOOLS = {
    "get_market_data": get_market_data,
    "analyze_structure": analyze_structure,
    "calculate_position_risk": calculate_position_risk,
    "scan_stocks": scan_stocks,
    "evaluate_position": evaluate_position,
    "get_sp500_list": get_sp500_list,
    "get_nifty50_list": get_nifty50_list,
    "get_major_etfs": get_major_etfs,
    "web_search": web_search,
    "rank_trades": rank_trades,

    # Live broker holdings (replaces the old JSON portfolio memory)
    "connect_broker": connect_broker,
    "fetch_holdings": fetch_holdings,
    "get_portfolio_snapshot": get_portfolio_snapshot,
    "plan_trade": plan_trade,
    "get_risk_settings": get_risk_settings,
    "set_risk_settings": set_risk_settings,

    "research_portfolio": research_portfolio,
    "get_research_status": get_research_status,
    "get_research_ticker": get_research_ticker,
    "get_research_report": get_research_report,

    "get_watchlist": get_watchlist,
    "add_to_watchlist": add_to_watchlist,

    "set_strategy": set_strategy,
    "get_strategy": get_strategy,

    "detect_market_regime": detect_market_regime,
    "calculate_position_size": calculate_position_size,
    "backtest_strategy": backtest_strategy,
    "check_distance_from_ath": check_distance_from_ath,
}
