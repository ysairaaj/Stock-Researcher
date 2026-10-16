from tools.market_data_engine import get_market_data
from tools.signal_engine import analyze_structure
from tools.risk_engine import calculate_position_risk
from tools.scanner_engine import scan_stocks
from tools.portfolio_manager import evaluate_position
from tools.universe_tools import get_sp500_list, get_nifty50_list, get_major_etfs
from tools.web_tools import web_search
from tools.ranking_engine import rank_trades
from tools.portfolio_memory import get_portfolio, add_position, remove_position
from tools.watchlist_memory import get_watchlist, add_to_watchlist
from agent.strategy_engine import set_strategy, get_strategy
from tools.regime_engine import detect_market_regime
from tools.position_sizing_engine import calculate_position_size
from tools.portfolio_value import get_portfolio_value
from tools.backtest_engine import backtest_strategy
from tools.ath_tools import check_distance_from_ath

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
    "get_portfolio": get_portfolio,
    "add_position": add_position,
    "remove_position": remove_position,

    "get_watchlist": get_watchlist,
    "add_to_watchlist": add_to_watchlist,

    "set_strategy": set_strategy,
    "get_strategy": get_strategy,

    "detect_market_regime": detect_market_regime,
    "calculate_position_size": calculate_position_size,
    "get_portfolio_value": get_portfolio_value,
    "backtest_strategy": backtest_strategy,
    "check_distance_from_ath": check_distance_from_ath,
}