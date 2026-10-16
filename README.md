# AI Trading Research Agent

An autonomous **AI-powered trading research assistant** that analyzes markets using structured technical analysis, risk management, and large language models running locally.

The system integrates a **local LLM**, technical analysis engines, portfolio management logic, and large-scale stock scanners to help identify **high-probability swing and position trading opportunities**.

The entire system runs **locally**, allowing private analysis without relying on external AI APIs.

---

# Overview

This project combines:

* A **local LLM reasoning agent**
* Multiple **technical analysis engines**
* A **portfolio risk management system**
* A **parallelized market scanner**
* A **strategy backtesting engine**

The AI can:

* Analyze individual stocks
* Scan large stock universes
* Rank trading opportunities
* Evaluate existing positions
* Detect market regimes
* Backtest strategies
* Manage portfolio risk

All reasoning is performed by a local LLM running through **Ollama**.

---

# System Architecture

```
User
  │
  ▼
LLM Reasoning Agent
  │
  ▼
Tool Layer
  │
  ├── Market Data Engine
  ├── Signal Engine
  ├── Risk Engine
  ├── Scanner Engine
  ├── Portfolio Manager
  ├── Market Regime Detector
  ├── Backtesting Engine
  └── Memory Engine
```

The LLM decides which tools to call based on user queries.

---

# Core Components

## Market Data Engine

Fetches historical and live market data using `yfinance`.

Responsibilities:

* Retrieve OHLCV data
* Fetch daily and weekly data
* Normalize data for analysis

---

## Signal Engine

Performs technical analysis on price data.

Includes:

* Swing high/low detection
* Support zone detection
* Fibonacci retracement levels
* ATR volatility calculation

This engine generates structured analysis used by the LLM.

---

## Risk Engine

Handles position sizing and risk calculations.

Calculates:

* Risk per share
* ATR-based stop losses
* Total trade risk
* Portfolio risk exposure
* Optimal position sizing

Risk is enforced using **percentage-based portfolio rules**.

---

## Scanner Engine

Scans large groups of stocks to find potential trading setups.

Features:

* Parallel scanning using `ThreadPoolExecutor`
* S&P500 scanning
* NIFTY50 scanning
* ETF scanning
* Crypto asset scanning

Only the **best candidates** are sent to the LLM for deeper analysis.

---

## Market Regime Detector

Determines overall market conditions.

Classifies markets into:

* Bull Trend
* Bear Trend
* Sideways
* High Volatility

Strategies are adjusted based on the detected regime.

---

## Portfolio Manager

Tracks existing positions and evaluates risk.

Capabilities:

* Analyze open positions
* Calculate portfolio exposure
* Suggest adding, reducing, or exiting positions
* Ensure portfolio risk limits are respected

---

## Backtesting Engine

Allows the AI to test strategies before recommending them.

Supported strategies include:

* Trend pullbacks
* Breakouts
* Mean reversion

Outputs include:

* Win rate
* Maximum drawdown
* Profit factor
* Risk-adjusted returns

---

## Memory Engine

Maintains conversation and strategy context.

Features:

* Stores conversation history
* Retains portfolio details
* Automatically trims old messages
* Maintains a rolling memory window

This allows the AI to remember user preferences and portfolio details.

---

# Local LLM Integration

The system runs using a local large language model via **Ollama**.

Recommended models:

Primary model:

```
qwen2.5:32b
```

Fallback model:

```
qwen2.5:14b
```

These models provide strong reasoning capabilities while running efficiently on modern GPUs.

---

# Hardware Requirements

Recommended GPU:

* NVIDIA RTX 4090
* 24GB VRAM

Typical model requirements:

| Model    | VRAM  |
| -------- | ----- |
| Qwen 14B | ~10GB |
| Qwen 32B | ~20GB |

CPU and RAM requirements are modest compared to GPU usage.

---

# Installation

## 1. Clone Repository

```
git clone <repo-url>
cd ai-trading-agent
```

---

## 2. Create Virtual Environment

```
python -m venv venv
```

Activate environment:

Windows

```
venv\Scripts\activate
```

---

## 3. Install Dependencies

```
pip install -r requirements.txt
```

Typical dependencies include:

```
yfinance
pandas
numpy
requests
```

---

## 4. Install Ollama

Download and install Ollama from:

https://ollama.ai

---

## 5. Pull the LLM Model

```
ollama pull qwen2.5:32b
```

Optional fallback:

```
ollama pull qwen2.5:14b
```

---

# Running the Agent

Start the trading agent:

```
python run_agent.py
```

You will enter an interactive prompt where you can query the system.

---

# Example Commands

Analyze a stock

```
analyze AAPL
```

Scan S&P500 for opportunities

```
scan sp500
```

Detect market regime

```
detect market regime
```

Backtest a strategy

```
backtest trend pullback on NVDA
```

Analyze a portfolio position

```
analyze my position in TSLA
```

---

# Example Analysis Output

```
Market Context
Bull Trend

Structure Analysis
Price is near structural support.

Support Zones
$107.11 | Medium | 6 touches

Volatility
Weekly ATR: $8.10

Risk Assessment
Stop loss: $94.50
Risk per share: $7.20

Trade Decision
High-quality pullback setup.
Reward-to-risk acceptable.
```

---

# Performance Features

Parallel scanning allows analysis of hundreds of stocks quickly.

Typical performance improvements:

Sequential scanning: ~5–10 minutes
Parallel scanning: ~10–30 seconds

---

# Design Philosophy

This system emphasizes:

* Structured technical analysis
* Strict risk management
* Data-driven decisions
* Tool-based reasoning
* Autonomous research workflows

The AI acts as a **research assistant**, not a black-box trading system.

---

# Future Improvements

Planned enhancements include:

* Multi-agent trading architecture
* Vector database for strategy memory
* Reinforcement learning from trade results
* Automated trade execution
* Macro-economic signal integration
* Options strategy analysis

---

# Disclaimer

This project is intended for **research and educational purposes only**.

It does not provide financial advice and should not be used for automated trading without extensive testing.

Always verify trading decisions independently.

---

# In-PROGRESS

* Proper memory implementation
* TODO list based research mode for longer queries 
* Improving backtesting methods
* General behavioural fixes where the agent does not fetch data , after fetching it once and storing it in memory .
