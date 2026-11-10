# Simple Crypto Backtester

A lightweight, dependency-minimal Python backtesting utility for cryptocurrency strategies. Built with **zero-lookahead execution**, **realistic friction modeling (maker/taker fees + slippage)**, and **institutional-grade risk metrics** out of the box.

---

## Key Features

- **Public Data Loader (`backtester/data_loader.py`)**:
  - Load historical OHLCV data from local **CSV** or **Parquet** files.
  - Fetch live or historical candlestick bars directly from **Binance's public REST API** (`/api/v3/klines`) without requiring an API key or third-party SDK.
- **Event-Driven Simulation Loop (`backtester/engine.py`)**:
  - Sequential bar-by-bar state machine tracking free cash, position size, collateral, unrealized/realized PnL, and trade-by-trade execution logs.
  - Supports long-only (`allow_short=False`) or long/short (`allow_short=True`) strategies.
- **Realistic Friction Modeling**:
  - Deducts configurable **maker/taker transaction fees** (e.g., `0.05%` per trade side) and applies adverse **execution slippage** (e.g., `0.02%` / 2 bps per fill) on every entry and exit.
- **Statistical Reporting (`backtester/metrics.py`)**:
  - **Total & Annualized Return** (benchmarked against Buy & Hold)
  - **Sharpe Ratio & Sortino Ratio** (annualized for 24/7 crypto markets)
  - **Maximum Drawdown (MDD)**
  - **Win Rate, Profit Factor, Average Trade PnL, and Total Fees Paid**

---

## Repository Structure

```text
simple-crypto-backtester/
├── README.md               # Quickstart guide + explanation of lookahead prevention
├── LICENSE                 # MIT License
├── requirements.txt        # pandas, numpy (minimal dependencies)
├── data/
│   └── sample_btc_1h.csv   # Bundled 30-day 1h BTCUSDT dataset for instant zero-setup testing
├── backtester/
│   ├── __init__.py
│   ├── engine.py           # Event-driven Backtester class & BacktestConfig
│   ├── metrics.py          # Standalone Sharpe, Sortino, MDD, Win Rate & Profit Factor calculators
│   └── data_loader.py      # CSV / Parquet / Public Binance REST fetch helper
└── run_backtest.py         # Entrypoint script demonstrating the workflow
```

---

## Quickstart

### 1. Install Dependencies

Set up virtual environment

```bash
python -m venv ./venv
source ./venv/bin/activate
```

Install requirements
```bash
python -m pip install -r requirements.txt
```

### 2. Run Instant Zero-Setup Backtest (Bundled Sample Data)

```bash
python run_backtest.py
```

### 3. Run Against Live Binance Public Data (No API Key Required)

```bash
python run_backtest.py --binance --symbol BTCUSDT --interval 1h --limit 720 --fee 0.0005 --slippage 0.0002
```

### 4. Programmatic Usage

```python
from backtester import Backtester, load_csv

# 1. Load standardized OHLCV data
df = load_csv("data/sample_btc_1h.csv")

# 2. Define a strategy function returning target signals: 1 (Long), 0 (Flat), -1 (Short)
def dual_ema_strategy(data):
    fast = data["close"].ewm(span=12, adjust=False).mean()
    slow = data["close"].ewm(span=36, adjust=False).mean()
    return (fast > slow).astype(int)

# 3. Configure realistic trading friction (0.05% fee + 0.02% slippage per fill)
engine = Backtester(
    initial_capital=10_000.0,
    fee_rate=0.0005,
    slippage_rate=0.0002,
    allow_short=False,
)

# 4. Run simulation and print report
result = engine.run(df, strategy=dual_ema_strategy)
print(result.summary())
```

---

## How Lookahead Bias is Prevented

One of the most common pitfalls in public backtesting scripts is **lookahead bias** (peeking into the future):

1. **The Classic Bug**: A strategy calculates an indicator using bar $t$'s closing price (`close[t]`) and immediately executes the buy/sell order at `close[t]`. In live trading, `close[t]` is only known at the exact instant bar $t$ finishes, at which point you can only fill at bar $t+1$'s opening price (`open[t+1]`) or later.
2. **Our Execution Model**:
   - At bar $t$, your strategy evaluates indicators using data available up to `close[t]` and emits a desired target state (`signal[t]`).
   - Inside `Backtester.run()`, target signals are shifted forward by one bar (`executable_target = signals.shift(1)`).
   - Consequently, a signal generated at the close of bar $t-1$ is executed at the **open of bar $t$ (`open[t]`)**.
3. **Adverse Slippage & Fee Deduction**:
   - **Buy Fills**: Executed at $\text{Fill Price} = \text{open}[t] \times (1 + \text{slippage\_rate})$.
   - **Sell Fills**: Executed at $\text{Fill Price} = \text{open}[t] \times (1 - \text{slippage\_rate})$.
   - **Transaction Fees**: Exchange fees ($\text{Notional} \times \text{fee\_rate}$) are deducted from cash on both entry and exit.

```
Bar t-1 Close                 Bar t Open                        Bar t Close
─────────────                 ──────────                        ───────────
Strategy computes             Order fills at:                   Position marked-to-market
signal[t-1] using  ────────►  open[t] * (1 ± slippage_rate) ──► at close[t] to update
data <= t-1                   minus maker/taker fee             bar-by-bar equity curve
```

---

## License

Released under the [MIT License](LICENSE).
