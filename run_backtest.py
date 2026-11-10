#!/usr/bin/env python3
"""
Entrypoint script demonstrating the Simple Crypto Backtester workflow:
1. Load historical OHLCV data (bundled CSV, custom CSV/Parquet, or live Binance public REST API)
2. Compute strategy signals without lookahead bias (EMA Crossover + RSI momentum filter)
3. Run the event-driven simulation with maker/taker fees and slippage
4. Print statistical risk/return reporting and sample trade log
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.data_loader import fetch_binance_ohlcv, load_csv, load_parquet
from backtester.engine import Backtester


def ema_rsi_strategy(
    df: pd.DataFrame,
    fast_span: int = 12,
    slow_span: int = 36,
    rsi_period: int = 14,
    rsi_exit_threshold: float = 78.0,
) -> pd.Series:
    """
    Example trend-following strategy with momentum exhaustion exit:
    - Enter/Hold LONG (1) when Fast EMA > Slow EMA and RSI < rsi_exit_threshold
    - Exit to FLAT (0) when Fast EMA <= Slow EMA or RSI >= rsi_exit_threshold

    Note: All indicators are computed strictly from historical `close` prices up to bar `t`.
    The Backtester engine shifts the resulting signal by 1 bar so a signal formed at
    `close[t]` is executed at `open[t+1]` with slippage and fees.
    """
    close = df["close"]
    fast_ema = close.ewm(span=fast_span, adjust=False).mean()
    slow_ema = close.ewm(span=slow_span, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / rsi_period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    signal = pd.Series(0, index=df.index, dtype=int)
    long_condition = (fast_ema > slow_ema) & (rsi < rsi_exit_threshold)
    signal.loc[long_condition] = 1
    # Warm-up period remains flat
    signal.iloc[:slow_span] = 0
    return signal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a realistic crypto strategy backtest with fees and slippage."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sample_btc_1h.csv",
        help="Path to local CSV or Parquet OHLCV file (default: data/sample_btc_1h.csv).",
    )
    parser.add_argument(
        "--binance",
        action="store_true",
        help="Fetch live historical klines from Binance public REST API instead of a local file.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Binance symbol when using --binance (default: BTCUSDT).",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Binance kline interval when using --binance (default: 1h).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of bars to fetch when using --binance (default: 500, max: 1000).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        help="Initial starting capital in USD/USDT (default: 10000.0).",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.0005,
        help="Taker/maker fee rate per trade side as decimal (default: 0.0005 = 0.05%%).",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=0.0002,
        help="Simulated adverse slippage per fill as decimal (default: 0.0002 = 0.02%%).",
    )
    parser.add_argument(
        "--fast-ema",
        type=int,
        default=12,
        help="Fast EMA period in bars (default: 12).",
    )
    parser.add_argument(
        "--slow-ema",
        type=int,
        default=36,
        help="Slow EMA period in bars (default: 36).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.binance:
        print(f"Fetching {args.limit} bars of {args.symbol} ({args.interval}) from Binance public API...")
        ohlcv = fetch_binance_ohlcv(symbol=args.symbol, interval=args.interval, limit=args.limit)
    else:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = Path(__file__).resolve().parent / data_path
        print(f"Loading historical OHLCV data from: {data_path}")
        if data_path.suffix.lower() in (".parquet", ".pq"):
            ohlcv = load_parquet(data_path)
        else:
            ohlcv = load_csv(data_path)

    start_ts = ohlcv["timestamp"].iloc[0]
    end_ts = ohlcv["timestamp"].iloc[-1]
    print(f"Loaded {len(ohlcv)} bars | Range: {start_ts} -> {end_ts}")
    print(
        f"Friction Config: Fee = {args.fee * 100:.3f}% per side | "
        f"Slippage = {args.slippage * 100:.3f}% per side\n"
    )

    engine = Backtester(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        position_size_pct=1.0,
        allow_short=False,
        close_on_last_bar=True,
        periods_per_year=8760,
    )

    result = engine.run(
        data=ohlcv,
        strategy=lambda df: ema_rsi_strategy(
            df, fast_span=args.fast_ema, slow_span=args.slow_ema
        ),
    )

    print(result.summary())

    if not result.trades.empty:
        print("\nCLOSED TRADES LOG (Last 10 Trades):")
        display_cols = [
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "fees",
            "net_pnl",
            "return_pct",
        ]
        preview = result.trades[display_cols].tail(10).copy()
        preview["entry_price"] = preview["entry_price"].map(lambda x: f"${x:,.2f}")
        preview["exit_price"] = preview["exit_price"].map(lambda x: f"${x:,.2f}")
        preview["fees"] = preview["fees"].map(lambda x: f"${x:,.2f}")
        preview["net_pnl"] = preview["net_pnl"].map(lambda x: f"${x:+,.2f}")
        preview["return_pct"] = preview["return_pct"].map(lambda x: f"{x * 100:+.2f}%")
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
