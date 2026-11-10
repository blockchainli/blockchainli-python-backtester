"""
Simple Crypto Backtester package.

Provides an event-driven OHLCV backtesting engine with realistic friction
modeling (maker/taker fees and slippage), zero-lookahead signal execution,
and standalone risk/performance metrics.
"""

from .data_loader import fetch_binance_ohlcv, load_csv, load_parquet, standardize_ohlcv
from .engine import BacktestConfig, Backtester, BacktestResult
from .metrics import (
    annualized_return,
    compute_summary_metrics,
    drawdown_series,
    format_metrics_report,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

__all__ = [
    "BacktestConfig",
    "Backtester",
    "BacktestResult",
    "annualized_return",
    "compute_summary_metrics",
    "drawdown_series",
    "fetch_binance_ohlcv",
    "format_metrics_report",
    "load_csv",
    "load_parquet",
    "max_drawdown",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "standardize_ohlcv",
    "total_return",
    "win_rate",
]
