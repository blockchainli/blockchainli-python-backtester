"""
Standalone risk and performance metric calculators for backtest evaluation.

Includes:
- Total & Annualized Return
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown (MDD) & Drawdown Series
- Win Rate & Profit Factor
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def total_return(equity_curve: pd.Series) -> float:
    """
    Compute cumulative return from an equity curve series.
    Returns a decimal (e.g., 0.25 for +25.0%).
    """
    if len(equity_curve) < 2 or equity_curve.iloc[0] == 0:
        return 0.0
    return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)


def annualized_return(equity_curve: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Compute Compound Annual Growth Rate (CAGR) assuming `periods_per_year` bars
    (default 8760 for 24/7 1-hour crypto bars).
    """
    n_periods = len(equity_curve) - 1
    if n_periods <= 0 or equity_curve.iloc[0] <= 0 or equity_curve.iloc[-1] <= 0:
        return 0.0
    cum_growth = float(equity_curve.iloc[-1] / equity_curve.iloc[0])
    years = n_periods / float(periods_per_year)
    if years <= 0:
        return 0.0
    return float(cum_growth ** (1.0 / years) - 1.0)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Compute annualized Sharpe Ratio from periodic bar returns.

    Args:
        returns: Series of periodic decimal returns (e.g. equity.pct_change()).
        risk_free_rate: Annualized risk-free rate (default 0.0).
        periods_per_year: Number of bars in a year (8760 for 1h crypto, 365 for 1d crypto).
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_per_period
    std = float(excess.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """
    Compute annualized Sortino Ratio using downside semi-deviation.

    Args:
        returns: Series of periodic decimal returns.
        risk_free_rate: Annualized target/risk-free rate (default 0.0).
        periods_per_year: Number of bars in a year (8760 for 1h crypto).
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = clean - rf_per_period
    downside_diff = np.minimum(0.0, excess.to_numpy(dtype=float))
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_diff))))
    if downside_deviation == 0.0 or np.isnan(downside_deviation):
        return 0.0
    return float((excess.mean() / downside_deviation) * np.sqrt(periods_per_year))


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute the percentage drawdown series from running peak equity.
    Values are <= 0.0 (e.g., -0.12 for a 12% drawdown).
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)
    running_max = equity_curve.cummax()
    return (equity_curve / running_max) - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Compute Maximum Drawdown (MDD) as a non-positive decimal (e.g., -0.153 for -15.3%).
    """
    dd = drawdown_series(equity_curve)
    if dd.empty:
        return 0.0
    return float(dd.min())


def win_rate(trades: pd.DataFrame) -> float:
    """
    Compute the fraction of closed trades with strictly positive net PnL.
    Returns a decimal in [0.0, 1.0].
    """
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    wins = int((trades["net_pnl"] > 0).sum())
    return float(wins / len(trades))


def profit_factor(trades: pd.DataFrame) -> float:
    """
    Compute Profit Factor = Gross Winning PnL / |Gross Losing PnL|.
    Returns np.inf if there are winning trades and zero losing trades, or 0.0 if no wins.
    """
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    gross_profit = float(trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum())
    gross_loss = float(abs(trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()))
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def compute_summary_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 8760,
) -> dict[str, Any]:
    """
    Compute a comprehensive dictionary of risk, return, and trade statistics.
    """
    equity = equity_df["equity"]
    returns = equity.pct_change().fillna(0.0)

    benchmark_return = 0.0
    if "close" in equity_df.columns and len(equity_df) >= 2:
        benchmark_return = float((equity_df["close"].iloc[-1] / equity_df["close"].iloc[0]) - 1.0)

    total_fees = float(equity_df["fees_paid"].sum()) if "fees_paid" in equity_df.columns else 0.0
    avg_trade_pnl = float(trades_df["net_pnl"].mean()) if not trades_df.empty else 0.0
    avg_trade_return = float(trades_df["return_pct"].mean()) if not trades_df.empty else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else float(initial_capital),
        "net_profit": float(equity.iloc[-1] - initial_capital) if not equity.empty else 0.0,
        "total_return": total_return(equity),
        "annualized_return": annualized_return(equity, periods_per_year=periods_per_year),
        "buy_and_hold_return": benchmark_return,
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "total_trades": int(len(trades_df)),
        "win_rate": win_rate(trades_df),
        "profit_factor": profit_factor(trades_df),
        "avg_trade_pnl": avg_trade_pnl,
        "avg_trade_return": avg_trade_return,
        "total_fees_paid": total_fees,
    }


def format_metrics_report(metrics: dict[str, Any]) -> str:
    """
    Render the metrics dictionary into a clean, human-readable ASCII summary table.
    """
    pf = metrics["profit_factor"]
    pf_str = "∞" if np.isinf(pf) else f"{pf:.2f}"

    lines = [
        "=" * 56,
        "             BACKTEST PERFORMANCE SUMMARY               ",
        "=" * 56,
        f"  Initial Capital        : ${metrics['initial_capital']:,.2f}",
        f"  Final Equity           : ${metrics['final_equity']:,.2f}",
        f"  Net Profit             : ${metrics['net_profit']:+,.2f}",
        f"  Total Return           : {metrics['total_return'] * 100:+.2f}%",
        f"  Annualized Return      : {metrics['annualized_return'] * 100:+.2f}%",
        f"  Buy & Hold Benchmark   : {metrics['buy_and_hold_return'] * 100:+.2f}%",
        "-" * 56,
        f"  Sharpe Ratio           : {metrics['sharpe_ratio']:.2f}",
        f"  Sortino Ratio          : {metrics['sortino_ratio']:.2f}",
        f"  Maximum Drawdown (MDD) : {metrics['max_drawdown'] * 100:.2f}%",
        "-" * 56,
        f"  Total Closed Trades    : {metrics['total_trades']}",
        f"  Win Rate               : {metrics['win_rate'] * 100:.2f}%",
        f"  Profit Factor          : {pf_str}",
        f"  Avg Trade Net PnL      : ${metrics['avg_trade_pnl']:+,.2f} ({metrics['avg_trade_return'] * 100:+.2f}%)",
        f"  Total Fees Paid        : ${metrics['total_fees_paid']:,.2f}",
        "=" * 56,
    ]
    return "\n".join(lines)
