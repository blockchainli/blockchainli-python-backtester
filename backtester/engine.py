"""
Event-driven sequential backtesting engine with realistic friction modeling
(transaction fees + slippage) and strict lookahead bias prevention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd

from .data_loader import standardize_ohlcv
from .metrics import compute_summary_metrics, format_metrics_report

StrategyCallable = Callable[[pd.DataFrame], Union[pd.Series, np.ndarray, list[int]]]


@dataclass(frozen=True)
class BacktestConfig:
    """
    Configuration parameters for the backtesting engine.

    Attributes:
        initial_capital: Starting cash in quote currency (e.g. USDT).
        fee_rate: Exchange fee rate per trade side (default 0.0005 = 0.05%).
        slippage_rate: Simulated adverse price slippage per fill (default 0.0002 = 0.02% / 2 bps).
        position_size_pct: Fraction of current available equity allocated per entry (0 < pct <= 1.0).
        allow_short: Whether target signal -1 opens a short position (if False, -1 is treated as 0 / flat).
        close_on_last_bar: Whether to force-close any open position at the final bar's close.
        periods_per_year: Annualization factor for Sharpe/Sortino (8760 for 1h bars, 365 for 1d bars).
    """

    initial_capital: float = 10_000.0
    fee_rate: float = 0.0005
    slippage_rate: float = 0.0002
    position_size_pct: float = 1.0
    allow_short: bool = False
    close_on_last_bar: bool = True
    periods_per_year: int = 8760

    def __post_init__(self) -> None:
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be > 0.")
        if self.fee_rate < 0 or self.fee_rate >= 0.5:
            raise ValueError("fee_rate must be in [0.0, 0.5).")
        if self.slippage_rate < 0 or self.slippage_rate >= 0.5:
            raise ValueError("slippage_rate must be in [0.0, 0.5).")
        if not (0.0 < self.position_size_pct <= 1.0):
            raise ValueError("position_size_pct must be in (0.0, 1.0].")


@dataclass
class BacktestResult:
    """
    Container for backtest outputs: bar-by-bar equity DataFrame, closed trade log,
    and computed statistical metrics.
    """

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, Any]
    config: BacktestConfig

    def summary(self) -> str:
        """Return a formatted text report of key risk & performance metrics."""
        return format_metrics_report(self.metrics)


class Backtester:
    """
    Sequential event-driven OHLCV backtesting engine.

    Lookahead Prevention Guarantee:
    -------------------------------
    A signal emitted at bar `t-1` (using data up to `close[t-1]`) cannot fill at
    `close[t-1]` in live trading without lookahead bias. This engine shifts target
    signals by 1 bar so orders triggered on bar `t-1` execute at `open[t]` plus
    adverse slippage and transaction fees.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0002,
        position_size_pct: float = 1.0,
        allow_short: bool = False,
        close_on_last_bar: bool = True,
        periods_per_year: int = 8760,
        config: Optional[BacktestConfig] = None,
    ) -> None:
        self.config = config or BacktestConfig(
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            position_size_pct=position_size_pct,
            allow_short=allow_short,
            close_on_last_bar=close_on_last_bar,
            periods_per_year=periods_per_year,
        )

    def _apply_slippage(self, raw_price: float, side: str) -> float:
        """Apply adverse slippage to an execution price ('buy' pays higher, 'sell' receives lower)."""
        if side == "buy":
            return raw_price * (1.0 + self.config.slippage_rate)
        if side == "sell":
            return raw_price * (1.0 - self.config.slippage_rate)
        raise ValueError(f"Unknown order side: {side}")

    def run(
        self,
        data: pd.DataFrame,
        strategy: Optional[StrategyCallable] = None,
        signal_col: str = "signal",
    ) -> BacktestResult:
        """
        Run the sequential simulation loop over historical OHLCV bars.

        Args:
            data: OHLCV DataFrame (must contain timestamp, open, high, low, close, volume).
            strategy: Optional callable taking `df` and returning target position signals
                      (1 for Long, 0 for Flat, -1 for Short). If omitted, `data[signal_col]` is used.
            signal_col: Column name to read target signals from when `strategy` is None.

        Returns:
            BacktestResult with `equity_curve`, `trades`, and `metrics`.
        """
        df = standardize_ohlcv(data)
        if strategy is not None:
            raw_signals = pd.Series(strategy(df), index=df.index)
        elif signal_col in data.columns:
            # Align signal column with standardized rows
            temp = data.copy()
            raw_signals = pd.Series(temp[signal_col].to_numpy(), index=df.index)
        else:
            raise ValueError(
                f"Provide either a `strategy` callable or a `{signal_col}` column in `data`."
            )

        # Normalize signals to {-1, 0, 1}
        signals = np.sign(pd.to_numeric(raw_signals, errors="coerce").fillna(0.0)).astype(int)
        if not self.config.allow_short:
            signals = signals.clip(lower=0)

        # LOOKAHEAD PREVENTION:
        # Target position evaluated at bar t-1 close is executed at bar t open.
        executable_target = signals.shift(1).fillna(0).astype(int).to_numpy()

        n_bars = len(df)
        timestamps = df["timestamp"].to_numpy()
        opens = df["open"].to_numpy(dtype=float)
        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)
        volumes = df["volume"].to_numpy(dtype=float)

        cash = float(self.config.initial_capital)
        position_dir = 0  # 1 = long, -1 = short, 0 = flat
        position_qty = 0.0  # signed quantity (+ for long, - for short)
        entry_price = 0.0
        entry_time = None
        entry_fee = 0.0
        entry_notional = 0.0

        trades_list: list[dict[str, Any]] = []
        equity_records: list[dict[str, Any]] = []

        fee_rate = self.config.fee_rate
        alloc_pct = self.config.position_size_pct

        def close_current_position(exec_raw_price: float, exec_time: Any, reason: str) -> float:
            nonlocal cash, position_dir, position_qty, entry_price, entry_time, entry_fee, entry_notional
            if position_dir == 0 or position_qty == 0.0:
                return 0.0

            abs_qty = abs(position_qty)
            if position_dir == 1:
                fill_price = self._apply_slippage(exec_raw_price, "sell")
                exit_notional = abs_qty * fill_price
                exit_fee = exit_notional * fee_rate
                gross_pnl = (fill_price - entry_price) * abs_qty
                cash += exit_notional - exit_fee
                side_label = "LONG"
            else:
                fill_price = self._apply_slippage(exec_raw_price, "buy")
                exit_notional = abs_qty * fill_price
                exit_fee = exit_notional * fee_rate
                gross_pnl = (entry_price - fill_price) * abs_qty
                # When short was opened, cash held collateral + initial proceeds; settle PnL
                cash += entry_notional + gross_pnl - exit_fee
                side_label = "SHORT"

            total_trade_fees = entry_fee + exit_fee
            net_pnl = gross_pnl - total_trade_fees
            return_pct = net_pnl / entry_notional if entry_notional > 0 else 0.0

            trades_list.append(
                {
                    "side": side_label,
                    "entry_time": entry_time,
                    "exit_time": exec_time,
                    "entry_price": entry_price,
                    "exit_price": fill_price,
                    "quantity": abs_qty,
                    "gross_pnl": gross_pnl,
                    "fees": total_trade_fees,
                    "net_pnl": net_pnl,
                    "return_pct": return_pct,
                    "exit_reason": reason,
                }
            )

            position_dir = 0
            position_qty = 0.0
            entry_price = 0.0
            entry_time = None
            entry_fee = 0.0
            entry_notional = 0.0
            return exit_fee

        def open_new_position(target_dir: int, exec_raw_price: float, exec_time: Any) -> float:
            nonlocal cash, position_dir, position_qty, entry_price, entry_time, entry_fee, entry_notional
            if target_dir == 0 or cash <= 0.0:
                return 0.0

            capital_to_deploy = cash * alloc_pct
            # Ensure capital_to_deploy covers both notional and entry fee:
            # notional * (1 + fee_rate) = capital_to_deploy
            notional = capital_to_deploy / (1.0 + fee_rate)
            fee = notional * fee_rate

            if target_dir == 1:
                fill_price = self._apply_slippage(exec_raw_price, "buy")
                qty = notional / fill_price
                cash -= notional + fee
                position_dir = 1
                position_qty = qty
            else:
                fill_price = self._apply_slippage(exec_raw_price, "sell")
                qty = notional / fill_price
                # Reserve collateral (`notional`) out of free cash and pay entry fee
                cash -= notional + fee
                position_dir = -1
                position_qty = -qty

            entry_price = fill_price
            entry_time = exec_time
            entry_fee = fee
            entry_notional = notional
            return fee

        # Event-driven bar-by-bar loop
        for i in range(n_bars):
            ts = timestamps[i]
            o, h, l, c, v = opens[i], highs[i], lows[i], closes[i], volumes[i]
            target_dir = int(executable_target[i])
            bar_fees = 0.0

            # 1. Execute transitions at the bar's OPEN price
            if target_dir != position_dir:
                if position_dir != 0:
                    bar_fees += close_current_position(o, ts, reason="signal_exit")
                if target_dir != 0:
                    bar_fees += open_new_position(target_dir, o, ts)

            # 2. Optionally force-close open position at the final bar's CLOSE price
            if i == n_bars - 1 and self.config.close_on_last_bar and position_dir != 0:
                bar_fees += close_current_position(c, ts, reason="end_of_data")

            # 3. Mark-to-market equity at bar CLOSE
            if position_dir == 1:
                position_value = abs(position_qty) * c
            elif position_dir == -1:
                unrealized_pnl = (entry_price - c) * abs(position_qty)
                position_value = entry_notional + unrealized_pnl
            else:
                position_value = 0.0

            total_equity = cash + position_value

            equity_records.append(
                {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "signal": int(signals.iloc[i]),
                    "position": position_dir,
                    "cash": cash,
                    "position_value": position_value,
                    "fees_paid": bar_fees,
                    "equity": total_equity,
                }
            )

        equity_df = pd.DataFrame(equity_records)
        trades_df = pd.DataFrame(
            trades_list,
            columns=[
                "side",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "gross_pnl",
                "fees",
                "net_pnl",
                "return_pct",
                "exit_reason",
            ],
        )

        metrics = compute_summary_metrics(
            equity_df=equity_df,
            trades_df=trades_df,
            initial_capital=self.config.initial_capital,
            periods_per_year=self.config.periods_per_year,
        )

        return BacktestResult(
            equity_curve=equity_df,
            trades=trades_df,
            metrics=metrics,
            config=self.config,
        )
