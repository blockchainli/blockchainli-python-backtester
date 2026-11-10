"""
Public OHLCV data loader supporting local CSV/Parquet files and keyless
Binance public REST API kline endpoints.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and standardize a raw DataFrame into canonical OHLCV format.

    Canonical columns:
        timestamp (UTC datetime64[ns, UTC]), open, high, low, close, volume (float64)
    Sorted chronologically with duplicate timestamps removed.
    """
    if df.empty:
        raise ValueError("OHLCV DataFrame is empty.")

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    # Map common aliases for timestamp
    alias_map = {
        "date": "timestamp",
        "datetime": "timestamp",
        "open_time": "timestamp",
        "time": "timestamp",
    }
    out = out.rename(columns={k: v for k, v in alias_map.items() if k in out.columns and "timestamp" not in out.columns})

    if "timestamp" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "timestamp"})

    missing = [col for col in REQUIRED_COLUMNS if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}. Found: {list(out.columns)}")

    # Parse timestamp (supports ISO strings or millisecond/second unix epochs)
    ts = out["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        unit = "ms" if ts.dropna().median() > 1e11 else "s"
        out["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        out["timestamp"] = pd.to_datetime(ts, utc=True)

    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

    out = out.dropna(subset=list(REQUIRED_COLUMNS))
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLCV price columns must be strictly positive.")

    return out[list(REQUIRED_COLUMNS)]


def load_csv(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load historical OHLCV data from a local CSV file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    raw_df = pd.read_csv(path)
    return standardize_ohlcv(raw_df)


def load_parquet(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load historical OHLCV data from a local Parquet file.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    raw_df = pd.read_parquet(path)
    return standardize_ohlcv(raw_df)


def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Union[int, str, None] = None,
    end_time: Union[int, str, None] = None,
    timeout: int = 15,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV candlesticks directly from Binance's public REST API
    without requiring an API key.

    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT').
        interval: Kline interval ('1m', '5m', '15m', '1h', '4h', '1d', etc.).
        limit: Number of bars to fetch (max 1000 per request on Binance).
        start_time: Optional start timestamp (ms epoch or ISO date string).
        end_time: Optional end timestamp (ms epoch or ISO date string).
        timeout: HTTP timeout in seconds.

    Returns:
        Standardized OHLCV DataFrame.
    """
    params: dict[str, Union[str, int]] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": min(max(int(limit), 1), 1000),
    }

    if start_time is not None:
        if isinstance(start_time, str):
            start_ms = int(pd.Timestamp(start_time, tz="UTC").timestamp() * 1000)
        else:
            start_ms = int(start_time)
        params["startTime"] = start_ms

    if end_time is not None:
        if isinstance(end_time, str):
            end_ms = int(pd.Timestamp(end_time, tz="UTC").timestamp() * 1000)
        else:
            end_ms = int(end_time)
        params["endTime"] = end_ms

    url = f"{BINANCE_KLINES_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "simple-crypto-backtester/1.0"})

    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not payload:
        raise ValueError(f"No kline data returned from Binance for {symbol} ({interval}).")

    # Binance kline format:
    # [open_time, open, high, low, close, volume, close_time, ...]
    records = [
        {
            "timestamp": int(row[0]),
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        }
        for row in payload
    ]
    return standardize_ohlcv(pd.DataFrame(records))
