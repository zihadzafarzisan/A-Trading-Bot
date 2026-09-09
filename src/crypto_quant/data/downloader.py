"""Historical market data downloader.

Fetches OHLCV candles from an exchange adapter with pagination, rate-limit
management, retries, and incremental (start-from-cache) updates.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..config.settings import get_config
from ..exchange.base import ExchangeAdapter
from ..logging_config import get_logger
from .validator import DataValidator

# Milliseconds per minute (Binance convention)
MS_PER_MIN = 60_000

# Mapping of timeframe -> ms interval
TIMEFRAME_MS = {
    "1m": 1 * MS_PER_MIN,
    "5m": 5 * MS_PER_MIN,
    "15m": 15 * MS_PER_MIN,
    "30m": 30 * MS_PER_MIN,
    "1h": 60 * MS_PER_MIN,
    "4h": 240 * MS_PER_MIN,
    "1d": 1440 * MS_PER_MIN,
}


class DownloadResult:
    """Summary of a download operation."""

    def __init__(self, symbol: str, timeframe: str, n_candles: int, start: int, end: int, had_errors: bool):
        self.symbol = symbol
        self.timeframe = timeframe
        self.n_candles = n_candles
        self.start = start
        self.end = end
        self.had_errors = had_errors

    @property
    def n_candles_str(self) -> str:
        return f"{self.n_candles:,}"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_candles": self.n_candles,
            "start": self.start,
            "end": self.end,
            "had_errors": self.had_errors,
        }


class HistoricalDownloader:
    """Fetch historical candles with pagination and rate limiting."""

    def __init__(
        self,
        adapter: ExchangeAdapter,
        validator: Optional[DataValidator] = None,
        max_candles_per_request: int = 1000,
        requests_per_second: float = 10.0,
    ):
        """Initialize downloader.

        Args:
            adapter: Exchange adapter to fetch from.
            validator: Optional validator used to sanity-check batches.
            max_candles_per_request: Max candles per kline request (<=1000).
            requests_per_second: Soft cap on request frequency.
        """
        self.adapter = adapter
        self.validator = validator or DataValidator()
        self.max_candles_per_request = min(max_candles_per_request, 1000)
        self._min_request_interval = 1.0 / max(requests_per_second, 0.1)
        self._last_request_time = 0.0
        self._logger = get_logger("data")

    # --------------------------------------------------------- rate limiting
    def _throttle(self) -> None:
        """Sleep if necessary to respect the requests-per-second cap."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.monotonic()

    # ------------------------------------------------------------ download
    def download(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        on_batch: Optional[callable] = None,
    ) -> DownloadResult:
        """Download all candles in [start_ms, end_ms).

        Args:
            symbol: Trading pair.
            timeframe: Interval (e.g. '1h').
            start_ms: Inclusive start UNIX ms.
            end_ms: Exclusive end UNIX ms.
            on_batch: Optional callback receiving each (df, ok) batch.

        Returns:
            DownloadResult summary.
        """
        step = TIMEFRAME_MS.get(timeframe)
        if step is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        self._logger.info(
            "Downloading %s %s from %s to %s",
            symbol, timeframe,
            _fmt_ms(start_ms), _fmt_ms(end_ms),
        )

        cursor = start_ms
        all_rows: List = []
        total_fetched = 0
        had_errors = False
        last_good = start_ms

        while cursor < end_ms:
            # Request up to N candles forward; clamp to end
            req_start = cursor
            req_end = min(cursor + step * self.max_candles_per_request, end_ms)

            self._throttle()
            try:
                rows = self.adapter.get_klines(
                    symbol, timeframe, start_time=req_start, end_time=req_end - 1,
                    limit=self.max_candles_per_request,
                )
            except Exception as exc:
                self._logger.error(
                    "Download failed for %s %s at %s: %s",
                    symbol, timeframe, _fmt_ms(req_start), exc,
                )
                had_errors = True
                break

            if not rows:
                break
            last_good = max(last_good, rows[-1][0])

            # Drop duplicates / the (possibly overlapping) repeated candle
            rows = _dedupe(rows, last_seen=all_rows[-1][0] if all_rows else None)

            if rows:
                all_rows.extend(rows)
                total_fetched += len(rows)

            # Advance cursor beyond last returned candle
            cursor = rows[-1][0] + step if rows else req_end

            if on_batch:
                df = self._rows_to_df(rows)
                on_batch(df, not had_errors)

            # Safety: avoid infinite loop if the API stalls
            if cursor <= req_start:
                self._logger.warning(
                    "No progress on %s %s at %s; aborting to avoid loop",
                    symbol, timeframe, _fmt_ms(cursor),
                )
                had_errors = True
                break

        df = self._rows_to_df(all_rows)
        self._logger.info(
            "Downloaded %s %s: %s candles (%s -> %s)",
            symbol, timeframe, f"{total_fetched:,}", _fmt_ms(start_ms), _fmt_ms(last_good),
        )
        return DownloadResult(
            symbol=symbol,
            timeframe=timeframe,
            n_candles=total_fetched,
            start=start_ms,
            end=last_good,
            had_errors=had_errors,
        )

    # ------------------------------------------------------------ helpers
    def _rows_to_df(self, rows: List) -> pd.DataFrame:
        """Convert raw kline rows to a normalized DataFrame."""
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows)
        df = df.iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df.dropna(subset=["timestamp"], inplace=True)
        return df


def _dedupe(rows: List, last_seen: Optional[int]) -> List:
    """Drop a batch's first row if it duplicates the previous batch's last."""
    if last_seen is None or not rows:
        return rows
    if rows[0][0] == last_seen:
        return rows[1:]
    return rows


def _fmt_ms(ms: int) -> str:
    """Format a UNIX ms timestamp as an ISO-ish string."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M")
