"""Market data repository.

Persists and retrieves raw OHLCV market data in SQLite, provides caching
lookups, and orchestrates incremental downloads by checking what is already
stored. Downloaded data is validated and cleaned before being persisted.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import func

from ..config.settings import get_config
from ..db.connection import DatabaseManager
from ..db.models import MarketData
from ..logging_config import get_logger
from .cleaner import DataCleaner
from .downloader import HistoricalDownloader
from .validator import DataValidator

# Interval milliseconds per timeframe
TIMEFRAME_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MarketDataRepository:
    """Persists and retrieves OHLCV market data."""

    def __init__(self, db: DatabaseManager):
        """Initialize repository with a database manager."""
        self.db = db
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self._logger = get_logger("data")

    # ------------------------------------------------------------ retrieval
    def load(self, symbol: str, timeframe: str, start_ms: Optional[int] = None,
             end_ms: Optional[int] = None, market_type: str = "spot") -> pd.DataFrame:
        """Load candles from the database as a DataFrame (sorted ascending)."""
        session = self.db.get_session()
        try:
            q = session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe,
                MarketData.market_type == market_type,
            )
            if start_ms is not None:
                q = q.filter(MarketData.timestamp >= start_ms)
            if end_ms is not None:
                q = q.filter(MarketData.timestamp < end_ms)

            rows = q.order_by(MarketData.timestamp).all()
            if not rows:
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            return pd.DataFrame([{
                "timestamp": r.timestamp,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            } for r in rows])
        finally:
            session.close()

    def count(self, symbol: str, timeframe: str, market_type: str = "spot") -> int:
        """Count stored candles for a symbol/timeframe."""
        session = self.db.get_session()
        try:
            return (session.query(MarketData)
                    .filter(MarketData.symbol == symbol,
                            MarketData.timeframe == timeframe,
                            MarketData.market_type == market_type)
                    .count())
        finally:
            session.close()

    def range(self, symbol: str, timeframe: str, market_type: str = "spot"
              ) -> Tuple[Optional[int], Optional[int]]:
        """Return (min_timestamp, max_timestamp) stored, or (None, None)."""
        session = self.db.get_session()
        try:
            lo = (session.query(func.min(MarketData.timestamp))
                  .filter(MarketData.symbol == symbol,
                          MarketData.timeframe == timeframe,
                          MarketData.market_type == market_type).scalar())
            hi = (session.query(func.max(MarketData.timestamp))
                  .filter(MarketData.symbol == symbol,
                          MarketData.timeframe == timeframe,
                          MarketData.market_type == market_type).scalar())
            return (int(lo) if lo is not None else None,
                    int(hi) if hi is not None else None)
        finally:
            session.close()

    def list_datasets(self, market_type: str = "spot") -> List[Dict[str, object]]:
        """List all stored (symbol, timeframe) datasets with counts and ranges."""
        session = self.db.get_session()
        try:
            rows = (
                session.query(
                    MarketData.symbol,
                    MarketData.timeframe,
                    func.count(MarketData.id),
                    func.min(MarketData.timestamp),
                    func.max(MarketData.timestamp),
                )
                .filter(MarketData.market_type == market_type)
                .group_by(MarketData.symbol, MarketData.timeframe)
                .all()
            )
            return [
                {"symbol": s, "timeframe": tf, "n_candles": n, "start": lo, "end": hi}
                for s, tf, n, lo, hi in rows
            ]
        finally:
            session.close()

    # ------------------------------------------------------------ persistence
    def save(self, df: pd.DataFrame, symbol: str, timeframe: str,
             market_type: str = "spot") -> int:
        """Insert candles, skipping existing timestamps. Returns rows inserted."""
        if df is None or df.empty:
            return 0

        session = self.db.get_session()
        try:
            existing_ts = set(ts for (ts,) in (
                session.query(MarketData.timestamp)
                .filter(MarketData.symbol == symbol,
                        MarketData.timeframe == timeframe,
                        MarketData.market_type == market_type)
                .all()
            ))

            to_add = []
            inserted = 0
            for _, row in df.iterrows():
                ts = int(row["timestamp"])
                if ts in existing_ts:
                    continue
                existing_ts.add(ts)
                to_add.append(MarketData(
                    symbol=symbol, timeframe=timeframe, timestamp=ts,
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"]), market_type=market_type,
                ))
                inserted += 1

            if to_add:
                session.add_all(to_add)
                session.commit()
            return inserted
        finally:
            session.close()

    def delete_range(self, symbol: str, timeframe: str, start_ms: int, end_ms: int,
                     market_type: str = "spot") -> int:
        """Delete candles in [start_ms, end_ms). Returns rows deleted."""
        session = self.db.get_session()
        try:
            deleted = (session.query(MarketData)
                       .filter(MarketData.symbol == symbol,
                               MarketData.timeframe == timeframe,
                               MarketData.market_type == market_type,
                               MarketData.timestamp >= start_ms,
                               MarketData.timestamp < end_ms)
                       .delete(synchronize_session=False))
            session.commit()
            return int(deleted)
        finally:
            session.close()

    # ------------------------------------------------------------ validation
    def validate(self, symbol: str, timeframe: str, market_type: str = "spot"):
        """Validate stored data for a symbol/timeframe, returning a report."""
        df = self.load(symbol, timeframe, market_type=market_type)
        return self.validator.validate_dataframe(df, symbol, timeframe)

    # ------------------------------------------------------------ orchestration
    def ensure_data(
        self,
        symbol: str,
        timeframe: str,
        start_ms: int,
        end_ms: int,
        downloader: HistoricalDownloader,
        market_type: str = "spot",
        incremental: bool = True,
    ) -> Dict[str, object]:
        """Ensure data in [start_ms, end_ms) exists in the database.

        Downloads missing portions, validates and cleans, then persists.

        Returns:
            Summary dict: {'requested_start', 'requested_end', 'stored_start',
                           'stored_end', 'n_candles', 'inserted', 'n_duplicates',
                           'had_errors'}.
        """
        interval = TIMEFRAME_MS.get(timeframe, 3_600_000)
        stored_lo, stored_hi = self.range(symbol, timeframe, market_type)

        # Determine download window
        download_start = start_ms
        download_end = end_ms
        already_covered = False

        if incremental and stored_lo is not None and stored_hi is not None:
            # Case 1: Requested range is entirely WITHIN the stored range
            if stored_lo <= start_ms and stored_hi + interval >= end_ms:
                already_covered = True

            # Case 2: Requested range is entirely AFTER the stored range
            elif stored_hi < start_ms:
                download_start = start_ms

            # Case 3: Requested range is entirely BEFORE the stored range
            elif end_ms <= stored_lo:
                download_start = start_ms
                download_end = end_ms

            # Case 4: Overlaps on the right (fetch the missing right part)
            elif stored_lo <= start_ms <= stored_hi + interval:
                download_start = stored_hi + interval

            # Case 5: Overlaps on the left (fetch the missing left part)
            elif start_ms < stored_lo:
                download_end = stored_lo
                # and maybe fetch the missing right part too? We only support one continuous fetch window here,
                # so we just fetch the whole requested range if it spans across the existing data.
                download_start = start_ms
                if end_ms > stored_hi + interval:
                    download_end = end_ms

        if already_covered:
            return {
                "requested_start": start_ms, "requested_end": end_ms,
                "stored_start": stored_lo, "stored_end": stored_hi,
                "n_candles": self.count(symbol, timeframe, market_type),
                "inserted": 0, "n_duplicates": 0, "had_errors": False,
                "note": "already covered",
            }

        if download_start >= download_end:
            return {
                "requested_start": start_ms, "requested_end": end_ms,
                "stored_start": stored_lo, "stored_end": stored_hi,
                "n_candles": 0, "inserted": 0, "n_duplicates": 0,
                "had_errors": False, "note": "window already covered",
            }

        # Download, accumulating batches into a frame
        batches: List[pd.DataFrame] = []
        result = downloader.download(symbol, timeframe, download_start, download_end,
                                     on_batch=lambda df, ok: batches.append(df))

        if batches:
            downloaded = pd.concat(batches, ignore_index=True).drop_duplicates("timestamp")
        else:
            downloaded = pd.DataFrame(columns=OHLCV_COLUMNS)

        if downloaded.empty:
            return {
                "requested_start": download_start, "requested_end": download_end,
                "stored_start": stored_lo, "stored_end": stored_hi,
                "n_candles": 0, "inserted": 0, "n_duplicates": 0,
                "had_errors": result.had_errors, "note": "no data returned",
            }

        # Validate + clean the downloaded frame
        vreport = self.validator.validate_dataframe(downloaded, symbol, timeframe)
        if vreport.error_count:
            self._logger.warning(
                "Validation errors for %s %s: %s", symbol, timeframe,
                [i.message for i in vreport.issues if i.severity == "error"],
            )
        cleaned, _creport = self.cleaner.clean(downloaded, symbol, timeframe)

        inserted = self.save(cleaned, symbol, timeframe, market_type)

        # Post-state
        new_lo, new_hi = self.range(symbol, timeframe, market_type)
        return {
            "requested_start": start_ms, "requested_end": end_ms,
            "stored_start": new_lo, "stored_end": new_hi,
            "n_candles": self.count(symbol, timeframe, market_type),
            "inserted": inserted,
            "n_duplicates": vreport.warning_count,
            "had_errors": result.had_errors,
        }

    def _on_batch(self, df: pd.DataFrame, ok: bool) -> None:
        """Callback hook for batch progress (kept lightweight)."""
        if not ok:
            self._logger.debug("Batch reported failure during download")
