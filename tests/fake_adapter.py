"""Deterministic in-memory exchange adapter for tests."""

import pandas as pd

from crypto_quant.exchange.base import ExchangeAdapter


class FakeAdapter(ExchangeAdapter):
    """In-memory adapter generating predictable OHLCV klines."""

    exchange_name = "fake"

    def __init__(self, n_candles: int, interval: int = 3_600_000, start: int = 1_600_000_000_000):
        self.n_candles = n_candles
        self.interval = interval
        self.start = start
        self.calls = 0
        self.first_request_start = None  # Track first request start for verification

    def _make_klines(self, start_time, end_time, limit):
        """Generate klines within the requested window."""
        self.calls += 1
        # Track the first request's start time for regression tests
        if self.first_request_start is None and start_time is not None:
            self.first_request_start = start_time
        rows = []
        last_ts = self.start + self.interval * (self.n_candles - 1)
        start_ts = max(start_time, self.start) if start_time is not None else self.start
        end_ts = min(end_time, last_ts) if end_time is not None else last_ts
        ts = start_ts
        while ts <= end_ts and len(rows) < limit:
            rows.append([ts, "100", "110", "90", "105", "1000",
                         ts + self.interval, "100000", 500, "500", "50000", "0"])
            ts += self.interval
        return rows

    def get_klines(self, symbol, timeframe, start_time=None, end_time=None, limit=1000):
        return self._make_klines(start_time, end_time or (self.start + self.interval * self.n_candles), limit)

    # --- stubs for unused abstract methods ---
    def get_server_time(self):
        return self.start

    def get_exchange_info(self):
        return {}

    def get_symbols(self):
        return ["BTCUSDT"]

    def get_funding_rate_history(self, symbol, start_time=None, end_time=None, limit=1000):
        return []

    def get_ohlcv_as_dataframe(self, symbol, timeframe, **kwargs):
        return pd.DataFrame()

    def close(self):
        pass
