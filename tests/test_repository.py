"""Tests for market data repository."""

import pytest
import pandas as pd

from crypto_quant.db.connection import DatabaseManager
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.data.downloader import HistoricalDownloader

from fake_adapter import FakeAdapter


@pytest.fixture
def db():
    """In-memory database for tests."""
    d = DatabaseManager(in_memory=True)
    d.create_tables()
    return d


@pytest.fixture
def repo(db):
    """Repository backed by in-memory DB."""
    return MarketDataRepository(db)


@pytest.fixture
def valid_df():
    """A clean OHLCV DataFrame."""
    base = 1_600_000_000_000
    step = 3_600_000
    rows = [
        {"timestamp": base + i * step, "open": 100.0, "high": 110.0,
         "low": 90.0, "close": 105.0, "volume": 1000.0}
        for i in range(5)
    ]
    return pd.DataFrame(rows)


class TestMarketDataRepository:
    """Test repository operations."""

    ONE_HOUR = 3_600_000
    START = 1_600_000_000_000

    def test_save_and_load(self, repo, valid_df):
        """Test saving and loading candles."""
        inserted = repo.save(valid_df, "BTCUSDT", "1h")
        assert inserted == 5

        loaded = repo.load("BTCUSDT", "1h")
        assert len(loaded) == 5
        assert loaded["close"].iloc[0] == 105.0

    def test_save_deduplicates(self, repo, valid_df):
        """Test saving duplicate data doesn't duplicate rows."""
        repo.save(valid_df, "BTCUSDT", "1h")
        inserted = repo.save(valid_df, "BTCUSDT", "1h")
        assert inserted == 0
        assert repo.count("BTCUSDT", "1h") == 5

    def test_count_and_range(self, repo, valid_df):
        """Test count and range methods."""
        repo.save(valid_df, "BTCUSDT", "1h")

        assert repo.count("BTCUSDT", "1h") == 5
        lo, hi = repo.range("BTCUSDT", "1h")
        assert lo == valid_df["timestamp"].min()
        assert hi == valid_df["timestamp"].max()

    def test_load_with_time_window(self, repo, valid_df):
        """Test loading with a time filter."""
        repo.save(valid_df, "BTCUSDT", "1h")
        start = int(valid_df["timestamp"].iloc[0])
        mid = int(valid_df["timestamp"].iloc[2])
        loaded = repo.load("BTCUSDT", "1h", start_ms=start, end_ms=mid)
        assert len(loaded) == 2  # first two candles

    def test_delete_range(self, repo, valid_df):
        """Test deleting a subset."""
        repo.save(valid_df, "BTCUSDT", "1h")
        start = int(valid_df["timestamp"].iloc[0])
        mid = int(valid_df["timestamp"].iloc[3])
        deleted = repo.delete_range("BTCUSDT", "1h", start, mid)
        assert deleted == 3
        assert repo.count("BTCUSDT", "1h") == 2

    def test_list_datasets(self, repo, valid_df):
        """Test listing stored datasets."""
        repo.save(valid_df, "BTCUSDT", "1h")
        datasets = repo.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["symbol"] == "BTCUSDT"
        assert datasets[0]["n_candles"] == 5

    def test_ensure_data_download_and_store(self, repo, db):
        """Test full ensure_data flow: download -> validate -> clean -> save."""
        adapter = FakeAdapter(n_candles=100, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        summary = repo.ensure_data(
            "BTCUSDT", "1h",
            start_ms=self.START,
            end_ms=self.START + 100 * self.ONE_HOUR,
            downloader=downloader,
        )
        assert summary["inserted"] == 100
        assert summary["n_candles"] == 100
        assert summary["had_errors"] is False

    def test_ensure_data_incremental(self, repo, db):
        """Test incremental update skips already-stored data."""
        adapter = FakeAdapter(n_candles=100, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        # First run downloads all 100
        repo.ensure_data("BTCUSDT", "1h", self.START, self.START + 100 * self.ONE_HOUR, downloader)
        assert repo.count("BTCUSDT", "1h") == 100
        calls_after_first = adapter.calls

        # Second run should detect coverage and skip
        summary = repo.ensure_data("BTCUSDT", "1h", self.START, self.START + 100 * self.ONE_HOUR, downloader)
        assert summary["inserted"] == 0
        assert summary["note"] == "already covered"
        assert adapter.calls == calls_after_first  # no additional requests

    def test_ensure_data_non_contiguous_range(self, repo, db):
        """Regression: requesting data AFTER existing non-contiguous range.

        Bug: When existing data is 2021 and requested range is 2025,
        the code incorrectly tried to download from 2021 (after stored end)
        instead of from 2025 (requested start).

        This test verifies that requesting a future date range works correctly.
        """
        # Step 1: Store some "old" data (simulating 2021)
        old_start = 1_609_459_200_000  # 2021-01-01 00:00:00 UTC
        adapter_old = FakeAdapter(n_candles=100, interval=self.ONE_HOUR, start=old_start)
        downloader_old = HistoricalDownloader(adapter_old, max_candles_per_request=1000)

        summary_old = repo.ensure_data(
            "BTCUSDT", "1h",
            start_ms=old_start,
            end_ms=old_start + 100 * self.ONE_HOUR,
            downloader=downloader_old,
        )
        assert summary_old["inserted"] == 100
        lo_old, hi_old = repo.range("BTCUSDT", "1h")
        assert lo_old == old_start
        assert hi_old == old_start + 99 * self.ONE_HOUR  # last candle timestamp

        # Step 2: Request data for a MUCH later period (2025)
        new_start = 1_735_689_600_000  # 2025-01-01 00:00:00 UTC
        new_end = new_start + 50 * self.ONE_HOUR  # 2025-01-03 02:00:00 UTC

        adapter_new = FakeAdapter(n_candles=50, interval=self.ONE_HOUR, start=new_start)
        downloader_new = HistoricalDownloader(adapter_new, max_candles_per_request=1000)

        summary_new = repo.ensure_data(
            "BTCUSDT", "1h",
            start_ms=new_start,
            end_ms=new_end,
            downloader=downloader_new,
        )

        # Step 3: Verify the new data was downloaded from the CORRECT start
        # The adapter should have been called with start=new_start, NOT old_end+interval
        assert adapter_new.first_request_start == new_start, (
            f"Downloader should start from requested start {new_start}, "
            f"but started from {adapter_new.first_request_start}"
        )

        # Step 4: Verify new data is stored correctly
        assert summary_new["inserted"] == 50
        assert summary_new["n_candles"] == 150  # 100 old + 50 new

        # Step 5: Verify the stored range covers BOTH periods
        lo_final, hi_final = repo.range("BTCUSDT", "1h")
        assert lo_final == old_start  # Still starts from old data
        assert hi_final == new_start + 49 * self.ONE_HOUR  # Ends at new data

        # Step 6: Verify we can load data from the new period specifically
        loaded_new = repo.load("BTCUSDT", "1h", start_ms=new_start, end_ms=new_end)
        assert len(loaded_new) == 50
        assert loaded_new["timestamp"].min() == new_start
        assert loaded_new["timestamp"].max() == new_start + 49 * self.ONE_HOUR

        # Step 7: Verify list_datasets shows correct range
        datasets = repo.list_datasets()
        assert len(datasets) == 1
        assert datasets[0]["n_candles"] == 150
        assert datasets[0]["start"] == old_start
        assert datasets[0]["end"] == new_start + 49 * self.ONE_HOUR

    def test_ensure_data_before_existing_range(self, repo, db):
        """Requesting data BEFORE existing range works correctly."""
        # Store some "new" data first
        new_start = 1_735_689_600_000  # 2025-01-01
        adapter_new = FakeAdapter(n_candles=50, interval=self.ONE_HOUR, start=new_start)
        downloader_new = HistoricalDownloader(adapter_new, max_candles_per_request=1000)

        repo.ensure_data("BTCUSDT", "1h", new_start, new_start + 50 * self.ONE_HOUR, downloader_new)
        assert repo.count("BTCUSDT", "1h") == 50

        # Now request data BEFORE the existing range
        old_start = 1_609_459_200_000  # 2021-01-01
        adapter_old = FakeAdapter(n_candles=100, interval=self.ONE_HOUR, start=old_start)
        downloader_old = HistoricalDownloader(adapter_old, max_candles_per_request=1000)

        summary_old = repo.ensure_data(
            "BTCUSDT", "1h",
            start_ms=old_start,
            end_ms=old_start + 100 * self.ONE_HOUR,
            downloader=downloader_old,
        )

        # Should download from old_start (not from new_start - interval)
        assert adapter_old.first_request_start == old_start
        assert summary_old["inserted"] == 100
        assert repo.count("BTCUSDT", "1h") == 150  # 100 old + 50 new

    def test_candle_count_not_capped_at_3000(self, repo, db):
        """Regression: verify candle count is not artificially capped."""
        # Request more than 3000 candles
        n_candles = 3500
        adapter = FakeAdapter(n_candles=n_candles, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        summary = repo.ensure_data(
            "BTCUSDT", "1h",
            start_ms=self.START,
            end_ms=self.START + n_candles * self.ONE_HOUR,
            downloader=downloader,
        )

        # Should have downloaded ALL candles, not capped at 3000
        assert summary["n_candles"] == n_candles
        assert summary["inserted"] == n_candles
        assert repo.count("BTCUSDT", "1h") == n_candles

    def test_status_reports_correct_range(self, repo, db):
        """Regression: status command reports actual stored range."""
        # Store data for a specific range
        start_ms = 1_735_689_600_000  # 2025-01-01
        n_candles = 1400  # ~58 days of hourly data
        adapter = FakeAdapter(n_candles=n_candles, interval=self.ONE_HOUR, start=start_ms)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        repo.ensure_data("BTCUSDT", "1h", start_ms, start_ms + n_candles * self.ONE_HOUR, downloader)

        # Verify list_datasets returns the correct range
        datasets = repo.list_datasets()
        assert len(datasets) == 1

        ds = datasets[0]
        assert ds["symbol"] == "BTCUSDT"
        assert ds["timeframe"] == "1h"
        assert ds["n_candles"] == n_candles
        assert ds["start"] == start_ms
        assert ds["end"] == start_ms + (n_candles - 1) * self.ONE_HOUR

        # Verify range() method
        lo, hi = repo.range("BTCUSDT", "1h")
        assert lo == start_ms
        assert hi == start_ms + (n_candles - 1) * self.ONE_HOUR
