"""Tests for historical data downloader."""

import pytest
import pandas as pd

from crypto_quant.data.downloader import HistoricalDownloader, DownloadResult
from fake_adapter import FakeAdapter


class TestHistoricalDownloader:
    """Test HistoricalDownloader."""

    ONE_HOUR = 3_600_000
    START = 1_600_000_000_000

    def test_download_all(self):
        """Test downloading all candles across multiple batches."""
        adapter = FakeAdapter(n_candles=2500, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        result = downloader.download(
            "BTCUSDT", "1h",
            start_ms=self.START,
            end_ms=self.START + 2500 * self.ONE_HOUR,
        )
        assert result.n_candles == 2500
        assert adapter.calls >= 3  # paginated

    def test_download_single_batch(self):
        """Test download under one batch uses one request."""
        adapter = FakeAdapter(n_candles=500, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=1000)

        result = downloader.download("BTCUSDT", "1h", self.START, self.START + 500 * self.ONE_HOUR)
        assert result.n_candles == 500
        assert adapter.calls == 1

    def test_no_double_counting_across_batches(self):
        """Test the duplicated candle at batch boundary is not double counted."""
        adapter = FakeAdapter(n_candles=1000, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter, max_candles_per_request=500)

        result = downloader.download("BTCUSDT", "1h", self.START, self.START + 1000 * self.ONE_HOUR)
        assert result.n_candles == 1000

    def test_invalid_timeframe(self):
        """Test invalid timeframe raises."""
        adapter = FakeAdapter(10)
        downloader = HistoricalDownloader(adapter)
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            downloader.download("BTCUSDT", "2h", self.START, self.START + self.ONE_HOUR)

    def test_download_result_record(self):
        """Test DownloadResult fields."""
        adapter = FakeAdapter(10, interval=self.ONE_HOUR, start=self.START)
        downloader = HistoricalDownloader(adapter)
        result = downloader.download("BTCUSDT", "1h", self.START, self.START + 10 * self.ONE_HOUR)
        assert isinstance(result, DownloadResult)
        assert result.n_candles == 10
        assert result.symbol == "BTCUSDT"
        assert result.timeframe == "1h"
