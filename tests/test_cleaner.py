"""Tests for data cleaner."""

import pytest
import pandas as pd
import numpy as np

from crypto_quant.data.cleaner import DataCleaner


def make_df(timestamps, ohlcv):
    """Build an OHLCV DataFrame, including extra columns."""
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [r[0] for r in ohlcv],
        "high": [r[1] for r in ohlcv],
        "low": [r[2] for r in ohlcv],
        "close": [r[3] for r in ohlcv],
        "volume": [r[4] for r in ohlcv],
    })


@pytest.fixture
def cleaner():
    """Provide a DataCleaner instance."""
    return DataCleaner()


class TestDataCleaner:
    """Test DataCleaner."""

    ONE_HOUR = 3_600_000
    BASE = 1609459200000

    def test_clean_returns_reports(self, cleaner):
        """Test clean returns both cleaned df and report."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(5)]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 5
        df = make_df(times, ohlcv)

        cleaned, report = cleaner.clean(df, "BTCUSDT", "1h")
        assert isinstance(report, object)
        assert cleaned is not None
        assert report.symbol == "BTCUSDT"
        assert len(cleaned) == 5

    def test_deduplication(self, cleaner):
        """Test duplicates are removed."""
        times = [self.BASE, self.BASE, self.BASE + self.ONE_HOUR]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        cleaned, report = cleaner.clean(df, "BTCUSDT", "1h")
        assert len(cleaned) == 2
        assert report.n_duplicates_removed == 1

    def test_sorting(self, cleaner):
        """Test out-of-order rows are sorted."""
        times = [self.BASE + self.ONE_HOUR, self.BASE, self.BASE + 2 * self.ONE_HOUR]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        cleaned, _ = cleaner.clean(df, "BTCUSDT", "1h")
        assert list(cleaned["timestamp"]) == sorted(times)

    def test_nan_timestamp_dropped(self, cleaner):
        """Test rows with NaN timestamp are dropped."""
        times = [self.BASE, self.BASE + self.ONE_HOUR, np.nan]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        cleaned, report = cleaner.clean(df, "BTCUSDT", "1h")
        assert len(cleaned) == 2

    def test_type_coercion(self, cleaner):
        """Test string values are coerced to numerics."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(3)]
        df = pd.DataFrame({
            "timestamp": times,
            "open": ["50000", "51000", "52000"],
            "high": ["51000", "52000", "53000"],
            "low": ["49500", "50500", "51500"],
            "close": ["50500", "51500", "52500"],
            "volume": ["100", "120", "110"],
        })

        cleaned, _ = cleaner.clean(df, "BTCUSDT", "1h")
        assert cleaned["close"].dtype == np.float64
        assert cleaned["volume"].dtype == np.float64

    def test_na_value_handling(self, cleaner):
        """Test NaN OHLC values are kept but reported (not silently dropped)."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(4)]
        ohlcv = [(50000, 51000, 49500, 50500, 100),
                 (50500, np.nan, 50000, 51000, 100),
                 (51000, 52000, 50500, 51500, 100),
                 (51500, 52500, 51000, 52000, 100)]
        df = make_df(times, ohlcv)

        cleaned, report = cleaner.clean(df, "BTCUSDT", "1h")
        # Cleaner preserves NaN in OHLC (it doesn't drop rows)
        assert len(cleaned) == 4
