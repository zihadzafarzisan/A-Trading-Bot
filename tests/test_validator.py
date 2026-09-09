"""Tests for market data validator."""

import pytest
import pandas as pd

from crypto_quant.data.validator import DataValidator, ValidationReport


def make_df(timestamps, ohlcv):
    """Build a valid OHLCV DataFrame."""
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": [r[0] for r in ohlcv],
        "high": [r[1] for r in ohlcv],
        "low": [r[2] for r in ohlcv],
        "close": [r[3] for r in ohlcv],
        "volume": [r[4] for r in ohlcv],
    })


@pytest.fixture
def validator():
    """Provide a DataValidator instance."""
    return DataValidator()


class TestDataValidator:
    """Test DataValidator."""

    # 1h interval klines for 12 consecutive hours
    ONE_HOUR = 3_600_000
    BASE = 1609459200000

    def test_empty_df(self, validator):
        """Test empty DataFrame."""
        df = pd.DataFrame()
        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert report.error_count == 1
        assert any(i.issue_type == "empty_dataset" for i in report.issues)

    def test_clean_data(self, validator):
        """Test clean data has no errors."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(10)]
        ohlcv = [(50000, 51000, 49500, 50500, 100) for _ in range(10)]
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert report.is_clean
        assert report.error_count == 0

    def test_duplicate_detection(self, validator):
        """Test duplicate timestamps are detected."""
        times = [self.BASE, self.BASE, self.BASE + self.ONE_HOUR]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "duplicate" for i in report.issues)

    def test_gap_detection(self, validator):
        """Test gaps are detected."""
        times = [
            self.BASE,
            self.BASE + self.ONE_HOUR,
            self.BASE + self.ONE_HOUR * 5,  # 4h gap
        ]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "gap" for i in report.issues)

    def test_out_of_order(self, validator):
        """Test out-of-order data is detected and sorted."""
        times = [
            self.BASE + self.ONE_HOUR,
            self.BASE,  # out of order
            self.BASE + self.ONE_HOUR * 2,
        ]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 3
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "out_of_order" for i in report.issues)

    def test_negative_values(self, validator):
        """Test negative prices are detected."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(3)]
        ohlcv = [(50000, 51000, 49500, 50500, 100),
                 (50500, 51500, -100, 51000, 100),  # negative low
                 (51000, 52000, 50500, 51500, 100)]
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "invalid_value" for i in report.issues)

    def test_high_low_inconsistency(self, validator):
        """Test high < low is detected."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(3)]
        ohlcv = [(50000, 51000, 49500, 50500, 100),
                 (50500, 49000, 51000, 51000, 100),  # high(49000) < low(51000)
                 (51000, 52000, 50500, 51500, 100)]
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "invalid_value" for i in report.issues)

    def test_missing_columns(self, validator):
        """Test missing columns are detected."""
        df = pd.DataFrame({"timestamp": [1, 2, 3]})  # missing OHLCV
        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any(i.issue_type == "missing_columns" for i in report.issues)

    def test_non_positive_volume(self, validator):
        """Test zero/negative volume is detected."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(3)]
        ohlcv = [(50000, 51000, 49500, 50500, 100),
                 (50500, 51500, 50000, 51000, 0),  # zero volume
                 (51000, 52000, 50500, 51500, 100)]
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        assert any("volume" in i.message for i in report.issues)

    def test_report_summary(self, validator):
        """Test report summary method."""
        times = [self.BASE + i * self.ONE_HOUR for i in range(10)]
        ohlcv = [(50000, 51000, 49500, 50500, 100)] * 10
        df = make_df(times, ohlcv)

        report = validator.validate_dataframe(df, "BTCUSDT", "1h")
        summary = report.summary()
        assert summary["symbol"] == "BTCUSDT"
        assert summary["timeframe"] == "1h"
        assert summary["n_candles"] == 10
        assert summary["clean"] is True
