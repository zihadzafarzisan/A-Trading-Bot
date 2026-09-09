"""Tests for validation utilities."""

import pytest

from crypto_quant.utils.validators import (
    validate_symbol,
    validate_timeframe,
    validate_date,
    validate_percentage,
    validate_positive,
    validate_non_negative,
    validate_leverage,
    validate_risk_per_trade,
    validate_capital,
    validate_ohlcv,
    validate_trade_record,
    validate_experiment_config,
)


class TestSymbolValidation:
    """Test symbol validation."""

    def test_valid_symbols(self):
        """Test valid symbol formats."""
        assert validate_symbol("BTCUSDT") is True
        assert validate_symbol("ETHUSDT") is True
        assert validate_symbol("BNBUSDT") is True
        assert validate_symbol("SOLUSDT") is True

    def test_invalid_symbols(self):
        """Test invalid symbol formats."""
        assert validate_symbol("BTC") is False
        assert validate_symbol("USDT") is False
        assert validate_symbol("123USDT") is False
        assert validate_symbol("") is False

    def test_case_insensitive(self):
        """Test case insensitive validation."""
        assert validate_symbol("btcusdt") is True


class TestTimeframeValidation:
    """Test timeframe validation."""

    def test_valid_timeframes(self):
        """Test valid timeframe formats."""
        assert validate_timeframe("1m") is True
        assert validate_timeframe("5m") is True
        assert validate_timeframe("15m") is True
        assert validate_timeframe("30m") is True
        assert validate_timeframe("1h") is True
        assert validate_timeframe("4h") is True
        assert validate_timeframe("1d") is True

    def test_invalid_timeframes(self):
        """Test invalid timeframe formats."""
        assert validate_timeframe("2h") is False
        assert validate_timeframe("1w") is False
        assert validate_timeframe("1M") is False
        assert validate_timeframe("") is False


class TestDateValidation:
    """Test date validation."""

    def test_valid_dates(self):
        """Test valid date formats."""
        assert validate_date("2020-01-01") is True
        assert validate_date("2026-09-08") is True
        assert validate_date("2024-12-31") is True

    def test_invalid_dates(self):
        """Test invalid date formats."""
        assert validate_date("01-01-2020") is False
        assert validate_date("2020/01/01") is False
        assert validate_date("invalid") is False
        assert validate_date("") is False


class TestPercentageValidation:
    """Test percentage validation."""

    def test_valid_percentages(self):
        """Test valid percentage values."""
        assert validate_percentage(0.0) is True
        assert validate_percentage(0.5) is True
        assert validate_percentage(1.0) is True
        assert validate_percentage(0.5, min_val=0.0, max_val=1.0) is True

    def test_invalid_percentages(self):
        """Test invalid percentage values."""
        assert validate_percentage(-0.1) is False
        assert validate_percentage(1.1) is False
        assert validate_percentage(0.5, min_val=0.6, max_val=1.0) is False


class TestRiskValidation:
    """Test risk-related validations."""

    def test_valid_leverage(self):
        """Test valid leverage values."""
        assert validate_leverage(1) is True
        assert validate_leverage(3) is True
        assert validate_leverage(5) is True

    def test_invalid_leverage(self):
        """Test invalid leverage values."""
        assert validate_leverage(0) is False
        assert validate_leverage(6) is False
        assert validate_leverage(10) is False

    def test_valid_risk_per_trade(self):
        """Test valid risk per trade values."""
        assert validate_risk_per_trade(0.01) is True
        assert validate_risk_per_trade(0.05) is True
        assert validate_risk_per_trade(0.1) is True

    def test_invalid_risk_per_trade(self):
        """Test invalid risk per trade values."""
        assert validate_risk_per_trade(0.001) is False
        assert validate_risk_per_trade(0.11) is False
        assert validate_risk_per_trade(0.0) is False

    def test_valid_capital(self):
        """Test valid capital values."""
        assert validate_capital(100) is True
        assert validate_capital(1000) is True
        assert validate_capital(10000) is True

    def test_invalid_capital(self):
        """Test invalid capital values."""
        assert validate_capital(0) is False
        assert validate_capital(50) is False
        assert validate_capital(-100) is False


class TestOHLCVValidation:
    """Test OHLCV data validation."""

    def test_valid_ohlcv(self):
        """Test valid OHLCV data."""
        assert validate_ohlcv(100, 105, 95, 103, 1000000) is True
        assert validate_ohlcv(50, 60, 40, 55, 500000) is True

    def test_invalid_ohlcv(self):
        """Test invalid OHLCV data."""
        # Negative values
        assert validate_ohlcv(-100, 105, 95, 103, 1000000) is False

        # High < Low
        assert validate_ohlcv(100, 90, 95, 103, 1000000) is False

        # High < Open
        assert validate_ohlcv(110, 105, 95, 103, 1000000) is False

        # Low > Close
        assert validate_ohlcv(100, 105, 110, 103, 1000000) is False


class TestTradeRecordValidation:
    """Test trade record validation."""

    def test_valid_trade_record(self):
        """Test valid trade record."""
        trade = {
            "symbol": "BTCUSDT",
            "market_type": "spot",
            "direction": "long",
            "timeframe": "1h",
            "entry_time": 1609459200000,
            "entry_price": 50000.0,
            "quantity": 0.1,
            "leverage": 1,
        }
        errors = validate_trade_record(trade)
        assert len(errors) == 0

    def test_invalid_trade_record(self):
        """Test invalid trade record."""
        trade = {
            "symbol": "BTCUSDT",
            "market_type": "invalid",
            "direction": "invalid",
            "timeframe": "1h",
            "entry_time": 1609459200000,
            "entry_price": -50000.0,
            "quantity": -0.1,
            "leverage": 1,
        }
        errors = validate_trade_record(trade)
        assert len(errors) > 0

    def test_missing_fields(self):
        """Test missing required fields."""
        trade = {"symbol": "BTCUSDT"}
        errors = validate_trade_record(trade)
        assert len(errors) > 0
        assert any("Missing required field" in e for e in errors)


class TestExperimentConfigValidation:
    """Test experiment configuration validation."""

    def test_valid_config(self):
        """Test valid experiment configuration."""
        config = {
            "strategy": {"type": "trend"},
            "data": {"start_date": "2020-01-01", "end_date": "2026-09-08"},
            "risk": {"starting_capital": 1000},
        }
        errors = validate_experiment_config(config)
        assert len(errors) == 0

    def test_missing_sections(self):
        """Test missing configuration sections."""
        config = {"strategy": {"type": "trend"}}
        errors = validate_experiment_config(config)
        assert len(errors) > 0
        assert any("Missing data configuration" in e for e in errors)

    def test_invalid_dates(self):
        """Test invalid dates in configuration."""
        config = {
            "strategy": {"type": "trend"},
            "data": {"start_date": "invalid", "end_date": "2026-09-08"},
            "risk": {"starting_capital": 1000},
        }
        errors = validate_experiment_config(config)
        assert any("Invalid start date" in e for e in errors)
