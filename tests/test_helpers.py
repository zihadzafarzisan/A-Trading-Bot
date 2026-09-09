"""Tests for helper utilities."""

import pytest
from datetime import datetime, timezone

from crypto_quant.utils.helpers import (
    generate_experiment_id,
    generate_strategy_id,
    generate_trade_id,
    timestamp_to_datetime,
    datetime_to_timestamp,
    format_currency,
    format_percentage,
    format_large_number,
    calculate_returns,
    calculate_log_returns,
    safe_divide,
    clamp,
    chunk_list,
    flatten_dict,
)


class TestIDGeneration:
    """Test ID generation functions."""

    def test_generate_experiment_id(self):
        """Test experiment ID generation."""
        exp_id = generate_experiment_id()
        assert exp_id.startswith("EXP-")
        assert len(exp_id) > 10

    def test_generate_strategy_id(self):
        """Test strategy ID generation."""
        strat_id = generate_strategy_id("TrendFollowing", 1)
        assert strat_id.startswith("STRAT-")
        assert len(strat_id) > 10

    def test_generate_trade_id(self):
        """Test trade ID generation."""
        trade_id = generate_trade_id()
        assert trade_id.startswith("TRADE-")
        assert len(trade_id) > 10

    def test_unique_ids(self):
        """Test that generated IDs are unique."""
        ids = set()
        for _ in range(100):
            ids.add(generate_experiment_id())
        assert len(ids) == 100


class TestTimestampConversion:
    """Test timestamp conversion functions."""

    def test_timestamp_to_datetime(self):
        """Test converting timestamp to datetime."""
        timestamp = 1609459200000  # 2021-01-01 00:00:00 UTC
        dt = timestamp_to_datetime(timestamp)

        assert isinstance(dt, datetime)
        assert dt.year == 2021
        assert dt.month == 1
        assert dt.day == 1

    def test_datetime_to_timestamp(self):
        """Test converting datetime to timestamp."""
        dt = datetime(2021, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        timestamp = datetime_to_timestamp(dt)

        assert isinstance(timestamp, int)
        assert timestamp == 1609459200000

    def test_roundtrip_conversion(self):
        """Test roundtrip conversion preserves value."""
        original = 1609459200000
        dt = timestamp_to_datetime(original)
        result = datetime_to_timestamp(dt)

        assert result == original


class TestFormatting:
    """Test formatting functions."""

    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1000) == "$1,000.00"
        assert format_currency(1234567.89) == "$1,234,567.89"
        assert format_currency(0.5, decimals=1) == "$0.5"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.65) == "65.00%"
        assert format_percentage(0.12345, decimals=1) == "12.3%"

    def test_format_large_number(self):
        """Test large number formatting."""
        assert format_large_number(1500) == "1.50K"
        assert format_large_number(1500000) == "1.50M"
        assert format_large_number(1500000000) == "1.50B"
        assert format_large_number(500) == "500.00"


class TestCalculations:
    """Test calculation functions."""

    def test_calculate_returns(self):
        """Test return calculation."""
        prices = [100, 110, 105, 115]
        returns = calculate_returns(prices)

        assert len(returns) == 3
        assert returns[0] == pytest.approx(0.1)  # 10%
        assert returns[1] == pytest.approx(-0.045454, rel=1e-4)  # -4.55%
        assert returns[2] == pytest.approx(0.095238, rel=1e-4)  # 9.52%

    def test_calculate_log_returns(self):
        """Test log return calculation."""
        import math
        prices = [100, 110]
        returns = calculate_log_returns(prices)

        assert len(returns) == 1
        assert returns[0] == pytest.approx(math.log(1.1))

    def test_empty_prices(self):
        """Test calculation with empty prices."""
        assert calculate_returns([]) == []
        assert calculate_returns([100]) == []


class TestUtilityFunctions:
    """Test utility functions."""

    def test_safe_divide(self):
        """Test safe division."""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, default=1.0) == 1.0

    def test_clamp(self):
        """Test value clamping."""
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_chunk_list(self):
        """Test list chunking."""
        lst = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(lst, 3)

        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7]

    def test_flatten_dict(self):
        """Test dictionary flattening."""
        nested = {"a": {"b": 1, "c": 2}, "d": 3}
        flat = flatten_dict(nested)

        assert flat == {"a.b": 1, "a.c": 2, "d": 3}
