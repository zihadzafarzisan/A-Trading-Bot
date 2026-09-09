"""Tests for Binance exchange adapter."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from crypto_quant.exchange.binance import BinanceAdapter, TIMEFRAME_MAP


class TestBinanceAdapter:
    """Test Binance adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create a test adapter instance."""
        return BinanceAdapter(market_type="spot", max_retries=2, backoff_factor=0.1)

    def test_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter.market_type == "spot"
        assert adapter.exchange_name == "binance"
        assert adapter.base_url == "https://api.binance.com"

    def test_futures_url(self):
        """Test futures adapter uses correct URL."""
        adapter = BinanceAdapter(market_type="futures")
        assert adapter.base_url == "https://fapi.binance.com"

    def test_invalid_market_type(self):
        """Test invalid market type raises error."""
        with pytest.raises(ValueError, match="Unsupported market_type"):
            BinanceAdapter(market_type="invalid")

    @patch('requests.Session.get')
    def test_get_server_time(self, mock_get, adapter):
        """Test fetching server time."""
        mock_get.return_value = Mock(status_code=200, json=lambda: {"serverTime": 1609459200000})
        result = adapter.get_server_time()
        assert result == 1609459200000
        assert mock_get.called

    @patch('requests.Session.get')
    def test_get_symbols(self, mock_get, adapter):
        """Test fetching trading symbols."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {
                "symbols": [
                    {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "ETHUSDT", "status": "TRADING", "quoteAsset": "USDT"},
                    {"symbol": "BTCBUSD", "status": "TRADING", "quoteAsset": "BUSD"},  # Not USDT
                    {"symbol": "XYZUSDT", "status": "HALT", "quoteAsset": "USDT"},  # Not trading
                ]
            }
        )
        symbols = adapter.get_symbols()
        assert "BTCUSDT" in symbols
        assert "ETHUSDT" in symbols
        assert "BTCBUSD" not in symbols
        assert "XYZUSDT" not in symbols

    @patch('requests.Session.get')
    def test_get_klines(self, mock_get, adapter):
        """Test fetching klines."""
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: [
                [1609459200000, "50000", "51000", "49500", "50500", "100", 1609462800000,
                 "5000000", 1000, "50", "2500000", "0"],
            ]
        )
        result = adapter.get_klines("BTCUSDT", "1h", limit=1)
        assert len(result) == 1
        assert result[0][0] == 1609459200000
        assert result[0][1] == "50000"

    @patch('requests.Session.get')
    def test_rate_limit_retry(self, mock_get, adapter):
        """Test 429 rate limit triggers retry."""
        mock_get.side_effect = [
            Mock(status_code=429, headers={}),
            Mock(status_code=200, json=lambda: {"serverTime": 1609459200000}),
        ]
        result = adapter.get_server_time()
        assert result == 1609459200000
        assert mock_get.call_count == 2

    @patch('requests.Session.get')
    def test_exhausted_retries(self, mock_get, adapter):
        """Test exhausted retries raises error."""
        mock_get.return_value = Mock(status_code=500)
        with pytest.raises(RuntimeError, match="Exhausted retries"):
            adapter.get_server_time()

    def test_get_available_timeframes(self, adapter):
        """Test available timeframes."""
        tfs = adapter.get_available_timeframes()
        assert "1h" in tfs
        assert "4h" in tfs
        assert "1d" in tfs

    def test_close(self, adapter):
        """Test closing the adapter."""
        adapter.close()  # Should not raise
