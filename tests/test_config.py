"""Tests for configuration system."""

import pytest
from pathlib import Path

from crypto_quant.config.settings import (
    AppConfig,
    MarketConfig,
    RiskConfig,
    DataConfig,
    load_config,
    save_config,
    get_config,
    reset_config,
)


class TestAppConfig:
    """Test AppConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AppConfig()

        assert config.market.exchange == "binance"
        assert config.market.spot is True
        assert config.market.futures is True
        assert config.risk.starting_capital == 1000.0
        assert config.risk.risk_per_trade == 0.01
        assert config.risk.max_open_positions == 3
        assert config.risk.max_leverage == 5
        assert config.trading.mode == "paper"
        assert config.trading.dry_run is True

    def test_market_config(self):
        """Test market configuration."""
        config = MarketConfig()

        assert config.exchange == "binance"
        assert config.spot is True
        assert config.futures is True
        assert config.default_market == "spot"

    def test_risk_config(self):
        """Test risk configuration."""
        config = RiskConfig()

        assert config.starting_capital == 1000.0
        assert config.risk_per_trade == 0.01
        assert config.max_open_positions == 3
        assert config.max_leverage == 5
        assert config.max_position_pct == 0.5
        assert config.daily_loss_limit == 100.0
        assert config.weekly_loss_limit == 500.0
        assert config.max_drawdown_pct == 0.25

    def test_data_config(self):
        """Test data configuration."""
        config = DataConfig()

        assert config.start_date == "2020-01-01"
        assert config.end_date == "2026-09-08"
        assert config.base_url == "https://api.binance.com"


class TestConfigLoading:
    """Test configuration loading and saving."""

    def test_load_default_config(self):
        """Test loading default configuration."""
        reset_config()
        config = get_config()

        assert isinstance(config, AppConfig)
        assert config.market.exchange == "binance"

    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        config = AppConfig()
        config_path = temp_dir / "test_config.yaml"

        save_config(config, config_path)
        loaded_config = load_config(config_path)

        assert loaded_config.market.exchange == config.market.exchange
        assert loaded_config.risk.starting_capital == config.risk.starting_capital

    def test_config_singleton(self):
        """Test config singleton pattern."""
        reset_config()
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_config_reset(self):
        """Test config reset."""
        reset_config()
        config1 = get_config()

        reset_config()
        config2 = get_config()

        assert config1 is not config2
