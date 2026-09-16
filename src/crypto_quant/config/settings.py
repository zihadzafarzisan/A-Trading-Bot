"""Configuration models using Pydantic."""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml


class RateLimitConfig(BaseModel):
    """API rate limit configuration."""
    requests_per_second: int = 10
    requests_per_minute: int = 1200


class RetryConfig(BaseModel):
    """Retry configuration."""
    max_attempts: int = 5
    backoff_factor: float = 2.0


class DataConfig(BaseModel):
    """Data engine configuration."""
    start_date: str = "2020-01-01"
    end_date: str = "2026-09-08"
    base_url: str = "https://api.binance.com"
    rate_limit: RateLimitConfig = RateLimitConfig()
    retry: RetryConfig = RetryConfig()


class UniverseConfig(BaseModel):
    """Asset universe configuration."""
    mode: str = "historical_top_n"
    top_n: int = 20
    fallback: str = "manual_list"


class TimeframeConfig(BaseModel):
    """Timeframe configuration."""
    primary: List[str] = Field(default_factory=lambda: ["1h", "4h", "1d"])
    available: List[str] = Field(default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])


class ResearchConfig(BaseModel):
    """Research period configuration."""
    start_date: str = "2020-01-01"
    end_date: str = "2026-09-08"
    train_end: str = "2024-12-31"
    validation_end: str = "2025-12-31"
    test_start: str = "2026-01-01"


class RiskConfig(BaseModel):
    """Risk management configuration."""
    starting_capital: float = 1000.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 3
    max_leverage: int = 5
    max_position_pct: float = 0.5
    daily_loss_limit: float = 100.0
    weekly_loss_limit: float = 500.0
    max_drawdown_pct: float = 0.25


class TradingConfig(BaseModel):
    """Trading mode configuration."""
    mode: str = "paper"  # signal_only, paper, live
    dry_run: bool = True


class FeesConfig(BaseModel):
    """Fee structure configuration."""
    spot_maker: float = 0.001
    spot_taker: float = 0.001
    futures_maker: float = 0.0002
    futures_taker: float = 0.0004
    slippage: float = 0.0005
    funding_rate: float = 0.0001


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    initial_capital: float = 1000.0
    commission: float = 0.001
    slippage: float = 0.0005
    execution_delay_bars: int = 1


class OptimizationConfig(BaseModel):
    """Strategy optimization configuration."""
    search_type: str = "grid"  # grid, random, bayesian
    max_combinations: int = 1000
    objective: str = "max_win_rate"
    min_trades: int = 100


class MLConfig(BaseModel):
    """Machine learning configuration."""
    models: List[str] = Field(default_factory=lambda: [
        "logistic_regression", "random_forest", "xgboost"
    ])
    test_size: float = 0.2
    validation_size: float = 0.15
    features: List[str] = Field(default_factory=lambda: [
        "rsi", "ema_distance", "atr_ratio", "volume_ratio",
        "volatility", "momentum", "market_regime", "hour",
        "day_of_week", "timeframe"
    ])


class MarketConfig(BaseModel):
    """Market configuration."""
    exchange: str = "binance"
    spot: bool = True
    futures: bool = True
    default_market: str = "spot"


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    output_dir: str = "./dashboard"
    auto_open: bool = False
    theme: str = "dark"


class FileHandlerConfig(BaseModel):
    """File handler logging configuration."""
    application: str = "logs/application.log"
    data: str = "logs/data.log"
    research: str = "logs/research.log"
    backtest: str = "logs/backtest.log"
    trading: str = "logs/trading.log"
    risk: str = "logs/risk.log"
    error: str = "logs/error.log"


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handlers: List[str] = Field(default_factory=lambda: ["console", "file"])
    file_handlers: FileHandlerConfig = FileHandlerConfig()


class AppConfig(BaseSettings):
    """Main application configuration."""
    market: MarketConfig = MarketConfig()
    data: DataConfig = DataConfig()
    universe: UniverseConfig = UniverseConfig()
    timeframes: TimeframeConfig = TimeframeConfig()
    research: ResearchConfig = ResearchConfig()
    risk: RiskConfig = RiskConfig()
    trading: TradingConfig = TradingConfig()
    fees: FeesConfig = FeesConfig()
    backtesting: BacktestConfig = BacktestConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    ml: MLConfig = MLConfig()
    dashboard: DashboardConfig = DashboardConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = {"env_prefix": "", "env_file": ".env", "extra": "ignore"}


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("config/settings.yaml")
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}
        return AppConfig(**config_data)
    else:
        return AppConfig()


def save_config(config: AppConfig, config_path: str | Path) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


# Global config instance
_config: AppConfig | None = None


def get_config(config_path: str | Path | None = None) -> AppConfig:
    """Get or create configuration singleton."""
    global _config
    if _config is None or config_path:
        _config = load_config(config_path)
    return _config


def reset_config() -> None:
    """Reset configuration singleton (for testing)."""
    global _config
    _config = None
