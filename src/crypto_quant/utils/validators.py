"""Validation utilities for the crypto quant terminal."""

from datetime import datetime, date
from typing import Any, List, Optional
import re


def validate_symbol(symbol: str) -> bool:
    """Validate trading symbol format (e.g., BTCUSDT)."""
    pattern = r'^[A-Z]{2,20}USDT$'
    return bool(re.match(pattern, symbol.upper()))


def validate_timeframe(timeframe: str) -> bool:
    """Validate timeframe format (e.g., 1h, 4h, 1d)."""
    valid_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}
    return timeframe in valid_timeframes


def validate_date(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
    """Validate date string format."""
    try:
        datetime.strptime(date_str, fmt)
        return True
    except ValueError:
        return False


def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 1.0) -> bool:
    """Validate percentage value is within range."""
    return min_val <= value <= max_val


def validate_positive(value: float) -> bool:
    """Validate value is positive."""
    return value > 0


def validate_non_negative(value: float) -> bool:
    """Validate value is non-negative."""
    return value >= 0


def validate_leverage(leverage: int, max_leverage: int = 5) -> bool:
    """Validate leverage is within allowed range."""
    return 1 <= leverage <= max_leverage


def validate_risk_per_trade(risk: float) -> bool:
    """Validate risk per trade percentage (0.01 = 1% to 0.1 = 10%)."""
    return 0.01 <= risk <= 0.1


def validate_capital(capital: float) -> bool:
    """Validate starting capital."""
    return capital >= 100  # Minimum $100


def validate_ohlcv(open: float, high: float, low: float, close: float, volume: float) -> bool:
    """Validate OHLCV data integrity."""
    if any(v < 0 for v in [open, high, low, close, volume]):
        return False
    if high < low:
        return False
    if high < open or high < close:
        return False
    if low > open or low > close:
        return False
    return True


def validate_trade_record(trade: dict) -> List[str]:
    """Validate a trade record, returning list of errors."""
    errors = []

    required_fields = [
        "symbol", "market_type", "direction", "timeframe",
        "entry_time", "entry_price", "quantity", "leverage"
    ]

    for field in required_fields:
        if field not in trade:
            errors.append(f"Missing required field: {field}")

    if "direction" in trade and trade["direction"] not in ["long", "short"]:
        errors.append(f"Invalid direction: {trade['direction']}")

    if "market_type" in trade and trade["market_type"] not in ["spot", "futures"]:
        errors.append(f"Invalid market type: {trade['market_type']}")

    if "entry_price" in trade and trade["entry_price"] <= 0:
        errors.append("Entry price must be positive")

    if "quantity" in trade and trade["quantity"] <= 0:
        errors.append("Quantity must be positive")

    return errors


def validate_experiment_config(config: dict) -> List[str]:
    """Validate experiment configuration."""
    errors = []

    if "strategy" not in config:
        errors.append("Missing strategy configuration")

    if "data" not in config:
        errors.append("Missing data configuration")

    if "risk" not in config:
        errors.append("Missing risk configuration")

    if "data" in config:
        data_config = config["data"]
        if "start_date" in data_config and not validate_date(data_config["start_date"]):
            errors.append(f"Invalid start date: {data_config['start_date']}")
        if "end_date" in data_config and not validate_date(data_config["end_date"]):
            errors.append(f"Invalid end date: {data_config['end_date']}")

    return errors
