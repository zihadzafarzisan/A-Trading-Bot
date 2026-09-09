"""Helper utilities for the crypto quant terminal."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def generate_experiment_id() -> str:
    """Generate unique experiment ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_input = f"{timestamp}{time.time_ns()}"
    hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    return f"EXP-{timestamp}-{hash_suffix.upper()}"


def generate_strategy_id(name: str, version: int = 1) -> str:
    """Generate unique strategy ID."""
    hash_input = f"{name}_{version}"
    hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:6]
    return f"STRAT-{hash_suffix.upper()}"


def generate_trade_id() -> str:
    """Generate unique trade ID."""
    timestamp = int(time.time() * 1000)
    hash_suffix = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
    return f"TRADE-{hash_suffix.upper()}"


def timestamp_to_datetime(timestamp_ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency string."""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a fraction (0..1) as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K"
    else:
        return f"{value:.2f}"


def calculate_returns(prices: List[float]) -> List[float]:
    """Calculate returns from price list."""
    if len(prices) < 2:
        return []
    return [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]


def calculate_log_returns(prices: List[float]) -> List[float]:
    """Calculate log returns from price list."""
    import math
    if len(prices) < 2:
        return []
    return [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]


def ensure_directory(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(file_path: str | Path) -> Dict[str, Any]:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], file_path: str | Path) -> None:
    """Save data to YAML file."""
    ensure_directory(Path(file_path).parent)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(file_path: str | Path) -> Any:
    """Load JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Save data to JSON file."""
    ensure_directory(Path(file_path).parent)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value."""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))
