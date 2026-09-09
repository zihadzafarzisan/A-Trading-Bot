"""Logging configuration for the crypto quant terminal."""

import logging
from pathlib import Path
from typing import Dict
from rich.console import Console
from rich.logging import RichHandler


class LogManager:
    """Manages application logging."""

    def __init__(self, config: dict | None = None):
        """Initialize log manager."""
        self.console = Console()
        self.loggers: Dict[str, logging.Logger] = {}

        # Default config
        self.config = config or {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "handlers": ["console", "file"],
            "file_handlers": {
                "application": "logs/application.log",
                "data": "logs/data.log",
                "research": "logs/research.log",
                "backtest": "logs/backtest.log",
                "trading": "logs/trading.log",
                "risk": "logs/risk.log",
                "error": "logs/error.log",
            }
        }

        # Create log directory
        Path("logs").mkdir(exist_ok=True)

    def get_logger(self, name: str, level: str | None = None) -> logging.Logger:
        """Get or create a logger."""
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level or self.config["level"]))

        # Rich console handler
        if "console" in self.config["handlers"]:
            rich_handler = RichHandler(
                console=self.console,
                rich_tracebacks=True,
                show_path=False,
            )
            rich_handler.setLevel(getattr(logging, level or self.config["level"]))
            logger.addHandler(rich_handler)

        # File handler
        if "file" in self.config["handlers"] and name in self.config["file_handlers"]:
            file_path = self.config["file_handlers"][name]
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(getattr(logging, level or self.config["level"]))
            formatter = logging.Formatter(self.config["format"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.loggers[name] = logger
        return logger

    def set_level(self, level: str) -> None:
        """Set logging level for all loggers."""
        for logger in self.loggers.values():
            logger.setLevel(getattr(logging, level))

    def log_trade(self, logger_name: str, trade_data: dict) -> None:
        """Log trade event with structured data."""
        logger = self.get_logger(logger_name)
        logger.info(
            f"TRADE: {trade_data.get('action', 'unknown')} "
            f"{trade_data.get('symbol', 'unknown')} "
            f"@ {trade_data.get('price', 0)} "
            f"qty={trade_data.get('quantity', 0)} "
            f"reason={trade_data.get('reason', 'none')}"
        )

    def log_risk_event(self, event_type: str, severity: str, message: str, context: dict | None = None) -> None:
        """Log risk management event."""
        logger = self.get_logger("risk")
        log_method = logger.critical if severity == "critical" else logger.warning
        log_method(f"[{event_type.upper()}] {message} | Context: {context}")


# Global log manager instance
_log_manager: LogManager | None = None


def get_log_manager(config: dict | None = None) -> LogManager:
    """Get or create log manager singleton."""
    global _log_manager
    if _log_manager is None:
        _log_manager = LogManager(config)
    return _log_manager


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Get a logger (convenience function)."""
    return get_log_manager().get_logger(name, level)


def setup_logging(level: str = "INFO") -> LogManager:
    """Setup logging with default configuration."""
    config = {
        "level": level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "handlers": ["console", "file"],
        "file_handlers": {
            "application": "logs/application.log",
            "data": "logs/data.log",
            "research": "logs/research.log",
            "backtest": "logs/backtest.log",
            "trading": "logs/trading.log",
            "risk": "logs/risk.log",
            "error": "logs/error.log",
        }
    }
    return get_log_manager(config)
