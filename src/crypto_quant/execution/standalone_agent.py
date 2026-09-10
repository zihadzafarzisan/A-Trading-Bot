"""Standalone monitoring & alerting agent.

Independent module that watches market conditions and system health, firing
alerts via log and optional webhook. Designed to run standalone or be imported
and used by worker.py or other orchestration code.

Monitors:
- Price anomalies: percentage change, standard deviation, volume spikes
- Risk limit breaches: drawdown, position concentration, margin usage
- System health: API connectivity, data feed status, broker status

Usage as script:
    python standalone_agent.py --config config.yaml

Usage as import:
    agent = StandaloneAgent(exchange_adapter=adapter, ...)
    agent.run_once()  # single check
    agent.run_loop()  # continuous monitoring
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from ..logging_config import get_logger
from ..utils.validators import validate_symbol, validate_timeframe

logger = get_logger("monitoring")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PriceAnomalyConfig:
    """Thresholds for price anomaly detection."""

    # Alert when price changes more than this percentage in `window_seconds`
    pct_change_threshold: float = 0.05  # 5%
    window_seconds: int = 60

    # Alert when price deviates more than N standard deviations from recent mean
    std_dev_threshold: float = 2.0
    std_dev_lookback: int = 100  # number of data points for rolling std

    # Alert when volume exceeds this multiple of recent average
    volume_spike_threshold: float = 3.0  # 3x average
    volume_lookback: int = 100


@dataclass
class RiskLimitConfig:
    """Configurable risk limits for monitoring."""

    # Maximum drawdown (fraction of peak equity)
    max_drawdown: float = 0.10  # 10%

    # Maximum single position as fraction of equity
    max_position_pct: float = 0.50  # 50%

    # Maximum margin utilization (used margin / free margin)
    max_margin_utilization: float = 0.80  # 80%

    # Maximum total exposure as fraction of equity
    max_total_exposure_pct: float = 1.0  # 100%


@dataclass
class HealthCheckConfig:
    """System health check settings."""

    # Timeout for API calls in seconds
    api_timeout: int = 5

    # How many consecutive failures before alerting
    failure_threshold: int = 3


@dataclass
class StandaloneAgentConfig:
    """Top-level configuration for the monitoring agent."""

    price_anomaly: PriceAnomalyConfig = field(default_factory=PriceAnomalyConfig)
    risk_limits: RiskLimitConfig = field(default_factory=RiskLimitConfig)
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)

    # Check interval in seconds
    check_interval: int = 60

    # Symbols to monitor
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])

    # Timeframe for price data
    timeframe: str = "1m"

    # Webhook URL (None = log only)
    webhook_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StandaloneAgentConfig":
        """Create config from a dictionary (e.g. loaded from YAML/JSON)."""
        price_cfg = PriceAnomalyConfig(**data.get("price_anomaly", {}))
        risk_cfg = RiskLimitConfig(**data.get("risk_limits", {}))
        health_cfg = HealthCheckConfig(**data.get("health_check", {}))
        return cls(
            price_anomaly=price_cfg,
            risk_limits=risk_cfg,
            health_check=health_cfg,
            check_interval=data.get("check_interval", 60),
            symbols=data.get("symbols", ["BTCUSDT"]),
            timeframe=data.get("timeframe", "1m"),
            webhook_url=data.get("webhook_url"),
        )

    @classmethod
    def from_env(cls) -> "StandaloneAgentConfig":
        """Create config with webhook URL from environment variable."""
        config = cls()
        env_url = os.environ.get("ALERT_WEBHOOK_URL")
        if env_url:
            config.webhook_url = env_url
        return config


# ---------------------------------------------------------------------------
# Alert Types
# ---------------------------------------------------------------------------
class AlertLevel:
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    """A monitoring alert."""

    level: str
    category: str  # price, risk, health
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Standalone Agent
# ---------------------------------------------------------------------------
class StandaloneAgent:
    """Monitoring & alerting agent.

    Watches market conditions and system health, firing alerts via log and
    optional webhook. Can run standalone or be imported by worker.py.
    """

    def __init__(
        self,
        exchange_adapter=None,
        price_source=None,
        broker=None,
        config: Optional[StandaloneAgentConfig] = None,
        webhook_url: Optional[str] = None,
    ):
        """Initialize the monitoring agent.

        Args:
            exchange_adapter: ExchangeAdapter for market data (public API).
            price_source: PriceSource for position-aware monitoring.
            broker: Broker instance for position/risk checks.
            config: Agent configuration.
            webhook_url: Override webhook URL (takes precedence over config).
        """
        self.config = config or StandaloneAgentConfig()
        self.exchange_adapter = exchange_adapter
        self.price_source = price_source
        self.broker = broker

        # Webhook URL: param > env > config
        self.webhook_url = (
            webhook_url
            or os.environ.get("ALERT_WEBHOOK_URL")
            or self.config.webhook_url
        )

        # Internal state
        self._price_history: Dict[str, List[float]] = {s: [] for s in self.config.symbols}
        self._volume_history: Dict[str, List[float]] = {s: [] for s in self.config.symbols}
        self._consecutive_failures: Dict[str, int] = {}
        self._alert_callbacks: List[Callable[[Alert], None]] = []

    # ------------------------------------------------------------------ public API
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Register a callback to be called on each alert."""
        self._alert_callbacks.append(callback)

    def run_once(self) -> List[Alert]:
        """Run a single monitoring check. Returns list of alerts fired."""
        alerts: List[Alert] = []

        # Price anomaly checks
        alerts.extend(self._check_price_anomalies())

        # Risk limit checks
        alerts.extend(self._check_risk_limits())

        # System health checks
        alerts.extend(self._check_system_health())

        # Fire all alerts
        for alert in alerts:
            self._fire_alert(alert)

        return alerts

    def run_loop(self) -> None:
        """Run continuous monitoring loop."""
        logger.info(
            "Starting monitoring loop (interval=%ds, symbols=%s)",
            self.config.check_interval,
            self.config.symbols,
        )
        try:
            while True:
                alerts = self.run_once()
                if alerts:
                    logger.info("Fired %d alerts this cycle", len(alerts))
                time.sleep(self.config.check_interval)
        except KeyboardInterrupt:
            logger.info("Monitoring loop stopped by user")

    # ------------------------------------------------------------------ price anomalies
    def _check_price_anomalies(self) -> List[Alert]:
        """Check for price anomalies across monitored symbols."""
        alerts: List[Alert] = []
        cfg = self.config.price_anomaly

        for symbol in self.config.symbols:
            try:
                current_price = self._get_price(symbol)
                if current_price is None:
                    continue

                # Update history
                history = self._price_history[symbol]
                history.append(current_price)
                if len(history) > max(cfg.std_dev_lookback, cfg.window_seconds * 10):
                    history.pop(0)

                # Check percentage change
                if len(history) >= 2:
                    prev_price = history[-2]
                    pct_change = abs(current_price - prev_price) / prev_price
                    if pct_change >= cfg.pct_change_threshold:
                        alerts.append(Alert(
                            level=AlertLevel.WARNING if pct_change < cfg.pct_change_threshold * 2 else AlertLevel.CRITICAL,
                            category="price",
                            message=f"{symbol} price moved {pct_change:.2%} (threshold: {cfg.pct_change_threshold:.2%})",
                            data={
                                "symbol": symbol,
                                "current_price": current_price,
                                "previous_price": prev_price,
                                "pct_change": pct_change,
                            },
                        ))

                # Check standard deviation deviation
                if len(history) >= cfg.std_dev_lookback:
                    recent = history[-cfg.std_dev_lookback:]
                    mean = sum(recent) / len(recent)
                    variance = sum((x - mean) ** 2 for x in recent) / len(recent)
                    std_dev = variance ** 0.5
                    if std_dev > 0:
                        z_score = abs(current_price - mean) / std_dev
                        if z_score >= cfg.std_dev_threshold:
                            alerts.append(Alert(
                                level=AlertLevel.WARNING,
                                category="price",
                                message=f"{symbol} price {z_score:.2f} std devs from mean (threshold: {cfg.std_dev_threshold})",
                                data={
                                    "symbol": symbol,
                                    "current_price": current_price,
                                    "mean": mean,
                                    "std_dev": std_dev,
                                    "z_score": z_score,
                                },
                            ))

                # Check volume spike
                current_volume = self._get_volume(symbol)
                if current_volume is not None:
                    vol_history = self._volume_history[symbol]
                    vol_history.append(current_volume)
                    if len(vol_history) > cfg.volume_lookback:
                        vol_history.pop(0)

                    if len(vol_history) >= cfg.volume_lookback:
                        avg_volume = sum(vol_history) / len(vol_history)
                        if avg_volume > 0 and current_volume >= avg_volume * cfg.volume_spike_threshold:
                            alerts.append(Alert(
                                level=AlertLevel.INFO,
                                category="price",
                                message=f"{symbol} volume spike: {current_volume:.0f} (threshold: {avg_volume * cfg.volume_spike_threshold:.0f})",
                                data={
                                    "symbol": symbol,
                                    "current_volume": current_volume,
                                    "average_volume": avg_volume,
                                    "spike_ratio": current_volume / avg_volume if avg_volume > 0 else 0,
                                },
                            ))

            except Exception as e:
                logger.error("Error checking price anomaly for %s: %s", symbol, e)

        return alerts

    # ------------------------------------------------------------------ risk limits
    def _check_risk_limits(self) -> List[Alert]:
        """Check for risk limit breaches."""
        alerts: List[Alert] = []
        cfg = self.config.risk_limits

        if self.broker is None:
            return alerts

        try:
            equity = self.broker.get_balance()
            positions = self.broker.get_positions()

            # Check drawdown
            peak_equity = getattr(self, "_peak_equity", equity)
            if equity > peak_equity:
                self._peak_equity = equity
                peak_equity = equity

            if peak_equity > 0:
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown >= cfg.max_drawdown:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        category="risk",
                        message=f"Drawdown {drawdown:.2%} exceeds limit {cfg.max_drawdown:.2%}",
                        data={
                            "drawdown": drawdown,
                            "max_drawdown": cfg.max_drawdown,
                            "peak_equity": peak_equity,
                            "current_equity": equity,
                        },
                    ))

            # Check position concentration
            for pos in positions:
                notional = abs(pos.get("notional", 0))
                if equity > 0:
                    position_pct = notional / equity
                    if position_pct >= cfg.max_position_pct:
                        alerts.append(Alert(
                            level=AlertLevel.WARNING,
                            category="risk",
                            message=f"Position {pos.get('symbol')} is {position_pct:.2%} of equity (limit: {cfg.max_position_pct:.2%})",
                            data={
                                "symbol": pos.get("symbol"),
                                "position_pct": position_pct,
                                "notional": notional,
                                "equity": equity,
                            },
                        ))

            # Check total exposure
            total_exposure = sum(abs(p.get("notional", 0)) for p in positions)
            if equity > 0:
                total_exposure_pct = total_exposure / equity
                if total_exposure_pct >= cfg.max_total_exposure_pct:
                    alerts.append(Alert(
                        level=AlertLevel.WARNING,
                        category="risk",
                        message=f"Total exposure {total_exposure_pct:.2%} exceeds limit {cfg.max_total_exposure_pct:.2%}",
                        data={
                            "total_exposure_pct": total_exposure_pct,
                            "total_exposure": total_exposure,
                            "equity": equity,
                        },
                    ))

        except Exception as e:
            logger.error("Error checking risk limits: %s", e)

        return alerts

    # ------------------------------------------------------------------ system health
    def _check_system_health(self) -> List[Alert]:
        """Check system health: API connectivity, data feed, broker status."""
        alerts: List[Alert] = []
        cfg = self.config.health_check

        # Check exchange adapter connectivity
        if self.exchange_adapter is not None:
            try:
                # Simple connectivity check - try to get a price
                test_symbol = self.config.symbols[0] if self.config.symbols else "BTCUSDT"
                self.exchange_adapter.get_ohlcv_as_dataframe(
                    test_symbol, self.config.timeframe, limit=1
                )
                self._consecutive_failures["exchange"] = 0
            except Exception as e:
                self._consecutive_failures["exchange"] = self._consecutive_failures.get("exchange", 0) + 1
                if self._consecutive_failures["exchange"] >= cfg.failure_threshold:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        category="health",
                        message=f"Exchange API unreachable ({self._consecutive_failures['exchange']} consecutive failures)",
                        data={"error": str(e)},
                    ))

        # Check price source
        if self.price_source is not None:
            try:
                test_symbol = self.config.symbols[0] if self.config.symbols else "BTCUSDT"
                self.price_source.get_price(test_symbol)
                self._consecutive_failures["price_source"] = 0
            except Exception as e:
                self._consecutive_failures["price_source"] = self._consecutive_failures.get("price_source", 0) + 1
                if self._consecutive_failures["price_source"] >= cfg.failure_threshold:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        category="health",
                        message=f"Price source unreachable ({self._consecutive_failures['price_source']} consecutive failures)",
                        data={"error": str(e)},
                    ))

        # Check broker connectivity
        if self.broker is not None:
            try:
                self.broker.get_balance()
                self._consecutive_failures["broker"] = 0
            except Exception as e:
                self._consecutive_failures["broker"] = self._consecutive_failures.get("broker", 0) + 1
                if self._consecutive_failures["broker"] >= cfg.failure_threshold:
                    alerts.append(Alert(
                        level=AlertLevel.CRITICAL,
                        category="health",
                        message=f"Broker unreachable ({self._consecutive_failures['broker']} consecutive failures)",
                        data={"error": str(e)},
                    ))

        return alerts

    # ------------------------------------------------------------------ helpers
    def _get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        # Try price source first (position-aware)
        if self.price_source is not None:
            try:
                return self.price_source.get_price(symbol)
            except Exception:
                pass

        # Fall back to exchange adapter (public API)
        if self.exchange_adapter is not None:
            try:
                df = self.exchange_adapter.get_ohlcv_as_dataframe(
                    symbol, self.config.timeframe, limit=1
                )
                if df is not None and not df.empty:
                    return float(df["close"].iloc[-1])
            except Exception:
                pass

        return None

    def _get_volume(self, symbol: str) -> Optional[float]:
        """Get current volume for a symbol."""
        if self.exchange_adapter is not None:
            try:
                df = self.exchange_adapter.get_ohlcv_as_dataframe(
                    symbol, self.config.timeframe, limit=1
                )
                if df is not None and not df.empty:
                    return float(df["volume"].iloc[-1])
            except Exception:
                pass
        return None

    def _fire_alert(self, alert: Alert) -> None:
        """Fire an alert via log, webhook, and callbacks."""
        # Log
        log_msg = f"[{alert.level.upper()}] [{alert.category}] {alert.message}"
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

        # Webhook
        if self.webhook_url:
            self._send_webhook(alert)

        # Callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("Alert callback error: %s", e)

    def _send_webhook(self, alert: Alert) -> None:
        """Send alert to webhook URL."""
        try:
            payload = json.dumps(alert.to_dict()).encode("utf-8")
            req = Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=5) as resp:
                logger.debug("Webhook sent (status=%d)", resp.status)
        except URLError as e:
            logger.error("Webhook failed: %s", e)
        except Exception as e:
            logger.error("Webhook error: %s", e)


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
def _load_config_from_file(path: str) -> StandaloneAgentConfig:
    """Load config from a JSON or YAML file."""
    import yaml

    with open(path, "r") as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    return StandaloneAgentConfig.from_dict(data)


def main():
    """Run standalone monitoring agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Standalone monitoring agent")
    parser.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="Symbols to monitor")
    parser.add_argument("--webhook", type=str, help="Webhook URL for alerts")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = _load_config_from_file(args.config)
    else:
        config = StandaloneAgentConfig.from_env()
        config.check_interval = args.interval
        config.symbols = args.symbols

    # Webhook override
    webhook_url = args.webhook or os.environ.get("ALERT_WEBHOOK_URL")

    # Create agent (no adapter/broker in standalone mode - health checks will warn)
    agent = StandaloneAgent(config=config, webhook_url=webhook_url)

    logger.info("Starting standalone monitoring agent")
    agent.run_loop()


if __name__ == "__main__":
    main()
