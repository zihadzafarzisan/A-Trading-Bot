"""Database models using SQLAlchemy."""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Boolean,
    ForeignKey,
    CheckConstraint,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


def utc_now() -> datetime:
    """Return timezone-aware UTC now (SQLite-compatible)."""
    return datetime.now(timezone.utc)


class MarketData(Base):
    """OHLCV market data."""
    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(Integer, nullable=False)  # Unix timestamp in ms
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    market_type = Column(String(10), nullable=False, default="spot")  # spot/futures
    created_at = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", "market_type", name="uq_market_data"),
        Index("ix_market_data_symbol_timeframe", "symbol", "timeframe"),
        Index("ix_market_data_timestamp", "timestamp"),
    )


class AssetUniverse(Base):
    """Historical asset universe tracking."""
    __tablename__ = "asset_universe"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    date = Column(String(10), nullable=False)  # YYYY-MM-DD
    rank = Column(Integer, nullable=False)
    market_cap = Column(Float)
    source = Column(String(50))  # coingecko, manual, etc.
    created_at = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_asset_universe"),
        Index("ix_asset_universe_date", "date"),
    )


class Feature(Base):
    """Calculated technical features."""
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(Integer, nullable=False)
    feature_name = Column(String(100), nullable=False)
    feature_value = Column(Float)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", "feature_name", name="uq_features"),
        Index("ix_features_symbol_timeframe", "symbol", "timeframe"),
    )


class Strategy(Base):
    """Strategy definitions."""
    __tablename__ = "strategies"

    id = Column(String(50), primary_key=True)  # STRAT-XXX
    name = Column(String(100), nullable=False)
    type = Column(String(50), nullable=False)  # trend, momentum, etc.
    description = Column(Text)
    parameters = Column(Text, nullable=False)  # JSON
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)


class Experiment(Base):
    """Research experiment tracking."""
    __tablename__ = "experiments"

    id = Column(String(50), primary_key=True)  # EXP-YYYY-XXXXXX
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False)
    config = Column(Text, nullable=False)  # JSON: universe, timeframe, risk, etc.
    start_date = Column(String(10), nullable=False)
    end_date = Column(String(10), nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending/running/completed/failed
    created_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime)

    strategy = relationship("Strategy", backref="experiments")
    trades = relationship("Trade", backref="experiment")
    results = relationship("ExperimentResult", backref="experiment", uselist=False)


class Trade(Base):
    """Individual trade records."""
    __tablename__ = "trades"

    id = Column(String(50), primary_key=True)  # TRADE-XXX
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    market_type = Column(String(10), nullable=False)  # spot/futures
    direction = Column(String(10), nullable=False)  # long/short
    timeframe = Column(String(10), nullable=False)
    entry_time = Column(Integer, nullable=False)  # Unix timestamp
    exit_time = Column(Integer)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False, default=1)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    fees = Column(Float, default=0)
    slippage = Column(Float, default=0)
    funding_cost = Column(Float, default=0)
    gross_pnl = Column(Float)
    net_pnl = Column(Float)
    return_pct = Column(Float)
    holding_period = Column(Integer)  # minutes
    exit_reason = Column(String(50))  # tp, sl, signal, time
    ml_probability = Column(Float)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    strategy = relationship("Strategy", backref="trades")

    __table_args__ = (
        Index("ix_trades_experiment", "experiment_id"),
        Index("ix_trades_symbol", "symbol"),
        Index("ix_trades_entry_time", "entry_time"),
    )


class ExperimentResult(Base):
    """Aggregated experiment results."""
    __tablename__ = "experiment_results"

    experiment_id = Column(String(50), ForeignKey("experiments.id"), primary_key=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    expectancy = Column(Float, default=0)
    net_return = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    sortino_ratio = Column(Float, default=0)
    avg_win = Column(Float, default=0)
    avg_loss = Column(Float, default=0)
    largest_win = Column(Float, default=0)
    largest_loss = Column(Float, default=0)
    avg_holding_period = Column(Float, default=0)
    total_fees = Column(Float, default=0)
    total_slippage = Column(Float, default=0)
    total_funding = Column(Float, default=0)
    results_json = Column(Text)  # Full metrics as JSON
    created_at = Column(DateTime, default=utc_now, nullable=False)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "net_return": self.net_return,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "avg_holding_period": self.avg_holding_period,
            "total_fees": self.total_fees,
            "total_slippage": self.total_slippage,
            "total_funding": self.total_funding,
        }


class MLModel(Base):
    """Machine learning model tracking."""
    __tablename__ = "ml_models"

    id = Column(String(50), primary_key=True)
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False)
    model_type = Column(String(50), nullable=False)  # logistic, rf, xgboost
    features_used = Column(Text, nullable=False)  # JSON array
    train_start = Column(String(10), nullable=False)
    train_end = Column(String(10), nullable=False)
    val_accuracy = Column(Float)
    test_accuracy = Column(Float)
    model_path = Column(String(500))
    created_at = Column(DateTime, default=utc_now, nullable=False)

    experiment = relationship("Experiment", backref="ml_models")


class RiskEvent(Base):
    """Risk management events."""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    event_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False)  # warning, critical
    message = Column(Text, nullable=False)
    context = Column(Text)  # JSON

    __table_args__ = (
        Index("ix_risk_events_timestamp", "timestamp"),
    )


class ResearchCandidate(Base):
    """A single strategy candidate evaluated during research/discovery.

    Enables exact reproduction of a discovery run: every candidate's strategy
    type, parameters, data context, and metrics are persisted.
    """
    __tablename__ = "research_candidates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(50), ForeignKey("experiments.id"), nullable=False)
    strategy_type = Column(String(50), nullable=False)
    params = Column(Text, nullable=False)       # JSON
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    market_type = Column(String(10), nullable=False)
    metrics = Column(Text, nullable=False)      # JSON (full metrics dict)
    rank = Column(Integer)
    passed_filters = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    experiment = relationship("Experiment", backref="candidates")

    __table_args__ = (
        Index("ix_research_candidates_experiment", "experiment_id"),
    )


class ExecutionTrade(Base):
    """Paper/Live execution trades."""
    __tablename__ = "execution_trades"

    id = Column(String(50), primary_key=True)
    strategy_id = Column(String(50), ForeignKey("strategies.id"), nullable=False)
    execution_mode = Column(String(20), nullable=False)  # paper, live
    symbol = Column(String(20), nullable=False)
    market_type = Column(String(10), nullable=False)
    timeframe = Column(String(10))
    direction = Column(String(10), nullable=False)
    entry_time = Column(Integer, nullable=False)
    exit_time = Column(Integer)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    quantity = Column(Float, nullable=False)
    leverage = Column(Integer, nullable=False, default=1)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    fees = Column(Float, default=0)
    funding = Column(Float, default=0)
    gross_pnl = Column(Float)
    order_ids = Column(Text)  # JSON array of exchange order IDs
    status = Column(String(20), nullable=False, default="open")  # open, closed, cancelled
    net_pnl = Column(Float)
    exit_reason = Column(String(50))
    created_at = Column(DateTime, default=utc_now, nullable=False)

    strategy = relationship("Strategy", backref="execution_trades")


class PaperAccount(Base):
    """Persisted paper account + worker state snapshot for status & recovery.

    One row per paper run. The worker upserts it every tick so `paper status`
    shows accurate live account state and a later run can recover the account
    (cash, positions) instead of silently starting over.
    """
    __tablename__ = "paper_accounts"

    run_id = Column(String(50), primary_key=True)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    market_type = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    mode = Column(String(20), nullable=False, default="replay")  # replay/realtime
    initial_capital = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    positions = Column(Text)          # JSON: open positions
    closed_trades = Column(Integer, nullable=False, default=0)
    worker_state = Column(Text)       # JSON: run flags, bars/signals/orders, peak equity
    risk_status = Column(Text)        # JSON: drawdown/exposure/kill-switch/events
    last_market_ts = Column(Integer)
    status = Column(String(20), nullable=False, default="running")  # running/stopped
    started_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        Index("ix_paper_accounts_updated", "updated_at"),
    )


class LiveAccount(Base):
    """Persisted live/testnet/dry-run account + worker state snapshot.

    Tracks live balances, equity, positions, risk status, environment (LIVE/TESTNET/DRY_RUN),
    and exchange synchronization metrics.
    """
    __tablename__ = "live_accounts"

    run_id = Column(String(50), primary_key=True)
    environment = Column(String(20), nullable=False)  # live, testnet, dry_run
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    market_type = Column(String(10), nullable=False)
    strategy = Column(String(50), nullable=False)
    initial_capital = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    positions = Column(Text)          # JSON: open positions
    closed_trades = Column(Integer, nullable=False, default=0)
    worker_state = Column(Text)       # JSON: orders placed, signals, error count
    risk_status = Column(Text)        # JSON: drawdown, daily/weekly loss, kill switch
    last_market_ts = Column(Integer)
    status = Column(String(20), nullable=False, default="running")  # running/stopped/halted
    started_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        Index("ix_live_accounts_updated", "updated_at"),
    )


class LiveOrder(Base):
    """Complete audit record of every order placed to Binance or Dry-Run."""
    __tablename__ = "live_orders"

    id = Column(String(64), primary_key=True)  # client_order_id
    exchange_order_id = Column(String(64))
    run_id = Column(String(50), nullable=False)
    environment = Column(String(20), nullable=False)  # live, testnet, dry_run
    strategy_id = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    market_type = Column(String(10), nullable=False)
    side = Column(String(10), nullable=False)  # buy, sell
    order_type = Column(String(20), nullable=False)  # market, limit
    requested_qty = Column(Float, nullable=False)
    executed_qty = Column(Float, default=0.0)
    requested_price = Column(Float)
    avg_fill_price = Column(Float)
    fee = Column(Float, default=0.0)
    status = Column(String(20), nullable=False)  # new, filled, partially_filled, rejected, canceled
    rejection_reason = Column(Text)
    latency_ms = Column(Float)
    idempotency_key = Column(String(64))
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    __table_args__ = (
        Index("ix_live_orders_symbol", "symbol"),
        Index("ix_live_orders_status", "status"),
        Index("ix_live_orders_created", "created_at"),
    )


class CarryPositionRecord(Base):
    """Durable lifecycle ledger for one delta-neutral cash-and-carry position."""
    __tablename__ = "carry_positions"

    position_id = Column(String(64), primary_key=True)
    symbol = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    spot_fill_price = Column(Float, nullable=False)
    futures_fill_price = Column(Float, nullable=False)
    entry_basis_spread_pct = Column(Float, nullable=False)
    leg_gap_ms = Column(Float, nullable=True)  # unhedged window between the two entry fills (ms)
    baseline_spot_qty = Column(Float, nullable=True)      # pre-existing spot inventory at entry
    baseline_futures_qty = Column(Float, nullable=True)   # pre-existing futures position at entry
    status = Column(String(20), nullable=False)
    opened_at = Column(DateTime, nullable=False)
    closed_at = Column(DateTime)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "status IN ('OPEN', 'UNWINDING', 'CLOSED', 'ORPHAN_UNWOUND', 'DESYNC_CLOSED')",
            name="ck_carry_position_status",
        ),
        Index("ix_carry_positions_symbol_status", "symbol", "status"),
        Index("ix_carry_positions_opened_at", "opened_at"),
    )


class CarryFundingPaymentRecord(Base):
    """One observed funding settlement attributable to a carry position."""
    __tablename__ = "carry_funding_payments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    position_id = Column(String(64), ForeignKey("carry_positions.position_id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    funding_rate = Column(Float, nullable=False)
    funding_payment_usdt = Column(Float, nullable=False)
    mark_price = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    position = relationship("CarryPositionRecord", backref="funding_payments")

    __table_args__ = (
        UniqueConstraint("position_id", "timestamp", name="uq_carry_funding_settlement"),
        Index("ix_carry_funding_symbol_timestamp", "symbol", "timestamp"),
    )


class ReconciliationEvent(Base):
    """Audit record of reconciliation checks and any detected discrepancies."""
    __tablename__ = "reconciliation_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=utc_now, nullable=False)
    is_clean = Column(Boolean, nullable=False)
    discrepancies = Column(Text)  # JSON array of discrepancies
    action_taken = Column(String(50))  # e.g. "halted_trading", "synced"

