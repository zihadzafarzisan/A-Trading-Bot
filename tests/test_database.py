"""Tests for database models and connection."""

import pytest
from datetime import datetime

from crypto_quant.db.connection import DatabaseManager
from crypto_quant.db.models import (
    MarketData,
    AssetUniverse,
    Feature,
    Strategy,
    Experiment,
    Trade,
    ExperimentResult,
)


class TestDatabaseConnection:
    """Test database connection management."""

    def test_create_database_file(self, temp_dir):
        """Test creating a new file-based database."""
        db_path = temp_dir / "test.db"
        db = DatabaseManager(str(db_path), in_memory=False)
        db.create_tables()
        assert db_path.exists()
        # Dispose engine to release file lock before cleanup
        db.engine.dispose()

    def test_create_database_in_memory(self):
        """Test creating an in-memory database."""
        db = DatabaseManager(in_memory=True)
        assert db.in_memory is True
        db.create_tables()

    def test_create_tables(self, test_db):
        """Test creating database tables."""
        assert test_db.table_exists("market_data")
        assert test_db.table_exists("strategies")
        assert test_db.table_exists("experiments")
        assert test_db.table_exists("trades")

    def test_get_session(self, test_db):
        """Test getting a database session."""
        session = test_db.get_session()
        assert session is not None
        session.close()

    def test_session_generator(self, test_db):
        """Test session generator for dependency injection."""
        gen = test_db.session_generator()
        session = next(gen)
        try:
            assert session is not None
        finally:
            try:
                next(gen)
            except StopIteration:
                pass


class TestMarketDataModel:
    """Test MarketData model."""

    def test_create_market_data(self, test_db):
        """Test creating market data record."""
        session = test_db.get_session()
        try:
            data = MarketData(
                symbol="BTCUSDT",
                timeframe="1h",
                timestamp=1609459200000,
                open=50000.0,
                high=51000.0,
                low=49500.0,
                close=50500.0,
                volume=1000000.0,
                market_type="spot",
            )
            session.add(data)
            session.commit()

            retrieved = session.query(MarketData).first()
            assert retrieved is not None
            assert retrieved.symbol == "BTCUSDT"
            assert retrieved.open == 50000.0
        finally:
            session.close()

    def test_market_data_unique_constraint(self, test_db):
        """Test unique constraint on market data."""
        session = test_db.get_session()
        try:
            data1 = MarketData(
                symbol="BTCUSDT",
                timeframe="1h",
                timestamp=1609459200000,
                open=50000.0,
                high=51000.0,
                low=49500.0,
                close=50500.0,
                volume=1000000.0,
                market_type="spot",
            )
            session.add(data1)
            session.commit()

            # Duplicate should raise error
            data2 = MarketData(
                symbol="BTCUSDT",
                timeframe="1h",
                timestamp=1609459200000,  # Same timestamp
                open=51000.0,
                high=52000.0,
                low=50500.0,
                close=51500.0,
                volume=1100000.0,
                market_type="spot",
            )
            session.add(data2)

            with pytest.raises(Exception):
                session.commit()
        finally:
            session.rollback()


class TestStrategyModel:
    """Test Strategy model."""

    def test_create_strategy(self, test_db):
        """Test creating strategy record."""
        session = test_db.get_session()
        try:
            strategy = Strategy(
                id="STRAT-TEST001",
                name="Trend Following",
                type="trend",
                description="Follows market trends",
                parameters='{"ema_fast": 20, "ema_slow": 50}',
                version=1,
            )
            session.add(strategy)
            session.commit()

            retrieved = session.query(Strategy).first()
            assert retrieved is not None
            assert retrieved.id == "STRAT-TEST001"
            assert retrieved.name == "Trend Following"
        finally:
            session.close()


class TestTradeModel:
    """Test Trade model."""

    def test_create_trade(self, test_db):
        """Test creating trade record."""
        session = test_db.get_session()
        try:
            # First create required strategy
            strategy = Strategy(
                id="STRAT-TEST001",
                name="Test Strategy",
                type="trend",
                parameters="{}",
                version=1,
            )
            session.add(strategy)
            session.commit()

            # Create experiment
            experiment = Experiment(
                id="EXP-TEST001",
                strategy_id="STRAT-TEST001",
                config="{}",
                start_date="2020-01-01",
                end_date="2026-09-08",
                status="completed",
            )
            session.add(experiment)
            session.commit()

            # Create trade
            trade = Trade(
                id="TRADE-TEST001",
                experiment_id="EXP-TEST001",
                strategy_id="STRAT-TEST001",
                symbol="BTCUSDT",
                market_type="spot",
                direction="long",
                timeframe="1h",
                entry_time=1609459200000,
                entry_price=50000.0,
                quantity=0.1,
                leverage=1,
            )
            session.add(trade)
            session.commit()

            retrieved = session.query(Trade).first()
            assert retrieved is not None
            assert retrieved.symbol == "BTCUSDT"
            assert retrieved.direction == "long"
        finally:
            session.close()


class TestExperimentResultModel:
    """Test ExperimentResult model."""

    def test_create_experiment_result(self, test_db):
        """Test creating experiment result record."""
        session = test_db.get_session()
        try:
            # First create required strategy
            strategy = Strategy(
                id="STRAT-TEST001",
                name="Test Strategy",
                type="trend",
                parameters="{}",
                version=1,
            )
            session.add(strategy)
            session.commit()

            # Create experiment
            experiment = Experiment(
                id="EXP-TEST001",
                strategy_id="STRAT-TEST001",
                config="{}",
                start_date="2020-01-01",
                end_date="2026-09-08",
                status="completed",
            )
            session.add(experiment)
            session.commit()

            # Create result
            result = ExperimentResult(
                experiment_id="EXP-TEST001",
                total_trades=100,
                winning_trades=65,
                losing_trades=35,
                win_rate=65.0,
                profit_factor=1.8,
                expectancy=0.5,
                net_return=50.0,
                max_drawdown=15.0,
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
            )
            session.add(result)
            session.commit()

            retrieved = session.query(ExperimentResult).first()
            assert retrieved is not None
            assert retrieved.total_trades == 100
            assert retrieved.win_rate == 65.0

            # Test to_dict method
            result_dict = retrieved.to_dict()
            assert result_dict["experiment_id"] == "EXP-TEST001"
            assert result_dict["total_trades"] == 100
        finally:
            session.close()
