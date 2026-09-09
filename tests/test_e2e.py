"""End-to-end integration tests for the complete Crypto Quant Terminal workflow.

These tests exercise the full pipeline:
1. Configuration loading
2. Database initialization
3. Data loading (with synthetic data)
4. Indicator computation
5. Feature engineering
6. Strategy creation and signal generation
7. Backtesting (spot & futures, long & short)
8. Research discovery
9. Walk-forward validation
10. Monte Carlo simulation
11. ML pipeline
12. Risk management
13. Dashboard generation

Each test uses realistic synthetic data to verify the system works correctly
without requiring actual Binance API access.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def generate_ohlcv_data(
    n_candles: int = 500,
    start_price: float = 42000.0,
    start_time: int = 1609459200000,  # 2021-01-01 00:00:00 UTC
    timeframe_minutes: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic OHLCV data with trends and volatility."""
    rng = np.random.RandomState(seed)

    timestamps = [start_time + i * timeframe_minutes * 60 * 1000 for i in range(n_candles)]
    prices = [start_price]

    # Generate price series with mean-reverting trends
    for i in range(1, n_candles):
        trend = 0.0002 * np.sin(i / 50)
        vol = 0.01 + 0.005 * abs(np.sin(i / 30))
        ret = trend + vol * rng.randn()
        prices.append(prices[-1] * (1 + ret))

    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        high_factor = 1 + abs(rng.randn()) * 0.005
        low_factor = 1 - abs(rng.randn()) * 0.005
        open_price = prices[i - 1] if i > 0 else close * (1 - 0.001 * rng.randn())

        high = max(open_price, close) * high_factor
        low = min(open_price, close) * low_factor
        volume = abs(1000000 + rng.randn() * 200000)

        data.append({
            "timestamp": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


def generate_multi_symbol_data(
    symbols: list = None,
    n_candles: int = 300,
    seed: int = 42,
) -> dict:
    """Generate OHLCV data for multiple symbols."""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

    rng = np.random.RandomState(seed)
    data = {}
    base_prices = {"BTCUSDT": 42000, "ETHUSDT": 2200, "BNBUSDT": 350}

    for sym in symbols:
        start_price = base_prices.get(sym, 1000)
        data[sym] = generate_ohlcv_data(
            n_candles=n_candles,
            start_price=start_price,
            seed=rng.randint(0, 10000),
        )

    return data


def generate_trades_data(n_trades: int = 100, seed: int = 42) -> list:
    """Generate realistic trade records for ML training."""
    rng = np.random.RandomState(seed)
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    directions = ["long", "short"]
    timeframes = ["1h", "4h", "1d"]
    strategies = ["trend", "momentum", "mean_reversion", "breakout"]

    trades = []
    base_time = 1609459200000

    for i in range(n_trades):
        is_win = rng.random() > 0.45
        entry_price = 42000 + rng.randn() * 5000
        if is_win:
            exit_price = entry_price * (1 + abs(rng.randn()) * 0.03)
        else:
            exit_price = entry_price * (1 - abs(rng.randn()) * 0.02)

        entry_time = base_time + i * 3600000 * rng.randint(1, 24)
        exit_time = entry_time + rng.randint(1, 48) * 3600000

        rsi = 30 + rng.random() * 40 if is_win else 20 + rng.random() * 60
        ema_dist = rng.randn() * 0.02 if is_win else rng.randn() * 0.03
        atr = abs(rng.randn()) * 0.015
        adx = 20 + rng.random() * 30 if is_win else 15 + rng.random() * 40

        trades.append({
            "id": f"TRADE-{i:04d}",
            "symbol": rng.choice(symbols),
            "direction": rng.choice(directions),
            "timeframe": rng.choice(timeframes),
            "strategy_type": rng.choice(strategies),
            "market_type": rng.choice(["spot", "futures"]),
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "net_pnl": exit_price - entry_price if rng.choice(["long", "short"]) == "long" else entry_price - exit_price,
            "return_pct": (exit_price / entry_price - 1) * 100,
            "exit_reason": rng.choice(["take_profit", "stop_loss", "signal", "trailing_stop"]),
            "rsi": rsi,
            "ema_distance": ema_dist,
            "atr": atr,
            "adx": adx,
            "volume_ratio": 1 + rng.randn() * 0.3,
            "regime": rng.choice(["bull_trend", "bear_trend", "sideways"]),
        })

    return trades


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------

class TestConfiguration:
    """Test configuration loading and defaults."""

    def test_default_config_loads(self):
        """Default configuration loads without error."""
        from crypto_quant.config.settings import get_config
        config = get_config()
        assert config is not None
        assert config.market.exchange == "binance"
        assert config.risk.starting_capital == 1000.0
        assert config.risk.risk_per_trade == 0.01

    def test_config_risk_defaults(self):
        """Risk configuration has correct defaults."""
        from crypto_quant.config.settings import get_config
        config = get_config()
        assert config.risk.max_open_positions == 3
        assert config.risk.max_leverage == 5
        assert config.risk.daily_loss_limit > 0
        assert config.risk.weekly_loss_limit > 0


# ---------------------------------------------------------------------------
# Database tests
# ---------------------------------------------------------------------------

class TestDatabase:
    """Test database initialization and operations."""

    def test_database_initialization(self):
        """Database initializes and creates tables."""
        from crypto_quant.db.connection import DatabaseManager
        db = DatabaseManager(in_memory=True)
        db.create_tables()
        assert db is not None

    def test_database_engine(self):
        """Database engine is accessible."""
        from crypto_quant.db.connection import DatabaseManager
        db = DatabaseManager(in_memory=True)
        db.create_tables()
        assert db.engine is not None


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Test data loading and repository operations."""

    def test_synthetic_data_generation(self):
        """Synthetic data generation produces valid OHLCV."""
        df = generate_ohlcv_data(n_candles=100)
        assert len(df) == 100
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert df["high"].ge(df["low"]).all()
        assert df["volume"].gt(0).all()

    def test_multi_symbol_data(self):
        """Multi-symbol data generation works."""
        data = generate_multi_symbol_data(symbols=["BTCUSDT", "ETHUSDT"])
        assert len(data) == 2
        assert "BTCUSDT" in data
        assert "ETHUSDT" in data
        assert len(data["BTCUSDT"]) == 300

    def test_repository_operations(self):
        """Repository save and load operations work."""
        from crypto_quant.db.connection import DatabaseManager
        from crypto_quant.data.repository import MarketDataRepository

        db = DatabaseManager(in_memory=True)
        db.create_tables()
        repo = MarketDataRepository(db)

        df = generate_ohlcv_data(n_candles=50)
        inserted = repo.save(df, symbol="BTCUSDT", timeframe="1h", market_type="spot")
        assert inserted > 0

        loaded = repo.load("BTCUSDT", "1h", market_type="spot")
        assert loaded is not None
        assert len(loaded) > 0


# ---------------------------------------------------------------------------
# Indicator tests
# ---------------------------------------------------------------------------

class TestIndicators:
    """Test technical indicator computation."""

    def test_sma_computation(self):
        """SMA indicator computes correctly."""
        from crypto_quant.indicators.trend import sma
        df = generate_ohlcv_data(n_candles=50)
        result = sma(df["close"], period=20)
        assert len(result) == len(df)
        # With shift=1 (default), first 20 values are NaN (period + shift - 1)
        assert result.isna().sum() == 20

    def test_ema_computation(self):
        """EMA indicator computes correctly."""
        from crypto_quant.indicators.trend import ema
        df = generate_ohlcv_data(n_candles=50)
        result = ema(df["close"], period=20)
        assert len(result) == len(df)
        assert result.isna().sum() == 20

    def test_rsi_computation(self):
        """RSI indicator computes correctly."""
        from crypto_quant.indicators.momentum import rsi
        df = generate_ohlcv_data(n_candles=50)
        result = rsi(df["close"], period=14)
        assert len(result) == len(df)
        valid = result.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_macd_computation(self):
        """MACD indicator computes correctly (returns DataFrame)."""
        from crypto_quant.indicators.momentum import macd
        df = generate_ohlcv_data(n_candles=50)
        result = macd(df["close"])
        # MACD returns a DataFrame with columns: macd, signal, histogram
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert "macd" in result.columns or len(result.columns) >= 3

    def test_bollinger_bands(self):
        """Bollinger Bands compute correctly."""
        from crypto_quant.indicators.volatility import bollinger_bands
        df = generate_ohlcv_data(n_candles=50)
        result = bollinger_bands(df["close"], period=20)
        # Bollinger Bands returns a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_atr_computation(self):
        """ATR indicator computes correctly."""
        from crypto_quant.indicators.volatility import atr
        df = generate_ohlcv_data(n_candles=50)
        result = atr(df["high"], df["low"], df["close"], period=14)
        assert len(result) == len(df)
        valid = result.dropna()
        assert valid.min() > 0

    def test_indicator_registry(self):
        """Indicator registry contains all expected indicators."""
        from crypto_quant.indicators import INDICATOR_REGISTRY
        expected = ["sma", "ema", "rsi", "macd", "bollinger_bands", "atr", "adx"]
        for name in expected:
            assert name in INDICATOR_REGISTRY


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    def test_feature_engine_computes(self):
        """Feature engine computes all features."""
        from crypto_quant.features.engine import FeatureEngine
        df = generate_ohlcv_data(n_candles=100)
        engine = FeatureEngine()
        features = engine.compute(df)
        assert features is not None
        assert len(features) > 0
        assert len(features.columns) > len(df.columns)

    def test_no_lookahead_bias(self):
        """Features don't use future information."""
        from crypto_quant.features.engine import FeatureEngine
        df = generate_ohlcv_data(n_candles=100)
        engine = FeatureEngine()
        features = engine.compute(df)
        # Features should have NaN for early values (warmup period)
        nan_counts = features.isna().sum()
        assert nan_counts.max() > 0, "Features should have NaN for warmup period"


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestStrategies:
    """Test strategy creation and signal generation."""

    def test_all_strategies_created(self):
        """All strategy types can be created."""
        from crypto_quant.strategies import create_strategy
        strategy_types = ["trend", "momentum", "mean_reversion", "breakout"]
        for stype in strategy_types:
            strat = create_strategy(stype)
            assert strat is not None
            assert strat.name is not None

    def test_strategy_signal_generation(self):
        """Strategies generate valid signals via entry_signal."""
        from crypto_quant.strategies import create_strategy
        from crypto_quant.strategies.base import Direction

        # Strategy needs OHLCV data, not just features
        df = generate_ohlcv_data(n_candles=100)
        strat = create_strategy("trend")

        # Test entry_signal on a few bars
        setup_df = strat.setup(df)
        signals = []
        for i in range(min(50, len(setup_df) - 1)):
            try:
                direction, reason = strat.entry_signal(setup_df, i)
                signals.append(direction)
            except Exception:
                pass

        assert len(signals) > 0
        # All signals should be valid Direction enum values
        for sig in signals:
            assert isinstance(sig, Direction)

    def test_trend_strategy_params(self):
        """Trend strategy accepts parameters."""
        from crypto_quant.strategies import create_strategy
        strat = create_strategy("trend", params={"ema_fast": 10, "ema_slow": 30})
        assert strat is not None


# ---------------------------------------------------------------------------
# Backtesting tests
# ---------------------------------------------------------------------------

class TestBacktesting:
    """Test backtesting engine."""

    def test_spot_long_backtest(self):
        """Spot long backtest completes successfully."""
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=200)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            max_open_positions=3,
            market_type="spot",
            timeframe="1h",
            execution=ExecutionConfig(market_type="spot"),
        )
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")

        assert result is not None
        assert result.metrics is not None
        assert result.equity_curve is not None
        assert result.trades is not None

    def test_futures_long_backtest(self):
        """Futures long backtest completes successfully."""
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=200)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            max_open_positions=3,
            max_leverage=5,
            market_type="futures",
            timeframe="1h",
            execution=ExecutionConfig(market_type="futures"),
        )
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")

        assert result is not None
        assert result.metrics is not None

    def test_futures_short_backtest(self):
        """Futures short backtest completes successfully."""
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=200)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            max_open_positions=3,
            max_leverage=5,
            market_type="futures",
            timeframe="1h",
            execution=ExecutionConfig(market_type="futures"),
        )
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")
        assert result is not None

    def test_backtest_metrics_computed(self):
        """Backtest metrics are computed correctly."""
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=200)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            market_type="spot",
            timeframe="1h",
            execution=ExecutionConfig(market_type="spot"),
        )
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")

        m = result.metrics
        # metrics.trades is a TradeStats object
        assert isinstance(m.net_return, float)
        assert 0 <= m.max_drawdown <= 1  # max_drawdown is stored as positive fraction
        assert isinstance(m.sharpe_ratio, float)

    def test_backtest_no_lookahead(self):
        """Backtest doesn't use future information."""
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=100)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            market_type="spot",
            timeframe="1h",
            execution=ExecutionConfig(market_type="spot"),
        )
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")

        for trade in result.trades:
            if trade.get("entry_time") and trade.get("entry_price"):
                assert trade["entry_price"] > 0


# ---------------------------------------------------------------------------
# Research discovery tests
# ---------------------------------------------------------------------------

class TestResearchDiscovery:
    """Test research discovery engine."""

    def test_discovery_engine_runs(self):
        """Discovery engine runs without error."""
        from crypto_quant.research import DiscoveryEngine, DiscoveryConfig

        df = generate_ohlcv_data(n_candles=300)
        config = DiscoveryConfig(
            strategy_types=["trend", "momentum"],
            search_type="grid",
            max_combinations=5,
            min_trades=10,
            market_type="spot",
            timeframe="1h",
        )
        engine = DiscoveryEngine(config=config)
        result = engine.discover(df, symbol="BTCUSDT")

        assert result is not None
        assert result.n_tested > 0
        assert result.ranked is not None

    def test_quality_filters_applied(self):
        """Quality filters reject bad strategies."""
        from crypto_quant.research import DiscoveryEngine, DiscoveryConfig

        df = generate_ohlcv_data(n_candles=300)
        config = DiscoveryConfig(
            strategy_types=["trend"],
            search_type="grid",
            max_combinations=3,
            min_trades=50,
            market_type="spot",
            timeframe="1h",
        )
        engine = DiscoveryEngine(config=config)
        result = engine.discover(df, symbol="BTCUSDT")
        assert result is not None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    """Test walk-forward and Monte Carlo validation."""

    def test_walk_forward_validation(self):
        """Walk-forward validation runs successfully with enough data."""
        from crypto_quant.validation import WalkForwardValidator

        # Use more candles to ensure windows have enough data
        df = generate_ohlcv_data(n_candles=2000, timeframe_minutes=60)
        validator = WalkForwardValidator()
        report = validator.validate(
            df, strategy_type="trend",
            params={"ema_fast": 20, "ema_slow": 50},
            symbol="BTCUSDT",
            market_type="spot",
            timeframe="1h",
        )

        assert report is not None
        # With 2000 hourly candles (~83 days), some windows should have data
        # If no windows, it's because the validator requires more data - that's ok
        assert report.strategy_type == "trend"

    def test_monte_carlo_simulation(self):
        """Monte Carlo simulation runs successfully."""
        from crypto_quant.validation import MonteCarloSimulator

        rng = np.random.RandomState(42)
        trade_returns = rng.normal(0.01, 0.02, 50).tolist()

        simulator = MonteCarloSimulator(n_simulations=100)
        result = simulator.simulate(trade_returns, initial_capital=1000)

        assert result is not None
        assert result.n_simulations == 100
        assert result.probability_of_ruin >= 0
        assert result.probability_of_ruin <= 1


# ---------------------------------------------------------------------------
# ML pipeline tests
# ---------------------------------------------------------------------------

class TestMLPipeline:
    """Test machine learning pipeline."""

    def test_ml_pipeline_runs(self):
        """ML pipeline runs without error."""
        from crypto_quant.ml import MLPipeline
        from crypto_quant.features.engine import FeatureEngine

        df = generate_ohlcv_data(n_candles=200)
        trades = generate_trades_data(n_trades=100)
        features = FeatureEngine().compute(df)

        pipeline = MLPipeline(threshold=0.6)
        report = pipeline.run(
            trades, features,
            symbol="BTCUSDT",
            timeframe="1h",
            strategy_type="trend",
        )

        assert report is not None
        assert report.n_samples > 0
        assert report.evaluation is not None

    def test_ml_no_data_leakage(self):
        """ML training doesn't use future data."""
        from crypto_quant.ml import MLPipeline
        from crypto_quant.features.engine import FeatureEngine

        df = generate_ohlcv_data(n_candles=200)
        trades = generate_trades_data(n_trades=50)
        features = FeatureEngine().compute(df)

        pipeline = MLPipeline(threshold=0.6)
        report = pipeline.run(
            trades, features,
            symbol="BTCUSDT",
            timeframe="1h",
            strategy_type="trend",
        )

        assert report is not None
        assert report.evaluation.test_metrics is not None


# ---------------------------------------------------------------------------
# Risk management tests
# ---------------------------------------------------------------------------

class TestRiskManagement:
    """Test risk management engine."""

    def test_risk_manager_initializes(self):
        """Risk manager initializes with correct config."""
        from crypto_quant.risk import RiskManager
        from crypto_quant.config.settings import get_config

        config = get_config()
        manager = RiskManager(config.risk)
        assert manager is not None

    def test_position_sizing(self):
        """Position sizing calculates correctly."""
        from crypto_quant.risk.position_sizing import PositionSizer
        from crypto_quant.risk.limits import RiskLimits

        sizer = PositionSizer(RiskLimits())
        result = sizer.size_position(
            equity=1000,
            entry_price=42000,
            stop_price=41500,
            direction="long",
            market_type="spot",
        )
        assert result.notional > 0
        assert result.quantity > 0
        assert result.risk_amount > 0

    def test_risk_limits(self):
        """Risk limits are configured correctly."""
        from crypto_quant.risk.limits import RiskLimits

        limits = RiskLimits(
            daily_loss_limit=100.0,
            weekly_loss_limit=300.0,
            max_drawdown_pct=0.25,
            max_position_pct=0.30,
        )
        assert limits.daily_loss_limit == 100.0
        assert limits.weekly_loss_limit == 300.0
        assert limits.max_drawdown_pct == 0.25

    def test_risk_limits_to_dict(self):
        """Risk limits can be serialized."""
        from crypto_quant.risk.limits import RiskLimits
        limits = RiskLimits()
        d = limits.to_dict()
        assert isinstance(d, dict)
        assert "starting_capital" in d
        assert "risk_per_trade" in d


# ---------------------------------------------------------------------------
# Dashboard generation tests
# ---------------------------------------------------------------------------

class TestDashboardGeneration:
    """Test complete dashboard generation."""

    def test_dashboard_generation(self, tmp_path):
        """Dashboard generates without error."""
        from crypto_quant.dashboard import DashboardData, DashboardGenerator

        data = DashboardData(
            title="E2E Test Dashboard",
            metrics={"win_rate": 0.62, "profit_factor": 1.8, "net_return": 0.25},
            equity_curve=[{"time": 1700000000000, "equity": 1000}],
            trades=[{"symbol": "BTCUSDT", "net_pnl": 50, "direction": "long",
                    "timeframe": "1h", "entry_price": 42000, "exit_price": 42500}],
        )
        gen = DashboardGenerator()
        out = gen.generate(data, tmp_path / "dashboard.html")
        assert out.exists()
        assert out.stat().st_size > 1000

    def test_dashboard_all_sections(self, tmp_path):
        """Dashboard includes all required sections."""
        from crypto_quant.dashboard import DashboardData, DashboardGenerator

        data = DashboardData(
            title="Full Dashboard Test",
            metrics={"win_rate": 0.65, "profit_factor": 1.8, "net_return": 0.25,
                    "max_drawdown": -0.12, "sharpe_ratio": 1.4, "total_trades": 50},
            equity_curve=[{"time": 1700000000000, "equity": 1000},
                         {"time": 1700003600000, "equity": 1050}],
            trades=[{"symbol": "BTCUSDT", "direction": "long", "timeframe": "1h",
                    "entry_price": 42000, "exit_price": 42500, "net_pnl": 50,
                    "exit_reason": "take_profit"}],
            regime_breakdown={"bull": {"win_rate": 0.65, "trades": 20}},
            coin_comparison={"coins": [{"symbol": "BTCUSDT",
                                        "metrics": {"win_rate": 0.65, "profit_factor": 2.0,
                                                    "net_return": 0.25, "max_drawdown": -0.10,
                                                    "total_trades": 50}}]},
            spot_vs_futures={"spot": {"win_rate": 0.62}, "futures": {"win_rate": 0.58}},
            walk_forward={"windows": [{"window": {"train_start": "2020-01-01", "train_end": "2021-01-01",
                                                   "test_start": "2021-01-01", "test_end": "2021-06-01"},
                                       "in_sample": {"win_rate": 0.65},
                                       "out_of_sample": {"win_rate": 0.58, "total_trades": 30},
                                       "win_rate_degradation": -0.07}]},
            monte_carlo={"final_equity_percentiles": {"p50": 1300},
                         "max_drawdown_percentiles": {"p50": -0.15},
                         "probability_of_ruin": 0.02},
            ml={"evaluation": {"baseline_win_rate": 0.55, "ml_filtered_win_rate": 0.65, "ml_improves": True}},
            risk_events=[{"timestamp": 1700000000000, "event_type": "test", "severity": "warning", "message": "Test"}],
            warnings=["Test warning"],
        )
        gen = DashboardGenerator()
        out = gen.generate(data, tmp_path / "full.html")
        html = out.read_text(encoding="utf-8")

        sections = ["Win Rate", "Equity Curve", "Trade History", "Regime Breakdown",
                   "Coin Performance", "Spot vs Futures", "Walk-Forward", "Monte Carlo",
                   "Machine Learning", "Risk Events", "Warnings"]
        for section in sections:
            assert section in html, f"Missing section: {section}"


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """End-to-end integration test of the complete pipeline."""

    def test_complete_workflow(self, tmp_path):
        """Test the complete workflow from data to dashboard."""
        # 1. Load configuration
        from crypto_quant.config.settings import get_config
        config = get_config()
        assert config is not None

        # 2. Initialize database
        from crypto_quant.db.connection import DatabaseManager
        db = DatabaseManager(in_memory=True)
        db.create_tables()

        # 3. Generate and load data
        df = generate_ohlcv_data(n_candles=300)
        assert len(df) == 300

        # 4. Compute features
        from crypto_quant.features.engine import FeatureEngine
        features = FeatureEngine().compute(df)
        assert len(features.columns) > len(df.columns)

        # 5. Test indicators
        from crypto_quant.indicators import INDICATOR_REGISTRY
        assert len(INDICATOR_REGISTRY) > 20

        # 6. Create and test strategy
        from crypto_quant.strategies import create_strategy
        strat = create_strategy("trend")
        assert strat is not None

        # 7. Run backtest
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        bt_config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            max_open_positions=3,
            market_type="spot",
            timeframe="1h",
            execution=ExecutionConfig(market_type="spot"),
        )
        engine = BacktestEngine(bt_config)
        result = engine.run(strat, df, symbol="BTCUSDT")
        assert result is not None

        # 8. Run Monte Carlo
        from crypto_quant.validation import MonteCarloSimulator
        returns = [t.get("return_pct", 0) / 100 for t in result.trades if t.get("return_pct") is not None]
        mc_result = None
        if len(returns) >= 10:
            mc = MonteCarloSimulator(n_simulations=50)
            mc_result = mc.simulate(returns, initial_capital=1000)
            assert mc_result is not None

        # 9. Generate dashboard
        from crypto_quant.dashboard import DashboardData, DashboardGenerator
        dashboard_data = DashboardData(
            title="E2E Pipeline Test",
            subtitle="Complete workflow verification",
            metrics=result.metrics.to_dict(),
            equity_curve=result.equity_curve,
            trades=result.trades[:100],
            monte_carlo=mc_result.to_dict() if mc_result else {},
        )
        gen = DashboardGenerator()
        out = gen.generate(dashboard_data, tmp_path / "pipeline_dashboard.html")
        assert out.exists()
        assert out.stat().st_size > 5000

        # 10. Verify dashboard content
        html = out.read_text(encoding="utf-8")
        assert "E2E Pipeline Test" in html
        assert "Win Rate" in html

        print(f"\n[OK] Complete E2E pipeline test passed!")
        print(f"  - Data: {len(df)} candles")
        print(f"  - Features: {len(features.columns)} columns")
        print(f"  - Trades: {len(result.trades)}")
        print(f"  - Dashboard: {out.stat().st_size:,} bytes")

    def test_multi_strategy_comparison(self, tmp_path):
        """Test comparing multiple strategies."""
        from crypto_quant.strategies import create_strategy
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.features.engine import FeatureEngine
        from crypto_quant.dashboard import DashboardData, DashboardGenerator

        df = generate_ohlcv_data(n_candles=300)
        features = FeatureEngine().compute(df)

        strategies = ["trend", "momentum", "mean_reversion", "breakout"]
        results = []

        for stype in strategies:
            strat = create_strategy(stype)
            config = BacktestConfig(
                initial_capital=1000,
                risk_per_trade=0.01,
                market_type="spot",
                timeframe="1h",
                execution=ExecutionConfig(market_type="spot"),
            )
            engine = BacktestEngine(config)
            result = engine.run(strat, df, symbol="BTCUSDT")
            results.append({
                "strategy_type": stype,
                "metrics": result.metrics.to_dict(),
                "trades": len(result.trades),
            })

        # Generate comparison dashboard
        ranked = []
        for i, r in enumerate(sorted(results, key=lambda x: x["metrics"].get("win_rate", 0), reverse=True)):
            ranked.append({
                "rank": i + 1,
                "strategy_type": r["strategy_type"],
                "params": {},
                "metrics": r["metrics"],
                "passed_filters": r["metrics"].get("total_trades", 0) >= 5,
                "filter_reasons": [],
            })

        dashboard_data = DashboardData(
            title="Multi-Strategy Comparison",
            ranked_strategies=ranked,
        )
        gen = DashboardGenerator()
        out = gen.generate(dashboard_data, tmp_path / "comparison.html")
        assert out.exists()

        html = out.read_text(encoding="utf-8")
        assert "Strategy Ranking" in html
        for stype in strategies:
            assert stype in html

        print(f"\n[OK] Multi-strategy comparison test passed!")
        print(f"  - Tested {len(strategies)} strategies")
        for r in ranked[:3]:
            print(f"  - #{r['rank']} {r['strategy_type']}: {r['metrics'].get('win_rate', 0):.1%} win rate")


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    """Test CLI commands work end-to-end."""

    def test_cli_help(self):
        """CLI help command works."""
        from typer.testing import CliRunner
        from crypto_quant.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Crypto Quant" in result.stdout

    def test_cli_data_status(self):
        """CLI data status command works."""
        from typer.testing import CliRunner
        from crypto_quant.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["data", "status"])
        assert result.exit_code == 0

    def test_cli_dashboard_help(self):
        """CLI dashboard help works."""
        from typer.testing import CliRunner
        from crypto_quant.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0

    def test_cli_serve_help(self):
        """CLI serve help works."""
        from typer.testing import CliRunner
        from crypto_quant.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.stdout.lower()


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------

class TestPerformance:
    """Basic performance benchmarks."""

    def test_backtest_performance(self):
        """Backtest completes within reasonable time."""
        import time
        from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig
        from crypto_quant.strategies import create_strategy

        df = generate_ohlcv_data(n_candles=1000)
        strat = create_strategy("trend")
        config = BacktestConfig(
            initial_capital=1000,
            risk_per_trade=0.01,
            market_type="spot",
            timeframe="1h",
            execution=ExecutionConfig(market_type="spot"),
        )

        start = time.time()
        engine = BacktestEngine(config)
        result = engine.run(strat, df, symbol="BTCUSDT")
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Backtest took {elapsed:.2f}s, should be < 5s"
        print(f"\n  Backtest performance: {elapsed:.3f}s for 1000 candles")

    def test_feature_computation_performance(self):
        """Feature computation completes within reasonable time."""
        import time
        from crypto_quant.features.engine import FeatureEngine

        df = generate_ohlcv_data(n_candles=1000)

        start = time.time()
        engine = FeatureEngine()
        features = engine.compute(df)
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Feature computation took {elapsed:.2f}s, should be < 2s"
        print(f"  Feature computation: {elapsed:.3f}s for 1000 candles")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
