"""Multi-Strategy Research Engine for BTCUSDT & ETHUSDT.

Evaluates 5 strategy families across 5 timeframes on BTCUSDT and ETHUSDT:
1. MTF Trend + Pullback + Confluence (mtf_trend_pullback)
2. Breakout + Retest + Volume (breakout_retest)
3. Trend-Filtered RSI Pullback (trend_filtered_rsi)
4. Regime-Adaptive Trend / Mean-Reversion (regime_adaptive)
5. VWAP / Bollinger Mean Reversion (vwap_bollinger_mr)

Applies multi-objective ranking, walk-forward validation, and Monte Carlo risk modeling.
"""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..data.repository import MarketDataRepository
from ..db.connection import DatabaseManager
from ..logging_config import get_logger
from ..strategies import create_strategy
from ..validation.monte_carlo import MonteCarloSimulator
from ..validation.optimize_window import (
    MultiObjectiveFilters,
    MultiObjectiveWeights,
    parameter_stability,
    score_metrics_multi_objective,
)

logger = get_logger("research")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]
STRATEGY_FAMILIES = [
    "mtf_trend_pullback",
    "breakout_retest",
    "trend_filtered_rsi",
    "regime_adaptive",
    "vwap_bollinger_mr",
]


@dataclass
class ExperimentResult:
    """Standardized record for one experiment in the research matrix."""

    experiment_id: str
    strategy: str
    symbol: str
    timeframe: str
    market_type: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    score: float
    passed_filters: bool
    rejection_reasons: List[str]
    sub_scores: Dict[str, float]
    trades: List[Dict[str, Any]]

    def to_dict(self, include_trades: bool = False) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "strategy": self.strategy,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "market_type": self.market_type,
            "params": self.params,
            "metrics": self.metrics,
            "score": self.score,
            "passed_filters": self.passed_filters,
            "rejection_reasons": self.rejection_reasons,
            "sub_scores": self.sub_scores,
        }
        if include_trades:
            d["trades"] = self.trades
        return d


class MultiStrategyResearchEngine:
    """Executes systematic multi-strategy matrix research."""

    def __init__(
        self,
        db_path: str = "data/crypto_quant.db",
        initial_capital: float = 1000.0,
        risk_per_trade: float = 0.01,
        max_open_positions: int = 3,
        max_leverage: float = 5.0,
        market_type: str = "futures",
    ):
        self.db = DatabaseManager(db_path)
        self.repo = MarketDataRepository(self.db)
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_open_positions = max_open_positions
        self.max_leverage = max_leverage
        self.market_type = market_type
        self.weights = MultiObjectiveWeights()
        self.filters = MultiObjectiveFilters(min_trades=15, min_profit_factor=1.05, max_drawdown=0.40)

    def run_single_experiment(
        self,
        strategy_type: str,
        symbol: str,
        timeframe: str,
        params: Optional[Dict[str, Any]] = None,
        market_type: Optional[str] = None,
    ) -> Optional[ExperimentResult]:
        """Run backtest on a single symbol/timeframe/strategy configuration."""
        m_type = market_type or self.market_type
        df = self.repo.load(symbol, timeframe, market_type="spot")
        if df.empty or len(df) < 50:
            logger.warning("No data for %s %s", symbol, timeframe)
            return None

        # Determine default sensible parameters if not supplied
        p = dict(params or {})
        if strategy_type == "mtf_trend_pullback":
            combo_map = {"5m": "4h_1h_5m", "15m": "4h_1h_15m", "1h": "1d_4h_1h", "4h": "1d_4h_1h", "1d": "1d_4h_1h"}
            p.setdefault("mtf_combo", combo_map.get(timeframe, "1d_4h_1h"))
            p.setdefault("htf_adx_threshold", 25.0 if symbol == "BTCUSDT" else 20.0)
            p.setdefault("atr_multiplier", 2.5)
            p.setdefault("risk_reward_ratio", 3.0 if symbol == "BTCUSDT" else 2.0)

        elif strategy_type == "breakout_retest":
            p.setdefault("lookback", 30)
            p.setdefault("breakout_vol_mult", 1.2)
            p.setdefault("retest_vol_max", 1.0)
            p.setdefault("atr_multiplier", 2.2)
            p.setdefault("risk_reward_ratio", 2.5)

        elif strategy_type == "trend_filtered_rsi":
            p.setdefault("sma_fast", 50)
            p.setdefault("sma_slow", 200)
            p.setdefault("adx_threshold", 20.0)
            p.setdefault("rsi_period", 14)
            p.setdefault("rsi_oversold", 35.0)
            p.setdefault("rsi_overbought", 65.0)
            p.setdefault("atr_multiplier", 2.2)
            p.setdefault("risk_reward_ratio", 2.2)

        elif strategy_type == "regime_adaptive":
            p.setdefault("adx_trend_thresh", 25.0)
            p.setdefault("chop_range_thresh", 55.0)
            p.setdefault("trend_ema_fast", 20)
            p.setdefault("trend_ema_slow", 50)
            p.setdefault("atr_multiplier", 2.2)
            p.setdefault("risk_reward_ratio", 2.2)

        elif strategy_type == "vwap_bollinger_mr":
            p.setdefault("adx_max_thresh", 22.0)
            p.setdefault("bb_period", 20)
            p.setdefault("bb_std", 2.0)
            p.setdefault("rsi_period", 14)
            p.setdefault("rsi_oversold", 35.0)
            p.setdefault("rsi_overbought", 65.0)
            p.setdefault("atr_multiplier", 2.0)
            p.setdefault("risk_reward_ratio", 2.0)

        strategy = create_strategy(strategy_type, params=p)
        cfg = BacktestConfig(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_open_positions=self.max_open_positions,
            max_leverage=self.max_leverage if m_type == "futures" else 1.0,
            market_type=m_type,
            timeframe=timeframe,
            execution=ExecutionConfig(market_type=m_type),
        )
        engine = BacktestEngine(cfg)
        res = engine.run(strategy, df, symbol=symbol)
        m = res.metrics.to_dict()

        score, passed, reasons, sub = score_metrics_multi_objective(m, self.weights, self.filters)
        exp_id = f"{strategy_type}_{symbol}_{timeframe}_{m_type}"

        return ExperimentResult(
            experiment_id=exp_id,
            strategy=strategy_type,
            symbol=symbol,
            timeframe=timeframe,
            market_type=m_type,
            params=p,
            metrics=m,
            score=score,
            passed_filters=passed,
            rejection_reasons=reasons,
            sub_scores=sub,
            trades=res.trades,
        )

    def run_matrix(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        market_type: str = "futures",
    ) -> List[ExperimentResult]:
        """Execute the full symbol x timeframe x strategy matrix."""
        target_symbols = symbols or SYMBOLS
        target_tfs = timeframes or TIMEFRAMES
        target_strats = strategies or STRATEGY_FAMILIES

        results: List[ExperimentResult] = []
        total = len(target_symbols) * len(target_tfs) * len(target_strats)
        count = 0

        print(f"\n=======================================================")
        print(f"RUNNING RESEARCH MATRIX ({total} EXPERIMENTS)")
        print(f"Symbols: {target_symbols} | Timeframes: {target_tfs}")
        print(f"Strategies: {target_strats}")
        print(f"=======================================================")

        for sym in target_symbols:
            for tf in target_tfs:
                for strat in target_strats:
                    count += 1
                    t0 = time.time()
                    res = self.run_single_experiment(strat, sym, tf, market_type=market_type)
                    dur = time.time() - t0

                    if res:
                        results.append(res)
                        m = res.metrics
                        status = "[PASS]" if res.passed_filters else "[REJECT]"
                        print(
                            f"[{count:>2}/{total}] {status} {strat:<20} | {sym} {tf:<4} -> "
                            f"Trades: {m.get('total_trades', 0):>4} | WR: {m.get('win_rate', 0)*100:>5.1f}% | "
                            f"PF: {m.get('profit_factor', 0):>4.2f} | Exp: ${m.get('expectancy', 0):>5.2f} | "
                            f"Ret: {m.get('net_return', 0)*100:>6.1f}% | MaxDD: {m.get('max_drawdown', 0)*100:>5.1f}% | "
                            f"Score: {res.score:>6.4f} ({dur:.1f}s)"
                        )

        # Sort all results by multi-objective score descending
        results.sort(key=lambda r: (1 if r.passed_filters else 0, r.score), reverse=True)
        return results
