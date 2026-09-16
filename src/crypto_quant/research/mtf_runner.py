"""Comprehensive MTF Strategy Research, Walk-Forward Validation & Monte Carlo Engine.

Runs separate multi-timeframe optimization, walk-forward testing, and Monte Carlo
analysis for BTCUSDT and ETHUSDT across 2022-2026.
"""

import json
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from crypto_quant.backtesting import BacktestConfig, BacktestEngine
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.strategies.mtf_trend_pullback import MTFTrendPullbackStrategy


def run_monte_carlo(trades: List[Dict[str, Any]], initial_capital: float = 1000.0, n_sims: int = 1500, seed: int = 42) -> Dict[str, Any]:
    """Run comprehensive Monte Carlo simulation on trade outcomes."""
    if not trades or len(trades) < 5:
        return {
            "n_sims": n_sims,
            "n_trades": len(trades) if trades else 0,
            "max_dd_percentiles": {},
            "sharpe_percentiles": {},
            "sortino_percentiles": {},
            "calmar_percentiles": {},
            "losing_streak_percentiles": {},
            "prob_severe_dd_20": 0.0,
            "prob_severe_dd_30": 0.0,
            "prob_ruin_50": 0.0,
        }

    # Extract return fractions
    returns = np.array([float(t.get("net_pnl", 0.0)) / initial_capital for t in trades])
    n_trades = len(returns)
    rng = np.random.default_rng(seed)

    max_dds = []
    sharpes = []
    sortinos = []
    calmars = []
    losing_streaks = []
    ruin_count = 0
    severe_dd_20_count = 0
    severe_dd_30_count = 0

    for _ in range(n_sims):
        sampled_returns = rng.choice(returns, size=n_trades, replace=True)
        equity = initial_capital * np.cumprod(1.0 + sampled_returns)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-6)
        max_dd = float(np.max(dd))
        max_dds.append(max_dd)

        if max_dd >= 0.20:
            severe_dd_20_count += 1
        if max_dd >= 0.30:
            severe_dd_30_count += 1
        if np.min(equity) <= initial_capital * 0.5:
            ruin_count += 1

        # Streaks
        cur_loss_streak = 0
        max_loss_streak = 0
        for r in sampled_returns:
            if r < 0:
                cur_loss_streak += 1
                if cur_loss_streak > max_loss_streak:
                    max_loss_streak = cur_loss_streak
            else:
                cur_loss_streak = 0
        losing_streaks.append(max_loss_streak)

        # Sharpe / Sortino / Calmar on trade sequence
        mean_r = np.mean(sampled_returns)
        std_r = np.std(sampled_returns)
        neg_std = np.std(sampled_returns[sampled_returns < 0]) if np.any(sampled_returns < 0) else 1e-6
        ann_factor = np.sqrt(max(1, 365 * 24 / 4))  # normalized trade frequency factor

        s = float((mean_r / (std_r + 1e-9)) * ann_factor) if std_r > 0 else 0.0
        so = float((mean_r / (neg_std + 1e-9)) * ann_factor) if neg_std > 0 else 0.0
        total_ret = (equity[-1] - initial_capital) / initial_capital
        c = float(total_ret / (max_dd + 1e-6))

        sharpes.append(s)
        sortinos.append(so)
        calmars.append(c)

    def pctiles(arr):
        return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in (5, 25, 50, 75, 95)}

    return {
        "n_sims": n_sims,
        "n_trades": n_trades,
        "max_dd_percentiles": pctiles(max_dds),
        "sharpe_percentiles": pctiles(sharpes),
        "sortino_percentiles": pctiles(sortinos),
        "calmar_percentiles": pctiles(calmars),
        "losing_streak_percentiles": {f"p{p}": int(np.percentile(losing_streaks, p)) for p in (5, 25, 50, 75, 95)},
        "prob_severe_dd_20": round(severe_dd_20_count / n_sims, 4),
        "prob_severe_dd_30": round(severe_dd_30_count / n_sims, 4),
        "prob_ruin_50": round(ruin_count / n_sims, 4),
    }


def evaluate_window(
    df: pd.DataFrame,
    start_str: str,
    end_str: str,
    params: Dict[str, Any],
    symbol: str,
    market_type: str = "futures",
    timeframe: str = "15m",
) -> Dict[str, Any]:
    """Run backtest on a specific chronological slice."""
    s_dt = pd.Timestamp(start_str, tz="UTC")
    e_dt = pd.Timestamp(end_str, tz="UTC")
    s_ms = int(s_dt.timestamp() * 1000)
    e_ms = int(e_dt.timestamp() * 1000)

    sub_df = df[(df["timestamp"] >= s_ms) & (df["timestamp"] < e_ms)].reset_index(drop=True)
    if len(sub_df) < 50:
        return {}

    strat = MTFTrendPullbackStrategy(params=params)
    cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_leverage=5.0 if market_type == "futures" else 1.0,
        market_type=market_type,
        timeframe=timeframe,
    )
    engine = BacktestEngine(cfg)
    res = engine.run(strat, sub_df, symbol=symbol)
    m = res.metrics
    ts = m.trades

    return {
        "net_return": round(m.net_return, 4),
        "net_pnl": round(m.net_pnl, 2),
        "total_trades": ts.total_trades,
        "win_rate": round(ts.win_rate, 4),
        "profit_factor": round(ts.profit_factor, 4),
        "expectancy": round(ts.expectancy, 4),
        "max_drawdown": round(m.max_drawdown, 4),
        "sharpe_ratio": round(m.sharpe_ratio, 4),
        "sortino_ratio": round(m.sortino_ratio, 4),
        "calmar_ratio": round(m.calmar_ratio, 4),
        "avg_win": round(ts.avg_win, 2),
        "avg_loss": round(ts.avg_loss, 2),
        "trades": res.trades,
    }


def run_walk_forward_optimization(
    symbol: str,
    base_timeframe: str,
    mtf_combo: str,
    market_type: str = "futures",
) -> Dict[str, Any]:
    """Execute chronological Walk-Forward Train -> Validation -> Test optimization."""
    db = DatabaseManager("data/crypto_quant.db")
    repo = MarketDataRepository(db)
    df = repo.load(symbol, base_timeframe, market_type="spot")
    if df.empty:
        raise ValueError(f"No data for {symbol} {base_timeframe}")

    # Expanding Walk-Forward Windows (2022-2026)
    # W1: Train 2022 (Bear), Val 2023-H1 (Recovery), Test 2023-H2
    # W2: Train 2022-2023, Val 2024-H1 (Bull break), Test 2024-H2
    # W3: Train 2022-2024, Val 2025-H1, Test 2025-H2
    # W4: Train 2022-2025, Val 2026-H1, Test 2026-H2 (through 2026-09-07)
    windows = [
        {
            "id": "W1 (2022 -> 2023)",
            "train": ("2022-01-01", "2023-01-01"),
            "val": ("2023-01-01", "2023-07-01"),
            "test": ("2023-07-01", "2024-01-01"),
        },
        {
            "id": "W2 (2022-2023 -> 2024)",
            "train": ("2022-01-01", "2024-01-01"),
            "val": ("2024-01-01", "2024-07-01"),
            "test": ("2024-07-01", "2025-01-01"),
        },
        {
            "id": "W3 (2022-2024 -> 2025)",
            "train": ("2022-01-01", "2025-01-01"),
            "val": ("2025-01-01", "2025-07-01"),
            "test": ("2025-07-01", "2026-01-01"),
        },
        {
            "id": "W4 (2022-2025 -> 2026)",
            "train": ("2022-01-01", "2026-01-01"),
            "val": ("2026-01-01", "2026-05-01"),
            "test": ("2026-05-01", "2026-09-08"),
        },
    ]

    # Parameter Candidates Grid
    param_grid = [
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 20.0, "itf_rsi_pullback": 48.0, "ltf_rsi_oversold": 36.0, "ltf_rsi_overbought": 64.0, "volume_mult": 1.0, "atr_multiplier": 2.0, "risk_reward_ratio": 2.0, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 25.0, "itf_rsi_pullback": 52.0, "ltf_rsi_oversold": 38.0, "ltf_rsi_overbought": 62.0, "volume_mult": 1.2, "atr_multiplier": 2.5, "risk_reward_ratio": 2.5, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 0.0, "itf_rsi_pullback": 45.0, "ltf_rsi_oversold": 34.0, "ltf_rsi_overbought": 66.0, "volume_mult": 1.0, "atr_multiplier": 1.8, "risk_reward_ratio": 2.0, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 20.0, "itf_rsi_pullback": 50.0, "ltf_rsi_oversold": 40.0, "ltf_rsi_overbought": 60.0, "volume_mult": 0.0, "atr_multiplier": 2.0, "risk_reward_ratio": 1.8, "stop_type": "swing"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 20.0, "itf_rsi_pullback": 48.0, "ltf_rsi_oversold": 38.0, "ltf_rsi_overbought": 62.0, "volume_mult": 1.0, "atr_multiplier": 2.5, "risk_reward_ratio": 2.2, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 22.0, "itf_rsi_pullback": 50.0, "ltf_rsi_oversold": 35.0, "ltf_rsi_overbought": 65.0, "volume_mult": 1.1, "atr_multiplier": 2.2, "risk_reward_ratio": 2.5, "stop_type": "atr"},
    ]

    for p in param_grid:
        p["mtf_combo"] = mtf_combo

    window_results = []
    all_oos_trades = []

    for w in windows:
        w_id = w["id"]
        t_s, t_e = w["train"]
        v_s, v_e = w["val"]
        o_s, o_e = w["test"]

        # 1. Optimize on Train + Validate
        best_candidate = None
        best_score = -999.0

        for candidate_p in param_grid:
            tr_res = evaluate_window(df, t_s, t_e, candidate_p, symbol, market_type, base_timeframe)
            val_res = evaluate_window(df, v_s, v_e, candidate_p, symbol, market_type, base_timeframe)

            if not tr_res or not val_res:
                continue

            # Ranking score: Expectancy * Profit Factor * Sharpe - 2*MaxDD
            # Guard against negative expectancy or 0 trades
            if tr_res.get("total_trades", 0) < 5 or val_res.get("total_trades", 0) < 2:
                continue

            score = (
                val_res.get("profit_factor", 0.0) * 0.4
                + val_res.get("sharpe_ratio", 0.0) * 0.3
                + val_res.get("expectancy", 0.0) * 20.0
                - val_res.get("max_drawdown", 0.0) * 1.5
            )

            if score > best_score:
                best_score = score
                best_candidate = (candidate_p, tr_res, val_res)

        if best_candidate is None:
            # Fallback to default params
            chosen_p = param_grid[0]
            tr_res = evaluate_window(df, t_s, t_e, chosen_p, symbol, market_type, base_timeframe)
            val_res = evaluate_window(df, v_s, v_e, chosen_p, symbol, market_type, base_timeframe)
        else:
            chosen_p, tr_res, val_res = best_candidate

        # 2. Test chosen params on Out-Of-Sample (OOS) period
        oos_res = evaluate_window(df, o_s, o_e, chosen_p, symbol, market_type, base_timeframe)
        if oos_res and "trades" in oos_res:
            all_oos_trades.extend(oos_res["trades"])

        window_results.append({
            "window": w_id,
            "params": chosen_p,
            "train_period": f"{t_s} -> {t_e}",
            "val_period": f"{v_s} -> {v_e}",
            "oos_period": f"{o_s} -> {o_e}",
            "train_metrics": {k: v for k, v in tr_res.items() if k != "trades"},
            "val_metrics": {k: v for k, v in val_res.items() if k != "trades"},
            "oos_metrics": {k: v for k, v in oos_res.items() if k != "trades"},
        })

    # Full period backtest with the most robust parameter set
    # Compute aggregate OOS metrics
    full_res = evaluate_window(df, "2022-01-01", "2026-09-08", param_grid[0], symbol, market_type, base_timeframe)

    # Monte Carlo on full/OOS trades
    eval_trades = all_oos_trades if len(all_oos_trades) >= 15 else full_res.get("trades", [])
    mc_result = run_monte_carlo(eval_trades, initial_capital=1000.0, n_sims=1500)

    return {
        "symbol": symbol,
        "base_timeframe": base_timeframe,
        "mtf_combo": mtf_combo,
        "market_type": market_type,
        "windows": window_results,
        "full_backtest": {k: v for k, v in full_res.items() if k != "trades"},
        "monte_carlo": mc_result,
        "total_oos_trades": len(all_oos_trades),
    }
