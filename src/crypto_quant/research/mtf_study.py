"""Comprehensive Multi-Timeframe Optimization & Walk-Forward Comparison.

Runs BTCUSDT & ETHUSDT across:
1. 4H -> 1H -> 15M
2. 4H -> 1H -> 5M
3. 1D -> 4H -> 1H

Calculates metrics, Walk-Forward stability, Monte Carlo uncertainty, and exports full reports.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from crypto_quant.backtesting import BacktestConfig, BacktestEngine
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.strategies.mtf_trend_pullback import MTFTrendPullbackStrategy


def run_mc_sim(trades: List[Dict[str, Any]], initial_capital: float = 1000.0, n_sims: int = 1500) -> Dict[str, Any]:
    """Run Monte Carlo simulation on trade outcomes."""
    if not trades or len(trades) < 5:
        return {
            "n_sims": n_sims, "n_trades": len(trades) if trades else 0,
            "max_dd": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
            "sharpe": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
            "sortino": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
            "calmar": {"p5": 0.0, "p50": 0.0, "p95": 0.0},
            "losing_streak": {"p50": 0, "p95": 0, "max": 0},
            "prob_severe_dd_20": 0.0, "prob_severe_dd_30": 0.0, "prob_ruin_50": 0.0,
        }

    returns = np.array([float(t.get("net_pnl", 0.0)) / initial_capital for t in trades])
    n_trades = len(returns)
    rng = np.random.default_rng(42)

    max_dds = []
    sharpes = []
    sortinos = []
    calmars = []
    losing_streaks = []
    ruin_count = 0
    dd20_count = 0
    dd30_count = 0

    for _ in range(n_sims):
        sampled = rng.choice(returns, size=n_trades, replace=True)
        equity = initial_capital * np.cumprod(1.0 + sampled)
        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / np.maximum(peak, 1e-6)
        mdd = float(np.max(dd))
        max_dds.append(mdd)

        if mdd >= 0.20:
            dd20_count += 1
        if mdd >= 0.30:
            dd30_count += 1
        if np.min(equity) <= initial_capital * 0.5:
            ruin_count += 1

        cur_l = max_l = 0
        for r in sampled:
            if r < 0:
                cur_l += 1
                if cur_l > max_l:
                    max_l = cur_l
            else:
                cur_l = 0
        losing_streaks.append(max_l)

        mean_r = np.mean(sampled)
        std_r = np.std(sampled)
        neg_std = np.std(sampled[sampled < 0]) if np.any(sampled < 0) else 1e-6
        ann_factor = np.sqrt(max(1, 365 * 24 / 4))

        s = float((mean_r / (std_r + 1e-9)) * ann_factor) if std_r > 0 else 0.0
        so = float((mean_r / (neg_std + 1e-9)) * ann_factor) if neg_std > 0 else 0.0
        tot_ret = (equity[-1] - initial_capital) / initial_capital
        c = float(tot_ret / (mdd + 1e-6))

        sharpes.append(s)
        sortinos.append(so)
        calmars.append(c)

    def p(arr):
        return {f"p{pct}": round(float(np.percentile(arr, pct)), 4) for pct in (5, 25, 50, 75, 95)}

    return {
        "n_sims": n_sims,
        "n_trades": n_trades,
        "max_dd": p(max_dds),
        "sharpe": p(sharpes),
        "sortino": p(sortinos),
        "calmar": p(calmars),
        "losing_streak": {
            "p50": int(np.percentile(losing_streaks, 50)),
            "p95": int(np.percentile(losing_streaks, 95)),
            "max": int(np.max(losing_streaks)),
        },
        "prob_severe_dd_20": round(dd20_count / n_sims, 4),
        "prob_severe_dd_30": round(dd30_count / n_sims, 4),
        "prob_ruin_50": round(ruin_count / n_sims, 4),
    }


def eval_slice(df: pd.DataFrame, s_str: str, e_str: str, params: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]:
    """Run backtest on chronological slice."""
    s_ms = int(pd.Timestamp(s_str, tz="UTC").timestamp() * 1000)
    e_ms = int(pd.Timestamp(e_str, tz="UTC").timestamp() * 1000)
    sub = df[(df["timestamp"] >= s_ms) & (df["timestamp"] < e_ms)].reset_index(drop=True)
    if len(sub) < 50:
        return {}

    strat = MTFTrendPullbackStrategy(params=params)
    cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_leverage=5.0,
        market_type="futures",
        timeframe=tf,
    )
    engine = BacktestEngine(cfg)
    res = engine.run(strat, sub, symbol=symbol)
    m = res.metrics
    t = m.trades

    return {
        "net_return": float(round(m.net_return, 4)),
        "net_pnl": float(round(m.net_pnl, 2)),
        "total_trades": int(t.total_trades),
        "win_rate": float(round(t.win_rate, 4)),
        "profit_factor": float(round(t.profit_factor, 4)),
        "expectancy": float(round(t.expectancy, 4)),
        "max_drawdown": float(round(m.max_drawdown, 4)),
        "sharpe_ratio": float(round(m.sharpe_ratio, 4)),
        "sortino_ratio": float(round(m.sortino_ratio, 4)),
        "calmar_ratio": float(round(m.calmar_ratio, 4)),
        "avg_win": float(round(t.avg_win, 2)),
        "avg_loss": float(round(t.avg_loss, 2)),
        "trades": res.trades,
    }


def run_full_study():
    db = DatabaseManager("data/crypto_quant.db")
    repo = MarketDataRepository(db)

    combos = [
        ("1d_4h_1h", "1h"),
        ("4h_1h_15m", "15m"),
        ("4h_1h_5m", "5m"),
    ]

    symbols = ["BTCUSDT", "ETHUSDT"]

    # Refined parameter grid
    grid = [
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 22.0, "itf_rsi_pullback": 45.0, "ltf_rsi_oversold": 35.0, "ltf_rsi_overbought": 65.0, "volume_mult": 1.0, "atr_multiplier": 2.0, "risk_reward_ratio": 2.5, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 25.0, "itf_rsi_pullback": 48.0, "ltf_rsi_oversold": 38.0, "ltf_rsi_overbought": 62.0, "volume_mult": 1.2, "atr_multiplier": 2.2, "risk_reward_ratio": 2.5, "stop_type": "atr"},
        {"htf_ema_fast": 50, "htf_ema_slow": 100, "htf_adx_threshold": 20.0, "itf_rsi_pullback": 50.0, "ltf_rsi_oversold": 36.0, "ltf_rsi_overbought": 64.0, "volume_mult": 1.1, "atr_multiplier": 2.5, "risk_reward_ratio": 2.5, "stop_type": "atr"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 20.0, "itf_rsi_pullback": 45.0, "ltf_rsi_oversold": 32.0, "ltf_rsi_overbought": 68.0, "volume_mult": 1.0, "atr_multiplier": 1.8, "risk_reward_ratio": 2.0, "stop_type": "swing"},
        {"htf_ema_fast": 20, "htf_ema_slow": 50, "htf_adx_threshold": 25.0, "itf_rsi_pullback": 50.0, "ltf_rsi_oversold": 35.0, "ltf_rsi_overbought": 65.0, "volume_mult": 1.0, "atr_multiplier": 2.5, "risk_reward_ratio": 3.0, "stop_type": "atr"},
    ]

    windows = [
        {"id": "W1 (2022 -> 2023)", "train": ("2022-01-01", "2023-01-01"), "val": ("2023-01-01", "2023-07-01"), "test": ("2023-07-01", "2024-01-01")},
        {"id": "W2 (2022-2023 -> 2024)", "train": ("2022-01-01", "2024-01-01"), "val": ("2024-01-01", "2024-07-01"), "test": ("2024-07-01", "2025-01-01")},
        {"id": "W3 (2022-2024 -> 2025)", "train": ("2022-01-01", "2025-01-01"), "val": ("2025-01-01", "2025-07-01"), "test": ("2025-07-01", "2026-01-01")},
        {"id": "W4 (2022-2025 -> 2026)", "train": ("2022-01-01", "2026-01-01"), "val": ("2026-01-01", "2026-05-01"), "test": ("2026-05-01", "2026-09-08")},
    ]

    all_study_results = {}

    for sym in symbols:
        all_study_results[sym] = {}
        for combo, tf in combos:
            print(f"\n=======================================================")
            print(f"EVALUATING: {sym} | Combo: {combo} | Base TF: {tf}")
            print(f"=======================================================")
            t0 = time.time()
            df = repo.load(sym, tf, market_type="spot")

            # Grid Search across train/val windows
            candidate_window_records = []
            oos_trades_collector = []

            for w in windows:
                w_id = w["id"]
                t_s, t_e = w["train"]
                v_s, v_e = w["val"]
                o_s, o_e = w["test"]

                best_p = None
                best_score = -999.0
                best_train_m = {}
                best_val_m = {}

                for p_idx, raw_p in enumerate(grid):
                    p = dict(raw_p)
                    p["mtf_combo"] = combo

                    tr = eval_slice(df, t_s, t_e, p, sym, tf)
                    va = eval_slice(df, v_s, v_e, p, sym, tf)

                    if not tr or not va:
                        continue

                    # Score on validation
                    exp = va.get("expectancy", 0.0)
                    pf = va.get("profit_factor", 0.0)
                    sh = va.get("sharpe_ratio", 0.0)
                    mdd = va.get("max_drawdown", 0.0)
                    n_t = va.get("total_trades", 0)

                    score = pf * 0.4 + sh * 0.3 + exp * 15.0 - mdd * 2.0
                    if score > best_score:
                        best_score = score
                        best_p = p
                        best_train_m = tr
                        best_val_m = va

                if best_p is None:
                    best_p = dict(grid[0])
                    best_p["mtf_combo"] = combo
                    best_train_m = eval_slice(df, t_s, t_e, best_p, sym, tf)
                    best_val_m = eval_slice(df, v_s, v_e, best_p, sym, tf)

                # Test on OOS
                oos_m = eval_slice(df, o_s, o_e, best_p, sym, tf)
                if oos_m and "trades" in oos_m:
                    oos_trades_collector.extend(oos_m["trades"])

                candidate_window_records.append({
                    "window": w_id,
                    "params": best_p,
                    "train": {k: v for k, v in best_train_m.items() if k != "trades"},
                    "val": {k: v for k, v in best_val_m.items() if k != "trades"},
                    "oos": {k: v for k, v in oos_m.items() if k != "trades"},
                })
                print(f"  {w_id} -> Train PF: {best_train_m.get('profit_factor', 0):.2f} | Val PF: {best_val_m.get('profit_factor', 0):.2f} | OOS PF: {oos_m.get('profit_factor', 0):.2f} | OOS Trades: {oos_m.get('total_trades', 0)}")

            # Full Period Backtest (2022-01-01 -> 2026-09-08) with most stable parameter set
            rep_p = candidate_window_records[-1]["params"] if candidate_window_records else grid[0]
            full_b = eval_slice(df, "2022-01-01", "2026-09-08", rep_p, sym, tf)
            full_trades = full_b.get("trades", [])

            # Monte Carlo simulation on actual trades (OOS + Full)
            mc_eval_trades = oos_trades_collector if len(oos_trades_collector) >= 15 else full_trades
            mc_stats = run_mc_sim(mc_eval_trades, initial_capital=1000.0, n_sims=1500)

            dur = time.time() - t0
            print(f"  -> Full Period Net Ret: {full_b.get('net_return', 0)*100:.1f}% | PF: {full_b.get('profit_factor', 0):.2f} | Sharpe: {full_b.get('sharpe_ratio', 0):.2f} | MaxDD: {full_b.get('max_drawdown', 0)*100:.1f}% | Trades: {full_b.get('total_trades', 0)} ({dur:.1f}s)")
            print(f"  -> Monte Carlo 95th MaxDD: {mc_stats['max_dd'].get('p95', 0)*100:.1f}% | Losing Streak p95: {mc_stats['losing_streak'].get('p95', 0)} | Ruin Prob: {mc_stats['prob_ruin_50']*100:.2f}%")

            all_study_results[sym][combo] = {
                "symbol": sym,
                "combo": combo,
                "base_timeframe": tf,
                "representative_params": rep_p,
                "windows": candidate_window_records,
                "full_backtest": {k: v for k, v in full_b.items() if k != "trades"},
                "monte_carlo": mc_stats,
                "total_oos_trades": len(oos_trades_collector),
            }

    # Save to disk as clean JSON artifact
    out_file = "data/mtf_study_results.json"
    with open(out_file, "w") as f:
        json.dump(all_study_results, f, indent=2)
    print(f"\nSaved full results to {out_file}")


if __name__ == "__main__":
    run_full_study()
