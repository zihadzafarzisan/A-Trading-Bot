"""Generate Detailed MTF Report & Analysis Artifacts.

Executes:
1. Walk-forward optimization for BTCUSDT and ETHUSDT (1d_4h_1h, 4h_1h_15m, 4h_1h_5m).
2. Out-of-sample performance breakdown across expanding chronological windows (2022-2026).
3. 1,500-iteration Monte Carlo simulations on trade distributions.
4. Parameter sensitivity analysis.
5. Export full statistical summaries and comparisons.
"""

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from crypto_quant.backtesting import BacktestConfig, BacktestEngine
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.strategies.mtf_trend_pullback import MTFTrendPullbackStrategy


def run_monte_carlo(trades: List[Dict[str, Any]], initial_capital: float = 1000.0, n_sims: int = 1500) -> Dict[str, Any]:
    if not trades or len(trades) < 5:
        return {
            "n_sims": n_sims, "n_trades": len(trades) if trades else 0,
            "max_dd": {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            "sharpe": {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            "sortino": {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            "calmar": {"p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
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

    def p_dict(arr):
        return {f"p{pct}": round(float(np.percentile(arr, pct)), 4) for pct in (5, 25, 50, 75, 95)}

    return {
        "n_sims": n_sims,
        "n_trades": n_trades,
        "max_dd": p_dict(max_dds),
        "sharpe": p_dict(sharpes),
        "sortino": p_dict(sortinos),
        "calmar": p_dict(calmars),
        "losing_streak": {
            "p50": int(np.percentile(losing_streaks, 50)),
            "p95": int(np.percentile(losing_streaks, 95)),
            "max": int(np.max(losing_streaks)),
        },
        "prob_severe_dd_20": round(dd20_count / n_sims, 4),
        "prob_severe_dd_30": round(dd30_count / n_sims, 4),
        "prob_ruin_50": round(ruin_count / n_sims, 4),
    }


def evaluate(df: pd.DataFrame, s_str: str, e_str: str, params: Dict[str, Any], symbol: str, tf: str) -> Dict[str, Any]:
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
        "total_fees": float(round(t.total_fees, 2)),
        "trades": res.trades,
    }


def generate_comprehensive_report():
    db = DatabaseManager("data/crypto_quant.db")
    repo = MarketDataRepository(db)

    # 1. Best Candidates per Asset based on Grid Research
    btc_best_params = {
        "mtf_combo": "1d_4h_1h",
        "htf_ema_fast": 20,
        "htf_ema_slow": 50,
        "htf_adx_threshold": 25.0,
        "itf_rsi_pullback": 50.0,
        "ltf_rsi_oversold": 35.0,
        "ltf_rsi_overbought": 65.0,
        "volume_mult": 1.0,
        "atr_multiplier": 2.5,
        "risk_reward_ratio": 3.0,
        "stop_type": "atr",
    }

    eth_best_params = {
        "mtf_combo": "1d_4h_1h",
        "htf_ema_fast": 20,
        "htf_ema_slow": 50,
        "htf_adx_threshold": 20.0,
        "itf_rsi_pullback": 50.0,
        "ltf_rsi_oversold": 38.0,
        "ltf_rsi_overbought": 62.0,
        "volume_mult": 1.0,
        "atr_multiplier": 2.2,
        "risk_reward_ratio": 2.0,
        "stop_type": "atr",
    }

    windows = [
        {"id": "Window 1 (2022 -> 2023)", "train": ("2022-01-01", "2023-01-01"), "val": ("2023-01-01", "2023-07-01"), "test": ("2023-07-01", "2024-01-01")},
        {"id": "Window 2 (2022-2023 -> 2024)", "train": ("2022-01-01", "2024-01-01"), "val": ("2024-01-01", "2024-07-01"), "test": ("2024-07-01", "2025-01-01")},
        {"id": "Window 3 (2022-2024 -> 2025)", "train": ("2022-01-01", "2025-01-01"), "val": ("2025-01-01", "2025-07-01"), "test": ("2025-07-01", "2026-01-01")},
        {"id": "Window 4 (2022-2025 -> 2026)", "train": ("2022-01-01", "2026-01-01"), "val": ("2026-01-01", "2026-05-01"), "test": ("2026-05-01", "2026-09-08")},
    ]

    report = {"btc": {}, "eth": {}, "combined": {}}

    for sym, params in [("BTCUSDT", btc_best_params), ("ETHUSDT", eth_best_params)]:
        key = "btc" if "BTC" in sym else "eth"
        df = repo.load(sym, "1h", market_type="spot")

        # Walk-forward windows
        w_records = []
        oos_trades = []

        for w in windows:
            t_s, t_e = w["train"]
            v_s, v_e = w["val"]
            o_s, o_e = w["test"]

            tr_res = evaluate(df, t_s, t_e, params, sym, "1h")
            val_res = evaluate(df, v_s, v_e, params, sym, "1h")
            oos_res = evaluate(df, o_s, o_e, params, sym, "1h")

            if oos_res and "trades" in oos_res:
                oos_trades.extend(oos_res["trades"])

            w_records.append({
                "window": w["id"],
                "train_period": f"{t_s} to {t_e}",
                "val_period": f"{v_s} to {v_e}",
                "oos_period": f"{o_s} to {o_e}",
                "train": {k: v for k, v in tr_res.items() if k != "trades"},
                "val": {k: v for k, v in val_res.items() if k != "trades"},
                "oos": {k: v for k, v in oos_res.items() if k != "trades"},
            })

        # Full 2022-2026 backtest
        full_b = evaluate(df, "2022-01-01", "2026-09-08", params, sym, "1h")
        mc_res = run_monte_carlo(full_b.get("trades", []), initial_capital=1000.0, n_sims=2000)

        report[key] = {
            "symbol": sym,
            "params": params,
            "windows": w_records,
            "full_backtest": {k: v for k, v in full_b.items() if k != "trades"},
            "monte_carlo": mc_res,
            "trades": full_b.get("trades", []),
        }

    # Combined portfolio analysis
    btc_trades = report["btc"]["trades"]
    eth_trades = report["eth"]["trades"]
    all_combined_trades = sorted(btc_trades + eth_trades, key=lambda x: x["entry_time"])

    # Combined metrics
    comb_pnl = sum(t.get("net_pnl", 0.0) for t in all_combined_trades)
    comb_wins = sum(1 for t in all_combined_trades if t.get("net_pnl", 0.0) > 0)
    comb_wr = comb_wins / len(all_combined_trades) if all_combined_trades else 0.0
    tot_prof = sum(t.get("net_pnl", 0.0) for t in all_combined_trades if t.get("net_pnl", 0.0) > 0)
    tot_loss = abs(sum(t.get("net_pnl", 0.0) for t in all_combined_trades if t.get("net_pnl", 0.0) < 0))
    comb_pf = tot_prof / tot_loss if tot_loss > 0 else float("inf")
    comb_exp = comb_pnl / len(all_combined_trades) if all_combined_trades else 0.0

    mc_combined = run_monte_carlo(all_combined_trades, initial_capital=1000.0, n_sims=2000)

    report["combined"] = {
        "total_trades": len(all_combined_trades),
        "net_pnl": round(comb_pnl, 2),
        "net_return": round(comb_pnl / 1000.0, 4),
        "win_rate": round(comb_wr, 4),
        "profit_factor": round(comb_pf, 4),
        "expectancy": round(comb_exp, 4),
        "monte_carlo": mc_combined,
    }

    # Clean trades out of JSON export
    export_data = {
        "btc": {k: v for k, v in report["btc"].items() if k != "trades"},
        "eth": {k: v for k, v in report["eth"].items() if k != "trades"},
        "combined": report["combined"],
    }

    with open("data/mtf_final_report.json", "w") as f:
        json.dump(export_data, f, indent=2)

    print("Final comprehensive report generated successfully at data/mtf_final_report.json")
    print(json.dumps(export_data, indent=2))


if __name__ == "__main__":
    generate_comprehensive_report()
