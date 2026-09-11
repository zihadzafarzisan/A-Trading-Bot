"""Execute Stage VWAP-A baseline architecture screening across 4 exact experiments.

Protocol: PROTOCOL-VWAP-MR-V1.0
Target Strategy: VWAPBollingerMRStrategy (slug: 'vwap_bollinger_mr')
Governance: Pre-OOS baseline screening only (2020/2022 -> 2025). 2026 data is strictly locked and excluded.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from crypto_quant.backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.strategies import create_strategy

# Protocol-Declared Exact Baseline Parameters (Zero Optimization)
VWAP_BASELINE_PARAMS: Dict[str, Any] = {
    "adx_max_thresh": 22.0,
    "bb_period": 20,
    "bb_std": 2.0,
    "vwap_period": 20,
    "rsi_period": 14,
    "rsi_oversold": 35.0,
    "rsi_overbought": 65.0,
    "atr_multiplier": 2.0,
    "risk_reward_ratio": 2.0,
    "stop_type": "atr",
}

# Standardized Execution Configuration
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_POSITIONS = 3
MAX_LEVERAGE = 5.0
MARKET_TYPE = "futures"

# Mandatory 2026 OOS Protection Boundary: 2025-12-31 23:59:59 UTC (2026 strictly excluded)
PRE_OOS_END_TIMESTAMP_MS = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)

# Exact Four Authorized Stage-A Screening Experiments
EXPERIMENTS = [
    {
        "exp_id": "EXP_VWAPA_BTC_4H",
        "symbol": "BTCUSDT",
        "timeframe": "4h",
        "start_str": "2020-01-01 00:00:00",
        "end_str": "2025-12-31 23:59:59",
    },
    {
        "exp_id": "EXP_VWAPA_BTC_1H",
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "start_str": "2020-01-01 00:00:00",
        "end_str": "2025-12-31 23:59:59",
    },
    {
        "exp_id": "EXP_VWAPA_ETH_4H",
        "symbol": "ETHUSDT",
        "timeframe": "4h",
        "start_str": "2022-01-01 00:00:00",
        "end_str": "2025-12-31 23:59:59",
    },
    {
        "exp_id": "EXP_VWAPA_ETH_1H",
        "symbol": "ETHUSDT",
        "timeframe": "1h",
        "start_str": "2022-01-01 00:00:00",
        "end_str": "2025-12-31 23:59:59",
    },
]


def run_stage_vwap_a(db_path: str = "data/crypto_quant.db") -> List[Dict[str, Any]]:
    """Execute the Stage-A screening experiments and generate output artifacts."""
    db = DatabaseManager(db_path)
    repo = MarketDataRepository(db)

    results: List[Dict[str, Any]] = []

    print("=" * 115)
    print("EXECUTING STAGE VWAP-A BASELINE SCREENING (PROTOCOL-VWAP-MR-V1.0)")
    print(f"Data Boundary: Pre-OOS Only (strictly < 2026-01-01 00:00:00 UTC)")
    print(f"Strategy: vwap_bollinger_mr | Baseline Params: {VWAP_BASELINE_PARAMS}")
    print("=" * 115)

    for exp in EXPERIMENTS:
        exp_id = exp["exp_id"]
        sym = exp["symbol"]
        tf = exp["timeframe"]
        s_ms = int(pd.Timestamp(exp["start_str"], tz="UTC").timestamp() * 1000)

        # 1. Load data via repository
        df_full = repo.load(sym, tf, market_type="spot")

        # 2. Strict Causal Truncation to Pre-OOS Boundary
        stage_a_df = df_full[(df_full["timestamp"] >= s_ms) & (df_full["timestamp"] < PRE_OOS_END_TIMESTAMP_MS)].reset_index(drop=True)

        # 2026 OOS Protection Assertion
        max_ts = stage_a_df["timestamp"].max()
        assert max_ts < PRE_OOS_END_TIMESTAMP_MS, f"OOS Violation in {exp_id}: max_ts {max_ts} >= {PRE_OOS_END_TIMESTAMP_MS}"

        # 3. Instantiate Strategy with Frozen Baseline Parameters
        params = dict(VWAP_BASELINE_PARAMS)
        strat_obj = create_strategy("vwap_bollinger_mr", params=params)

        # 4. Standardized Execution Configuration
        cfg = BacktestConfig(
            initial_capital=INITIAL_CAPITAL,
            risk_per_trade=RISK_PER_TRADE,
            max_open_positions=MAX_OPEN_POSITIONS,
            max_leverage=MAX_LEVERAGE,
            market_type=MARKET_TYPE,
            timeframe=tf,
            execution=ExecutionConfig(market_type=MARKET_TYPE),
        )

        # 5. Run Backtest
        engine = BacktestEngine(cfg)
        res = engine.run(strat_obj, stage_a_df, symbol=sym)
        m = res.metrics
        t = m.trades
        closed = res.trades

        # Directional & Cost Breakdowns
        long_trades = [tr for tr in closed if tr["direction"] == "long"]
        short_trades = [tr for tr in closed if tr["direction"] == "short"]
        wins = [tr for tr in closed if tr["net_pnl"] > 0]
        losses = [tr for tr in closed if tr["net_pnl"] < 0]

        gross_profit = sum(tr["net_pnl"] for tr in wins)
        gross_loss = abs(sum(tr["net_pnl"] for tr in losses))
        pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        # 6. Stage-A Screening Gate Rules
        n_rule_pass = bool(t.total_trades >= 30)
        catastrophic_fail = bool((pf < 0.75) and (m.max_drawdown > 0.75))
        stage_a_pass = bool(n_rule_pass and (not catastrophic_fail))

        # Evidence Classification
        if t.total_trades < 15:
            evidence_class = "INSUFFICIENT_EVIDENCE"
        elif 15 <= t.total_trades < 30:
            evidence_class = "MARGINAL_EVIDENCE"
        else:
            evidence_class = "ADEQUATE_EVIDENCE"

        rec = {
            "experiment_id": exp_id,
            "protocol": "PROTOCOL-VWAP-MR-V1.0",
            "strategy": "vwap_bollinger_mr",
            "symbol": sym,
            "timeframe": tf,
            "data_start": exp["start_str"],
            "data_end": exp["end_str"],
            "bars": len(stage_a_df),
            "params": params,
            "total_trades": int(t.total_trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "win_rate": round(float(t.win_rate), 4),
            "profit_factor": round(float(pf), 4),
            "expectancy": round(float(t.expectancy), 4),
            "net_pnl": round(float(m.net_pnl), 2),
            "net_return": round(float(m.net_return), 4),
            "max_drawdown": round(float(m.max_drawdown), 4),
            "sharpe": round(float(m.sharpe_ratio), 4),
            "sortino": round(float(m.sortino_ratio), 4),
            "calmar": round(float(m.calmar_ratio), 4),
            "gross_profit": round(float(gross_profit), 2),
            "gross_loss": round(float(gross_loss), 2),
            "fees": round(float(t.total_fees), 2),
            "slippage": round(float(t.total_slippage), 2),
            "funding": round(float(t.total_funding), 2),
            "liquidations": int(t.liquidations),
            "stage_a_n_ge_30": n_rule_pass,
            "stage_a_catastrophic": catastrophic_fail,
            "stage_a_status": "PASS" if stage_a_pass else "FAIL",
            "evidence_class": evidence_class,
        }
        results.append(rec)
        print(f"[{len(results)}/4] [{rec['stage_a_status']}] {exp_id:<22} -> Trades={t.total_trades:>4} | WR={t.win_rate*100:>5.1f}% | PF={pf:>5.2f} | Exp=${t.expectancy:>6.2f} | Ret={m.net_return*100:>6.1f}% | MaxDD={m.max_drawdown*100:>5.1f}% | Evidence={evidence_class}")

    # 7. Timeframe Combined Advancement Decisions
    tf_decisions = {}
    for tf_candidate in ["4h", "1h"]:
        tf_runs = [r for r in results if r["timeframe"] == tf_candidate]
        btc_run = next((r for r in tf_runs if r["symbol"] == "BTCUSDT"), None)
        eth_run = next((r for r in tf_runs if r["symbol"] == "ETHUSDT"), None)

        both_passed = bool(btc_run and eth_run and btc_run["stage_a_status"] == "PASS" and eth_run["stage_a_status"] == "PASS")
        advancement = "ADVANCES_TO_STAGE_B" if both_passed else "ELIMINATED"
        tf_decisions[tf_candidate] = {
            "btc_status": btc_run["stage_a_status"] if btc_run else "MISSING",
            "eth_status": eth_run["stage_a_status"] if eth_run else "MISSING",
            "advancement": advancement,
        }

    # 8. Export JSON Artifact
    json_path = "data/vwap_mr_stage_a_screening.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 9. Export Markdown Diagnostic Artifact
    md_path = "data/vwap_mr_stage_a_diagnostic.md"
    generate_diagnostic_markdown(results, tf_decisions, md_path)

    print("\n" + "=" * 115)
    print(f"STAGE VWAP-A SCREENING COMPLETE. Artifacts stored:")
    print(f"  1. {json_path}")
    print(f"  2. {md_path}")
    print("=" * 115)

    return results


def generate_diagnostic_markdown(results: List[Dict[str, Any]], tf_decisions: Dict[str, Any], output_path: str) -> None:
    """Generate the formal Stage-A diagnostic report."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Stage VWAP-A Baseline Screening Diagnostic Report",
        "",
        f"**Protocol:** `PROTOCOL-VWAP-MR-V1.0`  ",
        f"**Strategy Family:** `VWAPBollingerMRStrategy` (`vwap_bollinger_mr`)  ",
        f"**Execution Date:** {now_utc}  ",
        f"**OOS Status:** 2026 dataset strictly locked and excluded (`timestamp < 2026-01-01 00:00:00 UTC`).",
        "",
        "---",
        "",
        "## 1. Baseline Screening Summary",
        "",
        "| Experiment ID | Symbol | TF | Bars | Trades ($N$) | Win Rate | Profit Factor | Expectancy | Net Return | Max DD | Evidence Class | Stage-A Result |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        lines.append(
            f"| `{r['experiment_id']}` | {r['symbol']} | {r['timeframe']} | {r['bars']:,} | {r['total_trades']} | "
            f"{r['win_rate']*100:.1f}% | {r['profit_factor']:.4f} | ${r['expectancy']:.2f} | "
            f"{r['net_return']*100:.1f}% | {r['max_drawdown']*100:.1f}% | `{r['evidence_class']}` | **{r['stage_a_status']}** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Timeframe Architecture Advancement Decisions",
        "",
        "Under `PROTOCOL-VWAP-MR-V1.0` Section 9, a candidate timeframe advances to Stage-B parameter calibration **only if both BTCUSDT and ETHUSDT pass all Stage-A screening rules** ($N \\ge 30$ and non-catastrophic).",
        "",
        "| Timeframe | BTC Baseline Status | ETH Baseline Status | Both Passed Stage-A? | Stage-B Advancement Decision |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])

    for tf, d in tf_decisions.items():
        both_str = "YES" if d["advancement"] == "ADVANCES_TO_STAGE_B" else "NO"
        lines.append(f"| **{tf.upper()}** | **{d['btc_status']}** | **{d['eth_status']}** | **{both_str}** | **{d['advancement']}** |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Standardized Execution & Accounting Audit",
        "",
        "* **Market Type:** Futures (5x leverage, isolated margin)",
        "* **Capital & Risk:** $1,000 starting equity, 1.0% risk per trade, max 3 open positions",
        "* **Fees & Friction:** 0.04% taker fee per side, dynamic tick slippage embedded in fills, 8h funding rate accounting",
        "* **Execution Timing:** Signal at bar $i$ close; fill at bar $i+1$ open; conservative stop-first intra-bar priority",
        "* **Accounting Identity:** $\\text{Net PnL} = \\text{Gross Profit} - \\text{Gross Loss} = \\text{Final Equity} - \\text{Initial Capital}$",
        "",
        "---",
        "",
        "## 4. 2026 OOS Protection Verification",
        "",
        "* **OOS Status:** Fully locked and untouched.",
        "* **Maximum Timestamp Loaded:** Strictly $< 2026-01-01 00:00:00 UTC$ across all 4 experiments.",
        "* **Parameter Tuning Status:** Zero parameters tuned during Stage-A.",
        "",
        "---",
        "*Diagnostic report generated by `scripts/run_stage_vwap_a.py`.*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_stage_vwap_a()
