"""Execute Stage VWAP-B controlled 48-combination parameter calibration.

Protocol: PROTOCOL-VWAP-MR-V1.0
Strategy: VWAPBollingerMRStrategy (slug: 'vwap_bollinger_mr')
Governance: Pre-OOS Train & 2025 Validation slices only. 2026 data is strictly locked and excluded.
"""

import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from crypto_quant.backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from crypto_quant.data.repository import MarketDataRepository
from crypto_quant.db.connection import DatabaseManager
from crypto_quant.strategies import create_strategy
from crypto_quant.validation.optimize_window import (
    MultiObjectiveFilters,
    MultiObjectiveWeights,
    score_metrics_multi_objective,
)

# Standardized Execution Configuration
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
MAX_OPEN_POSITIONS = 3
MAX_LEVERAGE = 5.0
MARKET_TYPE = "futures"

# Hard Gate Filters (Validation Slice)
HARD_FILTERS = MultiObjectiveFilters(min_trades=15, min_profit_factor=1.05, max_drawdown=0.40)
SCORE_WEIGHTS = MultiObjectiveWeights()

# Mandatory 2026 OOS Protection Boundary: 2025-12-31 23:59:59 UTC (2026 strictly excluded)
PRE_OOS_END_TIMESTAMP_MS = int(pd.Timestamp("2026-01-01 00:00:00", tz="UTC").timestamp() * 1000)

# Exact Slicing Boundaries
BTC_TRAIN_START_MS = int(pd.Timestamp("2020-01-01 00:00:00", tz="UTC").timestamp() * 1000)
ETH_TRAIN_START_MS = int(pd.Timestamp("2022-01-01 00:00:00", tz="UTC").timestamp() * 1000)
TRAIN_END_MS = int(pd.Timestamp("2024-12-31 23:59:59", tz="UTC").timestamp() * 1000)
VAL_START_MS = int(pd.Timestamp("2025-01-01 00:00:00", tz="UTC").timestamp() * 1000)
VAL_END_MS = int(pd.Timestamp("2025-12-31 23:59:59", tz="UTC").timestamp() * 1000)

# Strategy Baseline Configuration
BASELINE_PARAMS: Dict[str, Any] = {
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

# The 4 Authorized Asset / Timeframe Tracks
TRACKS = [
    {"symbol": "BTCUSDT", "timeframe": "4h", "train_start_ms": BTC_TRAIN_START_MS},
    {"symbol": "ETHUSDT", "timeframe": "4h", "train_start_ms": ETH_TRAIN_START_MS},
    {"symbol": "BTCUSDT", "timeframe": "1h", "train_start_ms": BTC_TRAIN_START_MS},
    {"symbol": "ETHUSDT", "timeframe": "1h", "train_start_ms": ETH_TRAIN_START_MS},
]


def generate_48_combinations() -> List[Dict[str, Any]]:
    """Construct the exact 48 authorized parameter combinations from PROTOCOL-VWAP-MR-V1.0."""
    adx_vals = [20.0, 22.0, 25.0]           # 3
    bb_std_vals = [2.0, 2.2]                 # 2
    rsi_pairs = [(35.0, 65.0), (30.0, 70.0)] # 2 (oversold, overbought)
    atr_m_vals = [1.8, 2.2]                  # 2
    rr_vals = [1.5, 2.0]                     # 2

    combos = []
    for adx_t in adx_vals:
        for bb_s in bb_std_vals:
            for rsi_os, rsi_ob in rsi_pairs:
                for atr_m in atr_m_vals:
                    for rr in rr_vals:
                        combos.append({
                            "adx_max_thresh": adx_t,
                            "bb_period": 20,
                            "bb_std": bb_s,
                            "vwap_period": 20,
                            "rsi_period": 14,
                            "rsi_oversold": rsi_os,
                            "rsi_overbought": rsi_ob,
                            "atr_multiplier": atr_m,
                            "risk_reward_ratio": rr,
                            "stop_type": "atr",
                        })

    assert len(combos) == 48, f"Expected exactly 48 combinations, got {len(combos)}"
    return combos


def in_grid_neighbors(target_params: Dict[str, Any], grid: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return immediate in-grid neighbors differing in exactly one optimized dimension."""
    neighbors = []
    keys_to_compare = ["adx_max_thresh", "bb_std", "rsi_oversold", "atr_multiplier", "risk_reward_ratio"]
    for other in grid:
        if other == target_params:
            continue
        diffs = sum(1 for k in keys_to_compare if other.get(k) != target_params.get(k))
        if diffs == 1:
            neighbors.append(other)
    return neighbors


def compute_in_grid_stability(
    best_candidate: Dict[str, Any],
    all_evaluations: List[Dict[str, Any]],
    grid: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute stability strictly over in-grid neighbors from the validation evaluations."""
    best_params = best_candidate["params"]
    best_score = best_candidate["validation"]["score"]

    eval_by_param_str = {json.dumps(e["params"], sort_keys=True): e for e in all_evaluations}
    neighbors_params = in_grid_neighbors(best_params, grid)

    neighbor_rows = []
    for np_dict in neighbors_params:
        p_str = json.dumps(np_dict, sort_keys=True)
        if p_str in eval_by_param_str:
            ne = eval_by_param_str[p_str]
            n_val = ne["validation"]
            neighbor_rows.append({
                "combo_id": ne["combo_id"],
                "params": np_dict,
                "score": n_val["score"],
                "win_rate": n_val["win_rate"],
                "profit_factor": n_val["profit_factor"],
                "expectancy": n_val["expectancy"],
                "passed_filters": n_val["passed"],
                "delta_from_best": round(n_val["score"] - best_score, 4),
            })

    scores = [r["score"] for r in neighbor_rows]
    spread = float(np.std(scores)) if len(scores) > 1 else 0.0
    failing_neighbors_count = sum(1 for r in neighbor_rows if not r["passed_filters"])
    total_neighbors = len(neighbor_rows)

    is_isolated = bool(total_neighbors > 0 and (failing_neighbors_count / total_neighbors) >= 0.5 and best_score > 0.3)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "in_grid_neighbors": neighbor_rows,
        "neighbor_spread": round(spread, 4),
        "failing_neighbors_count": failing_neighbors_count,
        "total_neighbors": total_neighbors,
        "isolated_spike": is_isolated,
    }


def run_single_backtest(df: pd.DataFrame, symbol: str, timeframe: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single backtest and return standardized dictionary of metrics."""
    strat = create_strategy("vwap_bollinger_mr", params=params)
    cfg = BacktestConfig(
        initial_capital=INITIAL_CAPITAL,
        risk_per_trade=RISK_PER_TRADE,
        max_open_positions=MAX_OPEN_POSITIONS,
        max_leverage=MAX_LEVERAGE,
        market_type=MARKET_TYPE,
        timeframe=timeframe,
        execution=ExecutionConfig(market_type=MARKET_TYPE),
    )
    res = BacktestEngine(cfg).run(strat, df, symbol=symbol)
    m = res.metrics
    t = m.trades
    closed = res.trades

    long_trades = [tr for tr in closed if tr["direction"] == "long"]
    short_trades = [tr for tr in closed if tr["direction"] == "short"]
    wins = [tr for tr in closed if tr["net_pnl"] > 0]
    losses = [tr for tr in closed if tr["net_pnl"] < 0]

    gross_profit = sum(tr["net_pnl"] for tr in wins)
    gross_loss = abs(sum(tr["net_pnl"] for tr in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    score, passed, reasons, sub_scores = score_metrics_multi_objective(m.to_dict(), SCORE_WEIGHTS, HARD_FILTERS)

    if t.total_trades < 15:
        evidence_class = "INSUFFICIENT_EVIDENCE"
    elif 15 <= t.total_trades < 30:
        evidence_class = "MARGINAL_EVIDENCE"
    else:
        evidence_class = "ADEQUATE_EVIDENCE"

    return {
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
        "score": round(float(score), 4),
        "passed": bool(passed),
        "reasons": reasons,
        "evidence_class": evidence_class,
        "sub_scores": sub_scores,
    }


def run_stage_vwap_b(db_path: str = "data/crypto_quant.db") -> Dict[str, Any]:
    """Execute the full 48-combination calibration across 4 tracks."""
    db = DatabaseManager(db_path)
    repo = MarketDataRepository(db)

    grid_48 = generate_48_combinations()

    print("=" * 115)
    print("EXECUTING STAGE VWAP-B PARAMETER CALIBRATION (PROTOCOL-VWAP-MR-V1.0)")
    print(f"Combinations: 48 | Tracks: 4 (BTC 4H, ETH 4H, BTC 1H, ETH 1H) | Total Backtests: 48 x 4 x 2 = 384")
    print(f"Data Boundaries: Train (2020/2022-2024) | Validation (2025) | 2026 OOS: STRICTLY LOCKED")
    print("=" * 115)

    all_track_results: Dict[str, Any] = {}
    master_candidate_matrix: List[Dict[str, Any]] = []

    # Initialize master matrix rows with candidate parameters
    for idx, params in enumerate(grid_48):
        cid = f"C{idx+1:02d}"
        is_base = bool(params == BASELINE_PARAMS)
        master_candidate_matrix.append({
            "candidate_id": cid,
            "combo_index": idx + 1,
            "is_baseline": is_base,
            "params": params,
            "tracks": {},
        })

    for track in TRACKS:
        sym = track["symbol"]
        tf = track["timeframe"]
        t_start = track["train_start_ms"]
        track_key = f"{sym}_{tf.upper()}"

        print(f"\nProcessing Track: {track_key}")
        df_full = repo.load(sym, tf, market_type="spot")

        # Slice Train & Validation
        df_train = df_full[(df_full["timestamp"] >= t_start) & (df_full["timestamp"] <= TRAIN_END_MS)].reset_index(drop=True)
        df_val = df_full[(df_full["timestamp"] >= VAL_START_MS) & (df_full["timestamp"] <= VAL_END_MS)].reset_index(drop=True)

        # 2026 Protection assertions
        assert df_train["timestamp"].max() <= TRAIN_END_MS, f"Train OOS leak in {track_key}"
        assert df_val["timestamp"].max() <= VAL_END_MS, f"Validation OOS leak in {track_key}"
        assert df_val["timestamp"].max() < PRE_OOS_END_TIMESTAMP_MS, f"Strict 2026 OOS leak in {track_key}"

        print(f"  Train:      {pd.to_datetime(df_train['timestamp'].min(), unit='ms', utc=True).strftime('%Y-%m-%d')} to {pd.to_datetime(df_train['timestamp'].max(), unit='ms', utc=True).strftime('%Y-%m-%d')} ({len(df_train):,} bars)")
        print(f"  Validation: {pd.to_datetime(df_val['timestamp'].min(), unit='ms', utc=True).strftime('%Y-%m-%d')} to {pd.to_datetime(df_val['timestamp'].max(), unit='ms', utc=True).strftime('%Y-%m-%d')} ({len(df_val):,} bars)")

        evaluations = []
        for idx, params in enumerate(grid_48):
            cid = f"C{idx+1:02d}"
            # 1. Evaluate Train
            m_train = run_single_backtest(df_train, sym, tf, params)
            # 2. Evaluate Validation
            m_val = run_single_backtest(df_val, sym, tf, params)

            # Robustness / Degradation
            pf_deg = round(m_val["profit_factor"] - m_train["profit_factor"], 4)
            exp_deg = round(m_val["expectancy"] - m_train["expectancy"], 4)

            eval_entry = {
                "combo_id": cid,
                "combo_index": idx + 1,
                "params": params,
                "is_baseline": bool(params == BASELINE_PARAMS),
                "train": m_train,
                "validation": m_val,
                "degradation": {
                    "pf_delta": pf_deg,
                    "expectancy_delta": exp_deg,
                },
            }
            evaluations.append(eval_entry)
            master_candidate_matrix[idx]["tracks"][track_key] = eval_entry

        # Sort evaluations for this track: Passed validation first by validation score descending
        evaluations_sorted = sorted(
            evaluations,
            key=lambda e: (1 if e["validation"]["passed"] else 0, e["validation"]["score"]),
            reverse=True,
        )

        best_val_candidate = evaluations_sorted[0]
        passing_val_candidates = [e for e in evaluations_sorted if e["validation"]["passed"]]

        # Stability calculation
        stab_result = compute_in_grid_stability(best_val_candidate, evaluations, grid_48)

        all_track_results[track_key] = {
            "symbol": sym,
            "timeframe": tf,
            "total_combos": len(grid_48),
            "evaluations": evaluations_sorted,
            "passing_count": len(passing_val_candidates),
            "best_validation_candidate": best_val_candidate,
            "stability_result": stab_result,
        }

        pass_label = f"{len(passing_val_candidates)}/48 PASS"
        print(f"  -> Result: {pass_label} | Top: {best_val_candidate['combo_id']} (Val Score={best_val_candidate['validation']['score']}, PF={best_val_candidate['validation']['profit_factor']}, N={best_val_candidate['validation']['total_trades']}, Passed={best_val_candidate['validation']['passed']})")

    # Timeframe Decisions
    timeframe_decisions = {}
    for tf_label in ["4h", "1h"]:
        btc_res = all_track_results[f"BTCUSDT_{tf_label.upper()}"]
        eth_res = all_track_results[f"ETHUSDT_{tf_label.upper()}"]

        btc_pass = btc_res["passing_count"] > 0
        eth_pass = eth_res["passing_count"] > 0
        joint_pass = bool(btc_pass and eth_pass)

        timeframe_decisions[tf_label.upper()] = {
            "btc_passing_count": btc_res["passing_count"],
            "eth_passing_count": eth_res["passing_count"],
            "btc_top_candidate": btc_res["best_validation_candidate"]["combo_id"],
            "eth_top_candidate": eth_res["best_validation_candidate"]["combo_id"],
            "joint_timeframe_decision": "QUALIFIES_FOR_DUAL_ASSET_SHORTLIST" if joint_pass else "FAILED_DUAL_ASSET_VALIDATION",
        }

    # Export complete JSON artifact
    json_payload = {
        "protocol": "PROTOCOL-VWAP-MR-V1.0",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_combinations": len(grid_48),
        "tracks": all_track_results,
        "candidate_matrix": master_candidate_matrix,
        "timeframe_decisions": timeframe_decisions,
    }

    json_out_path = "data/vwap_mr_stage_b_optimization.json"
    with open(json_out_path, "w") as f:
        json.dump(json_payload, f, indent=2)

    # Export full Markdown diagnostic artifact
    md_out_path = "data/vwap_mr_stage_b_calibration.md"
    generate_stage_b_markdown(json_payload, md_out_path)

    print("\n" + "=" * 115)
    print(f"STAGE VWAP-B CALIBRATION COMPLETE.")
    print(f"Artifacts saved:")
    print(f"  1. JSON: {json_out_path}")
    print(f"  2. MD:   {md_out_path}")
    print("=" * 115)

    return json_payload


def generate_stage_b_markdown(payload: Dict[str, Any], output_path: str) -> None:
    """Generate comprehensive Stage-B Markdown calibration report."""
    now_utc = payload["timestamp_utc"]
    tracks = payload["tracks"]
    tf_dec = payload["timeframe_decisions"]
    matrix = payload["candidate_matrix"]

    lines = [
        "# Stage VWAP-B Parameter Calibration Diagnostic Report",
        "",
        f"**Protocol:** `PROTOCOL-VWAP-MR-V1.0`  ",
        f"**Strategy Family:** `VWAPBollingerMRStrategy` (`vwap_bollinger_mr`)  ",
        f"**Execution Date:** {now_utc}  ",
        f"**Total Combinations Evaluated:** 48  ",
        f"**Total Backtests Executed:** 48 combos $\\times$ 4 tracks $\\times$ 2 slices = 384  ",
        f"**OOS Status:** 2026 dataset strictly locked and excluded (`timestamp < 2026-01-01 00:00:00 UTC`).",
        "",
        "---",
        "",
        "## 1. Executive Summary & Dual-Asset Timeframe Decisions",
        "",
        "| Timeframe | BTC Passing Candidates | ETH Passing Candidates | Both Assets Passed? | Timeframe Decision |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]

    for tf_key, d in tf_dec.items():
        both_str = "YES" if d["joint_timeframe_decision"] == "QUALIFIES_FOR_DUAL_ASSET_SHORTLIST" else "NO"
        lines.append(
            f"| **{tf_key}** | {d['btc_passing_count']}/48 (Top: `{d['btc_top_candidate']}`) | "
            f"{d['eth_passing_count']}/48 (Top: `{d['eth_top_candidate']}`) | **{both_str}** | **`{d['joint_timeframe_decision']}`** |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Track Summary & Top Candidates",
        "",
    ])

    for track_name, t_data in tracks.items():
        best = t_data["best_validation_candidate"]
        tr = best["train"]
        va = best["validation"]
        stab = t_data["stability_result"]
        lines.extend([
            f"### Track: `{track_name}`",
            f"* **Passing Validation Count:** {t_data['passing_count']}/48 candidates",
            f"* **Top Candidate ID:** `{best['combo_id']}` {'(BASELINE)' if best['is_baseline'] else ''}",
            f"* **Parameters:** `{best['params']}`",
            f"* **Train Metrics:** $N={tr['total_trades']}$, $WR={tr['win_rate']*100:.1f}\\%$, $PF={tr['profit_factor']:.4f}$, $E=\\${tr['expectancy']:.2f}$, $\\text{{Ret}}={tr['net_return']*100:.1f}\\%$, $MDD={tr['max_drawdown']*100:.1f}\\%$",
            f"* **Validation (2025) Metrics:** $N={va['total_trades']}$, $WR={va['win_rate']*100:.1f}\\%$, $PF={va['profit_factor']:.4f}$, $E=\\${va['expectancy']:.2f}$, $\\text{{Ret}}={va['net_return']*100:.1f}\\%$, $MDD={va['max_drawdown']*100:.1f}\\%$, $\\text{{Score}}={va['score']:.4f}$",
            f"* **Gate Result:** **{'PASS' if va['passed'] else 'FAIL'}** (Reasons: `{va['reasons']}`)",
            f"* **In-Grid Neighbor Stability:** Spread = {stab['neighbor_spread']:.4f}, Failing Neighbors = {stab['failing_neighbors_count']}/{stab['total_neighbors']}, Isolated Spike = `{stab['isolated_spike']}`",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## 3. Complete 48-Candidate Performance Matrix",
        "",
        "| ID | ADX Max | BB Std | RSI (OS/OB) | ATR Mult | RR | BTC 4H Val PF (N) | ETH 4H Val PF (N) | BTC 1H Val PF (N) | ETH 1H Val PF (N) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for row in matrix:
        cid = row["candidate_id"]
        p = row["params"]
        b4 = row["tracks"]["BTCUSDT_4H"]["validation"]
        e4 = row["tracks"]["ETHUSDT_4H"]["validation"]
        b1 = row["tracks"]["BTCUSDT_1H"]["validation"]
        e1 = row["tracks"]["ETHUSDT_1H"]["validation"]

        b4_str = f"{b4['profit_factor']:.2f} ({b4['total_trades']})" + ("*" if b4["passed"] else "")
        e4_str = f"{e4['profit_factor']:.2f} ({e4['total_trades']})" + ("*" if e4["passed"] else "")
        b1_str = f"{b1['profit_factor']:.2f} ({b1['total_trades']})" + ("*" if b1["passed"] else "")
        e1_str = f"{e1['profit_factor']:.2f} ({e1['total_trades']})" + ("*" if e1["passed"] else "")

        base_mark = " (Base)" if row["is_baseline"] else ""
        lines.append(
            f"| `{cid}`{base_mark} | {p['adx_max_thresh']} | {p['bb_std']} | {p['rsi_oversold']}/{p['rsi_overbought']} | "
            f"{p['atr_multiplier']} | {p['risk_reward_ratio']} | {b4_str} | {e4_str} | {b1_str} | {e1_str} |"
        )

    lines.extend([
        "",
        "\\* indicates candidate passed all validation hard gates ($N \\ge 15, PF \\ge 1.05, E > 0, MDD \\le 40\\%, L=0$).",
        "",
        "---",
        "",
        "## 4. 2026 OOS Protection Verification",
        "",
        "* **OOS Status:** Fully locked and untouched.",
        "* **Maximum Timestamp Evaluated:** Strictly $\\le 2025-12-31 23:59:59 UTC$ across all 384 backtests.",
        "* **Next Step:** Awaiting formal human gatekeeper review.",
        "",
        "---",
        "*Report generated automatically by `scripts/run_stage_vwap_b.py`.*",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    run_stage_vwap_b()
