# OOS Readiness & Protocol Decision Memo — MTF Research Gate

**Date:** 2026-09-11
**Scope:** Read-only research-gate audit. No code modified, no backtests executed, no 2026 OOS data accessed, no parameters tuned.
**Candidate under review:** ETHUSDT Stage-B candidate **#56** (`1d_4h_1h` MTF Trend Pullback, 1H execution).

---

## A. ETH #56 Evidence Quality

### A.1 Train vs 2025 Validation

Source: `best_validation_candidate` and `evaluations[combo_index=56]` in `data/mtf_stage_b_optimization.json`, `ETHUSDT` section.

| Metric | Train (in-sample) | 2025 Validation | Δ (Train→Val) |
| :--- | :--- | :--- | :--- |
| Trades (N) | 40 | **17** | −57.5% |
| Win Rate | 42.5% | 35.3% | −7.2 ppt |
| Profit Factor | 1.4498 | **1.0565** | −27.1% |
| Expectancy | +2.0769 | **+0.3571** | −82.8% |
| Net Return | +8.31% | +0.61% | −92.7% |
| Max Drawdown | 27.46% | 26.93% | ≈ flat (improved) |
| Liquidations | 0 | 0 | — |
| Score | 0.1989 | 0.0874 | −56.1% |

**Validation summary:** all headline metrics decay from Train to Validation. Expectancy collapses by ~83%, net return by ~93%, and profit factor drops from 1.45 to 1.06. Only max drawdown is stable (~27%). The candidate is a break-even-to-slightly-positive configuration on 2025 data, not a high-conviction one.

### A.2 Margin above each hard gate (applied to VALIDATION metrics)

Gates defined in source: `MultiObjectiveFilters(min_trades=15, min_profit_factor=1.05, max_drawdown=0.40)` (`scripts/run_stage_mtf_b.py:35`, `src/crypto_quant/research/multi_strategy_engine.py:103`) plus expectancy and liquidation gates in `score_metrics_multi_objective` (`src/crypto_quant/validation/optimize_window.py:74`): **N ≥ 15, E > 0, PF ≥ 1.05, MDD ≤ 0.40, L == 0.**

| Hard Gate | Threshold | ETH #56 (Validation) | Margin | Pass? |
| :--- | :--- | :--- | :--- | :--- |
| **N** (trades) | ≥ 15 | 17 | **+2 trades** | ✓ |
| **PF** (profit factor) | ≥ 1.05 | 1.0565 | **+0.0065** | ✓ |
| **Expectancy** | > 0 | +0.3571 | **+0.3571** | ✓ |
| **MDD** (max drawdown) | ≤ 0.40 | 0.2693 | **+13.07 ppt** | ✓ |
| **Liquidations** | == 0 | 0 | 0 | ✓ |

All five gates are passed **technically**. However, two gates are passed with effectively zero margin.

### A.3 Why N = 17 and PF = 1.0565 are WEAK evidence despite passing

1. **Sample size is tiny.** 17 trades across an entire 2025 validation year ≈ 1.4 trades/month. With 17 trades, the profit factor and win rate carry enormous sampling error. The 95% binomial interval on a 35.3% win rate at N=17 spans roughly ±22 percentage points — the observed 35.3% (exactly 6/17 wins) is statistically indistinguishable from break-even or worse.
2. **PF margin is ~zero.** PF = 1.0565 sits just 0.0065 above the 1.05 gate — comfortably within noise of PF = 1.0 (break-even). A single losing trade or one less winner flips it below the gate. This is a pass *by rounding tolerance*, not by demonstrated edge.
3. **Expectancy is barely positive.** +0.36 per trade against the per-trade risk notional is negligible, and it represents an **82.8% decay** from train expectation. Net return over the whole 2025 year is +0.61% — economically flat.
4. **Selection bias / multiple comparisons.** #56 is the best-scoring of **72** grid combos, but only **3 of 72 (4.2%)** pass all validation gates, and those 3 tie at the identical score (0.0874). With 72 configurations tested, observing a marginal pass is consistent with luck under multiple testing, not a robust discovery.
5. **Weak plateau, not a clean peak.** The stability neighborhood (`stability_result.neighbors`) shows most small perturbations of #56 fail the gates (e.g. `htf_ema_fast=50`, `itf_rsi_pullback=50`, `ltf_rsi_oversold=32`, RR 2.0/3.0, `volume_mult=1.2` all → PF ≤ 1.0 or negative expectancy), while a few perturbations score *higher* than #56 (oversold=42 → score 0.110; stop_type="swing" → score 0.138). `isolated_spike = false`, `neighbor_spread = 0.4287`. The optimum is a shallow plateau — the exact parameterization frozen is arbitrary within a weak region, and better-scoring neighbors were excluded only because they fall **outside the approved 72-combo grid** (oversold=42 and stop_type="swing" are not in the grid). This is evidence of marginal robustness, not data corruption.

**Bottom line:** ETH #56 passes every hard gate but with negligible PF and sample margin, near-zero expectancy, 83% expectation decay, and a shallow optimum surrounded by failing neighbors. It is a *marginal* technical qualifier.

---

## B. Candidate Classification

**Classification: `MARGINAL RESEARCH CANDIDATE`**

- **Not ROBUST** — PF margin 0.0065 (≈ noise), N = 17 (tiny), expectancy decay −83%, net return +0.61%, flat/weak optimum, 3-of-72 passing tie.
- **Not REJECTED** — it does satisfy every approved hard gate on validation: N=17 ≥ 15, PF=1.0565 ≥ 1.05, E=+0.36 > 0, MDD=26.93% ≤ 40%, L=0.
- The classification follows strictly from the evidence above, not intuition.

---

## C. Protocol Question — Is an ETH-only OOS test explicitly permitted?

### C.1 What the protocol actually says (found wording)

The approved protocol is referenced as **"PROTOCOL-MTF-V1.5"** in `data/mtf_stage_a_diagnostic.md`. **No standalone protocol document body exists in the repository** — the only quotable approved protocol text recovered from the artifacts is PROTOCOL-MTF-V1.5 **Section 4**, quoted verbatim in `data/mtf_stage_a_diagnostic.md` (Section D), governing **architecture advancement from Stage-A to Stage-B**:

> *"An architecture advances to Stage MTF-B only if both its BTCUSDT and ETHUSDT Stage-A baseline experiments pass all approved Stage-A screening rules. If either asset fails a Stage-A screening rule, that architecture does not advance to Stage MTF-B."*

That rule governs the **Stage-A → B** architecture gate (already satisfied for `1d_4h_1h`), not the **Stage-B → OOS** step.

### C.2 What the Stage-B source enforces for candidate selection

`scripts/run_stage_mtf_b.py` freezes a `best_validation_candidate` **per-symbol, independently**. It runs the 72-combo grid over `["BTCUSDT", "ETHUSDT"]`, gates the **validation** slice with `MultiObjectiveFilters(...)`, and stores the best passing candidate per asset. There is:
- **No** dual-asset co-requirement encoded on OOS authorization.
- **No** clause requiring a BTC candidate as a precondition for an ETH OOS test.
- **No** code path or stored text stating ETH-only OOS is permitted **or** forbidden.

### C.3 Finding

The existing approved MTF protocol **does not explicitly authorize an ETH-only OOS test.** The only "both assets must qualify" provision is the Stage-A architecture-advancement rule (Section 4 above), which:
- is satisfied for `1d_4h_1h` (both BTC and ETH passed Stage-A), and
- by its own text applies to *architecture advancement*, not to *OOS execution* of a single asset's candidate.

Because the protocol body is not stored as a governance document and no section explicitly covers the "BTC has no qualified candidate, ETH has one marginal qualifier, no dual-asset candidate" situation, the ETH-only OOS path is **not explicitly authorized by the approved protocol**. Authorization cannot be assumed.

---

## D. Protocol Decision Required

**An ETH-only OOS test is NOT explicitly authorized by the existing approved protocol.**

A **new protocol decision is required** before any OOS execution for ETH #56. No new protocol is created in this memo — this memo only establishes that the decision is required.

The decision should resolve, at minimum: whether the partial-qualification state (only ETH qualifying) is acceptable grounds for a single-asset OOS, what statistical minimum for OOS trade count applies (given N=17 validation), and whether an ETH-only OOS can support any deployment posture absent a BTC leg.

---

## E. Research Documentation Consistency

Artifacts inspected: `data/mtf_stage_b_optimization.json`, `data/mtf_stage_a_screening.json`, `data/mtf_stage_a_diagnostic.md`, `data/mtf_final_report.json`, `data/mtf_study_results.json`, `data/final_post_mortem_report.md`, plus source `scripts/run_stage_mtf_b.py`, `src/crypto_quant/research/multi_strategy_engine.py`, `src/crypto_quant/validation/optimize_window.py`, `src/crypto_quant/research/mtf_study.py`.

| Check | Result |
| :--- | :--- |
| **Search-space ↔ stored params** | Consistent. Grid = 3×3×2×2×2 = **72 combos** (`run_stage_mtf_b.py`). ETH #56 params (`htf_adx_threshold=25.0, itf_rsi_pullback=45.0, ltf_rsi=(38,62), atr_multiplier=2.2, risk_reward_ratio=2.5, stop_type="atr"`) all lie inside the approved grid; `total_combos=72` and `len(evaluations)=72` match. |
| **Gates across sources** | Consistent. `min_trades=15, min_profit_factor=1.05, max_drawdown=0.40` appear identically in `run_stage_mtf_b.py:35`, `multi_strategy_engine.py:103`, and the scoring function in `optimize_window.py`. |
| **best_validation_candidate ↔ evaluations** | Consistent. `best_validation_candidate.combo_index = 56`; #56 is the top (tied) passing grid candidate (score 0.0874; only combos 56, 64, 72 pass, all tied). |
| **Stage-A ↔ Stage-B** | Consistent. Only `1d_4h_1h` advanced (as the diagnostic reports); Stage-B runs exclusively `1d_4h_1h`. |
| **Accounting identity** | Consistent with Stage-A diagnostic: PF = Total Profit / Total Loss; Net PnL = Total Profit − Total Loss = Final Equity − Initial Capital; per-trade `net_pnl = gross − entry_fee − exit_fee − funding`. |

**Genuine inconsistencies:** **None found.** All stored numbers are internally exact and reconcile with the documented accounting model.

**Observations (not inconsistencies, but relevant):**
- Only **3 of 72** grid combos pass validation gates, and they **tie** at score 0.0874 — a marginal, flat pass region.
- `stability_result.neighbors` shows two perturbations **outside the approved grid** (oversold=42, stop_type="swing") that score *higher* than #56 but were correctly never selected — the frozen #56 sits in a weak plateau, not a clean peak. This supports the `MARGINAL` classification and does not indicate a data defect.

---

## F. OOS Preflight Checklist (FOR PREPARATION ONLY — NOT EXECUTED)

An **ETH-only** OOS run, *should the required protocol decision authorize it*, must satisfy the following frozen contract. Nothing here is executed by this memo.

### F.1 Frozen Candidate — ETH #56 exact parameterization
Symbol: **ETHUSDT** · Architecture: `1d_4h_1h` (MTF Trend Pullback) · Execution timeframe: **1H**

| Param | Value |
| :--- | :--- |
| mtf_combo | `1d_4h_1h` |
| htf_ema_fast | 20 |
| htf_ema_slow | 50 |
| htf_adx_threshold | 25.0 |
| itf_ema_fast | 20 |
| itf_ema_slow | 50 |
| itf_rsi_period | 14 |
| itf_rsi_pullback | 45.0 |
| ltf_ema_fast | 20 |
| ltf_rsi_period | 14 |
| ltf_rsi_oversold | 38.0 |
| ltf_rsi_overbought | 62.0 |
| volume_mult | 1.0 |
| atr_multiplier | 2.2 |
| risk_reward_ratio | 2.5 |
| stop_type | `atr` |
| swing_lookback | 15 |

These must be fixed byte-for-byte. **No parameter may change** before or during OOS.

### F.2 Exact OOS date range
Per the approved Stage-B splitter (`run_stage_mtf_b.py`: `SplitDates(train_end="2024-12-31", validation_end="2025-12-31", test_end="2026-09-08")`), the OOS slice is the **unused test window**:

> **ETHUSDT OOS = 2025-12-31 / 2026-01-01 → 2026-09-08** (1H spot bars).

Only this slice may be loaded; it must have been strictly excluded from all training and validation.

### F.3 Execution configuration
- Market type: **spot** · Timeframe: **1H**
- Initial capital, risk-per-trade, max open positions, max leverage: use the identical values from `run_stage_mtf_b.py` `BacktestConfig` (must match Stage-B exactly so Train/Val/OOS are comparable).
- Execution config: `ExecutionConfig(market_type="spot")` — default taker friction model per Stage-A accounting.

### F.4 Required metrics (report every one)
`total_trades` (N), `win_rate`, `profit_factor`, `expectancy`, `net_return`, `max_drawdown`, `sharpe`, `sortino`, `calmar`, `liquidations`, `net_pnl`.

### F.5 Required accounting identities (must reconcile exactly)
- **Profit Factor = Total Profit / Total Loss** (using net per-trade PnL after fees & funding).
- **Net PnL = Total Profit − Total Loss = Final Equity − Initial Capital.**
- **Per-trade net_pnl = gross_pnl − entry_fee − exit_fee − funding.**
- `total_trades = long_trades + short_trades`; `win_rate = wins / total_trades`.
- Fees and funding are diagnostic (already deducted inside each trade's net_pnl); **never** subtracted a second time.

### F.6 Required post-OOS diagnostics
- OOS vs 2025 Validation vs Train degradation table (N, WR, PF, E, net return, MDD).
- Directional (long/short) breakdown.
- Exit-type distribution (stop-loss / take-profit / end-of-test).
- Costs vs edge attribution (gross PnL pre-costs vs fee+funding drag).
- Note explicitly that **OOS results are descriptive only** and cannot be used to re-tune.

### F.7 Prohibitions
- **NO parameter changes after OOS is run** — any adjustment voids the OOS.
- **NO use of OOS for model selection** — OOS is a single, one-shot acceptance test against the frozen candidate and pre-registered gates.
- **NO running multiple OOS variants and keeping the best** (selection across OOS windows).
- **NO new candidates, no grid expansion, no ETH #56 tuning** — all remain frozen pending the protocol decision.

---

*Generated by a read-only research-gate audit. Integrity: no code modified, no backtests executed, no 2026 OOS data accessed, no parameters tuned.*