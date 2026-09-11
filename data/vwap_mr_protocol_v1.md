# PROTOCOL-VWAP-MR-V1.0: Mean Reversion Research & Validation Protocol

**Document Version:** 1.0.0  
**Strategy Family:** `VWAPBollingerMRStrategy` (`src/crypto_quant/strategies/vwap_bollinger_mr.py`)  
**Status:** DRAFT / PENDING GATEKEEPER APPROVAL  
**Execution Authorization:** AUDIT & PROTOCOL ONLY. Zero backtests, zero optimization, zero OOS access authorized until formal approval.

---

## 1. Research Objectives & Core Question

### 1.1 Primary Research Question
Does the `VWAPBollingerMRStrategy` architecture provide a statistically robust, risk-adjusted, repeatable edge during non-trending regimes across core crypto markets (`BTCUSDT` and `ETHUSDT`) on intermediate timeframes (4H / 1H), or was the Step-2 baseline performance an artifact of small-sample variance?

### 1.2 Quantitative Target
Identify frozen parameter candidates that demonstrate:
1. Consistent outperformance above hard gatekeeping thresholds ($PF \ge 1.05, E > 0, MDD \le 40\%$).
2. Sufficient statistical power ($N \ge 15$ minimum gate, $N \ge 30$ preferred tier).
3. Parameter neighborhood stability (insensitivity to minor parameter perturbations).
4. Dual-asset cross-validation across both BTC and ETH.

---

## 2. Strategy Specification & Baseline Control

### 2.1 Strategy Under Test
* **Class:** `VWAPBollingerMRStrategy`
* **Architecture:** ADX regime-gated Bollinger-RSI extreme fading with fixed ATR stop-loss and fixed Risk-Reward take-profit.
* **Baseline Control Parameter Set (Default Combo):**
  * `adx_max_thresh`: `22.0`
  * `bb_period`: `20`
  * `bb_std`: `2.0`
  * `vwap_period`: `20`
  * `rsi_period`: `14`
  * `rsi_oversold`: `35.0`
  * `rsi_overbought`: `65.0`
  * `atr_multiplier`: `2.0`
  * `risk_reward_ratio`: `2.0`
  * `stop_type`: `"atr"`

---

## 3. Data Partitions & Strict Chronological Boundaries

All testing strictly enforces point-in-time chronological boundaries with zero lookahead bias:

| Asset | In-Sample Train Window | Validation Window | Out-Of-Sample (OOS) Status |
| :--- | :--- | :--- | :--- |
| **BTCUSDT** | `2020-01-01 00:00:00` $\rightarrow$ `2024-12-31 23:59:59` | `2025-01-01 00:00:00` $\rightarrow$ `2025-12-31 23:59:59` | **LOCKED & BLIND** (`2026-01-01+`) |
| **ETHUSDT** | `2022-01-01 00:00:00` $\rightarrow$ `2024-12-31 23:59:59` | `2025-01-01 00:00:00` $\rightarrow$ `2025-12-31 23:59:59` | **LOCKED & BLIND** (`2026-01-01+`) |

* **Zero-OOS Policy:** The 2026 OOS dataset must not be queried, filtered, loaded, backtested, or evaluated under this protocol. OOS evaluation requires an explicit future gatekeeper authorization after candidate parameter freezing.

---

## 4. Execution & Accounting Configuration

To maintain cross-strategy comparability with the project's standard research harness, the following standardized configuration is strictly enforced:

* **Market Type:** Futures (USDT-M Perpetual)
* **Initial Capital:** $1,000.00 per asset
* **Risk per Trade:** 1.0% of equity
* **Max Open Positions:** 3 concurrent positions
* **Leverage:** 5x isolated margin
* **Fee Model:** 0.04% taker fee per side (0.08% round-turn)
* **Slippage Model:** Embedded in fill price via standard repository execution model
* **Funding Model:** 8-hour funding rate accounting deducted per open position
* **Execution Timing:** Signal calculated at closed bar $i$; filled at next-bar `open[i+1]`
* **Intrabar Collision Priority:** Conservative stop-first assumption if high/low breaches both SL and TP within the same bar

---

## 5. Bounded Parameter Search Space (Stage-B Calibration)

To mitigate data-mining bias, the search space is strictly bounded to parameters addressing specific research hypotheses:

| Parameter | Type | Allowed Values | Combinations | Scientific Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `adx_max_thresh` | Optimized | `[20.0, 22.0, 25.0]` | 3 | Tests regime sensitivity to consolidation threshold |
| `bb_period` | Frozen | `[20]` | 1 | Standard 20-period moving average baseline |
| `bb_std` | Optimized | `[2.0, 2.2]` | 2 | Tests statistical extremity threshold for band probe |
| `rsi_period` | Frozen | `[14]` | 1 | Standard momentum lookback |
| `rsi_oversold` / `rsi_overbought` | Optimized | `[(35.0, 65.0), (30.0, 70.0)]` | 2 | Evaluates exhaustion threshold depth |
| `atr_multiplier` | Optimized | `[1.8, 2.2]` | 2 | Controls stop buffer against normal noise |
| `risk_reward_ratio` | Optimized | `[1.5, 2.0]` | 2 | Tests target distance vs probability of reversion |
| `stop_type` | Frozen | `["atr"]` | 1 | Fixed ATR stop model as currently implemented |

* **Total Stage-B Search Space:** $3 \times 1 \times 2 \times 1 \times 2 \times 2 \times 2 \times 1 = \mathbf{48\text{ combinations}}$ per asset.
* **Control:** Baseline default configuration is Combo #0 within this 48-combination grid.

---

## 6. Sample-Size Policy & Evidence Tiers

Trade counts ($N$) on the 1-year validation slice (`2025-01-01` to `2025-12-31`) are classified into three strict evidence tiers:

1. **Tier 1: $N < 15$ Trades $\rightarrow$ REJECT / INSUFFICIENT EVIDENCE**
   * Automatically triggers hard gate rejection (`Insufficient trades: N < 15`).
   * No candidate in this tier can advance, regardless of Profit Factor or Net Return.
2. **Tier 2: $15 \le N < 30$ Trades $\rightarrow$ MARGINAL RESEARCH EVIDENCE**
   * Technically satisfies the minimum trade gate.
   * Subject to strict statistical scrutiny; penalized in multi-objective scoring via sample-size confidence factor ($c_N = 1 - e^{-N/25}$).
   * Cannot be classified as ROBUST without strong cross-asset confirmation.
3. **Tier 3: $N \ge 30$ Trades $\rightarrow$ ADEQUATE SAMPLE SIZE**
   * Sufficient sample size for annual validation screening.

---

## 7. Hard Gatekeeping Criteria & Multi-Objective Scoring

### 7.1 Hard Gates (All Must Pass Simultaneously on Validation Slice)
1. **Trade Count ($N$):** $N \ge 15$
2. **Expectancy ($E$):** $E > \$0.00$ per trade
3. **Profit Factor ($PF$):** $PF \ge 1.0500$
4. **Max Drawdown ($MDD$):** $MDD \le 40.00\%$
5. **Liquidations ($L$):** $L = 0$

### 7.2 Multi-Objective Scoring Function
Candidates passing all hard gates are ranked via the deterministic repository utility function:
$$\text{Score} = c_N \times \left( w_{\text{exp}} u_{\text{exp}} + w_{\text{pf}} u_{\text{pf}} + w_{\text{dd}} p_{\text{dd}} + w_{\text{sh}} u_{\text{sh}} + w_{\text{so}} u_{\text{so}} + w_{\text{cal}} u_{\text{cal}} \right)$$

Where:
* $c_N = 1 - e^{-N/25}$ (sample size penalty)
* $u_{\text{exp}} = \tanh(E / 10)$
* $u_{\text{pf}} = \text{clip}((PF - 1.0) / 3.0, 0, 1)$
* $p_{\text{dd}} = \max(0, 1 - (MDD / 0.40)^2)$
* If any hard gate is failed: $\text{Score} = -1.0 + (c_N \times 0.1)$.

---

## 8. Parameter Stability Protocol

* **In-Grid Restriction:** Stability analysis must evaluate immediate parameter neighbors **strictly within the approved 48-combination search space**.
* **Prohibition:** Out-of-grid parameter evaluations are strictly forbidden. The search space cannot be expanded during stability verification.
* **Plateau Requirement:** A candidate is rejected if its passing score is an isolated spike where $\ge 50\%$ of in-grid neighbors fail the hard gates or show severe score degradation ($\Delta_{\text{score}} < -0.50$).

---

## 9. Advancement & Decision Rules

1. **Dual-Asset Advancement (Standard Requirement):**
   * Both `BTCUSDT` and `ETHUSDT` must produce at least one passing Stage-B validation candidate.
   * If both assets qualify, the top-scoring candidate from each asset is combined into a dual-asset candidate for OOS readiness review.
2. **Asymmetric / Single-Asset Outcome:**
   * If only one asset qualifies (e.g. BTC fails, ETH passes):
     * The strategy family is marked as **FAILED DUAL-ASSET VALIDATION**.
     * The qualifying single asset may be archived as a `MARGINAL ASSET-SPECIFIC CANDIDATE`.
     * **Zero OOS execution is authorized** without an explicit, separate governance protocol decision.
3. **Family Failure:**
   * If neither asset produces a passing candidate ($0/48$ on both), the `VWAPBollingerMRStrategy` family is formally **REJECTED** and closed.

---

## 10. Multiple-Testing Controls & Research Integrity

* **Predeclaration:** Search space ($M = 48$) is locked prior to experiment execution.
* **No Post-Hoc Adjustments:** Parameter ranges, indicators, and gates cannot be modified after running validation.
* **Anti-Lookahead Guarantee:** All signals generated on bar $i$ use indicators shifted by 1 bar (`shift=1`), filled on bar $i+1$ open.

---

## 11. Required Output Artifacts

Following authorization and execution of this protocol, the following standardized artifacts must be generated:
1. `data/vwap_mr_stage_a_screening.json` (Baseline comparison)
2. `data/vwap_mr_stage_b_optimization.json` (48-combo calibration across Train and Validation)
3. `data/vwap_mr_readiness_memo.md` (Formal gatekeeper review and candidate classification)

---

## 12. Protocol Summary & Freeze Declaration

* **Protocol Version:** `PROTOCOL-VWAP-MR-V1.0`
* **Strategy:** `VWAPBollingerMRStrategy`
* **Search Size:** 48 combinations $\times$ 2 assets = 96 evaluations per slice
* **Status:** **FROZEN / READY FOR REVIEW**
* **OOS Status:** **STRICTLY LOCKED**
