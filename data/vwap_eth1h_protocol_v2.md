# PROTOCOL-VWAP-ETH1H-V2.0: Single-Asset Pre-OOS Robustness & Validation Protocol

**Protocol Identity:** `PROTOCOL-VWAP-ETH1H-V2.0`  
**Target Strategy:** `VWAPBollingerMRStrategy` (`vwap_bollinger_mr`)  
**Target Candidate:** Candidate `C33` (`ETHUSDT` / 1H)  
**V2.0 Status:** `PROPOSED / NOT YET APPROVED`  
**2026 OOS Status:** `LOCKED — STRICTLY PROHIBITED`  
**Execution Authorization:** DOCUMENT & PROTOCOL DESIGN ONLY. Zero backtests, zero diagnostics, zero code modifications, zero 2026 OOS access authorized until explicit human governance sign-off.

---

## 1. Explicit Relationship to PROTOCOL-VWAP-MR-V1.0

### 1.1 Formal Closure & Failure of V1.0
`PROTOCOL-VWAP-MR-V1.0` is formally **CLOSED and FAILED** regarding its primary cross-asset, dual-asset hypothesis:
1. **4H Timeframe Track:** Completely eliminated across all 48 parameter configurations because both `BTCUSDT` ($N \le 14$) and `ETHUSDT` ($N \le 12$) failed the mandatory annual validation sample-size gate ($N \ge 15$).
2. **1H Timeframe Track (BTC Leg):** FAILED across 100% of tested parameter sets (0/48 passing; highest BTC 1H validation Profit Factor was $0.9660$).
3. **1H Timeframe Track (ETH Leg):** Produced 26 of 48 passing candidates, led by Candidate `C33` ($N = 49, PF = 1.6032, E = +\$2.54$).

### 1.2 Independent Research Hypothesis for V2.0
`PROTOCOL-VWAP-ETH1H-V2.0` is **NOT** a continuation or relaxation of the failed dual-asset approval. It constitutes an independent, newly scoped single-asset research protocol formulated around a distinct quantitative thesis:
> **Hypothesis:** The `VWAPBollingerMRStrategy` architecture exhibits asset-specific mean-reversion edge on `ETHUSDT` 1H during non-trending regimes due to specific ETH liquidity and volatility dynamics, despite the absence of a universal cross-asset edge on `BTCUSDT`.

---

## 2. Research Scope & Candidate Freeze

### 2.1 Scope Boundaries
* **Strategy:** `VWAPBollingerMRStrategy` (registered slug: `vwap_bollinger_mr`)
* **Asset:** `ETHUSDT` exclusively.
* **Market Type:** Futures (USDT-M Perpetual)
* **Execution Timeframe:** 1H (`1h`)
* **Candidate Under Evaluation:** **Candidate `C33` exclusively.**
* **Prohibitions:**
  * No `BTCUSDT`, `SOLUSDT`, or other assets.
  * No additional timeframes (no 5m, 15m, 4h, 1d).
  * No additional parameter grid combinations.
  * No parameter optimization, Bayesian search, or automated tuning.

### 2.2 Frozen Candidate `C33` Parameter Dictionary
Candidate `C33` is locked byte-for-byte as derived from the Stage-B optimization records:

```json
{
  "adx_max_thresh": 25.0,
  "bb_period": 20,
  "bb_std": 2.0,
  "vwap_period": 20,
  "rsi_period": 14,
  "rsi_oversold": 35.0,
  "rsi_overbought": 65.0,
  "atr_multiplier": 1.8,
  "risk_reward_ratio": 1.5,
  "stop_type": "atr"
}
```

* **Freeze Rule:** No parameter alterations, re-weightings, or neighbor substitutions are permitted before, during, or after the V2.0 pre-OOS robustness stage.

---

## 3. Standardized Execution Configuration

To maintain complete parity with repository backtest and accounting standards, the execution configuration is locked as:

* **Initial Capital:** $1,000.00
* **Risk per Trade:** 1.0% of portfolio equity
* **Maximum Open Positions:** 3 concurrent positions
* **Maximum Leverage:** 5x isolated margin
* **Fee Model:** 0.04% taker fee per side (0.08% round-turn)
* **Slippage Model:** Dynamic tick slippage embedded into fill prices via repository execution model
* **Funding Model:** 8-hour funding rate accounting deducted per open position
* **Execution Timing:** Signal evaluated on closed bar $i$; order filled at next-bar `open[i+1]`
* **Stop-Loss / Take-Profit Mechanics:** Fixed ATR stop-loss and static Risk-Reward take-profit price levels attached at entry
* **Intrabar Collision Priority:** Conservative stop-loss prioritization if bar high and low hit both SL and TP thresholds
* **Liquidation & PnL Accounting:** Standardized repository portfolio accounting ($\text{Net PnL} = \text{Gross Profit} - \text{Gross Loss} - \text{Fees} - \text{Funding}$)

---

## 4. 2026 Out-Of-Sample (OOS) Data Lock

* **Exact OOS Boundary:** `2026-01-01 00:00:00 UTC` (`timestamp = 1767225600000 ms`)
* **Absolute Prohibition:** Zero candles, trades, metrics, or summaries from on or after `2026-01-01 00:00:00 UTC` may be loaded, queried, inspected, or evaluated under V2.0.
* **Scope:** Protocol V2.0 is **strictly a pre-OOS protocol**. Approval of V2.0 does not grant permission to unlock or evaluate 2026 data.

---

## 5. Mandatory Pre-OOS Robustness Diagnostics (Pre-2026 Only)

Before any future OOS authorization can be considered by human governance, Candidate `C33` must undergo the following five pre-declared diagnostic audits using strictly historical pre-OOS data (`2022-01-01 00:00:00` to `2025-12-31 23:59:59 UTC`):

### Diagnostic A: Regime Attribution Audit
Evaluate frozen `C33` across pre-defined ADX regimes over the pre-OOS window:
1. Low-trend regime: $\text{ADX} < 20.0$
2. Moderate-trend / transition regime: $20.0 \le \text{ADX} < 25.0$
3. Strong-trend regime: $\text{ADX} \ge 25.0$
* **Required Metrics per Regime:** Total trades ($N$), win rate, profit factor, expectancy, net PnL, net return, MaxDD, Sharpe, Sortino, fees, slippage, funding, liquidations.
* **Objective:** Verify that positive edge is concentrated in low/moderate ADX environments and confirm the structural logic of the regime filter.

### Diagnostic B: Directional Symmetry Audit
Decompose all pre-OOS trades into Long and Short cohorts:
* **Required Metrics per Direction:** Trade count ($N$), win rate, profit factor, expectancy, gross profit, gross loss, net PnL, contribution percentage to total return.
* **Objective:** Determine whether profitability is balanced across both market directions or heavily dependent on an asymmetric market drift.

### Diagnostic C: Friction & Fee Stress Test
Evaluate frozen `C33` under progressive transaction cost stress levels:
1. **Level 1 (Baseline):** 0.04% taker fee per side (0.08% round-turn)
2. **Level 2 (Moderate Stress):** 0.06% taker fee per side (0.12% round-turn)
3. **Level 3 (Severe Stress):** 0.08% taker fee per side (0.16% round-turn)
* **Objective:** Measure edge resilience against exchange fee spikes, high volatility spreads, and severe execution drag.

### Diagnostic D: Chronological Sub-Window Stability Audit
Evaluate frozen `C33` across eight discrete, non-overlapping semi-annual sub-windows:
* `2022-H1` (`2022-01-01` to `2022-06-30`)
* `2022-H2` (`2022-07-01` to `2022-12-31`)
* `2023-H1` (`2023-01-01` to `2023-06-30`)
* `2023-H2` (`2023-07-01` to `2023-12-31`)
* `2024-H1` (`2024-01-01` to `2024-06-30`)
* `2024-H2` (`2024-07-01` to `2024-12-31`)
* `2025-H1` (`2025-01-01` to `2025-06-30`)
* `2025-H2` (`2025-07-01` to `2025-12-31`)
* **Required Metrics per Window:** $N$, win rate, profit factor, expectancy, net PnL, net return, MaxDD.
* **Objective:** Measure consistency across varying macro regimes without excluding unfavorable periods.

### Diagnostic E: Trade-Profit Concentration Audit
Quantify the degree to which 2025 validation performance depends on extreme outlier trades:
* **Required Metrics:** Total closed trades, total net PnL, PnL of top-1 trade (% of total), PnL of top-3 trades (% of total), PnL of top-5 trades (% of total), net PnL excluding top-5 trades.
* **Objective:** Confirm that the validation edge reflects a repeatable statistical distribution rather than a handful of unrepeatable outlier windfalls.

### Diagnostic F: In-Sample / Validation Context & Degradation Audit
Record the exact historical baseline metrics for `C33`:
* **Train (2022–2024):** $N = 151$, $PF = 0.9069$, $E = -\$0.42$, $\text{Net Return} = -6.34\%$, $\text{MaxDD} = 30.46\%$
* **Validation (2025):** $N = 49$, $PF = 1.6032$, $E = +\$2.54$, $\text{Net Return} = +12.46\%$, $\text{MaxDD} = 21.07\%$
* **Formal Disclosure:** The negative in-sample training performance ($PF = 0.9069$) and subsequent validation surge ($PF = 1.6032$) is explicitly flagged as a major regime-sensitivity concern requiring rigorous explanation in the diagnostic phase.

---

## 6. Multiple-Testing & Selection-Bias Disclosure

1. **Grid Search Selection Context:**
   * Candidate `C33` was selected from a discrete 48-combination calibration grid evaluated on 2025 validation data.
   * Consequently, the 2025 validation period **cannot be treated as a purely blind, independent test set**.
2. **Plateau vs Spurious Fit:**
   * While 26 of 48 combinations passed validation and 6 of 6 in-grid neighbors passed, this clustering constitutes *supportive exploratory evidence*, not proof of statistical significance.
3. **Purpose of V2.0 Pre-OOS Diagnostics:**
   * The diagnostics specified in Section 5 are designed to adversarially probe the stability and economic validity of `C33` prior to risking the uncompromised 2026 OOS dataset.

---

## 7. Future OOS Authorization Gate & Proposed Evaluation Framework

### 7.1 Formal Prerequisites for OOS Unlocking
The 2026 OOS dataset will remain strictly locked until ALL of the following criteria are satisfied:
1. Complete execution of all five pre-OOS diagnostics on pre-2026 data.
2. Production of a formal diagnostic review memo (`data/vwap_eth1h_pre_oos_diagnostic.md`).
3. Confirmation that zero parameter changes or code tweaks were made to `C33`.
4. Independent human gatekeeper sign-off explicitly authorizing the OOS test run.

### 7.2 Proposed OOS Success Criteria (Reference Only — Not Yet Authorized)
Should OOS evaluation be authorized in a future stage, `C33` must satisfy the following proposed single-shot thresholds on the 2026 OOS slice:
* **Trade Count ($N$):** $N \ge 15$
* **Profit Factor ($PF$):** $PF \ge 1.1000$
* **Expectancy ($E$):** $E > \$0.00$ per trade
* **Maximum Drawdown ($MDD$):** $MDD \le 30.0\%$
* **Liquidations ($L$):** $L = 0$

---

## 8. Post-OOS Failure & Anti-Retuning Policy

* **Single-Shot Verification:** The future OOS test is a strictly single-shot verification gate.
* **Prohibition on Retuning:** If `C33` fails the OOS gate, **no post-hoc parameter adjustments, threshold relaxations, or cherry-picking are permitted.**
* **Consequence of Failure:** If OOS fails, Candidate `C33` and the `VWAPBollingerMRStrategy` family will be permanently marked as **REJECTED AND ARCHIVED**. 2026 data will not be reused for model calibration.

---

## 9. Live Trading Boundary

* **No Automatic Live Deployment:** Successful completion of pre-OOS diagnostics and even a passing future OOS result **do not authorize live trading or real-capital deployment**.
* **Production Deployment Requirements:** Any live deployment requires an independent, dedicated Phase 13/14 production readiness review covering:
  * Binance API key management and endpoint security
  * Real-time WebSocket order-state reconciliation
  * Hard account-level loss circuit breakers and kill switches
  * Maximum notional exposure and leverage controls
  * Slippage protection, order rejection handlers, and network disconnection recovery

---

## 10. Governance Summary & Sign-Off Block

| Item | Specification |
| :--- | :--- |
| **Protocol ID** | `PROTOCOL-VWAP-ETH1H-V2.0` |
| **Strategy Slug** | `vwap_bollinger_mr` |
| **Asset & Timeframe** | `ETHUSDT` 1H |
| **Frozen Candidate** | `C33` (`ADX=25.0, BB=(20, 2.0), RSI=(14, 35/65), ATR=1.8, RR=1.5`) |
| **V2.0 Protocol Status** | **`PROPOSED / NOT YET APPROVED`** |
| **2026 OOS Status** | **`LOCKED — ZERO ACCESS AUTHORIZED`** |
| **Next Required Action** | Human gatekeeper review and formal approval of Protocol V2.0 |

---
*Document authored by Quantitative Research Governance. Stored at `data/vwap_eth1h_protocol_v2.md`.*
