# MTF ETH-Only OOS Protocol Decision Memo

**Date:** 2026-09-11  
**Scope:** Research Gate & Protocol Decision. No code modified, no backtests executed, no 2026 OOS data accessed, no parameters tuned, no OOS executed.  
**Subject:** Determination of OOS Evaluation Policy for ETHUSDT Stage-B Candidate #56 (`1d_4h_1h`, 1H execution).

---

## 1. Research Objective of Out-Of-Sample (OOS) Testing

The primary objective of the OOS evaluation phase in this quant research framework is to serve as a **single-shot, blind verification gate** to measure genuine edge generalization and protect against selection bias and overfitting accumulated during in-sample training and validation.

In the MTF research track (Stage-A and Stage-B), the overarching research objective was to develop and validate a **cross-asset, multi-timeframe structural trend-pullback strategy** across both core crypto assets (`BTCUSDT` and `ETHUSDT`). The Stage-A gatekeeper explicitly enforced dual-asset survival as a prerequisite for architectural advancement.

---

## 2. Protocol Classification

**Status:** **NEW PROTOCOL DECISION REQUIRED**

- **Not Explicitly Authorized:** The existing protocol (PROTOCOL-MTF-V1.5) does not explicitly authorize advancing a single-asset candidate to OOS when the other primary asset (BTC) fails to produce a passing candidate. While the Stage-B optimization script evaluates symbols independently in parallel loops, this is an artifact of execution architecture, not an explicit governance authorization to proceed with single-asset OOS evaluation.
- **Not Explicitly Prohibited:** The protocol does not contain an explicit prohibition forbidding single-asset OOS after Stage-B candidate selection.
- **Conclusion:** The protocol is silent regarding asymmetric (single-asset) qualification outcomes at Stage-B. Therefore, a **new protocol decision is required**.

---

## 3. Methodological Implications of ETH #56

Allowing ETH #56 to proceed to OOS in isolation presents severe methodological risks:

1. **Complete Absence of BTC Validation (0/72 Combinations):**
   - The strategy architecture failed across all 72 parameter permutations on BTCUSDT. This indicates that the MTF trend-pullback mechanism lacks cross-asset universality and is sensitive to asset-specific market microstructure or regime dynamics.
2. **Underpowered Sample Size ($N = 17$):**
   - 17 trades over the full 2025 validation year (~1.4 trades/month) is statistically underpowered. With such a small $N$, the standard error of the profit factor is substantial, making performance metrics highly vulnerable to small-sample variance.
3. **Marginal Profit Factor ($PF = 1.0565$ vs Gate $1.0500$):**
   - The candidate clears the gate by a razor-thin margin of **+0.0065**. A single losing trade would have caused the candidate to fail.
4. **Economic Expectancy Decay:**
   - Validation expectancy is only **+$0.36** per trade (down 82.8% from Train expectancy of +$2.08$), representing an economically negligible edge relative to risk.
5. **Multiple Testing & Search Space Dynamics:**
   - Only **3 of 72 (4.2%)** ETH combinations passed validation, and all three tied at the exact same score ($0.0874$) due to identical performance across varying ITF RSI settings (45, 50, 55).
   - The surrounding parameter neighborhood in ADX, ATR multiplier, and Risk-Reward ratio predominantly fails the hard gates. This indicates a flat, narrow plateau rather than a robust, stable parameter basin.
6. **Overall Assessment:**
   - Candidate #56 is classified as a **MARGINAL RESEARCH CANDIDATE**. Proceeding to OOS with a marginal candidate in the absence of corroborating cross-asset evidence carries a high probability of OOS degradation.

---

## 4. Policy Alternatives

### POLICY A — NO ETH-ONLY OOS (Strict Cross-Asset Requirement)
- **Rule:** Require a simultaneously qualified `BTCUSDT` + `ETHUSDT` pair before authorizing MTF OOS evaluation.
- **Implications:**
  - ETH #56 is frozen and archived at Stage-B without unlocking OOS.
  - The MTF `1d_4h_1h` strategy is marked as non-deployable in its current parameterization due to lack of cross-asset validation.
  - Preserves the purity of the 2026 OOS dataset for future strategy families and avoids spending OOS degrees of freedom on a statistically marginal candidate.

### POLICY B — ASSET-SPECIFIC OOS (Isolated Single-Asset Evaluation)
- **Rule:** Authorize a separately declared, asset-specific OOS evaluation exclusively for `ETHUSDT` Candidate #56, under strict research constraints.
- **Required Declarations:**
  - Explicitly record that this does **not** constitute dual-asset or cross-market validation.
  - Formally record that `BTCUSDT` is REJECTED and failed to qualify.
  - Formally record that ETH #56 enters OOS as a **MARGINAL RESEARCH CANDIDATE**.
  - Strict prohibition: OOS results on ETH can **never** be used to tune, re-optimize, or back-fit the BTC model or MTF parameters.
  - 2026 OOS dataset remains blind and locked until formal execution authorization is provided.

---

## 5. Formal Recommendation

**RECOMMENDATION: POLICY A (NO ETH-ONLY OOS)**

**Rationale:**
1. **Research Integrity & OOS Preservation:** The 2026 OOS data is a strictly finite, non-renewable statistical resource. Spending this validation gate on a candidate with $N=17$, $PF=1.0565$ (+0.0065 margin), and 0/72 BTC confirmation represents an inefficient and high-risk expenditure of OOS data.
2. **Failure of Core Premise:** The MTF research track was initiated on the hypothesis that multi-timeframe confirmation provides a robust structural edge across major crypto markets. A 0% pass rate on BTC disconfirms the cross-asset robustness hypothesis.
3. **High Probability of False Discovery:** Given that only 3/72 combinations passed and the optimum is a fragile plateau, the candidate exhibits hallmarks of validation curve-fitting.

*If leadership decides to override and adopt Policy B for diagnostic/academic interest, all caveats and prohibitions in Policy B must be strictly enforced.*

---

## 6. Corrected Preflight Execution & Boundary Specifications (For Reference Only)

Should a future protocol decision authorize an OOS evaluation, the preflight configuration is strictly defined as follows:

- **Execution Model:** Futures (USDT-M Perpetual)
- **Execution Timeframe:** 1H (`1d_4h_1h` architecture uses 1D macro trend, 4H intermediate filter, 1H entry execution)
- **Leverage:** 5x
- **Initial Capital:** $1,000.00
- **Risk per Trade:** 1.0%
- **Max Open Positions:** 3
- **Fee Model:** 0.04% taker fee (per side)
- **Funding & Slippage:** 8h funding rate accounting and dynamic tick slippage embedded in fill price
- **Execution Timing:** Next-bar open
- **Intra-bar Collision Rule:** Stop-loss prioritized over take-profit (conservative assumption)
- **Exact OOS Boundary:** `2025-12-31 00:00:00 UTC` to `2026-09-08 00:00:00 UTC` (as established by `SplitDates(train_end="2024-12-31", validation_end="2025-12-31", test_end="2026-09-08")`)

*Note: OOS data remains blind and has not been accessed.*

---

## 7. Status & Integrity Verification

- **Code modified:** NO
- **Backtests executed:** NO
- **2026 OOS accessed/queried:** NO
- **Parameters tuned:** NO
- **OOS Status:** **LOCKED / NOT EXECUTED**
