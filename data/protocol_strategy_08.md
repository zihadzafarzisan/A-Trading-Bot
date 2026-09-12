# PROTOCOL-STRATEGY-08-V1.1: Adaptive Displacement & Structural Trailing Protocol

## Canonical Protocol Identity

| Field | Value |
| :--- | :--- |
| Protocol ID | `PROTOCOL-STRATEGY-08-V1.1` |
| Version | `1.1.0` |
| Strategy | `AdaptiveDisplacementTrailingStrategy` |
| Slug | `adaptive_displacement_trailing` |
| Status | `DESIGN DRAFT — NOT APPROVED` |

## Authorization Summary (explicitly stated)

This document is a canonical design draft only. The following facts are explicitly recorded:

- No implementation has been authorized.
- No backtests have been authorized.
- No Stage-A execution has occurred.
- No Stage-B execution has occurred.
- 2026 Out-of-Sample (OOS) data remains blind and locked.
- No OOS authorization exists.

Persons or processes reading this document shall NOT infer any of the above
authorizations merely from the existence of this protocol document.

---

## 1. Execution Timeline

All signals are generated on **completed** bars and executed at the open of the
next bar.

- A signal is computed on the completed signal bar `k`.
- The resulting order is executed at the **next-bar open** `k+1`.
- No intrabar execution is permitted for signal-marked entries.

### 1.1 Entry Fill Price

For a long entry:

`FillPrice = open[k+1] × (1 + slippage)`

For a short entry:

`FillPrice = open[k+1] × (1 - slippage)`

Where `slippage` is the dynamic slippage parameter defined under Execution
Assumptions (Section 10) and `open[k+1]` is the open price of bar `k+1`.

---

## 2. Initial Risk, Initial Stop, and Take Profit

### 2.1 Initial Risk Amount

The initial per-trade risk `R` is defined exactly as:

`R = atr_multiplier × ATR_k(14)`

Where:
- `atr_multiplier` is the configured ATR multiplier parameter.
- `ATR_k(14)` is the 14-period Average True Range evaluated on the completed
  bar `k` (the signal bar).

This MUST NOT be simplified to `ATR_k(14)`; the `atr_multiplier` scaling is
mandatory and always present.

### 2.2 Initial Stop Price

The initial protective stop is placed at distance `R` from the fill price.

For a long:

`InitialStop = FillPrice - R`

For a short:

`InitialStop = FillPrice + R`

`InitialStop` is the initial stop price recorded at entry.

### 2.3 Take Profit Price

Take profit uses the configured risk-reward ratio applied to the initial risk
`R`:

`TakeProfit = FillPrice + (risk_reward_ratio × R)`   for a long

`TakeProfit = FillPrice - (risk_reward_ratio × R)`   for a short

For the C00 baseline, `risk_reward_ratio = 2.0`, producing a fixed 1:2
initial-risk-to-target structure.

---

## 3. Breakeven Logic

The breakeven stop is derived from the realized risk to-date:

`R = |FillPrice - InitialStopPrice|`

Where `InitialStopPrice` is the initial stop price recorded at entry
(Section 2.2).

The breakeven trigger distance is exactly `+1.0R`.

### 3.1 Breakeven Trigger Condition

The breakeven trigger is evaluated on the **completed-bar CLOSE** only.
High and low prices are NEVER used for the breakeven trigger, and the stop is
NEVER moved intrabar.

For a long:

`Close_m >= FillPrice + 1.0R`

For a short:

`Close_m <= FillPrice - 1.0R`

Where `Close_m` is the close of the completed bar `m` on which the trigger is
detected.

### 3.2 Breakeven Stop Transition

When the condition is triggered on the completed bar `m`, the active stop for
the following bar is set to the fill price:

`ActiveStop_(m+1) = FillPrice`

The stop transition becomes effective on the NEXT bar `m+1`. It is NOT applied
in the same bar `m`.

---

## 4. ATR Trailing Stop Logic

After breakeven has been activated, the active stop is ratcheted by a
14-period ATR trailing formula. The ATR trailing uses:

`ATR_j(14)`

evaluated on the completed bar `j`.

### 4.1 Trailing Candidate

For a long:

`TrailingCandidate_j = Close_j - (atr_multiplier × ATR_j(14))`

For a short:

`TrailingCandidate_j = Close_j + (atr_multiplier × ATR_j(14))`

### 4.2 Ratchet (Monotonic Stop)

For a long:

`ActiveStop_(j+1) = max(ActiveStop_j, TrailingCandidate_j)`

For a short:

`ActiveStop_(j+1) = min(ActiveStop_j, TrailingCandidate_j)`

The stop may NEVER move backwards. Trailing updates are based on completed
bars and become effective on the following bar. No intrabar trailing move is
permitted.

---

## 5. Fair Value Gap (FVG) Definition

A THREE-candle displacement structure is required. Let `t` denote the reference
completed bar index of the signal candle, and `EMA50` the 50-period EMA.

### 5.1 Bullish FVG

The bullish FVG configuration requires ALL of the following conditions on the
completed bar `t-1`:

`Close_(t-1) > Open_(t-1)`

AND

`Low_(t-1) > High_(t-3)`

AND

`Close_(t-1) > EMA50_(t-1)`

The bullish zone (the unfilled imbalance) is:

`[High_(t-3), Low_(t-1)]`

### 5.2 Bearish FVG

The bearish FVG configuration requires ALL of the following conditions on the
completed bar `t-1`:

`Close_(t-1) < Open_(t-1)`

AND

`High_(t-1) < Low_(t-3)`

AND

`Close_(t-1) < EMA50_(t-1)`

The bearish zone is:

`[High_(t-1), Low_(t-3)]`

---

## 6. FVG Mitigation Conditions

An entry is valid only when price is about to re-test / mitigate a previously
formed FVG. Let `k` denote the entry (signal) bar. Equality at the zone
boundaries is ALLOWED.

### 6.1 Long Mitigation

All of the following conditions must hold on the completed bar `k`:

`Low_k <= Low_(t-1)`

AND

`Low_k >= High_(t-3)`

AND

`Close_k > EMA50_k`

### 6.2 Short Mitigation

All of the following conditions must hold on the completed bar `k`:

`High_k >= High_(t-1)`

AND

`High_k <= Low_(t-3)`

AND

`Close_k < EMA50_k`

### 6.3 FVG Validity Window

An FVG formed at reference bar `t` is valid for mitigation only while:

`k ∈ [t, t + displacement_lookback - 1]`

If multiple valid FVGs exist at the same bar, the MOST RECENT active FVG is
used.

---

## 7. Kaufman Efficiency Ratio (ER) Regime Filter

The Kaufman Efficiency Ratio is computed over a fixed period:

`n = 10`

### 7.1 Directional Change

`DirectionalChange_k = |Close_k - Close_(k-n)|`

### 7.2 Total Volatility

`TotalVolatility_k = Σ_(r=0)^(n-1) |Close_(k-r) - Close_(k-r-1)|`

### 7.3 Efficiency Ratio

When `TotalVolatility_k > 0`:

`ER_k = DirectionalChange_k / TotalVolatility_k`

Otherwise (when `TotalVolatility_k = 0`):

`ER_k = 0.0`

The zero-denominator behavior MUST be explicit: if total volatility is zero,
the efficiency ratio is defined as `0.0` rather than throwing a division
error or producing NaN/infinity.

The ER regime filter requires `ER_k >= er_threshold` for a long candidate and
`ER_k <= -er_threshold` for a short candidate (i.e., directional momentum in
the trade direction).

---

## 8. Parameter Baseline (C00) and Optimization Grid

### 8.1 C00 Baseline Configuration

The C00 baseline is the canonical default parameter configuration. It is:

```json
{
  "candidate_id": "C00",
  "body_mult": 1.6,
  "volume_mult": 1.7,
  "volume_period": 20,
  "ema_period": 50,
  "er_period": 10,
  "er_threshold": 0.30,
  "displacement_lookback": 5,
  "atr_period": 14,
  "atr_multiplier": 2.2,
  "risk_reward_ratio": 2.0,
  "breakeven_trigger_r": 1.0,
  "stop_type": "atr_breakeven"
}
```

This JSON object MUST NOT be merged or re-keyed; every key and value is
preserved exactly.

### 8.2 Optimization Grid

The optimization grid is:

`body_mult = [1.4, 1.7]`

`volume_mult = [1.5, 1.8]`

`er_threshold = [0.25, 0.35, 0.45]`

`atr_multiplier = [1.8, 2.2, 2.5]`

This produces:

`2 × 2 × 3 × 3 = 36`

parameter combinations.

Adding the `C00` baseline yields:

`36 + 1 = 37`

total configurations. No additional optimization dimensions are introduced.

---

## 9. Stage-A and Stage-B Validation Gates

### 9.1 Stage-A Gate

For each candidate and each asset, Stage-A passes only if ALL of the following
hold:

`N >= 30`

AND NOT:

`PF < 0.75 AND MDD > 75%`

Where `N` is the number of trades, `PF` the profit factor, and `MDD` the
maximum drawdown state. **Both `BTCUSDT` and `ETHUSDT` must be evaluated** under
the Stage-A gate.

### 9.2 Stage-B Gate

A candidate passes Stage-B only if ALL of the following hold:

`N >= 15`

AND

`PF >= 1.0500`

AND

`Expectancy > 0`

AND

`MDD <= 30.00%`

AND

`Liquidations == 0`

### 9.3 Dual-Asset Intersection Rule (Final Candidate Set)

The final dual-asset candidate set MUST be the exact parameter-ID intersection:

`BTC_PASS_IDS ∩ ETH_PASS_IDS`

This is an EXACT parameter-ID intersection: a candidate ID is accepted only if
the SAME parameter ID passes Stage-B on BOTH assets.

The following methods are STRICTLY FORBIDDEN:

- union of pass sets
- independently selecting the best BTC and ETH candidate
- substitution
- nearest candidate
- manual replacement
- asset-specific candidate mismatch

The final candidate set must be identical on both assets by construction.

---

## 10. Data Partitions

### 10.1 Chronological Partitions

- BTCUSDT train: `2020-01-01 00:00:00 UTC` through `2024-12-31 23:59:59 UTC`
- ETHUSDT train: `2022-01-01 00:00:00 UTC` through `2024-12-31 23:59:59 UTC`
- Validation (both assets): `2025-01-01 00:00:00 UTC` through
  `2025-12-31 23:59:59 UTC`
- Blind Out-of-Sample (OOS): `2026-01-01 00:00:00 UTC` through
  `2026-08-31 23:59:59 UTC`

### 10.2 OOS Lock

The 2026 blind OOS window is LOCKED and cannot be used for:

- parameter tuning
- candidate selection
- threshold adjustment
- protocol modification
- strategy redesign

Zero bytes of 2026 data may be accessed, queried, or loaded prior to explicit
human OOS authorization (Section 12).

---

## 11. Execution Assumptions

The following canonical project execution assumptions apply and MUST NOT be
silently changed:

- Initial capital: `$1,000`
- Risk per trade: `1%`
- Maximum simultaneous positions: `3`
- Leverage: `5x`
- Taker fee: `0.04%` per side
- Dynamic slippage
- Funding interval: `8 hours`
- Conservative stop-first collision handling

---

## 12. Governance

### 12.1 Hard Risk Gates Are Not Overridable

Positive profitability does NOT override pre-declared hard risk gates. A
candidate that violates any Stage-A or Stage-B hard gate is rejected even if
it is profitable.

### 12.2 No Post-OOS Modification

No post-OOS activity is permitted after an OOS failure:

- retuning
- candidate swapping
- threshold relaxation
- protocol modification

Any OOS failure is final for the evaluated configuration.

### 12.3 Research Pipeline Order

The research pipeline is strictly ordered:

`Protocol`
→ `forensic verification`
→ `human approval`
→ `implementation`
→ `unit/invariant tests`
→ `historical validation`
→ `Stage-A`
→ `Stage-B`
→ `blind OOS authorization`
→ `OOS`
→ `robustness / Monte Carlo`
→ `paper trading`
→ `live trading`

No stage may be skipped.

### 12.4 Strategy #7 Immutability

Strategy #7 is permanently archived and locked. Its canonical OOS artifacts
MUST NOT be modified. For verification only, the following canonical SHA-256
hashes must remain unchanged:

- Runner: `13897d1d4683de86d89b3db7dae43c6f622a7cdc398f7fb409be7459bbc75879`
- OOS JSON: `ad55dc1038271f33bfb066144cf87004be58b93fde277c26d6c9dd3cfbc63fc3`
- OOS Markdown: `d22f61854e7fdf8f94416257f4d8eaa3013d13896ba59a82e874953e1c4f858a`
- Authorization: `a46d4bc7fd23b47009454bd7b58d9b4fdeda50bb068d736327f756e9437890a9`

If any of these hashes change, STOP and report a critical violation.

---

## 13. Disposition

This document is the single canonical Strategy #8 V1.1 protocol. No code is
implemented, no backtest is run, and no 2026 OOS authorization exists at the
time of this writing. Any downstream stage requires standing at the correct
point of the pipeline (Section 12.3) and securing explicit human approval.
