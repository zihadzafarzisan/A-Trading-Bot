# VWAP / Bollinger Mean Reversion Strategy: Source & Historical Implementation Audit (v2)

**Audit Scope:** Read-only source code and historical artifact verification.  
**Target Class:** `VWAPBollingerMRStrategy` in `src/crypto_quant/strategies/vwap_bollinger_mr.py`  
**Operational Rule:** Zero code modified, zero backtests executed, zero 2026 OOS data accessed, zero parameters tuned.

---

## 1. Source Code Implementation Audit

### 1.1 Class Specification & Metadata
* **File Location:** `src/crypto_quant/strategies/vwap_bollinger_mr.py`
* **Class Name:** `VWAPBollingerMRStrategy(BaseStrategy)`
* **Supported Markets:** `supports_spot = True`, `supports_futures = True`, `supports_short = True`
* **Default Timeframes:** `["15m", "1h", "4h"]`
* **Default Parameters:**
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
* **Declared Parameter Grid (`param_grid()`):**
  * `adx_max_thresh`: `[20.0, 22.0, 25.0]` (3 values)
  * `bb_period`: `[20]` (1 value)
  * `bb_std`: `[2.0, 2.2]` (2 values)
  * `rsi_period`: `[14]` (1 value)
  * `rsi_oversold`: `[30.0, 35.0]` (2 values)
  * `rsi_overbought`: `[65.0, 70.0]` (2 values)
  * `atr_multiplier`: `[1.5, 2.0, 2.5]` (3 values)
  * `risk_reward_ratio`: `[1.5, 2.0]` (2 values)
  * `stop_type`: `["atr"]` (1 value)
  * *Total Grid Combinations:* $3 \times 1 \times 2 \times 1 \times 2 \times 2 \times 3 \times 2 \times 1 = 144$ potential combinations (note: Step 3b evaluated a 24-combination subset).

---

### 1.2 Indicator Vectorization in `setup()`
All indicator columns are computed once during `setup(df)` using causal point-in-time shifts (`shift=1`):

1. **ADX (14):** `calc_adx(high, low, close, 14, shift=1)`
2. **RSI (14):** `calc_rsi(close, rsi_period, shift=1)`
3. **ATR (14):** `calc_atr(high, low, close, 14, shift=1)` -> stored as `out["atr_14"]`
4. **VWAP (20):** `calc_vwap(high, low, close, volume, vwap_period, shift=1)` -> rolling 20-bar volume-weighted price
5. **Bollinger Bands (20, 2.0):** `calc_bb(close, bb_period, bb_std, shift=1)` -> produces `bb_mid`, `bb_upper`, `bb_lower`, `bb_width`

---

### 1.3 Exact Signal & Trigger Logic in `entry_signal()`

* **Warmup Guard:**
  ```python
  if i < 30:
      return Direction.NONE, ""
  ```
  Returns `Direction.NONE` for the first 30 bars of the backtest slice.

* **Regime Filter:**
  $$\text{range\_regime}_t = (\text{ADX}_{t-1} \le \text{adx\_max\_thresh})$$

* **Long Setup Condition (`_long_setup`):**
  $$\text{Low}_{t-1} \le \text{BB\_Lower}_{t-1} \quad \land \quad \text{RSI}_{t-1} \le \text{RSI\_Oversold} \quad \land \quad \text{Close}_{t-1} \ge \text{Open}_{t-1} \quad \land \quad \text{range\_regime}_t$$

* **Short Setup Condition (`_short_setup`):**
  $$\text{High}_{t-1} \ge \text{BB\_Upper}_{t-1} \quad \land \quad \text{RSI}_{t-1} \ge \text{RSI\_Overbought} \quad \land \quad \text{Close}_{t-1} \le \text{Open}_{t-1} \quad \land \quad \text{range\_regime}_t$$

* **Long / Short Symmetry:**
  The trigger structure is strictly symmetric in its boolean conditions:
  * Long checks `Low <= Lower Band`, `RSI <= Oversold`, and a green rejection bar (`Close >= Open`).
  * Short checks `High >= Upper Band`, `RSI >= Overbought`, and a red rejection bar (`Close <= Open`).

---

### 1.4 Crucial Source Facts on VWAP & BB Mid-Band
1. **VWAP Usage:**
   * `out["vwap"]` is computed in `setup()`.
   * **Fact:** `out["vwap"]` is **never referenced** in `_long_setup`, `_short_setup`, `entry_signal()`, or anywhere in the strategy logic. Despite the strategy's name, VWAP plays zero active role in signal generation or exits.
2. **Bollinger Middle Band Usage:**
   * `out["bb_mid"]` is computed in `setup()`.
   * **Fact:** `out["bb_mid"]` is **never referenced** in entry or exit logic. The strategy does not target the middle band for take-profit or trailing exits.

---

### 1.5 Stop-Loss & Take-Profit Implementation

1. **Stop Loss:**
   * Derived in `BaseStrategy.compute_stop_loss()` using `StopLossSpec(stop_type=StopType.ATR, atr_multiplier=...)`.
   * Long Stop Price: $\text{Close}_i - (\text{atr\_multiplier} \times \text{ATR}_{14, i})$
   * Short Stop Price: $\text{Close}_i + (\text{atr\_multiplier} \times \text{ATR}_{14, i})$
   * **Fixed vs Trailing:** The stop price is calculated once at the close of signal bar $i$ and attached to the position. In `BacktestEngine._manage_exits()`, this price level remains **strictly fixed** throughout the life of the position. It is **NOT** a trailing stop.
2. **Take Profit:**
   * Derived in `BaseStrategy.compute_take_profit()` using `TakeProfitSpec(mode="rr", risk_reward_ratio=...)`.
   * Defined by risk distance: $\text{Risk} = |\text{Close}_i - \text{StopLoss}_i|$
   * Long TP Price: $\text{Close}_i + (\text{Risk} \times \text{risk\_reward\_ratio})$
   * Short TP Price: $\text{Close}_i - (\text{Risk} \times \text{risk\_reward\_ratio})$
   * Enforced as a static limit exit in `BacktestEngine._manage_exits()`.

---

## 2. Causality & Execution Timing

* **Indicator Shifts:**
  All indicator outputs are explicitly shifted by 1 bar (`shift=1`), ensuring values at index $i$ represent information as of the close of bar $i-1$.
* **Price Series Shifts:**
  The OHLC checks in `setup()` explicitly shift prices (`out["low"].shift(1)`, `out["close"].shift(1)`, `out["open"].shift(1)`), ensuring that the reaction candle evaluation strictly observes closed bar $i-1$.
* **Execution Flow in `BacktestEngine`:**
  1. Signal generated at the close of bar $i$ using data through bar $i-1$.
  2. Order is placed into `pending`.
  3. Order is filled at `open[i+1]` on the next bar.
  4. Intrabar collision rule: If both stop-loss and take-profit thresholds are breached within the same bar, the engine conservatively assumes the stop-loss was hit first.
* **Lookahead Assessment:** **Zero lookahead bias detected.** The causal structure is rigorous and completely point-in-time.

---

## 3. Historical Empirical Evidence (Verified from Artifacts)

### 3.1 Step 2: Full-Period Baseline Sweep (Futures, Default Parameters)
Source: `data/50_experiments_report.json`

| Experiment ID | Symbol | TF | Trades ($N$) | Win Rate | Profit Factor | Expectancy | Net Return | Max DD | Net PnL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_05` | BTCUSDT | 5m | 1,112 | 33.5% | 0.5390 | −$0.53 | −58.6% | 66.4% | −$586.00 |
| `EXP_10` | BTCUSDT | 15m | 351 | 32.5% | 0.6533 | −$0.86 | −30.1% | 44.9% | −$301.28 |
| `EXP_15` | BTCUSDT | 1h | 192 | 33.9% | 0.8420 | −$0.81 | −15.6% | 42.2% | −$155.54 |
| **`EXP_20`** | **BTCUSDT** | **4h** | **64** | **42.2%** | **1.3337** | **+$1.98** | **+12.7%** | **21.6%** | **+$126.76** |
| `EXP_25` | BTCUSDT | 1d | 3 | 100.0% | $\infty$ | +$19.93 | +6.0% | 5.1% | +$59.78 |
| `EXP_30` | ETHUSDT | 5m | 1,210 | 31.5% | 0.6042 | −$0.53 | −64.1% | 73.6% | −$640.76 |
| `EXP_35` | ETHUSDT | 15m | 368 | 32.1% | 0.7279 | −$0.83 | −30.5% | 49.1% | −$305.07 |
| `EXP_40` | ETHUSDT | 1h | 127 | 38.6% | 1.1349 | +$0.75 | +9.5% | 25.9% | +$94.69 |
| **`EXP_45`** | **ETHUSDT** | **4h** | **43** | **62.8%** | **3.1028** | **+$9.42** | **+40.5%** | **18.2%** | **+$405.12** |
| `EXP_50` | ETHUSDT | 1d | 2 | 0.0% | 0.0000 | −$10.07 | −2.0% | 3.9% | −$20.13 |

**Key Observation:** 4H was the only timeframe showing consistent profitability across both BTC ($PF=1.33$) and ETH ($PF=3.10$). Sub-4H timeframes suffered severe degradation and fee erosion, while 1D produced negligible sample sizes ($N \le 3$).

---

### 3.2 Step 3b: Train (2020/2022–2024) vs Validation (2025)
Source: `data/step_3b_optimization_report.json`

Both targets evaluated a 24-combination grid on 4H futures. For both assets, Candidate **#19** (`adx_max_thresh=25.0, atr_multiplier=1.8, risk_reward_ratio=2.0`) was the top validation scorer:

#### Target `BTC_4H_VWAP_MR` (Candidate #19)
* **Train Partition (2020–2024):**
  * $N = 80$, $\text{Win Rate} = 45.00\%$, $\text{Profit Factor} = 1.5514$, $\text{Expectancy} = +\$3.1684$, $\text{Net Return} = +25.35\%$, $\text{MaxDD} = 21.37\%$
  * Gate Result: **PASSED** (Score = $0.3247$)
* **Validation Partition (2025):**
  * $N = 14$, $\text{Win Rate} = 35.71\%$, $\text{Profit Factor} = 1.0869$, $\text{Expectancy} = +\$0.5079$, $\text{Net Return} = +0.71\%$, $\text{MaxDD} = 20.18\%$
  * Gate Result: **FAILED** strictly on trade count ($N = 14 < 15$ gate). Note that $PF \ge 1.05$, $E > 0$, and $MDD \le 40\%$ were technically satisfied.

#### Target `ETH_4H_VWAP_MR` (Candidate #19)
* **Train Partition (2022–2024):**
  * $N = 56$, $\text{Win Rate} = 53.57\%$, $\text{Profit Factor} = 2.0906$, $\text{Expectancy} = +\$5.9716$, $\text{Net Return} = +33.44\%$, $\text{MaxDD} = 19.77\%$
  * Gate Result: **PASSED** (Score = $0.4055$)
* **Validation Partition (2025):**
  * $N = 12$, $\text{Win Rate} = 33.33\%$, $\text{Profit Factor} = 0.8809$, $\text{Expectancy} = -\$0.8224$, $\text{Net Return} = -0.99\%$, $\text{MaxDD} = 18.68\%$
  * Gate Result: **FAILED** on multiple gates ($N = 12 < 15$, $PF = 0.8809 < 1.05$, $E = -\$0.82 \le 0$).

---

## 4. Structured Research Categorization

To maintain strict scientific rigor, all observations are categorized into four distinct levels:

### Category A: Facts Established by Source Code
1. `VWAPBollingerMRStrategy` is an ADX-filtered Bollinger-RSI extreme fader.
2. `out["vwap"]` is calculated but **never used** in entry or exit logic.
3. `out["bb_mid"]` is calculated but **never used** in entry or exit logic.
4. Stop-loss is a **fixed price level** derived from ATR at entry, not a trailing stop.
5. Take-profit is a **fixed price level** derived from risk distance multiplied by a static $R:R$ ratio.
6. Indicator and price shifts enforce strict point-in-time causality with zero lookahead bias.

### Category B: Historical Empirical Observations
1. In Step 2 (full period), the 4H timeframe produced positive returns and profit factors ($PF=1.33$ BTC, $PF=3.10$ ETH), whereas lower timeframes (5m, 15m, 1h) suffered negative expectancy or high fee drag.
2. In Step 3b (2025 validation), both BTC ($N=14$) and ETH ($N=12$) generated fewer than 15 trades over the 1-year validation window, causing both to fail the $N \ge 15$ gate.
3. In Step 3b (2025 validation), ETH Candidate #19 degraded to $PF=0.88$ and negative expectancy ($-\$0.82$).

### Category C: Research Hypotheses (Untested Ideas)
1. *Sample Size Hypothesis:* The 4H timeframe in isolation may generate insufficient trade opportunities over standard 1-year validation windows to reliably satisfy sample size gates ($N \ge 15$).
2. *Exit Mechanism Hypothesis:* Exiting via fixed $R:R$ multiples during low-volatility range regimes may hold trades too long when price reaches the mean, potentially converting winning mean-reversions into stop-outs. *(Note: This is an unverified hypothesis, not an established fact).*
3. *VWAP Integration Hypothesis:* Explicitly incorporating VWAP as a filter or target level might provide additional structural information that is currently absent.

### Category D: Proposed Future Protocol Objectives
1. Formalize a multi-stage research protocol (Stage-A Baseline Screening $\rightarrow$ Stage-B Parameter Optimization $\rightarrow$ Frozen Candidate Selection $\rightarrow$ OOS Verification) specifically tailored for Mean Reversion strategies.
2. Formulate explicit criteria for resolving the trade-frequency vs timeframe dilemma without prematurely modifying code or violating dataset blind gates.

---

## 5. Audit Conclusion & Gate Status

* **Source Code State:** Verified exact, clean, and causal.
* **Empirical Baseline State:** Strong historical 4H full-period evidence, but limited by small 2025 validation sample sizes ($N=14$ and $N=12$).
* **Integrity Confirmation:**
  * Code modified: **NO**
  * Backtests executed: **NO**
  * 2026 OOS accessed: **NO**
  * Parameters tuned: **NO**
