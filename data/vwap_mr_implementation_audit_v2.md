# VWAP / Bollinger Mean Reversion Strategy: Source & Historical Implementation Audit (v2)

**Audit Scope:** Read-only source code and historical artifact verification.  
**Target Class:** `VWAPBollingerMRStrategy` in `src/crypto_quant/strategies/vwap_bollinger_mr.py`  
**Operational Rule:** Zero code modified, zero backtests executed, zero 2026 OOS data accessed, zero parameters tuned.

---

## 1. Source Code Implementation Audit

### 1.1 Class Specification & Strategy Registry
* **File Location:** `src/crypto_quant/strategies/vwap_bollinger_mr.py`
* **Class Name:** `VWAPBollingerMRStrategy(BaseStrategy)`
* **Strategy Type Slug:** `"vwap_bollinger_mr"`
* **Registration:** Registered in `src/crypto_quant/strategies/registry.py` under `"vwap_bollinger_mr"` mapping to `VWAPBollingerMRStrategy`.
* **Supported Markets:** `supports_spot = True`, `supports_futures = True`, `supports_short = True`
* **Default Timeframes:** `["15m", "1h", "4h"]`

---

### 1.2 Default Parameters & Parameter Grid
* **Default Parameter Dictionary:**
  * `adx_max_thresh`: `22.0` (float)
  * `bb_period`: `20` (int)
  * `bb_std`: `2.0` (float)
  * `vwap_period`: `20` (int)
  * `rsi_period`: `14` (int)
  * `rsi_oversold`: `35.0` (float)
  * `rsi_overbought`: `65.0` (float)
  * `atr_multiplier`: `2.0` (float)
  * `risk_reward_ratio`: `2.0` (float)
  * `stop_type`: `"atr"` (str)

* **Declared Parameter Grid (`param_grid()` class method):**
  ```python
  {
      "adx_max_thresh": [20.0, 22.0, 25.0],
      "bb_period": [20],
      "bb_std": [2.0, 2.2],
      "rsi_period": [14],
      "rsi_oversold": [30.0, 35.0],
      "rsi_overbought": [65.0, 70.0],
      "atr_multiplier": [1.5, 2.0, 2.5],
      "risk_reward_ratio": [1.5, 2.0],
      "stop_type": ["atr"],
  }
  ```
  *Total search space defined in class:* $3 \times 1 \times 2 \times 1 \times 2 \times 2 \times 3 \times 2 \times 1 = 144$ combinations.

---

### 1.3 Technical Indicators & Setup Vectorization
Inside `setup(self, df: pd.DataFrame) -> pd.DataFrame`:
1. **ADX (14):** `calc_adx(out["high"], out["low"], out["close"], 14, shift=1)` -> `out["adx"]`
2. **RSI (14):** `calc_rsi(out["close"], rsi_p, shift=1)` -> `out["rsi"]`
3. **ATR (14):** `calc_atr(out["high"], out["low"], out["close"], 14, shift=1)` -> `out["atr_14"]`
4. **VWAP (20):** `calc_vwap(out["high"], out["low"], out["close"], out["volume"], vwap_p, shift=1)` -> `out["vwap"]`
   * *Formula in `indicators/volume.py`:* $\text{typical} = (H+L+C)/3$; $\text{VWAP} = \text{rolling\_sum}(\text{typical} \times V, 20) / \text{rolling\_sum}(V, 20)$
5. **Bollinger Bands (20, 2.0):** `calc_bb(out["close"], bb_p, bb_s, shift=1)` -> `out["bb_mid"]`, `out["bb_upper"]`, `out["bb_lower"]`, `out["bb_width"]`

---

### 1.4 Entry & Trigger Logic in `entry_signal()`

* **Warmup Constraint:**
  ```python
  if i < 30:
      return Direction.NONE, ""
  ```
  Bars $0$ to $29$ are skipped.

* **Regime Filter:**
  $$\text{range\_regime}_t = (\text{ADX}_{t-1} \le \text{adx\_max\_thresh})$$

* **Long Setup Condition (`_long_setup`):**
  $$\text{Low}_{t-1} \le \text{BB\_Lower}_{t-1} \quad \land \quad \text{RSI}_{t-1} \le \text{RSI\_Oversold} \quad \land \quad \text{Close}_{t-1} \ge \text{Open}_{t-1} \quad \land \quad \text{range\_regime}_t$$

* **Short Setup Condition (`_short_setup`):**
  $$\text{High}_{t-1} \ge \text{BB\_Upper}_{t-1} \quad \land \quad \text{RSI}_{t-1} \ge \text{RSI\_Overbought} \quad \land \quad \text{Close}_{t-1} \le \text{Open}_{t-1} \quad \land \quad \text{range\_regime}_t$$

* **Long/Short Symmetry:**
  The entry logic is strictly symmetric:
  * Long tests Lower Band breach + RSI $\le$ Oversold + Green rejection bar (`Close >= Open`).
  * Short tests Upper Band breach + RSI $\ge$ Overbought + Red rejection bar (`Close <= Open`).

---

### 1.5 Crucial Source Facts on VWAP & BB Mid-Band Usage
1. **VWAP Usage:**
   * `out["vwap"]` is calculated in `setup()`.
   * **Fact:** `out["vwap"]` is **never referenced** in `_long_setup`, `_short_setup`, `entry_signal()`, or exit logic. VWAP is completely unused in actual execution.
2. **Bollinger Middle Band Usage:**
   * `out["bb_mid"]` is calculated in `setup()`.
   * **Fact:** `out["bb_mid"]` is **never referenced** in entry or exit logic. The strategy does not exit or revert to `bb_mid`.

---

### 1.6 Risk Management, Stop-Loss, and Take-Profit Mechanics

1. **Stop Loss:**
   * Handled by `BaseStrategy.compute_stop_loss()` with `StopLossSpec(stop_type=StopType.ATR, atr_multiplier=...)`.
   * Signal Bar $i$ computes a **fixed price level**:
     * Long Stop: $\text{Close}_i - (\text{atr\_multiplier} \times \text{ATR}_{14, i})$
     * Short Stop: $\text{Close}_i + (\text{atr\_multiplier} \times \text{ATR}_{14, i})$
   * In `BacktestEngine._manage_exits()`, this stop level remains **strictly fixed** for the duration of the trade. It is **NOT** a trailing stop.
2. **Take Profit:**
   * Handled by `BaseStrategy.compute_take_profit()` with `TakeProfitSpec(mode="rr", risk_reward_ratio=...)`.
   * Defined by risk distance: $\text{Risk} = |\text{Close}_i - \text{StopLoss}_i|$
   * Long TP: $\text{Close}_i + (\text{Risk} \times \text{risk\_reward\_ratio})$
   * Short TP: $\text{Close}_i - (\text{Risk} \times \text{risk\_reward\_ratio})$
   * Enforced as a static limit exit in `BacktestEngine._manage_exits()`.

---

### 1.7 Execution Timing & Causality
* **Shift/Lookahead Protection:**
  All indicator functions apply `shift=1`, and price trigger expressions explicitly check `out["low"].shift(1)`, `out["close"].shift(1)`, `out["open"].shift(1)`. Thus, when evaluating bar $i$, all signals strictly depend on data at or prior to closed bar $i-1$.
* **Execution Flow:**
  Signals generated at the close of bar $i$ are queued in `pending` and filled at `open[i+1]` on the next bar.
* **Intrabar Priority:**
  `BacktestEngine._manage_exits()` checks stop-loss prior to take-profit if both bounds are penetrated within the same bar (`hit_sl and hit_tp` $\rightarrow$ stop-loss assumed first).
* **Lookahead Assessment:** **Zero lookahead bias.** Fully causal point-in-time implementation.

---

## 2. Historical Evidence Audit

### 2.1 Step 2: Full-Period Baseline Sweep (Futures, Default Parameters)
Source: `data/50_experiments_report.json`

| Experiment ID | Symbol | TF | Trades ($N$) | Win Rate | Profit Factor | Expectancy | Net Return | Max DD | Net PnL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP_05` | BTCUSDT | 5m | 1,112 | 33.45% | 0.5390 | −$0.53 | −58.60% | 66.41% | −$586.00 |
| `EXP_10` | BTCUSDT | 15m | 351 | 32.48% | 0.6533 | −$0.86 | −30.13% | 44.91% | −$301.28 |
| `EXP_15` | BTCUSDT | 1h | 192 | 33.85% | 0.8420 | −$0.81 | −15.55% | 42.23% | −$155.54 |
| **`EXP_20`** | **BTCUSDT** | **4h** | **64** | **42.19%** | **1.3337** | **+$1.98** | **+12.68%** | **21.63%** | **+$126.76** |
| `EXP_25` | BTCUSDT | 1d | 3 | 100.0% | $\infty$ | +$19.93 | +5.98% | 5.09% | +$59.78 |
| `EXP_30` | ETHUSDT | 5m | 1,210 | 31.49% | 0.6042 | −$0.53 | −64.08% | 73.60% | −$640.76 |
| `EXP_35` | ETHUSDT | 15m | 368 | 32.07% | 0.7279 | −$0.83 | −30.51% | 49.09% | −$305.07 |
| `EXP_40` | ETHUSDT | 1h | 127 | 38.58% | 1.1349 | +$0.75 | +9.47% | 25.87% | +$94.69 |
| **`EXP_45`** | **ETHUSDT** | **4h** | **43** | **62.79%** | **3.1028** | **+$9.42** | **+40.51%** | **18.17%** | **+$405.12** |
| `EXP_50` | ETHUSDT | 1d | 2 | 0.0% | 0.0000 | −$10.07 | −2.01% | 3.91% | −$20.13 |

**Historical Baseline Summary:**
* On 4H, both assets showed positive net returns, profit factors $> 1.30$, and moderate drawdowns ($\le 21.63\%$).
* On 5M and 15M, high trade counts suffered catastrophic negative expectancy due to fee drag and false mean-reversions.
* On 1D, trade count was virtually zero ($N \le 3$).

---

### 2.2 Step 3b: Optimization & 2025 Validation Screen
Source: `data/step_3b_optimization_report.json`

A 24-combination grid on 4H futures was evaluated for BTC and ETH:

#### Target `BTC_4H_VWAP_MR` (Best Candidate: Combo #19)
* **Parameters:** `adx_max_thresh=25.0, atr_multiplier=1.8, risk_reward_ratio=2.0, bb_period=20, bb_std=2.0, rsi_oversold=35.0, rsi_overbought=65.0`
* **Train (2020-01-01 to 2024-12-31):**
  * $N = 80$, $\text{WR} = 45.00\%$, $\text{PF} = 1.5514$, $\text{Exp} = +\$3.1684$, $\text{Net Return} = +25.35\%$, $\text{MaxDD} = 21.37\%$
  * Gate Status: **PASSED** (Score = $0.3247$)
* **Validation (2025-01-01 to 2025-12-31):**
  * $N = 14$, $\text{WR} = 35.71\%$, $\text{PF} = 1.0869$, $\text{Exp} = +\$0.5079$, $\text{Net Return} = +0.71\%$, $\text{MaxDD} = 20.18\%$
  * Gate Status: **FAILED** strictly on trade count ($N = 14 < 15$). Note: $PF \ge 1.05$, $E > 0$, and $MDD \le 40\%$ met thresholds.

#### Target `ETH_4H_VWAP_MR` (Best Candidate: Combo #19)
* **Parameters:** `adx_max_thresh=25.0, atr_multiplier=1.8, risk_reward_ratio=2.0, bb_period=20, bb_std=2.0, rsi_oversold=35.0, rsi_overbought=65.0`
* **Train (2022-01-01 to 2024-12-31):**
  * $N = 56$, $\text{WR} = 53.57\%$, $\text{PF} = 2.0906$, $\text{Exp} = +\$5.9716$, $\text{Net Return} = +33.44\%$, $\text{MaxDD} = 19.77\%$
  * Gate Status: **PASSED** (Score = $0.4055$)
* **Validation (2025-01-01 to 2025-12-31):**
  * $N = 12$, $\text{WR} = 33.33\%$, $\text{PF} = 0.8809$, $\text{Exp} = -\$0.8224$, $\text{Net Return} = -0.99\%$, $\text{MaxDD} = 18.68\%$
  * Gate Status: **FAILED** ($N = 12 < 15$, $PF = 0.8809 < 1.05$, $E \le 0$).

---

## 3. Strict Research Categorization

### Category A: Facts Established by Source Code
1. `VWAPBollingerMRStrategy` is an ADX-filtered Bollinger-RSI extreme fader.
2. `out["vwap"]` is calculated in `setup()` but **never used** in entry or exit logic.
3. `out["bb_mid"]` is calculated in `setup()` but **never used** in entry or exit logic.
4. Stop-loss is a **fixed price level** calculated at entry from ATR, not a trailing stop.
5. Take-profit is a **fixed price level** based on static risk-reward ratio multiples.
6. Vectorized indicators and price shifts enforce strict point-in-time causality with zero lookahead bias.

### Category B: Historical Empirical Observations
1. In Step 2, 4H was the only timeframe demonstrating positive expectancy across both BTC ($PF=1.33$) and ETH ($PF=3.10$). Sub-4H timeframes suffered heavy fee drag.
2. In Step 3b, 4H validation produced $N=14$ on BTC and $N=12$ on ETH, causing both to fail the $N \ge 15$ gate.
3. ETH 4H validation showed severe performance decay ($PF$ dropped from $2.09$ to $0.88$).

### Category C: Research Hypotheses (Untested Concepts)
1. *Trade Frequency Bottleneck Hypothesis:* The 4H timeframe in isolation generates insufficient trades over standard 1-year validation windows to reliably meet the $N \ge 15$ gate.
2. *Exit Mechanism Hypothesis:* Exiting via fixed $R:R$ multiples during range regimes may hold trades too long when price reaches the mean, potentially converting winning mean-reversions into stop-outs. *(Untested hypothesis)*.
3. *VWAP Integration Hypothesis:* Actively utilizing VWAP as an entry filter or reversion target may provide useful structural information that is currently omitted. *(Untested hypothesis)*.

### Category D: Protocol Design Objectives
1. Formulate a clean, staged protocol (`PROTOCOL-VWAP-MR-V1.0`) specifying Stage-A baseline screening, Stage-B bounded calibration, and explicit candidate selection rules.
2. Predeclare all parameter bounds, exit models, and advancement criteria before running any experiments.
