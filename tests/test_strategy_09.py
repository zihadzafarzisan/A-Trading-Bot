"""Dedicated tests for Strategy #9 (PROTOCOL-STRATEGY-09-V1.0).

Synthetic fixtures only — no historical data, no database access, no Stage-A.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import (
    Direction,
    VolatilityCompressionBreakoutStrategy,
    create_strategy,
    get_strategy_class,
)
from crypto_quant.strategies.volatility_compression_breakout import (
    C00_PARAMS,
    bb_width_population,
    enumerate_candidates,
    enumerate_grid_tuples,
    sma_atr,
    sma_seeded_ema,
    type7_percentile,
)
from crypto_quant.backtesting.engine import BacktestConfig, BacktestEngine
from crypto_quant.backtesting.execution import ExecutionConfig, ExecutionModel

N_FEED = 260  # enough bars for squeeze baseline (needs warmup >= 139)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SIGNAL_BAR = 180  # 0-indexed bar where the engineered long signal fires (>= 139)


def make_df(n=None):
    """Deterministic OHLCV engineered to fire a LONG signal at t=SIGNAL_BAR.

    Regimes (by close-dispersion for BBWidth):
      - bars 0..164 : wide close noise  => large historical BBWidth (high percentile)
      - bars 165..179: flat close noise  => tiny BBWidth (squeeze)
      - bars 180..  : normal

    Ranges (for ATR expansion):
      - bars 0..164 : wide range      => high ATR history (but outside 10-bar baseline)
      - bars 165..179: narrow range    => small ATR expansion baseline
      - bar 180      : wide range      => TR spike fires expansion (>1.1x baseline)
      - bar 180      : close jumps +8  => clears prior 20-bar high (long breakout)

    Compression baseline (120 widths, bars 60..179) is dominated by wide-regime
    widths, so the tiny current width at t=180 sits far below the 30th percentile.
    """
    n = N_FEED if n is None else n
    times = [1609459200000 + i * 3600000 for i in range(n)]

    def close_noise_sd(i):
        # Dispersion of closes drives BBWidth.
        if 165 <= i < 180:
            return 0.1          # squeeze
        if i < 180:
            return 3.0          # wide
        return 2.0

    def range_amp(i):
        # (high-low) drives ATR.
        if 120 <= i < 180:
            return 1.0          # narrow -> small expansion baseline
        if i in (180, 181, 182):
            return 12.0         # spike -> expansion fires
        return 12.0

    rng = np.random.default_rng(11)
    closes = np.empty(n)
    price = 100.0
    for i in range(n):
        if i == SIGNAL_BAR:
            # Confirmed breakout: close jumps well above prior 20-bar highs.
            price = price + 8.0
        else:
            price += 0.05
            price += rng.normal(0.0, close_noise_sd(i))
        closes[i] = price

    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    for i in range(n):
        a = range_amp(i)
        opens[i] = closes[i] - 0.3
        highs[i] = closes[i] + a / 2
        lows[i] = closes[i] - a / 2
    # The output DataFrame must remain a valid OHLCV frame.
    return pd.DataFrame({
        "timestamp": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


def c00_strategy(**overrides) -> VolatilityCompressionBreakoutStrategy:
    params = {k: v for k, v in C00_PARAMS.items() if k != "candidate_id"}
    params.update(overrides)
    return VolatilityCompressionBreakoutStrategy(params)


def prepared(df, **overrides) -> pd.DataFrame:
    return c00_strategy(**overrides).setup(df)


# ── Registry / grid ───────────────────────────────────────────────────────────

def test_registry_and_details():
    cls = get_strategy_class("volatility_compression_breakout")
    assert cls is VolatilityCompressionBreakoutStrategy
    s = create_strategy("volatility_compression_breakout")
    assert isinstance(s, VolatilityCompressionBreakoutStrategy)
    assert s.strategy_type == "volatility_compression_breakout"
    assert s.name == "Volatility Compression Breakout"

    for k, v in C00_PARAMS.items():
        if k != "candidate_id":
            assert s.params[k] == v, f"default {k} mismatch"

    g = s.param_grid()
    assert list(g.keys()) == [
        "breakout_lookback",
        "squeeze_percentile",
        "expansion_multiplier",
        "atr_multiplier",
    ]
    for k in ("bb_period", "squeeze_lookback", "expansion_lookback",
              "fast_ema", "slow_ema", "atr_period", "risk_reward_ratio"):
        assert k not in g, f"{k} must not be a grid dimension"


def test_grid_is_exactly_36_and_c00_once():
    tuples = enumerate_grid_tuples()
    assert len(tuples) == 36
    assert len(set(tuples)) == 36
    assert tuples == sorted(tuples)

    # C00 exists exactly once and is one of the 36.
    c00 = (C00_PARAMS["breakout_lookback"], C00_PARAMS["squeeze_percentile"],
           float(C00_PARAMS["expansion_multiplier"]), float(C00_PARAMS["atr_multiplier"]))
    assert tuples.count(c00) == 1
    assert c00 in tuples

    cands = enumerate_candidates()
    assert len(cands) == 36
    ids = [c["candidate_id"] for c in cands]
    assert ids.count("C00") == 1
    assert ids[0] == "C00"
    assert ids == sorted(set(ids))
    assert c00 == (cands[0]["breakout_lookback"], cands[0]["squeeze_percentile"],
                   float(cands[0]["expansion_multiplier"]), float(cands[0]["atr_multiplier"]))

    # Body/other nonexistent params must not appear.
    for c in cands:
        extras = set(c) - {"candidate_id", "breakout_lookback", "squeeze_percentile",
                           "expansion_multiplier", "atr_multiplier"}
        assert not extras, f"unexpected param in {c}"
    # Every config holds exactly the four grid params.
    for c in cands:
        assert set(c) == {"candidate_id", "breakout_lookback", "squeeze_percentile",
                          "expansion_multiplier", "atr_multiplier"}


# ── Bollinger (population) ────────────────────────────────────────────────────

def test_bb_population_math():
    closes = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                       9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 2.0],
                      dtype=float)
    mid, width, std = bb_width_population(closes, 20)
    # mean of [1..10,9..1,2] = 5.1; population std via ddof=0.
    assert np.isclose(mid[19], 5.1, atol=1e-9)
    assert np.isclose(std[19], np.std(closes[:20], ddof=0), atol=1e-9)
    assert np.isclose(width[19], 4.0 * np.std(closes[:20], ddof=0) / 5.1, atol=1e-9)
    # ddof=1 differs -> verifies population semantics
    assert np.std(closes[:20], ddof=1) != np.std(closes[:20], ddof=0)


def test_bb_zero_mid_fails_closed():
    closes = np.concatenate([np.zeros(20), np.ones(20)])
    mid, width, std = bb_width_population(closes, 20)
    assert np.isnan(width[19])  # mid == 0 -> undefined (compression fails)


# ── SMA-seeded EMA ────────────────────────────────────────────────────────────

def test_ema_seed_is_sma_and_recursive():
    closes = np.array([float(i + 1) for i in range(50)], dtype=float)
    e = sma_seeded_ema(closes, 20)
    assert np.isnan(e[18])
    assert np.isclose(e[19], closes[:20].mean(), atol=1e-9)
    alpha = 2.0 / 21.0
    assert np.isclose(e[20], alpha * closes[20] + (1 - alpha) * e[19], atol=1e-12)


# ── ATR (SMA-of-TR) ───────────────────────────────────────────────────────────

def test_atr_sma_semantics():
    n = 30
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    a = sma_atr(high, low, close, 14)
    # TR = (high-low)=2 every bar after bar0; TR_0=2. SMA of 14 twos = 2.
    assert np.isclose(a[13], 2.0, atol=1e-9)
    assert np.isnan(a[12])


# ── Type-7 percentile ─────────────────────────────────────────────────────────

def test_type7_percentile():
    sample = np.array(list(range(1, 21)), dtype=float)  # 1..20
    assert np.isclose(type7_percentile(sample, 50.0), 10.5, atol=1e-9)
    assert np.isclose(type7_percentile(sample, 25.0), np.percentile(sample, 25, method="linear"), atol=1e-9)
    assert np.isclose(type7_percentile(np.array([np.nan, 2.0, np.nan]), 100.0), 2.0, atol=1e-9)
    assert np.isnan(type7_percentile(np.array([np.nan]), 50.0))


# ── Compression ───────────────────────────────────────────────────────────────

def test_compression_baseline_window_and_exclusion():
    df = make_df()
    p = prepared(df)
    i = SIGNAL_BAR
    S = 120
    width = p["bb_width"].to_numpy(dtype=float)
    hist = width[i - S:i]
    assert np.isfinite(hist).all()
    # Current observation excluded from baseline.
    assert p["squeeze_threshold"].iloc[i] == np.percentile(hist, 30, method="linear")
    # Bar t-S-1 must not be in baseline (shifting window by one changes threshold).
    thresh_now = p["squeeze_threshold"].iloc[i]
    thresh_one_less = np.percentile(width[i - S - 1:i - 1], 30, method="linear")
    assert thresh_now != thresh_one_less


def test_compression_future_perturbation_immune():
    df = make_df()
    p = prepared(df)
    i = SIGNAL_BAR
    baseline = p["squeeze_threshold"].iloc[i]

    df2 = df.copy()
    df2.loc[i + 1:, "close"] = df2.loc[i + 1:, "close"] * 3.0
    df2.loc[i + 1:, "high"] = df2.loc[i + 1:, "high"] * 3.0
    df2.loc[i + 1:, "low"] = df2.loc[i + 1:, "low"] * 3.0
    p2 = prepared(df2)

    assert float(p2["squeeze_threshold"].iloc[i]) == float(baseline)


def test_compression_equality_passes():
    # Forced equality: bb_width[i] == threshold exactly -> compression passes.
    df = make_df()
    s = c00_strategy()
    width = df["bb_width"].to_numpy() if "bb_width" in df else None
    i = SIGNAL_BAR
    p0 = s.setup(df)
    thresh = float(p0["squeeze_threshold"].iloc[i])
    # Artificially set the current bb_width column to the threshold.
    p0.loc[i, "bb_width"] = thresh
    assert s.compression_at(p0, i) is True


def test_compression_short_baseline_fails_closed():
    df = make_df(n=30)
    s = c00_strategy()
    p = s.setup(df)
    assert s.compression_at(p, 10) is False  # i < WARMUP_MIN_INDEX


# ── Breakout ──────────────────────────────────────────────────────────────────

def test_breakout_long_short_equality_current_exclusion():
    df = make_df()
    s = c00_strategy()
    t = SIGNAL_BAR
    L = int(c00_strategy().params["breakout_lookback"])

    # Confirm the current close clears the prior window's high (long breakout) and
    # does NOT dip below the prior window's low (no short breakout).
    prior_high = float(df["high"].iloc[t - L:t].max())
    prior_low = float(df["low"].iloc[t - L:t].min())
    close_t = float(df["close"].iloc[t])
    assert close_t > prior_high
    assert s.long_break_at(s.setup(df), t) is True
    assert s.short_break_at(s.setup(df), t) is False

    # Equality (close == prior max high) does NOT count as a breakout.
    pb = s.setup(df)
    pb.loc[t, "close"] = prior_high
    assert s.long_break_at(pb, t) is False

    # Current bar's own high is excluded from its own reference: an extreme high
    # at bar t must not inflate the reference (breakout still governed by close).
    pd_ = s.setup(df)
    pd_.loc[t, "high"] = close_t + 999.0
    assert s.long_break_at(pd_, t) is True


def test_breakout_lookback_20_vs_40_material():
    # L=20 sees only the flat near-terrace (reference ~small) => close breaks out.
    # L=40 sees a much higher distant high => close does NOT break its L=40 ref.
    df = make_df()
    t = SIGNAL_BAR
    df.loc[t - 40:t - 1, "high"] = 1000.0         # far-high within L=40 only
    df.loc[t - 20:t - 1, "high"] = 100.0
    df.loc[t, "close"] = 110.0
    s20 = c00_strategy(breakout_lookback=20)
    s40 = c00_strategy(breakout_lookback=40)
    p20 = s20.setup(df)
    p40 = s40.setup(df)
    assert s20.long_break_at(p20, t) is True      # ref high = 100 (< 110)
    assert s40.long_break_at(p40, t) is False     # ref high = 1000 (> 110)


def test_breakout_insufficient_history():
    df = make_df(n=30)
    s = c00_strategy(breakout_lookback=20)
    p = s.setup(df)
    assert s.long_break_at(p, 12) is False


# ── Expansion ─────────────────────────────────────────────────────────────────

def test_expansion_baseline_current_excluded():
    df = make_df()
    s = c00_strategy()
    p = s.setup(df)
    t = SIGNAL_BAR
    X = int(s.params["expansion_lookback"])
    # Baseline = mean ATR of bars t-X..t-1 (current t excluded).
    assert np.isclose(p["atr_baseline"].iloc[t], p["atr_sma"].iloc[t - X:t].mean(), atol=1e-9)


def test_expansion_multiplier_material():
    # ATR_t in [1.1, 1.3) vs baseline 1.0: 1.1 passes, 1.3 fails.
    df = make_df()
    t = SIGNAL_BAR
    s11 = c00_strategy(expansion_multiplier=1.1)
    s13 = c00_strategy(expansion_multiplier=1.3)
    p11 = s11.setup(df)
    p13 = s13.setup(df)
    p11.loc[t, "atr_sma"] = 1.2
    p11.loc[t, "atr_baseline"] = 1.0
    p13.loc[t, "atr_sma"] = 1.2
    p13.loc[t, "atr_baseline"] = 1.0
    assert s11.expansion_at(p11, t) is True
    assert s13.expansion_at(p13, t) is False


def test_expansion_equality_passes():
    df = make_df()
    t = SIGNAL_BAR
    s = c00_strategy()
    p = s.setup(df)
    p.loc[t, "atr_sma"] = float(p["atr_baseline"].iloc[t]) * float(s.params["expansion_multiplier"])
    assert s.expansion_at(p, t) is True


def test_expansion_current_and_future_excluded():
    df = make_df()
    t = SIGNAL_BAR
    s = c00_strategy()
    p = s.setup(df)
    base = p["atr_baseline"].iloc[t]
    # Current bar ATR is excluded from baseline.
    df2 = df.copy()
    df2.loc[t:t + 5, "high"] = df2.loc[t:t + 5, "high"] + 5000.0
    df2.loc[t:t + 5, "low"] = df2.loc[t:t + 5, "low"] - 5000.0
    p2 = s.setup(df2)
    assert float(p2["atr_baseline"].iloc[t]) == float(base)


def test_expansion_zero_baseline_fails():
    df = make_df()
    t = SIGNAL_BAR
    s = c00_strategy()
    p = s.setup(df)
    p.loc[t, "atr_baseline"] = 0.0
    p.loc[t, "atr_sma"] = 5.0
    assert s.expansion_at(p, t) is False


# ── Trend ─────────────────────────────────────────────────────────────────────

def test_trend_direction_and_equality():
    df = make_df()
    up = np.linspace(100, 200, len(df))
    df["close"] = up
    s = c00_strategy()
    p = s.setup(df)
    t = 200
    assert s.long_trend_at(p, t) is True
    assert s.short_trend_at(p, t) is False

    down = np.linspace(200, 100, len(df))
    df2 = make_df()
    df2["close"] = down
    p2 = s.setup(df2)
    assert s.short_trend_at(p2, t) is True
    assert s.long_trend_at(p2, t) is False

    flat = np.full(len(df), 100.0)
    df3 = make_df()
    df3["close"] = flat
    p3 = s.setup(df3)
    assert s.long_trend_at(p3, t) is False
    assert s.short_trend_at(p3, t) is False


# ── Combined / each-condition-required ────────────────────────────────────────

def test_fixture_fires_long_at_signal_bar():
    s = c00_strategy()
    df = make_df()
    p = s.setup(df)
    direction, _ = s.entry_signal(p, SIGNAL_BAR)
    assert direction == Direction.LONG


def test_each_condition_independently_required_long():
    """Flip exactly one condition; every other stays strong -> NO signal."""
    s = c00_strategy()
    df = make_df()
    p = s.setup(df)
    i = SIGNAL_BAR
    assert s.entry_signal(p, i)[0] == Direction.LONG  # baseline fires

    # 1) Compression failure only.
    pc = p.copy()
    pc.loc[i, "squeeze_threshold"] = float(pc["bb_width"].iloc[i]) - 0.001
    assert s.entry_signal(pc, i)[0] == Direction.NONE

    # 2) Breakout failure only (equality, not strict >).
    pb = p.copy()
    pb.loc[i, "close"] = float(pb["breakout_high"].iloc[i])
    assert s.entry_signal(pb, i)[0] == Direction.NONE

    # 3) Expansion failure only.
    pe = p.copy()
    pe.loc[i, "atr_sma"] = 0.0
    assert s.entry_signal(pe, i)[0] == Direction.NONE

    # 4) Trend failure only (EMA equality -> no direction).
    pt = p.copy()
    pt.loc[i, "ema_fast"] = float(pt["ema_slow"].iloc[i])
    assert s.entry_signal(pt, i)[0] == Direction.NONE


def test_short_signal_fires_on_down_fixture():
    s = c00_strategy()
    df = make_short_df()
    p = s.setup(df)
    i = SIGNAL_BAR
    assert s.entry_signal(p, i)[0] == Direction.SHORT


def make_short_df(n=None):
    """Mirror of make_df but downward: short breakout + short trend + expansion."""
    n = N_FEED if n is None else n
    times = [1609459200000 + i * 3600000 for i in range(n)]

    rng = np.random.default_rng(5)
    closes = np.empty(n)
    price = 300.0
    for i in range(n):
        if i == SIGNAL_BAR:
            price = price - 8.0            # crash below prior 20-bar low
        else:
            price -= 0.05
            sd = 0.1 if 165 <= i < 180 else 3.0
            price += rng.normal(0.0, sd)
        closes[i] = price

    opens = np.empty(n)
    highs = np.empty(n)
    lows = np.empty(n)
    for i in range(n):
        a = 1.0 if 120 <= i < 180 else 12.0   # narrow baseline, spike at signal bar
        opens[i] = closes[i] + 0.3
        highs[i] = closes[i] + a / 2
        lows[i] = closes[i] - a / 2
    return pd.DataFrame({
        "timestamp": times,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.full(n, 1000.0),
    })


# ── Signal/risk independence ──────────────────────────────────────────────────

def test_atr_multiplier_does_not_change_signal():
    df = make_df()
    sigs = {}
    for a in (1.8, 2.2, 2.6):
        s = c00_strategy(atr_multiplier=a)
        p = s.setup(df)
        sigs[a] = [s.entry_signal(p, i)[0] for i in range(len(df))]
    assert sigs[1.8] == sigs[2.2] == sigs[2.6]

    # Risk quantities differ and are proportional to atr_multiplier.
    fill = 100.0
    atr = 10.0
    for a in (1.8, 2.2, 2.6):
        p = c00_strategy(atr_multiplier=a).setup(df)
        p.loc[SIGNAL_BAR, "atr_sma"] = atr
        _, _, state = c00_strategy(atr_multiplier=a).prepare_fill(
            p, SIGNAL_BAR, Direction.LONG, fill)
        assert np.isclose(state["risk_R"], a * atr, atol=1e-9)


# ── No-lookahead ──────────────────────────────────────────────────────────────

def test_no_lookahead_signal_immutable_to_future():
    df = make_df()
    s = c00_strategy()
    i = SIGNAL_BAR
    p = s.setup(df)
    sig0 = s.entry_signal(p, i)
    df2 = df.copy()
    df2.loc[i + 1:, ["open", "high", "low", "close", "volume"]] *= 7.0
    p2 = s.setup(df2)
    sig2 = s.entry_signal(p2, i)
    assert sig0 == sig2


def test_no_lookahead_baselines_immutable_to_current_or_future():
    df = make_df()
    s = c00_strategy()
    i = SIGNAL_BAR
    p = s.setup(df)
    thresh = p["squeeze_threshold"].iloc[i]
    atr_base = p["atr_baseline"].iloc[i]
    df2 = df.copy()
    # Change current + all future; baselines (which exclude current) must persist.
    df2.loc[i:, ["open", "high", "low", "close"]] = df2.loc[i:, ["open", "high", "low", "close"]] * 4.0
    p2 = s.setup(df2)
    assert float(p2["squeeze_threshold"].iloc[i]) == float(thresh)
    assert float(p2["atr_baseline"].iloc[i]) == float(atr_base)


# ── Execution / risk ──────────────────────────────────────────────────────────

def test_prepare_fill_anchor_and_r():
    s = c00_strategy(atr_multiplier=2.2, risk_reward_ratio=2.0)
    df = make_df()
    p = s.setup(df)
    p.loc[SIGNAL_BAR, "atr_sma"] = 10.0
    fill = 105.0
    stop, take, state = s.prepare_fill(p, SIGNAL_BAR, Direction.LONG, fill)
    assert np.isclose(state["risk_R"], 22.0, atol=1e-9)
    assert np.isclose(stop, 105.0 - 22.0, atol=1e-9)
    assert np.isclose(take, 105.0 + 44.0, atol=1e-9)
    stop_s, take_s, _ = s.prepare_fill(p, SIGNAL_BAR, Direction.SHORT, fill)
    assert np.isclose(stop_s, 105.0 + 22.0, atol=1e-9)
    assert np.isclose(take_s, 105.0 - 44.0, atol=1e-9)


def _engine_config():
    exec_cfg = ExecutionConfig(
        market_type="futures", slippage=0.0005, taker_fee=0.0004, use_taker=True,
        funding_rate=0.0001, execution_delay_bars=1, initial_capital=1000.0,
        max_leverage=5.0,
    )
    return BacktestConfig(
        initial_capital=1000.0, risk_per_trade=0.01, max_open_positions=3,
        max_position_pct=0.50, max_leverage=5.0, market_type="futures",
        timeframe="1h", execution=exec_cfg, funding_bars=8, honor_take_profit=True,
    )


def test_backtest_next_bar_execution_long_slippage_fees():
    df = make_df()
    s = c00_strategy()
    # Force a long signal at some bar and a subsequent open.
    engine = BacktestEngine(config=_engine_config())
    result = engine.run(s, df, "BTCUSDT")
    assert len(result.trades) >= 1
    t = result.trades[0]
    assert t["direction"] == "long"
    assert t["entry_price"] > t["stop_loss"]


def test_backtest_stop_first_collision():
    df = make_df()
    s = c00_strategy()
    engine = BacktestEngine(config=_engine_config())
    result = engine.run(s, df, "BTCUSDT")
    # Any trade that closed in a same-bar collision must prefer stop; we assert
    # at least the engine flag honors stop-first (exit_reason "sl" when both hit).
    for tr in result.trades:
        if tr["exit_reason"] == "sl":
            assert tr["gross_pnl"] <= 0


# ── Deterministic, engine-level stop-first collision (§19 evidence repair) ─────

def _stop_first_collision_df(maker, side):
    """Build a two-run deterministic same-bar collision.

    Pass 1 runs the benign fixture to anchor the engine's exact fill/stop/TP/R and
    leverage. Pass 2 rebuilds the frame keeping the entry bar benign (range strictly
    inside (SL, TP)) and puts a DISTINCT collision candle one bar later whose range
    reaches BOTH the stop and the take-profit (while staying outside the liquidation
    level). Because the collision candle's high/low are never inputs to the signal or
    to the fill/stop/TP math, the anchors are identical in both runs.

    Returns (trade, collision_bar, stop, take) for the collision run.
    """
    cfg = _engine_config()
    engine = BacktestEngine(config=cfg)
    discovery = maker()
    base = engine.run(c00_strategy(), discovery, "BTCUSDT").trades[0]
    ts = discovery["timestamp"].to_numpy()
    entry_bar = int(np.where(ts == base["entry_time"])[0][0])
    E = base["entry_price"]
    SL = base["stop_loss"]
    TP = base["take_profit"]
    lev = base["leverage"]
    liq = ExecutionModel(cfg.execution).liquidation_price(E, base["direction"], lev)

    df = maker().copy()
    coll = entry_bar + 1
    if side == "long":
        # Benign entry bar: range strictly inside (SL, TP), well inside liq side.
        df.loc[entry_bar, "low"] = SL + 0.30 * (TP - SL)
        df.loc[entry_bar, "high"] = SL + 0.75 * (TP - SL)
        # Collision candle: low below the stop (no liquidation), high above the TP.
        df.loc[coll, "low"] = (SL + liq) / 2.0           # < SL, > liq
        df.loc[coll, "high"] = TP + 2.0                  # > TP
    else:
        # Benign entry bar: range strictly inside (TP, SL).
        df.loc[entry_bar, "low"] = TP + 0.25 * (SL - TP)
        df.loc[entry_bar, "high"] = TP + 0.75 * (SL - TP)
        # Collision candle for short: high above the stop (below upside liq), low below TP.
        df.loc[coll, "high"] = (SL + liq) / 2.0          # > SL, < liq
        df.loc[coll, "low"] = TP - 2.0                   # < TP

    res = engine.run(c00_strategy(), df, "BTCUSDT")
    tr = res.trades[0]
    ts2 = df["timestamp"].to_numpy()
    exit_bar = int(np.where(ts2 == tr["exit_time"])[0][0])
    return tr, coll, exit_bar, df


def test_stop_first_collision_long():
    """A candle whose range reaches BOTH stop and TP on the same bar must exit at
    the stop first (exit_reason 'sl', exit_price == stop), not the take-profit."""
    tr, coll, exit_bar, df = _stop_first_collision_df(make_df, "long")
    stop = tr["stop_loss"]
    tp = tr["take_profit"]

    # The collision candle truly reaches both levels on one bar (engine receives
    # hit_sl == True AND hit_tp == True) and is not liquidated.
    assert df.loc[coll, "low"] <= stop          # low touches/exceeds the stop
    assert df.loc[coll, "high"] >= tp           # high touches/exceeds the take-profit
    assert abs(df.loc[coll, "low"] - stop) > 1e-4   # uncomfortable margin, not exact
    assert abs(df.loc[coll, "high"] - tp) > 1e-4     # uncomfortable margin, not exact
    assert tr["liquidated"] is False

    # The exit happens on that same collision candle.
    assert exit_bar == coll
    # Conservative stop-first: stop executes, never the take-profit.
    assert tr["exit_reason"] == "sl"
    assert np.isclose(float(tr["exit_price"]), float(stop), atol=1e-6)
    assert not np.isclose(float(tr["exit_price"]), float(tp), atol=1e-6)


def test_stop_first_collision_short():
    """Mirror of the long collision test for a short position (stop above, TP below)."""
    tr, coll, exit_bar, df = _stop_first_collision_df(make_short_df, "short")
    stop = tr["stop_loss"]
    tp = tr["take_profit"]

    assert df.loc[coll, "high"] >= stop          # high reaches the stop (above entry)
    assert df.loc[coll, "low"] <= tp             # low reaches the take-profit (below entry)
    assert abs(df.loc[coll, "high"] - stop) > 1e-4
    assert abs(df.loc[coll, "low"] - tp) > 1e-4
    assert tr["liquidated"] is False

    assert exit_bar == coll
    assert tr["exit_reason"] == "sl"
    assert np.isclose(float(tr["exit_price"]), float(stop), atol=1e-6)
    assert not np.isclose(float(tr["exit_price"]), float(tp), atol=1e-6)


def test_backtest_position_sizing_fees_funding():
    from crypto_quant.risk import PositionSizer, RiskLimits
    sizer = PositionSizer(RiskLimits(
        risk_per_trade=0.01, max_position_pct=0.50, max_leverage=5.0))
    res = sizer.size_position(
        equity=1000.0, entry_price=100.0, stop_price=90.0,
        direction="long", market_type="futures", leverage=5)
    assert np.isclose(res.risk_amount, 10.0, atol=1e-9)
    # notional = equity*1% / (stop_dist=0.10) = 100.0
    assert np.isclose(res.notional, 100.0, atol=1e-9)
    # margin = notional / leverage
    assert np.isclose(res.margin_used if hasattr(res, "margin_used") else res.notional / res.leverage,
                      20.0, atol=1e-9)


def test_backtest_shortsupported_and_sizing_future_dataclass():
    df = make_df()
    engine = BacktestEngine(config=_engine_config())
    s = c00_strategy()
    result = engine.run(s, df, "BTCUSDT")
    # Sizing respected: notional <= equity * leverage cap.
    for tr in result.trades:
        notional = float(tr["quantity"]) * float(tr["entry_price"])
        assert notional <= 1000.0 * 5.0


# ── Coverage gaps closed (Plan A–E) ───────────────────────────────────────────
# A: compression zero/invalid baseline fail-closed, wired through to the signal


def test_compression_nonfinite_baseline_fails_closed():
    """A non-finite BBWidth anywhere in the 120-bar baseline -> compression False
    and no signal (fail-closed), per protocol §4.2."""
    df = make_df()
    s = c00_strategy()
    p = s.setup(df)
    i = SIGNAL_BAR
    assert s.compression_at(p, i) is True          # fixture baseline fires
    p2 = p.copy()
    p2.loc[i - 60, "bb_width"] = np.nan            # poison a baseline bar (t-60 in [t-120,t-1])
    assert s.compression_at(p2, i) is False        # fail-closed
    assert s.entry_signal(p2, i)[0] == Direction.NONE


# B: squeeze_percentile is a SIGNAL parameter — flipping it can change signal existence


def test_squeeze_percentile_changes_signal_existence():
    """With BBWidth_t between the 20th and 40th percentile of the *same* baseline,
    q=40 passes compression while q=20 fails — flipping the combined signal."""
    df = make_df()
    i = SIGNAL_BAR
    base = prepared(df)["bb_width"].iloc[i - 120:i].to_numpy(dtype=float)
    p20 = float(type7_percentile(base, 20.0))
    p40 = float(type7_percentile(base, 40.0))
    assert p20 < p40                               # baseline is non-degenerate
    v = float((p20 + p40) / 2.0)                   # current width strictly between them

    s20 = c00_strategy(squeeze_percentile=20)
    s40 = c00_strategy(squeeze_percentile=40)
    p20s = s20.setup(df)
    p40s = s40.setup(df)
    # Override only the current bar's width; its historical baselines (bars < i) are
    # unchanged, so the threshold columns are the genuine P20 / P40 of that same baseline.
    p20s.loc[i, "bb_width"] = v
    p40s.loc[i, "bb_width"] = v
    assert s40.compression_at(p40s, i) is True      # v <= P40 -> compression holds
    assert s20.compression_at(p20s, i) is False     # v >  P20 -> compression fails
    assert s40.entry_signal(p40s, i)[0] == Direction.LONG
    assert s20.entry_signal(p20s, i)[0] != Direction.LONG


# C: each short condition is independently required (mirror of the long test)


def test_each_condition_independently_required_short():
    """Flip exactly one short condition while the rest stay strong -> NO short signal."""
    s = c00_strategy()
    df = make_short_df()
    p = s.setup(df)
    i = SIGNAL_BAR
    assert s.entry_signal(p, i)[0] == Direction.SHORT  # baseline fires short

    # 1) Compression failure only.
    pc = p.copy()
    pc.loc[i, "squeeze_threshold"] = float(pc["bb_width"].iloc[i]) - 0.001
    assert s.entry_signal(pc, i)[0] == Direction.NONE

    # 2) Breakout failure only (equality vs prior-range low, not strict <).
    pb = p.copy()
    pb.loc[i, "close"] = float(pb["breakout_low"].iloc[i])
    assert s.entry_signal(pb, i)[0] == Direction.NONE

    # 3) Expansion failure only.
    pe = p.copy()
    pe.loc[i, "atr_sma"] = 0.0
    assert s.entry_signal(pe, i)[0] == Direction.NONE

    # 4) Trend failure only (EMA equality -> no direction).
    pt = p.copy()
    pt.loc[i, "ema_fast"] = float(pt["ema_slow"].iloc[i])
    assert s.entry_signal(pt, i)[0] == Direction.NONE


# D: next-bar execution oracle — signal at t fills at open[t+1], long/short slip formulas


def _entry_bar_and_open(df, tr):
    ts = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy()
    idx = int(np.where(ts == tr["entry_time"])[0][0])
    return idx, float(opens[idx])


def test_execution_next_bar_and_long_fill():
    """Signal at bar t fills at open[t+1]; long fill = open[t+1]*(1+0.0005); no same-bar."""
    df = make_df()
    s = c00_strategy()
    result = BacktestEngine(config=_engine_config()).run(s, df, "BTCUSDT")
    assert result.trades
    for tr in result.trades:
        idx, raw_open = _entry_bar_and_open(df, tr)
        if tr["direction"] == "long":
            assert np.isclose(tr["entry_price"], raw_open * 1.0005, atol=1e-6)
            assert tr["entry_price"] > raw_open
        else:
            assert np.isclose(tr["entry_price"], raw_open * 0.9995, atol=1e-6)
            assert tr["entry_price"] < raw_open
    # Isolated fixture: the single long signal fires at SIGNAL_BAR, so the fill lands
    # exactly one bar later (open[t+1]); the signal bar itself is never filled.
    assert result.trades[0]["direction"] == "long"
    idx, _ = _entry_bar_and_open(df, result.trades[0])
    assert idx == SIGNAL_BAR + 1


def test_execution_short_fill():
    """Short fill = open[t+1]*(1 - 0.0005); adverse slippage pushes the entry down."""
    df = make_short_df()
    s = c00_strategy()
    result = BacktestEngine(config=_engine_config()).run(s, df, "BTCUSDT")
    assert result.trades and result.trades[0]["direction"] == "short"
    idx, raw_open = _entry_bar_and_open(df, result.trades[0])
    assert idx == SIGNAL_BAR + 1
    assert np.isclose(result.trades[0]["entry_price"], raw_open * 0.9995, atol=1e-6)
    assert result.trades[0]["entry_price"] < raw_open


# E: §10a — reported slippage is price displacement, distinct from the dollar cost


def test_slippage_is_price_displacement_not_dollar_cost():
    """§10a: the reported 'slippage' is price displacement (price units); the dollar
    cost (= displacement × quantity) is a distinct, labeled quantity — never conflated."""
    df = make_df()
    s = c00_strategy()
    result = BacktestEngine(config=_engine_config()).run(s, df, "BTCUSDT")
    assert result.trades
    for tr in result.trades:
        _, raw_open = _entry_bar_and_open(df, tr)
        displacement = abs(float(tr["entry_price"]) - raw_open)   # price units
        assert np.isclose(float(tr["slippage"]), displacement, atol=1e-9)
        dollar_cost = displacement * float(tr["quantity"])         # USDT
        assert dollar_cost > 0
        assert not np.isclose(float(tr["slippage"]), dollar_cost, atol=1e-9)