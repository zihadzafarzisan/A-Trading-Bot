"""Oracle / unit tests for PROTOCOL-STRATEGY-09-V1.1 (Strategy #9 V1.1).

Covers the invariants the frozen protocol (``data/protocol_strategy_09_v1_1.md``,
SHA-256 ``465cb1a43158acce4488777fcbce48b80abf630350ba37017fdd0f70f5e09e34``)
declares as non-negotiable:

  1. Strict running-max monotonicity of the Chandelier stop under ATR expansion (F-01).
  2. Breakeven ratchet arming at +1.5R with permanent latching, level ``Fill*(1+/-0.0020)`` (F-02).
  3. Conservative stop-first same-bar collision resolution.
  4. Strict no-look-ahead (signal evaluates bar ``t`` close, fill at ``open_{t+1}``).
  5. Fail-closed warm-up bound (``t < 139`` never signals).
  6. The full 36-candidate grid (C00..C35) instantiates with correct parameters.

Plus correctness guards that pin the protocol fixes:
  - The frozen entry uses ``expansion_multiplier = 1.1`` (NOT the risk ``atr_multiplier``).
  - The breakout is a plain strict ``>``/``<`` cross (V1.0 verbatim, no freshness gate).
  - The engine wiring (``prepare_fill``/``update_stop``) carries the running-max state.

DATA FIREWALL: All fixtures are synthetic. No raw market data (train / 2025
validation / 2026 OOS) is accessed by this suite.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies.base import Direction
from crypto_quant.strategies.volatility_compression_breakout_v1_1 import (
    BREAKEVEN_BUFFER, BREAKEVEN_TRIGGER_R, CANDIDATE_TUPLES, DOMINATED_EXIT,
    EXIT_CONFIGS, VolatilityCompressionBreakoutV1_1,
    V1_1ExitManager, break_even_level, candidate_grid, chandelier_trail,
    resolve_collision, running_max_stop, type7_percentile,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no market data)
# --------------------------------------------------------------------------- #
def vcb_long_dataset(n=400) -> pd.DataFrame:
    """Deterministic synthetic OHLCV that fires a V1.1 LONG exactly at bar 259.

    Structure (all values explicit, no randomization):
      bars 0..235   : wide high-volatility uptrend  -> high BB-width baseline,
                      EMA20 > EMA50 holds throughout.
      bars 236..258 : tight compression (low BB width, low ATR baseline, low prior highs).
      bar  259      : expansion + breakout up bar. FIRES a LONG here.
      bars 260+     : mild drift (keeps EMA uptrend for the tail).
    """
    n = int(n)
    close = np.zeros(n); high = np.zeros(n); low = np.zeros(n)
    for i in range(236):
        c = 100.0 + i * 0.15 + 2.0 * np.sin(i / 4.0)
        close[i] = c; high[i] = c + 3.5; low[i] = c - 3.5
    c0 = close[235]
    for i in range(236, 259):
        c = c0 + (i - 235) * 0.02
        close[i] = c; high[i] = c + 0.05; low[i] = c - 0.05
    c = close[258] + 1.5
    close[259] = c; high[259] = c + 0.1; low[259] = close[258] + 0.4
    for i in range(260, n):
        c = close[i - 1] + 0.05
        close[i] = c; high[i] = c + 0.4; low[i] = c - 0.4
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close})


def prepared(df: pd.DataFrame, strategy: VolatilityCompressionBreakoutV1_1) -> pd.DataFrame:
    return strategy.setup(df)


def golden_frame(i=200, close_hi=102.0, close_lo=98.0, atr=1.2, avg=1.0,
                 expansion_multiplier=1.1) -> pd.DataFrame:
    """A hand-built, deterministic prepared-like frame where every entry gate is set
    to PASS at bar ``i`` EXCEPT the ones the caller wants to probe.

    Returns a DataFrame of length ``i + 1`` whose indicator columns are constant so
    compression / expansion / trend hold trivially while ``close`` / ``breakout_*``
    are controllable. ``expansion_multiplier`` lets the callers confirm the frozen 1.1.
    """
    n = i + 1
    close = np.full(n, float(close_lo))
    close[i] = float(close_hi)
    high = close + 1.0
    low = close - 1.0
    df = pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "bb_width": np.full(n, 1.0),
        "squeeze_q": np.full(n, 1.0),                 # bb_width <= squeeze_q -> compression
        "atr_14": np.full(n, float(atr)),
        "avg_atr": np.full(n, float(avg)),            # avg > 0
        "ema_fast": np.full(n, 100.0),
        "ema_slow": np.full(n, 90.0),                 # fast > slow -> long trend
        "breakout_hh": np.full(n, 100.0),             # close[i] controlled vs this
        "breakout_ll": np.full(n, 0.0),               # close >> 0 -> no short breakout
    })
    return df


# =========================================================================== #
# 1. Strict running-max monotonicity under ATR expansion (F-01)
# =========================================================================== #
class TestRunningMaxMonotonicity:
    def test_long_stop_never_recedes_under_atr_spike(self):
        # A flat high-water mark with a rising ATR makes the RAW chandelier recede
        # (HH fixed, CH = HH - c*ATR drops as ATR rises). The running-max effective
        # stop must NOT recede (protocol F-01).
        fill, R, m = 100.0, 5.0, 3.0
        mgr = V1_1ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=m)
        mgr.armed = True  # force arming so the ratchet does not dominate the trail
        highs = [112.0, 112.0, 112.0, 112.0]
        lows = [110.0, 110.0, 110.0, 110.0]
        closes = [112.0, 113.0, 114.0, 115.0]
        atrs = [2.0, 2.6, 3.2, 4.0]
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "atr_14": atrs})
        stops = [mgr.scan_bar(df, t) for t in range(len(df))]
        assert all(stops[t] <= stops[t + 1] for t in range(len(stops) - 1))  # non-decreasing
        assert stops[-1] >= stops[0]
        # the raw chandelier DID recede — proving the running-max held the stop
        raw = [chandelier_trail(Direction.LONG, max(highs[:t + 1]), atrs[t], m)
               for t in range(len(df))]
        assert raw[-1] < raw[0]
        assert stops[-1] >= 112.0 - m * 4.0          # >= the receded raw, i.e. not below it

    def test_short_stop_never_recedes_under_atr_spike(self):
        fill, R, m = 100.0, 5.0, 3.0
        mgr = V1_1ExitManager(Direction.SHORT, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=m)
        mgr.armed = True
        lows = [90.0, 90.0, 90.0, 90.0]
        highs = [92.0, 92.0, 92.0, 92.0]
        closes = [90.0, 89.0, 88.0, 87.0]
        atrs = [2.0, 2.6, 3.2, 4.0]
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes, "atr_14": atrs})
        stops = [mgr.scan_bar(df, t) for t in range(len(df))]
        assert all(stops[t] >= stops[t + 1] for t in range(len(stops) - 1))  # non-increasing
        assert stops[-1] <= stops[0]
        raw = [chandelier_trail(Direction.SHORT, min(lows[:t + 1]), atrs[t], m)
               for t in range(len(df))]
        assert raw[-1] > raw[0]

    def test_running_max_stop_helper_is_monotonic_by_construction(self):
        assert running_max_stop(Direction.LONG, prior=105.0, rbe_if_armed=100.2, chandelier=103.0) == 105.0
        assert running_max_stop(Direction.LONG, prior=105.0, rbe_if_armed=100.2, chandelier=108.0) == 108.0
        assert running_max_stop(Direction.SHORT, prior=95.0, rbe_if_armed=99.8, chandelier=97.0) == 95.0
        assert running_max_stop(Direction.SHORT, prior=95.0, rbe_if_armed=99.8, chandelier=92.0) == 92.0


# =========================================================================== #
# 2. Breakeven ratchet: trigger at +1.5R and persistent latching (F-02)
# =========================================================================== #
class TestBreakevenRatchet:
    def test_arms_at_plus_1_5r_and_level_is_fill_times_1_plus_buffer(self):
        fill, R = 100.0, 5.0
        trigger = fill + BREAKEVEN_TRIGGER_R * R   # 107.5
        mgr = V1_1ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        assert mgr.armed is False
        assert break_even_level(fill, Direction.LONG, BREAKEVEN_BUFFER) == pytest.approx(100.2)
        assert break_even_level(fill, Direction.SHORT, BREAKEVEN_BUFFER) == pytest.approx(99.8)
        # just below the trigger must NOT arm
        df_below = pd.DataFrame({"high": [106.0], "low": [100.0],
                                 "close": [trigger - 0.01], "atr_14": [2.5]})
        mgr.scan_bar(df_below, 0)
        assert mgr.armed is False
        # reaching +1.5R arms it and floors the stop at the ratchet level
        df_at = pd.DataFrame({"high": [101.0], "low": [99.0],
                              "close": [trigger + 0.1], "atr_14": [2.5]})
        mgr.scan_bar(df_at, 0)
        assert mgr.armed is True
        # effective stop is floored at the ratchet level once armed
        assert mgr.effective_stop() >= 100.2 - 1e-9

    def test_latching_is_permanent_after_retrace(self):
        fill, R = 100.0, 5.0
        mgr = V1_1ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        high = [114.0, 108.0, 102.0, 100.5]
        low = [110.0, 104.0, 100.0, 99.0]
        close = [114.0, 107.0, 101.5, 100.0]
        atrs = [2.0, 2.0, 2.0, 2.0]
        df = pd.DataFrame({"high": high, "low": low, "close": close, "atr_14": atrs})
        stops = [mgr.scan_bar(df, t) for t in range(len(df))]
        assert mgr.armed is True
        assert stops[-1] >= 100.2 - 1e-9
        assert mgr.effective_stop() >= 100.2 - 1e-9

    def test_short_ratchet_level_and_latch(self):
        fill, R = 100.0, 5.0
        mgr = V1_1ExitManager(Direction.SHORT, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        trigger = fill - BREAKEVEN_TRIGGER_R * R   # 92.5
        df = pd.DataFrame({"high": [96.0], "low": [90.0],
                           "close": [trigger - 0.1], "atr_14": [2.0]})
        mgr.scan_bar(df, 0)
        assert mgr.armed is True
        assert mgr.effective_stop() <= 99.8 + 1e-9   # capped at the (short) ratchet floor
        assert break_even_level(fill, Direction.SHORT, 0.0020) == pytest.approx(99.8)


# =========================================================================== #
# 3. Conservative stop-first same-bar collision
# =========================================================================== #
class TestCollisionOrdering:
    def test_stop_wins_over_alt_on_same_bar(self):
        outcome, fill = resolve_collision(Direction.LONG, open_price=101.0,
                                          high=108.0, low=99.0,
                                          stop_level=100.0, alt_level=106.0)
        assert outcome == "stop"
        assert fill == pytest.approx(100.0)

    def test_gap_adverse_fill(self):
        outcome, fill = resolve_collision(Direction.LONG, open_price=99.0,
                                          high=99.5, low=98.0, stop_level=100.0)
        assert outcome == "stop"
        assert fill == pytest.approx(99.0)          # open adverse (below stop)
        o2, f2 = resolve_collision(Direction.SHORT, open_price=101.0,
                                   high=102.0, low=100.5, stop_level=100.0)
        assert o2 == "stop" and f2 == pytest.approx(101.0)

    def test_no_touch_no_fill(self):
        outcome, fill = resolve_collision(Direction.LONG, open_price=102.0,
                                          high=104.0, low=101.5,
                                          stop_level=100.0, alt_level=106.0)
        assert outcome is None

    def test_alt_fills_when_stop_not_touched(self):
        outcome, fill = resolve_collision(Direction.LONG, open_price=104.0,
                                          high=108.0, low=103.0,
                                          stop_level=100.0, alt_level=106.0)
        assert outcome == "alt" and fill == pytest.approx(106.0)


# =========================================================================== #
# 4. Strict no-look-ahead (signal bar t close -> fill bar t+1 open)
# =========================================================================== #
class TestNoLookAhead:
    def test_signal_fires_and_is_immune_to_future_bars(self):
        df = vcb_long_dataset()
        s = VolatilityCompressionBreakoutV1_1(params={"exit_config": "X1"})
        out = prepared(df, s)
        i = 259
        d, reason = s.entry_signal(out, i)
        assert d == Direction.LONG
        out2 = out.copy()
        out2.loc[out2.index >= i + 1, ["high", "low", "close"]] *= 1.5
        d2, r2 = s.entry_signal(out2, i)
        assert d2 == d and r2 == reason

    def test_fill_uses_open_of_next_bar_with_slip(self):
        s = VolatilityCompressionBreakoutV1_1(params={"exit_config": "X1"})
        assert s.fill_price(100.0, Direction.LONG) == pytest.approx(100.05)
        assert s.fill_price(100.0, Direction.SHORT) == pytest.approx(99.95)

    def test_exit_scan_bar_invariant_to_future_bars(self):
        df = vcb_long_dataset()
        s = VolatilityCompressionBreakoutV1_1(params={"exit_config": "X1"})
        out = prepared(df, s)
        fill = s.fill_price(float(df["open"].iloc[260]), Direction.LONG)
        R = s.risk_distance(float(out["atr_14"].iloc[259]))
        mgr = V1_1ExitManager(Direction.LONG, fill, R, entry_bar=260,
                              lookback_trail=s.lookback_trail,
                              trail_atr_mult=s.trail_atr_mult)
        for t in range(260, 280):
            mgr.scan_bar(out, t)
        stop_before = mgr.effective_stop()
        out2 = out.copy()
        out2.loc[out2.index >= 280, ["high", "low", "close", "atr_14"]] *= 0.3
        mgr2 = V1_1ExitManager(Direction.LONG, fill, R, entry_bar=260,
                               lookback_trail=s.lookback_trail,
                               trail_atr_mult=s.trail_atr_mult)
        for t in range(260, 280):
            mgr2.scan_bar(out2, t)
        assert mgr2.effective_stop() == pytest.approx(stop_before)


# =========================================================================== #
# 5. Fail-closed warm-up (bar 138 vs 139) + frozen entry semantics
# =========================================================================== #
class TestWarmUp:
    def _strategy(self):
        return VolatilityCompressionBreakoutV1_1(params={"exit_config": "X1"})

    def test_no_signal_before_139_any_data(self):
        s = self._strategy()
        df = vcb_long_dataset()
        out = prepared(df, s)
        for j in (0, 1, 50, 137, 138):
            assert s.entry_signal(out, j)[0] == Direction.NONE

    def test_signal_allowed_at_warmup_boundary(self):
        # Bar 139 is legitimate to evaluate (>= WARM_UP); on a smooth ramp the gate
        # conditions do not hold, so it deterministically returns NONE (no crash).
        s = self._strategy()
        n = 200
        df = pd.DataFrame({
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n) + 1,
            "low": np.linspace(99, 109, n) - 1,
            "close": np.linspace(100.5, 110.5, n),
        })
        out = prepared(df, s)
        assert s.entry_signal(out, 139)[0] in (Direction.LONG, Direction.SHORT, Direction.NONE)
        assert s.entry_signal(out, 138)[0] == Direction.NONE

    def test_breakout_is_plain_strict_cross_verbatim_v10(self):
        # The frozen entry preserves V1.0's breakout: a strict `>` cross of close_t over
        # the current-bar-excluded trailing high. There is NO freshness gate — a bar
        # that ALREADY holds above the range still fires (V1.0 long_break_at semantics).
        s = self._strategy()
        i = 200
        # breakout_hh[i] = 100 constant; close[i]=102 crosses it.
        df = golden_frame(i=i, close_hi=102.0)
        d, _ = s.entry_signal(df, i)
        assert d == Direction.LONG
        # A strong close on the PREVIOUS bar (already above the range) must NOT block
        # a signal at i — proving no fresh-cross requirement.
        df2 = golden_frame(i=i, close_hi=102.0)
        df2.loc[i - 1, "close"] = 103.0     # prev bar also above breakout_hh (100)
        df2.loc[i - 1, "high"] = 104.0      # not relevant to breakout ref (excl. current)
        d2, _ = s.entry_signal(df2, i)
        assert d2 == Direction.LONG
        # Strictness: close[i] <= breakout_hh -> no long breakout.
        df3 = golden_frame(i=i, close_hi=100.0)
        d3, _ = s.entry_signal(df3, i)
        assert d3 != Direction.LONG

    def test_expansion_uses_frozen_1_1_not_risk_atr_multiplier(self):
        # The frozen signal expansion multiplier is 1.1 (protocol §5a). The grid's RISK
        # atr_multiplier (2.0 / 2.5 / 3.0) must NOT drive the expansion threshold.
        s = VolatilityCompressionBreakoutV1_1(params={"atr_multiplier": 3.0, "exit_config": "X1"})
        i = 200
        # atr_14[i]=1.2 vs avg_atr[i]=1.0 -> 1.2 >= 1.1 passes (and 1.2 < 3.0, so a
        # buggy implementation using the risk atr_multiplier would fail).
        df = golden_frame(i=i, atr=1.2, avg=1.0)
        d, _ = s.entry_signal(df, i)
        assert d == Direction.LONG
        # Boundary: equality at 1.1 passes.
        df_eq = golden_frame(i=i, atr=1.1, avg=1.0)
        assert s.entry_signal(df_eq, i)[0] == Direction.LONG
        # Just below the frozen 1.1 fails (fail-closed, no signal) even though a risk
        # multiplier of 3.0 would trivially be satisfied anywhere above 0.
        df_below = golden_frame(i=i, atr=1.09, avg=1.0)
        assert s.entry_signal(df_below, i)[0] != Direction.LONG


# =========================================================================== #
# 6. 36-candidate grid instantiation
# =========================================================================== #
class TestCandidateGrid:
    def test_exactly_36_candidates_contiguous(self):
        grid = candidate_grid()
        assert len(grid) == 36
        assert [c.id for c in grid] == ["C%02d" % i for i in range(36)]

    def test_no_duplicates_full_cross(self):
        from itertools import product
        grid = candidate_grid()
        cells = [(c.breakout_lookback, c.squeeze_percentile, c.expansion_multiplier,
                  c.atr_multiplier, c.exit_config) for c in grid]
        assert len(set(cells)) == 36
        cross = set(product([20, 40], [30, 40], [1.1], [2.0, 2.5, 3.0], ["X1", "X2", "X3"]))
        assert set(cells) == cross

    def test_c00_anchor(self):
        c00 = CANDIDATE_TUPLES[0]
        assert c00.id == "C00"
        assert (c00.breakout_lookback, c00.squeeze_percentile, c00.expansion_multiplier,
                c00.atr_multiplier) == (20, 30, 1.1, 2.5)
        assert (c00.lookback_trail, c00.trail_atr_mult) == EXIT_CONFIGS["X1"]

    def test_all_candidates_instantiate(self):
        for cand in CANDIDATE_TUPLES:
            s = VolatilityCompressionBreakoutV1_1(params=cand.params)
            assert s.breakout_lookback == cand.breakout_lookback
            assert s.squeeze_percentile == cand.squeeze_percentile
            assert s.atr_multiplier == cand.atr_multiplier
            assert s.lookback_trail == cand.lookback_trail
            assert s.trail_atr_mult == cand.trail_atr_mult
            assert cand.expansion_multiplier == 1.1
            assert cand.exit_config in EXIT_CONFIGS

    def test_dominated_pair_rejected(self):
        with pytest.raises(ValueError):
            VolatilityCompressionBreakoutV1_1(params={
                "breakout_lookback": 20, "squeeze_percentile": 30,
                "atr_multiplier": 2.0, "lookback_trail": 10, "trail_atr_mult": 2.0,
            })


# =========================================================================== #
# 7. Engine wiring: prepare_fill / update_stop carry the running-max state
# =========================================================================== #
class TestEngineWiring:
    def test_prepare_fill_seeds_state_and_no_take_profit(self):
        s = VolatilityCompressionBreakoutV1_1(params={"atr_multiplier": 2.5, "exit_config": "X1"})
        df = vcb_long_dataset()
        out = prepared(df, s)
        out.loc[259, "atr_14"] = 10.0
        fill = 105.0
        stop, tp, state = s.prepare_fill(out, 259, Direction.LONG, fill)
        assert np.isclose(state["risk_R"], 25.0, atol=1e-9)     # 2.5 * 10
        assert np.isclose(stop, fill - 25.0, atol=1e-9)          # Stop0 = Fill - R
        assert tp is None                                        # uncapped right tail
        assert state["fill_price"] == 105.0
        assert state["breakeven_armed"] is False
        assert np.isclose(state["prior_stop"], stop, atol=1e-12) # base case Stop_{t0+1}

    def test_update_stop_advances_active_stop_monotonically(self):
        s = VolatilityCompressionBreakoutV1_1(params={"atr_multiplier": 2.5, "exit_config": "X1"})
        df = vcb_long_dataset()
        out = prepared(df, s)
        out.loc[259, "atr_14"] = 10.0
        fill = s.fill_price(float(out["open"].iloc[260]), Direction.LONG)
        stop, _, state = s.prepare_fill(out, 259, Direction.LONG, fill)

        class _FakePos:
            def __init__(self):
                self.direction = "long"
                self.entry_bar = 260
                self.active_stop = None
                self.extra_state = dict(state)

        pos = _FakePos()
        # The manager must evolve the stop from Stop0 through several bars.
        active = []
        for bar in range(261, 300):
            s.update_stop(out, bar, pos)
            active.append(pos.active_stop)
        assert len(active) == len(set(np.round(active, 6))) or all(
            active[t] <= active[t + 1] for t in range(len(active) - 1)
        )
        assert all(active[t] <= active[t + 1] for t in range(len(active) - 1))  # monotone


# =========================================================================== #
# Extra: Type-7 percentile consistency with numpy (auto-regression guard)
# =========================================================================== #
class TestType7Percentile:
    def test_matches_numpy_linear(self):
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        for q in (0.0, 0.1, 0.3, 0.5, 0.75, 1.0):
            assert type7_percentile(x, q) == pytest.approx(np.quantile(x, q, method="linear"))