"""Oracle / unit tests for PROTOCOL-STRATEGY-10-V1.0 (Strategy #10).

Covers the invariants the design memo (``data/memo_strategy_10_v1_0_design.md``)
declares as non-negotiable:

  1. Flat / zero-range candles (Delta == 0) -> wick ratio 0.0, signal suppressed.
  2. Micro-doji suppression: range < min_range_atr_mult * ATR(14) kills the signal even
     with a large wick-rejection ratio.
  3. Pure hammer / shooting star bodies verify the exact wick-ratio arithmetic.
  4. Strict no-look-ahead: Shift(1) swing levels are immune to the current/future bar.
  5. Breakeven ratchet arms and latches at +1.0R, monotonic, with fee/slip/funding covered.
  6. Fail-closed warm-up: zero signals for ``t < max(lookback, 50, 14) + 1``.
  7. The 36-candidate grid (C00..C35) equals the full 3x3x2x2 Cartesian cross, C00 anchored.

DATA FIREWALL: All fixtures are synthetic. No raw market data (train / 2025 validation /
2026 OOS) is accessed by this suite.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies.base import Direction
from crypto_quant.strategies.liquidity_sweep_mean_reversion import (
    BREAKEVEN_BUFFER, BREAKEVEN_TRIGGER_R, CANDIDATE_GRID, EXHAUSTION_RATIOS,
    EXIT_MODES, FLAT_RANGE_EPS, LOOKBACKS, MIN_RANGE_ATR_MULTS,
    LiquiditySweepMeanReversionStrategy as Strategy,
    breakeven_level, candidate_grid, cost_covered, ratchet_stop, safe_ratio,
    stop_0, take_profit_level, warm_up_for,
)
from crypto_quant.strategies.base import Direction


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no market data)
# --------------------------------------------------------------------------- #
def long_dataset(n=240) -> pd.DataFrame:
    """Deterministic synthetic OHLCV that fires a LONG exactly at bar 150.

    Swing low of 100 planted at bar 120 (inside the prior window [102, 149]); the
    signal bar 150 sweeps below it (low=99) and rejects back inside with a long lower
    wick (6/7.2 = 0.833) and a close in the upper half of the bar.
    """
    n = int(n)
    open_ = np.full(n, 104.0)
    high = np.full(n, 105.0)
    low = np.full(n, 103.0)
    close = np.full(n, 104.0)
    low[120] = 100.0                       # prior swing low
    i = 150
    open_[i] = 105.0
    close[i] = 105.3
    high[i] = 106.2
    low[i] = 99.0
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


class _FakePos:
    def __init__(self, direction, entry_bar, state):
        self.direction = direction
        self.entry_bar = int(entry_bar)
        self.active_stop = None
        self.extra_state = dict(state)


# =========================================================================== #
# 1. Flat / zero-range candles (Delta == 0 -> ratio 0.0, no signal)
# =========================================================================== #
class TestFlatRange:
    def test_safe_ratio_flat_denominator_is_zero(self):
        r = safe_ratio(pd.Series([3.0, 2.0]), pd.Series([0.0, 0.0]))
        assert r.iloc[0] == 0.0 and r.iloc[1] == 0.0

    def test_flat_candle_suppresses_signal(self):
        n = 200
        open_ = np.full(n, 100.0); high = np.full(n, 100.0)
        low = np.full(n, 100.0); close = np.full(n, 100.0)
        # make bar i a zero-range bar
        i = 150
        df = long_dataset()
        # force a true flat bar at i while preserving the prior swing-low structure
        df.loc[i, ["open", "high", "low", "close"]] = 100.0
        s = Strategy(params={})
        out = s.setup(df)
        assert out["total_range"].iloc[i] == 0.0
        assert out["upper_wick_ratio"].iloc[i] == 0.0
        assert out["lower_wick_ratio"].iloc[i] == 0.0
        assert s.entry_signal(out, i)[0] == Direction.NONE

    def test_sub_eps_range_fails_closed(self):
        s = Strategy(params={})
        # an intentionally degenerate prepared frame with a tiny diagonal (<=EPS not
        # producible cheaply in raw OHLCV, so probe the guard directly on safe_ratio).
        assert safe_ratio(pd.Series([1.0]), pd.Series([FLAT_RANGE_EPS / 2.0])).iloc[0] == 0.0


# =========================================================================== #
# 2. Micro-doji suppression (range < min_range_atr_mult * ATR -> no signal)
# =========================================================================== #
class TestMicroDoji:
    def _star_frame(self, atr) -> pd.DataFrame:
        """A prepared frame where bar i is a shooting star that passes every short gate
        EXCEPT the dynamic minimum-range filter, whose strength is set by ``atr``."""
        n = 200
        df = pd.DataFrame({
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
            # preset indicator columns (as setup would produce)
            "swing_high": np.full(n, 101.0),
            "swing_low": np.full(n, 99.0),
            "sma_50": np.full(n, 100.0),
            "midpoint": np.full(n, 100.0),
            "atr_14": np.full(n, float(atr)),
            "total_range": np.full(n, 2.0),
            "upper_wick_ratio": np.full(n, 0.75),   # >= theta (0.45)
            "lower_wick_ratio": np.full(n, 0.0),
        })
        # bar i: a compact shooting star sweeping above the prior high
        i = 150
        df.loc[i, "open"] = 100.0
        df.loc[i, "close"] = 100.5
        df.loc[i, "high"] = 102.0                  # high > swing_high (101) -> sweep
        df.loc[i, "low"] = 100.0
        df.loc[i, "total_range"] = 2.0             # delta = 2
        df.loc[i, "upper_wick_ratio"] = 0.75       # 1.5 / 2.0
        return df

    def test_micro_candle_suppressed(self):
        s = Strategy(params={"min_range_atr_mult": 0.5})
        warm = s.warm_up
        i = max(150, warm)
        df = self._star_frame(atr=10.0)            # 0.5*10 = 5 > delta(2) -> suppress
        d, _ = s.entry_signal(df, i)
        assert d == Direction.NONE

    def test_equivalent_larger_candle_passes(self):
        s = Strategy(params={"min_range_atr_mult": 0.5})
        warm = s.warm_up
        i = max(150, warm)
        df = self._star_frame(atr=2.0)             # 0.5*2 = 1 <= delta(2) -> allowed
        d, _ = s.entry_signal(df, i)
        assert d == Direction.SHORT


# =========================================================================== #
# 3. Pure hammer / shooting star wick-ratio arithmetic
# =========================================================================== #
class TestWickRatioArithmetic:
    def test_hammer(self):
        df = pd.DataFrame({
            "open": np.array([99.0]), "close": np.array([100.0]),
            "high": np.array([101.0]), "low": np.array([95.0]),
        })
        s = Strategy(params={})
        out = s.setup(df)
        assert out["total_range"].iloc[0] == 6.0
        assert out["upper_wick"].iloc[0] == pytest.approx(1.0)
        assert out["lower_wick"].iloc[0] == pytest.approx(4.0)
        assert out["upper_wick_ratio"].iloc[0] == pytest.approx(1.0 / 6.0)
        assert out["lower_wick_ratio"].iloc[0] == pytest.approx(4.0 / 6.0)

    def test_shooting_star(self):
        df = pd.DataFrame({
            "open": np.array([100.0]), "close": np.array([99.0]),
            "high": np.array([105.0]), "low": np.array([98.0]),
        })
        s = Strategy(params={})
        out = s.setup(df)
        assert out["total_range"].iloc[0] == 7.0
        assert out["upper_wick_ratio"].iloc[0] == pytest.approx(5.0 / 7.0)
        assert out["lower_wick_ratio"].iloc[0] == pytest.approx(1.0 / 7.0)

    def test_doji_equal_body(self):
        df = pd.DataFrame({
            "open": np.array([100.0]), "close": np.array([100.0]),
            "high": np.array([103.0]), "low": np.array([97.0]),
        })
        s = Strategy(params={})
        out = s.setup(df)
        assert out["body_top"].iloc[0] == pytest.approx(100.0)
        assert out["body_bot"].iloc[0] == pytest.approx(100.0)
        assert out["upper_wick_ratio"].iloc[0] == pytest.approx(3.0 / 6.0)
        assert out["lower_wick_ratio"].iloc[0] == pytest.approx(3.0 / 6.0)


# =========================================================================== #
# 4. Strict no-look-ahead (Shift(1) swing levels)
# =========================================================================== #
class TestNoLookAhead:
    def test_perturbing_current_bar_does_not_change_swing_levels(self):
        raw = long_dataset()
        s = Strategy(params={})
        o1 = s.setup(raw)
        raw2 = raw.copy()
        raw2.loc[150, "high"] = 999.0              # current-bar high perturbed
        raw2.loc[150, "low"] = 1.0                 # current-bar low perturbed
        o2 = s.setup(raw2)
        # swing levels at bar 150 use only [102..149] — immune to bar 150
        assert o2["swing_high"].iloc[150] == o1["swing_high"].iloc[150]
        assert o2["swing_low"].iloc[150] == o1["swing_low"].iloc[150]
        assert abs(o2["swing_high"].iloc[150] - 105.0) < 1e-9
        assert abs(o2["swing_low"].iloc[150] - 100.0) < 1e-9

    def test_perturbing_future_bars_does_not_change_swing_levels(self):
        raw = long_dataset()
        s = Strategy(params={})
        o1 = s.setup(raw)
        raw3 = raw.copy()
        raw3.loc[160:, "high"] *= 3.0
        raw3.loc[160:, "low"] /= 3.0
        raw3.loc[160:, "close"] *= 2.0
        o3 = s.setup(raw3)
        assert o3["swing_high"].iloc[150] == o1["swing_high"].iloc[150]
        assert o3["swing_low"].iloc[150] == o1["swing_low"].iloc[150]

    def test_fires_long_and_direction_immune_to_future(self):
        raw = long_dataset()
        s = Strategy(params={})
        o1 = s.setup(raw)
        d1, reason1 = s.entry_signal(o1, 150)
        assert d1 == Direction.LONG and reason1 == "sweep_reject_long"
        assert o1["entry_signal"].iloc[150] == 1.0
        o4 = o1.copy()
        o4.loc[160:, ["high", "low", "close"]] *= 2.0
        d4, reason4 = s.entry_signal(o4, 150)
        assert d4 == d1 and reason4 == reason1


# =========================================================================== #
# 5. Breakeven ratchet: +1.0R arming, permanent latch, monotonic, cost-covered
# =========================================================================== #
class TestBreakevenRatchet:
    def test_arms_exactly_at_plus_1_0r_long(self):
        fill, R = 100.0, 5.0
        rbe = breakeven_level(fill, Direction.LONG, BREAKEVEN_BUFFER)   # 100.2
        state = {
            "fill_price": fill, "entry_price": fill, "initial_r": R,
            "prior_stop": 99.0, "stop_direction": "long",
            "breakeven_armed": False, "break_even_level": rbe, "signal_bar": 0,
        }
        # j = current_bar_index - 1 addresses the completed bar's close directly.
        closes = [0.0, 104.99, 105.0, 103.0, 101.0]   # j: 0..4 ; +1R=105.0
        df = pd.DataFrame({
            "open": [100.0] * 5, "high": [106.0] * 5,
            "low": [99.0] * 5, "close": closes,
        })
        s = Strategy(params={})
        pos = _FakePos("long", entry_bar=1, state=state)
        # current_bar_index=2 -> completed bar j=1 -> close 104.99 (below +1R): no arm
        s.update_stop(df, 2, pos, state)
        assert state["breakeven_armed"] is False
        # current_bar_index=3 -> completed bar j=2 -> close 105.0 (== +1R): arms
        s.update_stop(df, 3, pos, state)
        assert state["breakeven_armed"] is True
        assert pos.active_stop == pytest.approx(rbe, abs=1e-9)
        # latched: lower closes (103.0 / 101.0) cannot loosen the clamped stop
        prev = pos.active_stop
        for cb in range(4, 5):
            s.update_stop(df, cb, pos, state)
            assert state["breakeven_armed"] is True
            assert pos.active_stop == pytest.approx(prev, abs=1e-9)

    def test_short_ratchet_and_monotonicity(self):
        fill, R = 100.0, 5.0
        rbe = breakeven_level(fill, Direction.SHORT, BREAKEVEN_BUFFER)  # 99.8
        state = {
            "fill_price": fill, "entry_price": fill, "initial_r": R,
            "prior_stop": 101.0, "stop_direction": "short",
            "breakeven_armed": False, "break_even_level": rbe, "signal_bar": 0,
        }
        closes = [0.0, 95.01, 95.0, 97.0, 96.0]          # j: 0..4 ; -1R = 95.0
        df = pd.DataFrame({
            "open": [100.0] * 5, "high": [101.5] * 5,
            "low": [94.0] * 5, "close": closes,
        })
        s = Strategy(params={})
        pos = _FakePos("short", entry_bar=1, state=state)
        s.update_stop(df, 2, pos, state)            # j=1 close 95.01 (above -1R): no arm
        assert state["breakeven_armed"] is False
        s.update_stop(df, 3, pos, state)            # j=2 close 95.0 (== -1R): arms
        assert state["breakeven_armed"] is True
        assert pos.active_stop == pytest.approx(rbe, abs=1e-9)
        # monotonic short: never rises back above the ratchet floor after further close
        prev = pos.active_stop
        s.update_stop(df, 4, pos, state)            # j=3 close 97.0 -> clamp to prior
        assert state["breakeven_armed"] is True
        assert pos.active_stop == pytest.approx(prev, abs=1e-9)

    def test_ratchet_level_covers_friction_and_is_latched(self):
        fill = 100.0
        gain_long = cost_covered(breakeven_level(fill, Direction.LONG), fill, Direction.LONG)
        gain_short = cost_covered(breakeven_level(fill, Direction.SHORT), fill, Direction.SHORT)
        # 0.0020 (~0.20%) covers round-trip 0.18% (0.08 fees + 0.10 slip) + funding drag
        assert gain_long >= 0.0018 * fill
        assert gain_short >= 0.0018 * fill
        # ratchet is strictly monotonic by construction
        assert ratchet_stop(98.0, 100.2, Direction.LONG) == pytest.approx(100.2)
        assert ratchet_stop(101.0, 100.2, Direction.LONG) == pytest.approx(101.0)
        assert ratchet_stop(102.0, 99.8, Direction.SHORT) == pytest.approx(99.8)
        assert ratchet_stop(99.0, 99.8, Direction.SHORT) == pytest.approx(99.0)

    def test_prepare_fill_seeds_state_and_levels(self):
        raw = long_dataset()
        s = Strategy(params={})                      # C00 anchor: lookback48, SM50
        out = s.setup(raw)
        i = 150
        fill = float(out["open"].iloc[i + 1])        # 104.0, next-bar open
        # §9.1 target polarity: a LONG target must clear the fill. long_dataset's flat
        # closes put SMA50 == fill, which would (correctly) fail closed — so stamp a
        # monotonic target above the fill to exercise the state-seeding path.
        out.loc[i, "sma_50"] = fill + 10.0           # 114.0 -> target above fill
        stop, tp, state = s.prepare_fill(out, i, Direction.LONG, fill)
        exp_stop = 99.0 * (1 - 0.0010)               # Low_t * (1 - 0.0010)
        assert stop == pytest.approx(exp_stop)
        assert state["entry_price"] == pytest.approx(fill)
        assert state["initial_r"] == pytest.approx(abs(fill - stop))
        assert state["breakeven_armed"] is False
        exp_tp = float(out["sma_50"].iloc[i])        # exit_mode == SMA50
        assert tp == pytest.approx(exp_tp)
        assert take_profit_level(float(out["sma_50"].iloc[i]),
                                 float(out["midpoint"].iloc[i]), "SMA50") == pytest.approx(exp_tp)
        assert take_profit_level(float(out["sma_50"].iloc[i]),
                                 float(out["midpoint"].iloc[i]), "MIDPOINT") == pytest.approx(
                                      float(out["midpoint"].iloc[i]))

    def test_stop_0_buffer_direction(self):
        assert stop_0(105.0, 99.0, Direction.LONG) == pytest.approx(99.0 * (1 - 0.0010))
        assert stop_0(105.0, 99.0, Direction.SHORT) == pytest.approx(105.0 * (1 + 0.0010))


# =========================================================================== #
# 6. Fail-closed warm-up bound
# =========================================================================== #
class TestWarmUp:
    def test_zero_signals_before_warmup_all_lookbacks(self):
        raw = long_dataset()
        for lb in LOOKBACKS:
            s = Strategy(params={"lookback": lb})
            warm = warm_up_for(lb)
            out = s.setup(raw)
            for j in (0, 1, warm - 2, warm - 1):
                assert s.entry_signal(out, j)[0] == Direction.NONE, (lb, j)
                assert out["entry_signal"].iloc[j] == 0.0

    def test_fires_at_or_after_warmup(self):
        s = Strategy(params={"lookback": 48})
        assert s.warm_up == 51
        raw = long_dataset()
        out = s.setup(raw)
        assert s.entry_signal(out, 51)[0] == Direction.NONE     # no setup here, safe
        assert s.entry_signal(out, 150)[0] == Direction.LONG


# =========================================================================== #
# 7. Grid orthogonality: 36 == 3x3x2x2 Cartesian cross, C00 anchored
# =========================================================================== #
class TestCandidateGrid:
    def test_exactly_36_contiguous_ids(self):
        grid = candidate_grid()
        assert len(grid) == 36
        assert [c["id"] for c in grid] == ["C%02d" % i for i in range(36)]

    def test_full_orthogonal_cross(self):
        from itertools import product
        grid = candidate_grid()
        cells = {(c["lookback"], c["exhaustion_ratio"], c["min_range_atr_mult"], c["exit_mode"])
                 for c in grid}
        cross = set(product(LOOKBACKS, EXHAUSTION_RATIOS, MIN_RANGE_ATR_MULTS, EXIT_MODES))
        assert len(cells) == 36
        assert cells == cross

    def test_c00_anchor(self):
        c00 = CANDIDATE_GRID[0]
        assert c00["id"] == "C00"
        assert (c00["lookback"], c00["exhaustion_ratio"],
                c00["min_range_atr_mult"], c00["exit_mode"]) == (48, 0.45, 0.5, "SMA50")

    def test_all_candidates_instantiate(self):
        for cand in CANDIDATE_GRID:
            params = {k: v for k, v in cand.items() if k != "id"}
            s = Strategy(params=params)
            assert s.lookback == cand["lookback"]
            assert s.exhaustion_ratio == cand["exhaustion_ratio"]
            assert s.min_range_atr_mult == cand["min_range_atr_mult"]
            assert s.exit_mode == cand["exit_mode"]

    def test_invalid_params_rejected(self):
        with pytest.raises(ValueError):
            Strategy(params={"lookback": 13})
        with pytest.raises(ValueError):
            Strategy(params={"exhaustion_ratio": 0.30})
        with pytest.raises(ValueError):
            Strategy(params={"min_range_atr_mult": 1.0})
        with pytest.raises(ValueError):
            Strategy(params={"exit_mode": "VWAP"})


# =========================================================================== #
# 8. §9.1 target-polarity fail-closed: a non-monotonic target aborts the entry
# =========================================================================== #
class TestTargetPolarityFailClosed:
    """PROTOCOL-STRATEGY-10-V1.0 §9.1 (FROZEN).

    SMA50 (trend anchor) and Midpoint (range anchor) can each land on the *wrong*
    side of the fill depending on regime. A take-profit for a Long must clear the
    fill; for a Short it must sit below the fill. If the target is at or on the
    wrong side of the fill the level is unreachable as profit, so ``prepare_fill``
    must fail closed -> ``(None, None, {})`` and the entry is aborted.
    """

    def _frame(self, fill, long_ok, short_ok) -> pd.DataFrame:
        raw = long_dataset()
        s = Strategy(params={})
        out = s.setup(raw)
        i = 150
        # Force a monotonic midpoint so only the exit_mode line under test decides.
        out.loc[i, "midpoint"] = fill + 5.0 if long_ok else fill - 5.0
        if long_ok:
            out.loc[i, "sma_50"] = fill + 5.0
        if short_ok:
            out.loc[i, "sma_50"] = fill - 5.0
        return out, i

    def test_long_aborts_when_fill_at_or_below_target(self):
        s = Strategy(params={})
        out, i = self._frame(104.0, long_ok=False, short_ok=True)   # sma=99 (< fill)
        # SMA50 target below the fill -> non-monotonic LONG -> abort
        stop, tp, state = s.prepare_fill(out, i, Direction.LONG, 104.0)
        assert stop is None and tp is None and state == {}

    def test_long_aborts_when_target_exactly_at_fill(self):
        raw = long_dataset()
        s = Strategy(params={})
        out = s.setup(raw)
        i = 150
        fill = float(out["open"].iloc[i + 1])       # 104.0
        out.loc[i, "sma_50"] = fill                  # target == fill -> unreachable
        stop, tp, state = s.prepare_fill(out, i, Direction.LONG, fill)
        assert stop is None and tp is None and state == {}

    def test_short_aborts_when_fill_at_or_above_target(self):
        s = Strategy(params={})
        out, i = self._frame(104.0, long_ok=True, short_ok=False)  # sma=109 (> fill)
        # SMA50 target above the fill -> non-monotonic SHORT -> abort
        stop, tp, state = s.prepare_fill(out, i, Direction.SHORT, 104.0)
        assert stop is None and tp is None and state == {}

    def test_monotonic_targets_pass(self):
        s = Strategy(params={})                      # exit_mode == SMA50
        # LONG with target above fill -> allowed
        out, i = self._frame(104.0, long_ok=True, short_ok=False)  # sma=109
        stop, tp, state = s.prepare_fill(out, i, Direction.LONG, 104.0)
        assert tp == pytest.approx(109.0) and state != {}
        # SHORT with target below fill -> allowed
        out, i = self._frame(104.0, long_ok=False, short_ok=True)  # sma=99
        stop, tp, state = s.prepare_fill(out, i, Direction.SHORT, 104.0)
        assert tp == pytest.approx(99.0) and state != {}