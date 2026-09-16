"""Oracle / unit tests for PROTOCOL-STRATEGY-09-V1.2 (Strategy #9 V1.2).

Covers the invariants the frozen protocol (``data/protocol_strategy_09_v1_2.md``,
SHA-256 ``17a255d50b72b255a3490af297799985cca9b075623456af363c5ffc28ad008e``)
declares as non-negotiable:

  1. Exact classical Wilder RMA ADX(14) against synthetic deterministic step functions.
  2. The ADX regime gate suppresses entry when ``ADX <= threshold`` (strict, fail-closed)
     and permits entry when ``ADX > threshold``.
  3. Strict running-max monotonicity of the Chandelier stop under ATR expansion (F-01).
  4. Breakeven ratchet arming at +1.5R with permanent latching, level ``Fill*(1+/-0.0020)`` (F-02).
  5. Conservative stop-first same-bar collision resolution.
  6. Fail-closed warm-up bound (``t < 139`` never signals) + no-look-ahead.
  7. The full 24-candidate grid (C00..C23) instantiates with exact orthogonal parameters,
     C00 anchored, ``X4=(10,2.0)`` re-included, ``squeeze_percentile=30`` and
     ``atr_multiplier=2.5`` fixed.

Plus correctness guards that pin the V1.2 fixes (Wilder ADX over SMA-smoothed; the
regime gate is a disable/enable only; ``expansion_multiplier=1.1`` not the risk width).

DATA FIREWALL: All fixtures are synthetic. No raw market data (train / 2025
validation / 2026 OOS) is accessed by this suite.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies.base import Direction
from crypto_quant.strategies.volatility_compression_breakout_v1_2 import (
    ADX_THRESHOLDS, BREAKEVEN_BUFFER, BREAKEVEN_TRIGGER_R, CANDIDATE_TUPLES,
    EXIT_CONFIGS, RISK_ATR_MULTIPLIER, SQUEEZE_PERCENTILE, VolatilityCompressionBreakoutV1_2,
    V1_2ExitManager, break_even_level, candidate_grid, chandelier_trail,
    resolve_collision, running_max_stop, wilder_adx,
)


# --------------------------------------------------------------------------- #
# Independent reference implementation of Wilder's ADX (published formula)
# --------------------------------------------------------------------------- #
def _ref_adx(high, low, close, period=14):
    """Reference Wilder ADX written directly from the Wilder publication (bar-1-based)."""
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2 * period:
        return out
    tr = np.zeros(n); dmp = np.zeros(n); dmm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        dmp[i] = up if (up > dn and up > 0) else 0.0
        dmm[i] = dn if (dn > up and dn > 0) else 0.0
    str_ = np.zeros(n); sp = np.zeros(n); sm = np.zeros(n)
    for i in range(period, n):
        if i == period:
            str_[i] = float(np.sum(tr[1:period + 1]))
            sp[i] = float(np.sum(dmp[1:period + 1]))
            sm[i] = float(np.sum(dmm[1:period + 1]))
        else:
            str_[i] = str_[i - 1] + (tr[i] - str_[i - 1]) / period
            sp[i] = sp[i - 1] + (dmp[i] - sp[i - 1]) / period
            sm[i] = sm[i - 1] + (dmm[i] - sm[i - 1]) / period
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if str_[i] <= 0:
            continue
        dpi = 100.0 * sp[i] / str_[i]
        dmi = 100.0 * sm[i] / str_[i]
        if dpi + dmi <= 0:
            continue
        dx[i] = 100.0 * abs(dpi - dmi) / (dpi + dmi)
    first = 2 * period - 1
    if n >= first + 1:
        out[first] = float(np.mean(dx[period:first + 1]))
        for i in range(first + 1, n):
            if not np.isfinite(dx[i]):
                out[i] = np.nan
                continue
            out[i] = out[i - 1] + (dx[i] - out[i - 1]) / period
    return out


def _deterministic_ohlc(n=300, seed=7):
    """Seeded-but-deterministic OHLC with uptrends, downtrends, and flats."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.6, n))
    # inject a sustained strong downleg and upleg so directional strength is meaningful
    close[80:130] = close[79] - np.linspace(0, 8, 50)
    close[150:210] = close[149] + np.linspace(0, 9, 60)
    high = close + np.abs(rng.normal(0.3, 0.4, n))
    low = close - np.abs(rng.normal(0.3, 0.4, n))
    return {"high": high, "low": low, "close": close}


# --------------------------------------------------------------------------- #
# Synthetic fixtures (no market data)
# --------------------------------------------------------------------------- #
def vcb_long_dataset(n=400) -> pd.DataFrame:
    """Deterministic synthetic OHLCV that fires a V1.2 LONG exactly at bar 259
    (uptrend on bars 0..235 keeps ADX high; compression 236..258; breakout 259)."""
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


def prepared(df: pd.DataFrame, strategy: VolatilityCompressionBreakoutV1_2) -> pd.DataFrame:
    return strategy.setup(df)


def golden_frame(i=200, close_hi=102.0, close_lo=98.0, atr=1.2, avg=1.0, adx=40.0,
                 expansion_multiplier=1.1) -> pd.DataFrame:
    """Hand-built deterministic prepared-like frame passing every entry gate at bar
    ``i`` EXCEPT the ones the caller wants to probe, plus a controllable ADX column."""
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
        "adx_14": np.full(n, float(adx)),
    })
    return df


# =========================================================================== #
# 1. Exact Wilder ADX(14) calculation
# =========================================================================== #
class TestWilderADX:
    def test_nan_before_stabilization_window(self):
        o = _deterministic_ohlc(n=300)
        adx = wilder_adx(o["high"], o["low"], o["close"], period=14)
        for j in range(2 * 14 - 1):
            assert np.isnan(adx.iloc[j])        # fail-closed until ADX is defined

    def test_matches_independent_reference_across_mixed_series(self):
        o = _deterministic_ohlc(n=300, seed=11)
        adx = wilder_adx(o["high"], o["low"], o["close"], period=14)
        ref = _ref_adx(o["high"], o["low"], o["close"], 14)
        # compare only where the reference is finite (both defined), at all bars
        mask = np.isfinite(ref)
        assert mask.sum() > 0
        assert np.allclose(adx.iloc[mask], ref[mask], rtol=1e-9, atol=1e-9)

    def test_pure_uptrend_adx_converges_to_100(self):
        # Strictly increasing high/low -> -DM=0 always, +DI=100, DX=100, ADX -> 100.
        n = 200
        close = np.linspace(100, 140, n)
        high = close + 0.5
        low = close - 0.5
        adx = wilder_adx(high, low, close, period=14)
        assert adx.iloc[-1] == pytest.approx(100.0, abs=1e-6)
        assert adx.iloc[-1] > 99.0

    def test_zero_direction_flat_series_fails_closed(self):
        # Constant prices -> TR>0 but +DM=-DM=0 -> +DI + -DI = 0 -> DX undefined =>
        # FAIL-CLOSED to NaN per protocol §3b (the regime gate therefore suppresses entry
        # in a zero-direction market rather than emitting a spurious trend value).
        close = np.full(200, 100.0)
        high = close + 1.0
        low = close - 1.0
        adx = wilder_adx(high, low, close, period=14)
        assert np.isnan(adx.iloc[-1])
        assert not np.isfinite(adx.iloc[-1])

    def test_regime_gate_reads_the_same_adx(self):
        # setup() must expose the ADX column used by the entry regime gate.
        s = VolatilityCompressionBreakoutV1_2(params={"exit_config": "X1"})
        df = vcb_long_dataset()
        out = s.setup(df)
        assert "adx_14" in out.columns
        direct = wilder_adx(df["high"], df["low"], df["close"], 14)
        assert np.allclose(out["adx_14"].iloc[27:], direct.iloc[27:], rtol=1e-9, atol=1e-9)


# =========================================================================== #
# 2. ADX regime gate suppression / permission
# =========================================================================== #
class TestRegimeGate:
    def _strategy(self, threshold=25):
        return VolatilityCompressionBreakoutV1_2(
            params={"adx_threshold": threshold, "exit_config": "X1"})

    def test_blocks_entry_when_adx_below_threshold(self):
        s = self._strategy(threshold=25)
        i = 200
        d, _ = s.entry_signal(golden_frame(i=i, adx=20.0), i)
        assert d == Direction.NONE

    def test_permits_entry_when_adx_above_threshold(self):
        s = self._strategy(threshold=25)
        i = 200
        d, _ = s.entry_signal(golden_frame(i=i, adx=40.0), i)
        assert d == Direction.LONG

    def test_strict_boundary_adx_equals_threshold_blocks(self):
        # Gate is `ADX > threshold` (strict). Equality must NOT fire (fail-closed).
        s = self._strategy(threshold=25)
        i = 200
        d, _ = s.entry_signal(golden_frame(i=i, adx=25.0), i)
        assert d == Direction.NONE

    def test_nonfinite_adx_fails_closed(self):
        s = self._strategy(threshold=25)
        i = 200
        df = golden_frame(i=i, adx=40.0)
        df.loc[i, "adx_14"] = float("nan")
        d, _ = s.entry_signal(df, i)
        assert d == Direction.NONE

    def test_regime_gate_is_a_disable_only(self):
        # Same bars: only the ADX value differs; the signal flips purely on the gate.
        s = self._strategy(threshold=25)
        i = 200
        assert s.entry_signal(golden_frame(i=i, adx=40.0), i)[0] == Direction.LONG
        assert s.entry_signal(golden_frame(i=i, adx=20.0), i)[0] == Direction.NONE

    def test_gate_also_suppresses_on_end_to_end_setup_data(self):
        s = self._strategy(threshold=25)
        df = vcb_long_dataset()
        out = prepared(df, s)
        # natural strong uptrend -> ADX is high -> signal fires at 259
        assert s.entry_signal(out, 259)[0] == Direction.LONG
        assert float(out["adx_14"].iloc[259]) > 25
        # suppress: force ADX below threshold -> gate must block the SAME selection
        out2 = out.copy()
        out2.loc[259, "adx_14"] = 5.0
        assert s.entry_signal(out2, 259)[0] == Direction.NONE


# =========================================================================== #
# 3. Strict running-max monotonicity under ATR expansion (F-01)
# =========================================================================== #
class TestRunningMaxMonotonicity:
    def test_long_stop_never_recedes_under_atr_spike(self):
        fill, R, m = 100.0, 5.0, 3.0
        mgr = V1_2ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=m)
        mgr.armed = True  # force arming so the ratchet does not dominate the trail
        df = pd.DataFrame({
            "high": [112.0, 112.0, 112.0, 112.0],
            "low": [110.0, 110.0, 110.0, 110.0],
            "close": [112.0, 113.0, 114.0, 115.0],
            "atr_14": [2.0, 2.6, 3.2, 4.0],
        })
        stops = [mgr.scan_bar(df, t) for t in range(len(df))]
        assert all(stops[t] <= stops[t + 1] for t in range(len(stops) - 1))  # non-decreasing
        raw = [chandelier_trail(Direction.LONG, 112.0, atrs, m) for atrs in [2.0, 2.6, 3.2, 4.0]]
        assert raw[-1] < raw[0]                                # raw trail DID recede
        assert stops[-1] >= 112.0 - m * 4.0                    # running-max held it

    def test_short_stop_never_recedes_under_atr_spike(self):
        fill, R, m = 100.0, 5.0, 3.0
        mgr = V1_2ExitManager(Direction.SHORT, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=m)
        mgr.armed = True
        df = pd.DataFrame({
            "low": [90.0, 90.0, 90.0, 90.0],
            "high": [92.0, 92.0, 92.0, 92.0],
            "close": [90.0, 89.0, 88.0, 87.0],
            "atr_14": [2.0, 2.6, 3.2, 4.0],
        })
        stops = [mgr.scan_bar(df, t) for t in range(len(df))]
        assert all(stops[t] >= stops[t + 1] for t in range(len(stops) - 1))  # non-increasing

    def test_running_max_stop_helper_is_monotonic_by_construction(self):
        assert running_max_stop(Direction.LONG, prior=105.0, rbe_if_armed=100.2, chandelier=103.0) == 105.0
        assert running_max_stop(Direction.LONG, prior=105.0, rbe_if_armed=100.2, chandelier=108.0) == 108.0
        assert running_max_stop(Direction.SHORT, prior=95.0, rbe_if_armed=99.8, chandelier=97.0) == 95.0
        assert running_max_stop(Direction.SHORT, prior=95.0, rbe_if_armed=99.8, chandelier=92.0) == 92.0


# =========================================================================== #
# 4. Breakeven ratchet: trigger at +1.5R and persistent latching (F-02)
# =========================================================================== #
class TestBreakevenRatchet:
    def test_arms_at_plus_1_5r_and_level_is_fill_times_1_plus_buffer(self):
        fill, R = 100.0, 5.0
        trigger = fill + BREAKEVEN_TRIGGER_R * R   # 107.5
        mgr = V1_2ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        assert mgr.armed is False
        assert break_even_level(fill, Direction.LONG, BREAKEVEN_BUFFER) == pytest.approx(100.2)
        assert break_even_level(fill, Direction.SHORT, BREAKEVEN_BUFFER) == pytest.approx(99.8)
        df_below = pd.DataFrame({"high": [106.0], "low": [100.0],
                                 "close": [trigger - 0.01], "atr_14": [2.5]})
        mgr.scan_bar(df_below, 0)
        assert mgr.armed is False
        df_at = pd.DataFrame({"high": [101.0], "low": [99.0],
                              "close": [trigger + 0.1], "atr_14": [2.5]})
        mgr.scan_bar(df_at, 0)
        assert mgr.armed is True
        assert mgr.effective_stop() >= 100.2 - 1e-9

    def test_latching_is_permanent_after_retrace(self):
        fill, R = 100.0, 5.0
        mgr = V1_2ExitManager(Direction.LONG, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        df = pd.DataFrame({
            "high": [114.0, 108.0, 102.0, 100.5],
            "low": [110.0, 104.0, 100.0, 99.0],
            "close": [114.0, 107.0, 101.5, 100.0],
            "atr_14": [2.0, 2.0, 2.0, 2.0],
        })
        for t in range(len(df)):
            mgr.scan_bar(df, t)
        assert mgr.armed is True
        assert mgr.effective_stop() >= 100.2 - 1e-9   # floored at the ratchet level

    def test_short_ratchet_level_and_latch(self):
        fill, R = 100.0, 5.0
        mgr = V1_2ExitManager(Direction.SHORT, fill, R, entry_bar=0,
                              lookback_trail=20, trail_atr_mult=3.0)
        trigger = fill - BREAKEVEN_TRIGGER_R * R   # 92.5
        df = pd.DataFrame({"high": [96.0], "low": [90.0],
                           "close": [trigger - 0.1], "atr_14": [2.0]})
        mgr.scan_bar(df, 0)
        assert mgr.armed is True
        assert mgr.effective_stop() <= 99.8 + 1e-9   # capped at the (short) ratchet floor


# =========================================================================== #
# 5. Conservative stop-first same-bar collision
# =========================================================================== #
class TestCollisionOrdering:
    def test_stop_wins_over_alt_on_same_bar(self):
        outcome, fill = resolve_collision(Direction.LONG, open_price=101.0,
                                          high=108.0, low=99.0,
                                          stop_level=100.0, alt_level=106.0)
        assert outcome == "stop"
        assert fill == pytest.approx(100.0)

    def test_gap_adverse_fill(self):
        o, f = resolve_collision(Direction.LONG, open_price=99.0,
                                 high=99.5, low=98.0, stop_level=100.0)
        assert o == "stop" and f == pytest.approx(99.0)
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
# 6. Fail-closed warm-up (bar 138 vs 139) + no-look-ahead
# =========================================================================== #
class TestWarmUpAndNoLookAhead:
    def _strategy(self):
        return VolatilityCompressionBreakoutV1_2(params={"exit_config": "X1"})

    def test_no_signal_before_139_any_data(self):
        s = self._strategy()
        # A frame that would fire at i>=139 (all gates + ADX pass) — but must NOT at 138.
        for j in (0, 1, 50, 137, 138):
            assert s.entry_signal(golden_frame(i=140, adx=40.0), j)[0] == Direction.NONE

    def test_signal_allowed_at_warmup_boundary(self):
        s = self._strategy()
        # Bar 139 is legitimate to evaluate; on a passing golden frame it fires LONG.
        assert s.entry_signal(golden_frame(i=139, adx=40.0), 139)[0] == Direction.LONG
        # Bar 138 is below the warm-up bound -> always NONE even though gates look passable.
        assert s.entry_signal(golden_frame(i=139, adx=40.0), 138)[0] == Direction.NONE

    def test_signal_is_immune_to_future_bars(self):
        s = self._strategy()
        df = vcb_long_dataset()
        out = prepared(df, s)
        d, reason = s.entry_signal(out, 259)
        assert d == Direction.LONG
        out2 = out.copy()
        out2.loc[out2.index >= 260, ["high", "low", "close"]] *= 1.5
        d2, r2 = s.entry_signal(out2, 259)
        assert d2 == d and r2 == reason

    def test_fill_uses_open_of_next_bar_with_slip(self):
        s = self._strategy()
        assert s.fill_price(100.0, Direction.LONG) == pytest.approx(100.05)
        assert s.fill_price(100.0, Direction.SHORT) == pytest.approx(99.95)


# =========================================================================== #
# 7. 24-candidate grid instantiation (C00..C23)
# =========================================================================== #
class TestCandidateGrid:
    def test_exactly_24_candidates_contiguous(self):
        grid = candidate_grid()
        assert len(grid) == 24
        assert [c.id for c in grid] == ["C%02d" % i for i in range(24)]

    def test_no_duplicates_full_cross(self):
        from itertools import product
        grid = candidate_grid()
        cells = [(c.breakout_lookback, c.adx_threshold, c.exit_config) for c in grid]
        assert len(set(cells)) == 24
        cross = set(product([20, 40], sorted(ADX_THRESHOLDS), ["X1", "X2", "X3", "X4"]))
        assert set(cells) == cross                     # no gaps, no duplicates

    def test_c00_anchor(self):
        c00 = CANDIDATE_TUPLES[0]
        assert c00.id == "C00"
        assert (c00.breakout_lookback, c00.squeeze_percentile, c00.expansion_multiplier,
                c00.adx_threshold) == (20, 30, 1.1, 25)
        assert (c00.lookback_trail, c00.trail_atr_mult) == EXIT_CONFIGS["X1"]

    def test_fixed_dimensions_across_grid(self):
        for c in CANDIDATE_TUPLES:
            assert c.squeeze_percentile == SQUEEZE_PERCENTILE == 30
            assert c.expansion_multiplier == 1.1
            assert c.atr_multiplier == RISK_ATR_MULTIPLIER == 2.5
            assert (c.lookback_trail, c.trail_atr_mult) == EXIT_CONFIGS[c.exit_config]

    def test_all_candidates_instantiate_and_pass_validation(self):
        for cand in CANDIDATE_TUPLES:
            s = VolatilityCompressionBreakoutV1_2(params=cand.params)
            s.validate_params()
            assert s.breakout_lookback == cand.breakout_lookback
            assert s.adx_threshold == cand.adx_threshold
            assert s.squeeze_percentile == 30
            assert s.atr_multiplier == 2.5
            assert s.lookback_trail == cand.lookback_trail
            assert s.trail_atr_mult == cand.trail_atr_mult

    def test_x4_pair_now_permitted(self):
        # X4=(10, 2.0) was excluded in V1.1 as dominated; V1.2 re-includes it (protocol §5c).
        s = VolatilityCompressionBreakoutV1_2(params={
            "breakout_lookback": 20, "squeeze_percentile": 30,
            "adx_threshold": 25, "lookback_trail": 10, "trail_atr_mult": 2.0,
        })
        s.validate_params()
        assert (s.lookback_trail, s.trail_atr_mult) == (10, 2.0)

    def test_nonfrozen_grid_dimensions_rejected(self):
        with pytest.raises(ValueError):
            VolatilityCompressionBreakoutV1_2(params={"squeeze_percentile": 40})
        with pytest.raises(ValueError):
            VolatilityCompressionBreakoutV1_2(params={"adx_threshold": 35})
        with pytest.raises(ValueError):
            VolatilityCompressionBreakoutV1_2(params={"atr_multiplier": 3.0})


# =========================================================================== #
# 8. Engine wiring: prepare_fill / update_stop carry the running-max state
# =========================================================================== #
class TestEngineWiring:
    def test_prepare_fill_seeds_state_and_no_take_profit(self):
        s = VolatilityCompressionBreakoutV1_2(params={"exit_config": "X1"})
        df = vcb_long_dataset()
        out = prepared(df, s)
        out.loc[259, "atr_14"] = 10.0
        fill = 105.0
        stop, tp, state = s.prepare_fill(out, 259, Direction.LONG, fill)
        assert np.isclose(state["risk_R"], 25.0, atol=1e-9)     # 2.5 * 10
        assert np.isclose(stop, fill - 25.0, atol=1e-9)          # Stop0 = Fill - a*ATR
        assert tp is None                                        # uncapped right tail
        assert state["fill_price"] == 105.0
        assert state["breakeven_armed"] is False
        assert np.isclose(state["prior_stop"], stop, atol=1e-12) # base case Stop_{t0+1}

    def test_compute_take_profit_is_none(self):
        s = VolatilityCompressionBreakoutV1_2(params={"exit_config": "X1"})
        assert s.compute_take_profit(None, 0, Direction.LONG, 100.0) is None

    def test_update_stop_advances_active_stop_monotonically(self):
        s = VolatilityCompressionBreakoutV1_2(params={"exit_config": "X1"})
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
        active = []
        for bar in range(261, 300):
            s.update_stop(out, bar, pos)
            active.append(pos.active_stop)
        assert all(active[t] <= active[t + 1] for t in range(len(active) - 1))  # monotone


# =========================================================================== #
# Extra: Type-7 percentile consistency with numpy (auto-regression guard)
# =========================================================================== #
class TestType7Percentile:
    def test_matches_numpy_linear(self):
        from crypto_quant.strategies.volatility_compression_breakout_v1_2 import type7_percentile
        x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        for q in (0.0, 0.1, 0.3, 0.5, 0.75, 1.0):
            assert type7_percentile(x, q) == pytest.approx(np.quantile(x, q, method="linear"))