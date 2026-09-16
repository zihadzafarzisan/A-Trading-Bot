"""Unit, qualification, causality, and regression tests for Strategy #8 V1.3.

Covers:
- Parameter registration & validation (including body_period and volume_period)
- Body qualification mechanics: normal, threshold failure, equality, zero baseline,
  exact 20-bar window (t-21..t-2), t-1 exclusion, future bar exclusion
- Volume qualification mechanics: normal, threshold failure, equality, zero baseline,
  exact 20-bar window (t-21..t-2), t-1 exclusion, future bar exclusion
- Integration: body/volume failure blocks entry, mitigation does not require qualification,
  materiality of body_mult and volume_mult
- Regression: bullish FVG, bearish FVG, signed ER filter, next-bar execution, slippage,
  initial stop, TP, breakeven trigger, ATR trailing ratchet, position sizing, fees, funding,
  leverage cap, and trade collision safety.
"""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.strategies import (
    AdaptiveDisplacementTrailingStrategy,
    Direction,
    create_strategy,
    get_strategy_class,
)
from crypto_quant.strategies.adaptive_displacement_trailing import C00_PARAMS
from crypto_quant.backtesting.engine import BacktestConfig, BacktestEngine
from crypto_quant.backtesting.execution import ExecutionConfig
from crypto_quant.backtesting.portfolio import Portfolio, Position


# ── Fixtures & Helpers ─────────────────────────────────────────────────────────

def _make_base_df(n_bars: int = 120) -> pd.DataFrame:
    """Create a neutral baseline DataFrame with uniform bars."""
    timestamps = [1609459200000 + i * 3600000 for i in range(n_bars)]
    # Default bar: open=100.0, close=100.5 (body=0.5), high=101.0, low=99.5, volume=1000.0
    opens = [100.0] * n_bars
    closes = [100.5] * n_bars
    highs = [101.0] * n_bars
    lows = [99.5] * n_bars
    volumes = [1000.0] * n_bars
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _create_bullish_fvg_scenario(
    t: int = 65,
    body_mult: float = 1.6,
    volume_mult: float = 1.7,
    body_period: int = 20,
    volume_period: int = 20,
    baseline_body: float = 2.0,
    baseline_volume: float = 1000.0,
    t1_body: float = 4.0,       # 4.0 >= 1.6 * 2.0 = 3.2 (Qualifies)
    t1_volume: float = 2000.0,   # 2000 >= 1.7 * 1000 = 1700 (Qualifies)
) -> tuple[pd.DataFrame, AdaptiveDisplacementTrailingStrategy]:
    """Build a deterministic DataFrame with a bullish FVG at reference bar t.
    
    t = reference bar (e.g. 65, ensuring EMA50 is fully warmed up)
    t-1 = displacement candle (bar 64)
    t-21..t-2 = baseline window (bars 44..63)
    t-3 = prior structure candle (bar 62)
    """
    n_bars = max(120, t + 15)
    df = _make_base_df(n_bars)

    # Establish baseline bars t-21 to t-2 (indices 44 to 63 for t=65)
    for i in range(t - 21, t - 1):
        df.loc[i, "open"] = 100.0
        df.loc[i, "close"] = 100.0 + baseline_body
        df.loc[i, "high"] = 100.0 + baseline_body + 0.5
        df.loc[i, "low"] = 99.5
        df.loc[i, "volume"] = baseline_volume

    # Bar t-3 (prior structure bar):
    df.loc[t - 3, "open"] = 100.0
    df.loc[t - 3, "close"] = 102.0
    df.loc[t - 3, "high"] = 103.0
    df.loc[t - 3, "low"] = 99.0
    df.loc[t - 3, "volume"] = baseline_volume

    # Bar t-1 (displacement candle):
    # Geometric bullish FVG: Close_(t-1) > Open_(t-1), Low_(t-1) > High_(t-3), Close_(t-1) > EMA50_(t-1)
    df.loc[t - 1, "open"] = 104.0
    df.loc[t - 1, "close"] = 104.0 + t1_body
    df.loc[t - 1, "high"] = 104.0 + t1_body + 1.0
    df.loc[t - 1, "low"] = 103.5  # Low_(t-1)=103.5 > High_(t-3)=103.0 (gap of 0.5)
    df.loc[t - 1, "volume"] = t1_volume

    # Bar t (reference bar / first mitigation opportunity):
    df.loc[t, "open"] = 107.0
    df.loc[t, "close"] = 107.5
    df.loc[t, "high"] = 108.0
    df.loc[t, "low"] = 103.2  # mitigates zone [103.0, 103.5]
    df.loc[t, "volume"] = 1000.0

    strat = AdaptiveDisplacementTrailingStrategy({
        "body_mult": body_mult,
        "volume_mult": volume_mult,
        "body_period": body_period,
        "volume_period": volume_period,
        "displacement_lookback": 5,
        "er_threshold": 0.20,
    })
    return df, strat


def _create_bearish_fvg_scenario(
    t: int = 65,
    body_mult: float = 1.6,
    volume_mult: float = 1.7,
    baseline_body: float = 2.0,
    baseline_volume: float = 1000.0,
    t1_body: float = 4.0,
    t1_volume: float = 2000.0,
) -> tuple[pd.DataFrame, AdaptiveDisplacementTrailingStrategy]:
    """Build a deterministic DataFrame with a bearish FVG at reference bar t."""
    n_bars = max(120, t + 15)
    # Start prices high and trend down so Close < EMA50
    timestamps = [1609459200000 + i * 3600000 for i in range(n_bars)]
    opens = [250.0 - i * 0.5 for i in range(n_bars)]
    closes = [opens[i] - baseline_body for i in range(n_bars)]
    highs = [opens[i] + 0.5 for i in range(n_bars)]
    lows = [closes[i] - 0.5 for i in range(n_bars)]
    volumes = [baseline_volume] * n_bars

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    # Bar t-3 (prior structure bar):
    df.loc[t - 3, "open"] = 150.0
    df.loc[t - 3, "close"] = 148.0
    df.loc[t - 3, "high"] = 151.0
    df.loc[t - 3, "low"] = 147.0
    df.loc[t - 3, "volume"] = baseline_volume

    # Bar t-1 (bearish displacement candle):
    # Geometric bearish FVG: Close_(t-1) < Open_(t-1), High_(t-1) < Low_(t-3), Close_(t-1) < EMA50_(t-1)
    df.loc[t - 1, "open"] = 146.0
    df.loc[t - 1, "close"] = 146.0 - t1_body
    df.loc[t - 1, "high"] = 146.5  # High_(t-1)=146.5 < Low_(t-3)=147.0 (gap of 0.5)
    df.loc[t - 1, "low"] = 146.0 - t1_body - 1.0
    df.loc[t - 1, "volume"] = t1_volume

    # Bar t (reference bar):
    df.loc[t, "open"] = 140.0
    df.loc[t, "close"] = 139.5
    df.loc[t, "high"] = 146.8  # mitigates zone [146.5, 147.0]
    df.loc[t, "low"] = 138.0
    df.loc[t, "volume"] = 1000.0

    strat = AdaptiveDisplacementTrailingStrategy({
        "body_mult": body_mult,
        "volume_mult": volume_mult,
        "body_period": 20,
        "volume_period": 20,
        "displacement_lookback": 5,
        "er_threshold": 0.20,
    })
    return df, strat


# ── 1. Registry & Parameter Validation Tests ──────────────────────────────────

def test_registry_integration():
    """Verify Strategy #8 is properly registered and instantiated."""
    cls = get_strategy_class("adaptive_displacement_trailing")
    assert cls is AdaptiveDisplacementTrailingStrategy

    strat = create_strategy("adaptive_displacement_trailing")
    assert isinstance(strat, AdaptiveDisplacementTrailingStrategy)
    assert strat.strategy_type == "adaptive_displacement_trailing"
    assert strat.name == "Adaptive Displacement Trailing"

    # Default parameters must match C00 specification exactly
    for k, v in C00_PARAMS.items():
        if k != "candidate_id":
            assert strat.params[k] == v, f"Default param {k} mismatch: {strat.params[k]} vs {v}"

    # Verify param grid dimensions: 2 * 2 * 3 * 3 = 36
    grid = strat.param_grid()
    assert len(grid["body_mult"]) == 2
    assert len(grid["volume_mult"]) == 2
    assert len(grid["er_threshold"]) == 3
    assert len(grid["atr_multiplier"]) == 3
    combos = len(grid["body_mult"]) * len(grid["volume_mult"]) * len(grid["er_threshold"]) * len(grid["atr_multiplier"])
    assert combos == 36


def test_param_validation():
    """Verify parameter validation bounds per Protocol Section 8."""
    # Valid
    strat = AdaptiveDisplacementTrailingStrategy({"body_mult": 1.4, "volume_mult": 1.5, "body_period": 20})
    strat.validate_params()

    # Invalid body_mult
    with pytest.raises(ValueError, match="body_mult must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"body_mult": 0.0}).validate_params()
    with pytest.raises(ValueError, match="body_mult must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"body_mult": -1.0}).validate_params()

    # Invalid volume_mult
    with pytest.raises(ValueError, match="volume_mult must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"volume_mult": 0.0}).validate_params()

    # Invalid body_period
    with pytest.raises(ValueError, match="body_period must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"body_period": 0}).validate_params()
    with pytest.raises(ValueError, match="body_period must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"body_period": -5}).validate_params()

    # Invalid volume_period
    with pytest.raises(ValueError, match="volume_period must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"volume_period": 0}).validate_params()

    # Invalid er_threshold
    with pytest.raises(ValueError, match="er_threshold out of range"):
        AdaptiveDisplacementTrailingStrategy({"er_threshold": 1.5}).validate_params()
    with pytest.raises(ValueError, match="er_threshold out of range"):
        AdaptiveDisplacementTrailingStrategy({"er_threshold": -0.1}).validate_params()

    # Invalid atr_multiplier
    with pytest.raises(ValueError, match="atr_multiplier must be > 0"):
        AdaptiveDisplacementTrailingStrategy({"atr_multiplier": 0.0}).validate_params()


# ── 2. Body Qualification Mechanics ───────────────────────────────────────────

def test_body_qualification_normal():
    """Verify body qualification passes when Body_(t-1) >= body_mult * AvgBody."""
    # baseline body = 2.0, body_mult = 1.6 -> threshold = 3.2. Body = 4.0 -> PASS
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=2.0, t1_body=4.0)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is not None
    assert zone == (103.0, 103.5)


def test_body_qualification_threshold_failure():
    """Verify body qualification FAILS when Body_(t-1) < body_mult * AvgBody."""
    # baseline body = 2.0, body_mult = 1.6 -> threshold = 3.2. Body = 3.1 -> FAIL
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=2.0, t1_body=3.1)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is None, "Body below threshold should not qualify FVG"


def test_body_qualification_equality():
    """Verify body qualification PASSES when Body_(t-1) == body_mult * AvgBody (equality allowed)."""
    # baseline body = 2.0, body_mult = 1.6 -> threshold = 3.2000000. Body = 3.2000000 -> PASS
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=2.0, t1_body=3.2)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is not None, "Exact equality on body qualification must pass per Protocol §5.1"


def test_body_qualification_zero_baseline():
    """Verify zero body baseline FAILS qualification per Protocol Section 5.1."""
    # baseline body = 0.0 -> AvgBody = 0.0 -> FAIL
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=0.0, t1_body=4.0)
    # Ensure bar t-3 (index 62) also has body = 0.0 so the 20-bar baseline is strictly 0.0
    df.loc[62, "open"] = 100.0
    df.loc[62, "close"] = 100.0
    df.loc[62, "high"] = 100.5
    df.loc[62, "low"] = 99.5
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is None, "Zero body baseline must fail qualification per Protocol §5.1"


def test_body_exact_20_bar_baseline_window():
    """Prove that body baseline window is strictly t-21 ... t-2 (20 bars).
    
    Bar t-22 must NOT affect baseline.
    Bars t-21 through t-2 must affect baseline.
    Bar t-1 must NOT affect baseline.
    Bar t must NOT affect baseline.
    Future bars must NOT affect baseline.
    """
    t = 65
    df, strat = _create_bullish_fvg_scenario(t=t, body_mult=1.6, baseline_body=2.0, t1_body=4.0)
    prep = strat.setup(df)

    # Expected baseline for t=65 ends at t-2=63, using 20 bars: 44..63 (t-21..t-2)
    avg_body_expected = float((prep["close"].iloc[44:64] - prep["open"].iloc[44:64]).abs().mean())
    avg_body_actual = float(prep["avg_body_20"].iloc[t - 2])
    assert np.isclose(avg_body_actual, avg_body_expected, atol=1e-12)
    assert np.isclose(avg_body_actual, 2.0, atol=1e-12)

    # 1. Bar t-22 (index 43): Mutate bar 43 to body=100.0 -> must NOT change avg_body at t-2
    df_perturbed_past = df.copy()
    df_perturbed_past.loc[t - 22, "close"] = df_perturbed_past.loc[t - 22, "open"] + 100.0
    prep_past = strat.setup(df_perturbed_past)
    assert np.isclose(prep_past["avg_body_20"].iloc[t - 2], 2.0, atol=1e-12), "Bar t-22 must not enter baseline"

    # 2. Boundary bars t-21 (index 44) and t-2 (index 63): Mutating them MUST change avg_body
    df_perturbed_start = df.copy()
    df_perturbed_start.loc[t - 21, "close"] = df_perturbed_start.loc[t - 21, "open"] + 4.0
    prep_start = strat.setup(df_perturbed_start)
    assert prep_start["avg_body_20"].iloc[t - 2] != 2.0, "Bar t-21 must enter baseline"

    df_perturbed_end = df.copy()
    df_perturbed_end.loc[t - 2, "close"] = df_perturbed_end.loc[t - 2, "open"] + 4.0
    prep_end = strat.setup(df_perturbed_end)
    assert prep_end["avg_body_20"].iloc[t - 2] != 2.0, "Bar t-2 must enter baseline"

    # 3. Bar t-1 (displacement candle, index 64): Mutating t-1 body must NOT change baseline
    df_perturbed_t1 = df.copy()
    df_perturbed_t1.loc[t - 1, "close"] = df_perturbed_t1.loc[t - 1, "open"] + 50.0
    prep_t1 = strat.setup(df_perturbed_t1)
    assert np.isclose(prep_t1["avg_body_20"].iloc[t - 2], 2.0, atol=1e-12), "Displacement candle t-1 must be excluded from baseline"

    # 4. Bar t (reference bar, index 65) & future: Mutating them must NOT change baseline
    df_perturbed_future = df.copy()
    df_perturbed_future.loc[t:, "close"] = df_perturbed_future.loc[t:, "open"] + 99.0
    prep_future = strat.setup(df_perturbed_future)
    assert np.isclose(prep_future["avg_body_20"].iloc[t - 2], 2.0, atol=1e-12), "Future bars must not enter baseline"


# ── 3. Volume Qualification Mechanics ─────────────────────────────────────────

def test_volume_qualification_normal():
    """Verify volume qualification passes when Volume_(t-1) >= volume_mult * AvgVolume."""
    # baseline volume = 1000.0, volume_mult = 1.7 -> threshold = 1700. Volume = 2000.0 -> PASS
    df, strat = _create_bullish_fvg_scenario(t=65, volume_mult=1.7, baseline_volume=1000.0, t1_volume=2000.0)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is not None


def test_volume_qualification_threshold_failure():
    """Verify volume qualification FAILS when Volume_(t-1) < volume_mult * AvgVolume."""
    # baseline volume = 1000.0, volume_mult = 1.7 -> threshold = 1700. Volume = 1650.0 -> FAIL
    df, strat = _create_bullish_fvg_scenario(t=65, volume_mult=1.7, baseline_volume=1000.0, t1_volume=1650.0)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is None, "Volume below threshold should not qualify FVG"


def test_volume_qualification_equality():
    """Verify volume qualification PASSES when Volume_(t-1) == volume_mult * AvgVolume (equality allowed)."""
    # baseline volume = 1000.0, volume_mult = 1.7 -> threshold = 1700. Volume = 1700.0 -> PASS
    df, strat = _create_bullish_fvg_scenario(t=65, volume_mult=1.7, baseline_volume=1000.0, t1_volume=1700.0)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is not None, "Exact equality on volume qualification must pass per Protocol §5.2"


def test_volume_qualification_zero_baseline():
    """Verify zero volume baseline FAILS qualification per Protocol Section 5.2."""
    # baseline volume = 0.0 -> AvgVolume = 0.0 -> FAIL
    df, strat = _create_bullish_fvg_scenario(t=65, volume_mult=1.7, baseline_volume=0.0, t1_volume=2000.0)
    prep = strat.setup(df)
    zone = strat._fvg_bull_zone(prep, 65)
    assert zone is None, "Zero volume baseline must fail qualification per Protocol §5.2"


def test_volume_exact_20_bar_baseline_window():
    """Prove that volume baseline window is strictly t-21 ... t-2 (20 bars)."""
    t = 65
    df, strat = _create_bullish_fvg_scenario(t=t, volume_mult=1.7, baseline_volume=1000.0, t1_volume=2000.0)
    prep = strat.setup(df)

    # Expected baseline for t=65 ends at t-2=63, using 20 bars: 44..63 (t-21..t-2)
    avg_vol_expected = float(prep["volume"].iloc[44:64].mean())
    avg_vol_actual = float(prep["avg_vol_20"].iloc[t - 2])
    assert np.isclose(avg_vol_actual, avg_vol_expected, atol=1e-12)
    assert np.isclose(avg_vol_actual, 1000.0, atol=1e-12)

    # 1. Bar t-22 (index 43): Mutate bar 43 volume to 500,000 -> must NOT change avg_vol at t-2
    df_perturbed_past = df.copy()
    df_perturbed_past.loc[t - 22, "volume"] = 500_000.0
    prep_past = strat.setup(df_perturbed_past)
    assert np.isclose(prep_past["avg_vol_20"].iloc[t - 2], 1000.0, atol=1e-12), "Bar t-22 must not enter volume baseline"

    # 2. Boundary bars t-21 (index 44) and t-2 (index 63): Mutating them MUST change avg_vol
    df_perturbed_start = df.copy()
    df_perturbed_start.loc[t - 21, "volume"] = 3000.0
    prep_start = strat.setup(df_perturbed_start)
    assert prep_start["avg_vol_20"].iloc[t - 2] != 1000.0, "Bar t-21 must enter volume baseline"

    df_perturbed_end = df.copy()
    df_perturbed_end.loc[t - 2, "volume"] = 3000.0
    prep_end = strat.setup(df_perturbed_end)
    assert prep_end["avg_vol_20"].iloc[t - 2] != 1000.0, "Bar t-2 must enter volume baseline"

    # 3. Bar t-1 (displacement candle, index 64): Mutating t-1 volume must NOT change baseline
    df_perturbed_t1 = df.copy()
    df_perturbed_t1.loc[t - 1, "volume"] = 999_999.0
    prep_t1 = strat.setup(df_perturbed_t1)
    assert np.isclose(prep_t1["avg_vol_20"].iloc[t - 2], 1000.0, atol=1e-12), "Displacement candle t-1 must be excluded from volume baseline"

    # 4. Future bars: Mutating them must NOT change baseline
    df_perturbed_future = df.copy()
    df_perturbed_future.loc[t:, "volume"] = 888_888.0
    prep_future = strat.setup(df_perturbed_future)
    assert np.isclose(prep_future["avg_vol_20"].iloc[t - 2], 1000.0, atol=1e-12), "Future bars must not enter volume baseline"


# ── 4. Bearish FVG Qualification ──────────────────────────────────────────────

def test_bearish_fvg_body_and_volume_qualification():
    """Verify body and volume qualification on bearish displacement candle."""
    # Both qualify: baseline body=2, vol=1000. t1_body=4.0 (>= 1.6*2=3.2), t1_vol=2000 (>= 1.7*1000=1700)
    df, strat = _create_bearish_fvg_scenario(t=65, t1_body=4.0, t1_volume=2000.0)
    prep = strat.setup(df)
    zone = strat._fvg_bear_zone(prep, 65)
    assert zone is not None
    assert zone == (146.5, 147.0)

    # Body failure blocks bearish FVG
    df_body_fail, _ = _create_bearish_fvg_scenario(t=65, t1_body=3.0, t1_volume=2000.0)
    prep_body_fail = strat.setup(df_body_fail)
    assert strat._fvg_bear_zone(prep_body_fail, 65) is None

    # Volume failure blocks bearish FVG
    df_vol_fail, _ = _create_bearish_fvg_scenario(t=65, t1_body=4.0, t1_volume=1600.0)
    prep_vol_fail = strat.setup(df_vol_fail)
    assert strat._fvg_bear_zone(prep_vol_fail, 65) is None

    # Zero body baseline fails bearish FVG
    df_zero_body, _ = _create_bearish_fvg_scenario(t=65, baseline_body=0.0, t1_body=4.0)
    df_zero_body.loc[62, "open"] = 150.0
    df_zero_body.loc[62, "close"] = 150.0
    df_zero_body.loc[62, "high"] = 150.5
    df_zero_body.loc[62, "low"] = 149.5
    prep_zero_body = strat.setup(df_zero_body)
    assert strat._fvg_bear_zone(prep_zero_body, 65) is None


# ── 5. Integration Tests: Qualification & Mitigation ──────────────────────────

def test_body_failure_blocks_entry_signal():
    """Prove that displacement body failure blocks subsequent entry signal at mitigation candle."""
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=2.0, t1_body=3.0)  # fails: 3.0 < 3.2
    prep = strat.setup(df)
    direction, reason = strat.entry_signal(prep, 65)
    assert direction == Direction.NONE
    assert reason == ""


def test_volume_failure_blocks_entry_signal():
    """Prove that displacement volume failure blocks subsequent entry signal at mitigation candle."""
    df, strat = _create_bullish_fvg_scenario(t=65, volume_mult=1.7, baseline_volume=1000.0, t1_volume=1500.0)  # fails: 1500 < 1700
    prep = strat.setup(df)
    direction, reason = strat.entry_signal(prep, 65)
    assert direction == Direction.NONE
    assert reason == ""


def test_mitigation_candle_does_not_require_body_or_volume_qualification():
    """Prove that mitigation candle k does NOT have body or volume requirements.
    
    Only the displacement candle t-1 must qualify.
    """
    df, strat = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)
    # Set mitigation bar t (bar 65) to tiny body (0.01) and low volume (1.0)
    df.loc[65, "open"] = 107.0
    df.loc[65, "close"] = 107.01  # body = 0.01 (tiny)
    df.loc[65, "volume"] = 1.0    # volume = 1.0 (negligible)
    df.loc[65, "low"] = 103.2     # mitigates zone [103.0, 103.5]
    df.loc[65, "high"] = 108.0

    prep = strat.setup(df)
    direction, reason = strat.entry_signal(prep, 65)
    assert direction == Direction.LONG
    assert reason == "long_fvg_mitigation"


def test_body_mult_materiality():
    """Prove that body_mult materially changes FVG qualification.
    
    Under body_mult=1.4, baseline=2.0 -> threshold=2.8 -> body=3.0 QUALIFIES.
    Under body_mult=1.7, baseline=2.0 -> threshold=3.4 -> body=3.0 FAILS.
    """
    df, _ = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, t1_body=3.0, t1_volume=2500.0)

    strat_loose = AdaptiveDisplacementTrailingStrategy({"body_mult": 1.4, "volume_mult": 1.5})
    prep_loose = strat_loose.setup(df)
    assert strat_loose._fvg_bull_zone(prep_loose, 65) is not None

    strat_strict = AdaptiveDisplacementTrailingStrategy({"body_mult": 1.7, "volume_mult": 1.5})
    prep_strict = strat_strict.setup(df)
    assert strat_strict._fvg_bull_zone(prep_strict, 65) is None


def test_volume_mult_materiality():
    """Prove that volume_mult materially changes FVG qualification.
    
    Under volume_mult=1.5, baseline=1000 -> threshold=1500 -> volume=1600 QUALIFIES.
    Under volume_mult=1.8, baseline=1000 -> threshold=1800 -> volume=1600 FAILS.
    """
    df, _ = _create_bullish_fvg_scenario(t=65, baseline_volume=1000.0, t1_body=4.0, t1_volume=1600.0)

    strat_loose = AdaptiveDisplacementTrailingStrategy({"body_mult": 1.4, "volume_mult": 1.5})
    prep_loose = strat_loose.setup(df)
    assert strat_loose._fvg_bull_zone(prep_loose, 65) is not None

    strat_strict = AdaptiveDisplacementTrailingStrategy({"body_mult": 1.4, "volume_mult": 1.8})
    prep_strict = strat_strict.setup(df)
    assert strat_strict._fvg_bull_zone(prep_strict, 65) is None


def test_combined_qualification_truth_table():
    """Explicitly verify combined qualification truth table (C1, C2, C3, C4).
    
    C1: Body PASS (4.0 >= 3.2), Vol PASS (2000 >= 1700) => Eligible (zone formed)
    C2: Body FAIL (3.0 < 3.2),  Vol PASS (2000 >= 1700) => Blocked (zone is None)
    C3: Body PASS (4.0 >= 3.2), Vol FAIL (1500 < 1700) => Blocked (zone is None)
    C4: Body FAIL (3.0 < 3.2),  Vol FAIL (1500 < 1700) => Blocked (zone is None)
    """
    # C1: Both pass
    df_c1, s_c1 = _create_bullish_fvg_scenario(t=65, t1_body=4.0, t1_volume=2000.0)
    p_c1 = s_c1.setup(df_c1)
    assert s_c1._fvg_bull_zone(p_c1, 65) is not None, "C1: body pass + vol pass must qualify"
    dir_c1, _ = s_c1.entry_signal(p_c1, 65)
    assert dir_c1 == Direction.LONG, "C1: eligible for long entry"

    # C2: Body fail, volume pass
    df_c2, s_c2 = _create_bullish_fvg_scenario(t=65, t1_body=3.0, t1_volume=2000.0)
    p_c2 = s_c2.setup(df_c2)
    assert s_c2._fvg_bull_zone(p_c2, 65) is None, "C2: body fail must block zone"
    dir_c2, _ = s_c2.entry_signal(p_c2, 65)
    assert dir_c2 == Direction.NONE, "C2: blocked from entry"

    # C3: Body pass, volume fail
    df_c3, s_c3 = _create_bullish_fvg_scenario(t=65, t1_body=4.0, t1_volume=1500.0)
    p_c3 = s_c3.setup(df_c3)
    assert s_c3._fvg_bull_zone(p_c3, 65) is None, "C3: volume fail must block zone"
    dir_c3, _ = s_c3.entry_signal(p_c3, 65)
    assert dir_c3 == Direction.NONE, "C3: blocked from entry"

    # C4: Both fail
    df_c4, s_c4 = _create_bullish_fvg_scenario(t=65, t1_body=3.0, t1_volume=1500.0)
    p_c4 = s_c4.setup(df_c4)
    assert s_c4._fvg_bull_zone(p_c4, 65) is None, "C4: both fail must block zone"
    dir_c4, _ = s_c4.entry_signal(p_c4, 65)
    assert dir_c4 == Direction.NONE, "C4: blocked from entry"


# ── 6. Causality & No-Lookahead Tests ─────────────────────────────────────────

def test_causality_and_no_lookahead():
    """Verify that signal and indicator state at target bar depends strictly on bars <= target_bar.
    
    Perturbing bars target_bar+1..end must NOT change setup features or signals at target_bar.
    """
    df, strat = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)
    prep_orig = strat.setup(df)
    target_bar = 65
    sig_orig, reason_orig = strat.entry_signal(prep_orig, target_bar)

    # Mutate all future bars after target_bar
    df_perturbed = df.copy()
    df_perturbed.loc[target_bar + 1 :, ["open", "high", "low", "close", "volume"]] *= 5.0
    prep_perturbed = strat.setup(df_perturbed)
    sig_perturbed, reason_perturbed = strat.entry_signal(prep_perturbed, target_bar)

    assert sig_orig == sig_perturbed
    assert reason_orig == reason_perturbed
    for col in ["ema_50", "atr_14", "vol_sma_20", "avg_body_20", "avg_vol_20", "er"]:
        val_orig = prep_orig.loc[target_bar, col]
        val_pert = prep_perturbed.loc[target_bar, col]
        assert np.isclose(val_orig, val_pert, atol=1e-12, equal_nan=True), f"Lookahead detected in column {col}"


# ── 7. Regression Tests: Execution, Risk & Portfolio ──────────────────────────

def test_regression_signed_er_filter():
    """Verify signed Kaufman Efficiency Ratio regime filter (Section 7)."""
    # Upward trend produces positive ER
    strat = AdaptiveDisplacementTrailingStrategy({"er_threshold": 0.30})
    closes_up = pd.Series([100.0 + i * 2.0 for i in range(20)])
    er_up = strat._compute_er(closes_up, 10)
    assert er_up.iloc[-1] == 1.0  # monotonically increasing -> ER = 1.0

    # Downward trend produces negative ER
    closes_down = pd.Series([200.0 - i * 2.0 for i in range(20)])
    er_down = strat._compute_er(closes_down, 10)
    assert er_down.iloc[-1] == -1.0  # monotonically decreasing -> ER = -1.0

    # Zero total volatility produces 0.0 without error
    closes_flat = pd.Series([100.0] * 20)
    er_flat = strat._compute_er(closes_flat, 10)
    assert er_flat.iloc[-1] == 0.0


def test_regression_fvg_validity_window():
    """Verify FVG validity window k in [t, t + lookback - 1] per Section 6.3."""
    t = 65
    lookback = 5
    df, strat = _create_bullish_fvg_scenario(t=t, body_mult=1.6, baseline_body=2.0, t1_body=4.0)
    prep = strat.setup(df)

    # Valid within window: k = t (65), t+1 (66), t+2 (67), t+3 (68), t+4 (69)
    for k in range(t, t + lookback):
        zone = strat._active_fvg_zone(prep, k, Direction.LONG)
        assert zone is not None, f"FVG should be active at bar {k}"

    # Expired beyond window: k = t + 5 (70)
    zone_expired = strat._active_fvg_zone(prep, t + lookback, Direction.LONG)
    assert zone_expired is None, f"FVG should expire at bar {t + lookback}"


def test_regression_most_recent_active_fvg_resolution():
    """Verify that when multiple active FVGs exist at bar k, the MOST RECENT is selected (§6.3)."""
    # Create scenario with two active FVGs: FVG1 formed at t=65, FVG2 formed at t=68
    df, strat = _create_bullish_fvg_scenario(t=65, body_mult=1.6, baseline_body=2.0, t1_body=4.0)

    # Form second FVG at t=68:
    # Displacement candle at t1=67, prior structure candle at t3=65 (high=108.0)
    df.loc[67, "open"] = 110.0
    df.loc[67, "close"] = 116.0  # body = 6.0 >= 1.6 * avg_body
    df.loc[67, "high"] = 117.0
    df.loc[67, "low"] = 109.0    # Low_(67)=109.0 > High_(65)=108.0 -> Bullish FVG2 zone: [108.0, 109.0]
    df.loc[67, "volume"] = 2500.0

    prep = strat.setup(df)

    # Both FVG1 (formed t=65) and FVG2 (formed t=68) are valid at bar k=69 (within lookback=5)
    zone_65 = strat._fvg_bull_zone(prep, 65)
    zone_68 = strat._fvg_bull_zone(prep, 68)
    assert zone_65 is not None
    assert zone_68 is not None
    assert zone_65 != zone_68

    # Active zone at bar 69 must resolve to the MOST RECENT (zone_68)
    active_zone = strat._active_fvg_zone(prep, 69, Direction.LONG)
    assert active_zone == zone_68, f"Active zone must be most recent FVG {zone_68}, got {active_zone}"


def test_regression_prepare_fill_and_risk_parameters():
    """Verify fill-price-anchored initial risk, stop, and take profit (Section 2)."""
    strat = AdaptiveDisplacementTrailingStrategy({
        "atr_multiplier": 2.2,
        "risk_reward_ratio": 2.0,
    })
    df = _make_base_df(70)
    prep = strat.setup(df)
    prep.loc[65, "atr_14"] = 10.0

    fill_price = 105.0
    stop, take, state = strat.prepare_fill(prep, signal_bar_index=65, direction=Direction.LONG, actual_fill_price=fill_price)

    # R = 2.2 * 10.0 = 22.0
    assert state["risk_R"] == 22.0
    # Stop = 105.0 - 22.0 = 83.0
    assert stop == 83.0
    # TP = 105.0 + 2.0 * 22.0 = 149.0
    assert take == 149.0
    assert state["breakeven_triggered"] is False
    assert state["trailing_active"] is False


def test_atr_multiplier_does_not_alter_signal_existence():
    """Prove that atr_multiplier does NOT alter signal existence and only affects risk quantities."""
    df, _ = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)

    strat_18 = AdaptiveDisplacementTrailingStrategy({"atr_multiplier": 1.8})
    strat_22 = AdaptiveDisplacementTrailingStrategy({"atr_multiplier": 2.2})
    strat_25 = AdaptiveDisplacementTrailingStrategy({"atr_multiplier": 2.5})

    p_18 = strat_18.setup(df)
    p_22 = strat_22.setup(df)
    p_25 = strat_25.setup(df)

    # Verify signals are identical across all bars
    for bar_idx in range(len(df)):
        sig_18, r_18 = strat_18.entry_signal(p_18, bar_idx)
        sig_22, r_22 = strat_22.entry_signal(p_22, bar_idx)
        sig_25, r_25 = strat_25.entry_signal(p_25, bar_idx)
        assert sig_18 == sig_22 == sig_25, f"Signal mismatch at bar {bar_idx} across atr_multipliers"
        assert r_18 == r_22 == r_25, f"Reason mismatch at bar {bar_idx} across atr_multipliers"

    # Verify prepare_fill produces different R values proportional to atr_multiplier
    fill_price = 100.0
    p_22.loc[65, "atr_14"] = 5.0
    _, _, state_18 = strat_18.prepare_fill(p_22, 65, Direction.LONG, fill_price)
    _, _, state_22 = strat_22.prepare_fill(p_22, 65, Direction.LONG, fill_price)
    _, _, state_25 = strat_25.prepare_fill(p_22, 65, Direction.LONG, fill_price)

    assert state_18["risk_R"] == 1.8 * 5.0  # 9.0
    assert state_22["risk_R"] == 2.2 * 5.0  # 11.0
    assert state_25["risk_R"] == 2.5 * 5.0  # 12.5



def test_regression_breakeven_and_trailing_ratchet():
    """Verify breakeven stop move at +1.0R and subsequent ATR trailing ratchet (Sections 3 & 4)."""
    strat = AdaptiveDisplacementTrailingStrategy({
        "atr_multiplier": 2.2,
        "risk_reward_ratio": 2.0,
        "breakeven_trigger_r": 1.0,
    })
    df = _make_base_df(75)
    prep = strat.setup(df)
    prep["atr_14"] = 10.0

    pos = Position(
        symbol="TESTUSDT",
        direction="long",
        market_type="futures",
        quantity=1.0,
        entry_price=100.0,
        entry_time=1609459200000,
        entry_bar=60,
        stop_loss=78.0,
        take_profit=144.0,
        initial_stop=78.0,
        active_stop=78.0,
        extra_state={
            "fill_price": 100.0,
            "risk_R": 22.0,
            "breakeven_triggered": False,
            "trailing_active": False,
            "stop_direction": "long",
        },
    )

    # Bar 66: close = 120.0 (< fill + 1.0*R = 122.0) -> Breakeven NOT triggered
    prep.loc[65, "close"] = 120.0
    strat.update_stop(prep, bar_index=66, position=pos)
    assert pos.active_stop == 78.0
    assert pos.extra_state["breakeven_triggered"] is False

    # Bar 67: close = 123.0 (>= fill + 1.0*R = 122.0) -> Breakeven TRIGGERED, stop moves to 100.0
    prep.loc[66, "close"] = 123.0
    strat.update_stop(prep, bar_index=67, position=pos)
    assert pos.active_stop == 100.0
    assert pos.extra_state["breakeven_triggered"] is True
    assert pos.extra_state["trailing_active"] is True

    # Bar 68: ATR trailing candidate = close_j - atr_mult * atr_j = 140.0 - 2.2 * 10.0 = 118.0
    # Stop moves from 100.0 to 118.0
    prep.loc[67, "close"] = 140.0
    strat.update_stop(prep, bar_index=68, position=pos)
    assert pos.active_stop == 118.0

    # Bar 69: price pulls back: close_j = 130.0 -> candidate = 130.0 - 22.0 = 108.0
    # Ratchet must NOT loosen stop (stop remains 118.0)
    prep.loc[68, "close"] = 130.0
    strat.update_stop(prep, bar_index=69, position=pos)
    assert pos.active_stop == 118.0, "ATR trailing ratchet must never loosen active stop"


def test_regression_backtest_execution_and_fees():
    """Verify next-bar execution, taker fee, slippage, and position sizing via BacktestEngine."""
    df, strat = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)

    # Ensure price reaches TP on subsequent bar
    df.loc[66, "open"] = 108.0
    df.loc[66, "high"] = 160.0
    df.loc[66, "close"] = 155.0
    df.loc[66, "low"] = 107.0

    exec_cfg = ExecutionConfig(
        market_type="futures",
        slippage=0.0005,
        taker_fee=0.0004,
        use_taker=True,
        funding_rate=0.0001,
        execution_delay_bars=1,
        initial_capital=1000.0,
        max_leverage=5.0,
    )
    backtest_cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_position_pct=0.50,
        max_leverage=5.0,
        market_type="futures",
        timeframe="1h",
        execution=exec_cfg,
        funding_bars=8,
        honor_take_profit=True,
    )

    engine = BacktestEngine(config=backtest_cfg)
    result = engine.run(strat, df, symbol="BTCUSDT")

    # Trade should have executed and filled at bar 66 open + slippage
    assert len(result.trades) >= 1
    trade = result.trades[0]
    # Signal was generated at bar 65, execution at bar 66 open
    assert trade["direction"] == "long"
    expected_fill = 108.0 * (1.0 + 0.0005)
    assert np.isclose(float(trade["entry_price"]), expected_fill, atol=1e-4)
    # Fee paid on entry
    assert float(trade["fees"]) > 0
    # Position sizing respected risk
    notional = float(trade["quantity"]) * float(trade["entry_price"])
    assert notional <= 1000.0 * 5.0  # leverage cap 5x


# ── 8. Gap Repair Evidence Tests (Gaps 1–5) ───────────────────────────────────

def test_regression_stop_first_collision():
    """Gap 1: Verify conservative stop-first exit on same-bar SL and TP collision."""
    df, strat = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)

    # Bar 66: fill bar at open=108.0
    df.loc[66, "open"] = 108.0
    df.loc[66, "high"] = 109.0
    df.loc[66, "low"] = 107.0
    df.loc[66, "close"] = 108.0

    # Bar 67: BOTH stop-loss and take-profit are breached on the same bar.
    # Initial fill ~ 108.054, stop ~ 101.11, take_profit ~ 121.95, liquidation ~ 86.4
    # Setting low=98.0 breaches stop (98.0 <= 101.11) without liquidating (98.0 > 86.4)
    # Setting high=125.0 breaches take profit (125.0 >= 121.95)
    df.loc[67, "open"] = 108.0
    df.loc[67, "high"] = 125.0  # hit_tp = True
    df.loc[67, "low"] = 98.0    # hit_sl = True
    df.loc[67, "close"] = 108.0

    exec_cfg = ExecutionConfig(
        market_type="futures",
        slippage=0.0005,
        taker_fee=0.0004,
        use_taker=True,
        funding_rate=0.0001,
        execution_delay_bars=1,
        initial_capital=1000.0,
        max_leverage=5.0,
    )
    backtest_cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_position_pct=0.50,
        max_leverage=5.0,
        market_type="futures",
        timeframe="1h",
        execution=exec_cfg,
        funding_bars=8,
        honor_take_profit=True,
    )

    engine = BacktestEngine(config=backtest_cfg)
    result = engine.run(strat, df, symbol="BTCUSDT")

    assert len(result.trades) == 1
    trade = result.trades[0]
    # In collision, engine MUST exit via stop loss ("sl"), NOT take profit ("tp")
    assert trade["exit_reason"] == "sl", f"Expected exit_reason 'sl', got {trade['exit_reason']}"
    # Exit price must equal the stop price, NOT the take profit price
    assert np.isclose(float(trade["exit_price"]), float(trade["stop_loss"]), atol=1e-4)
    assert not np.isclose(float(trade["exit_price"]), float(trade["take_profit"]), atol=1e-4)
    # Trade realized a loss, not a profit
    assert float(trade["gross_pnl"]) < 0


def test_regression_short_entry_slippage():
    """Gap 2: Verify short-side adverse slippage: short_fill = next_bar_open * (1 - 0.0005)."""
    df, strat = _create_bearish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)

    # Bar 66: execution bar following signal at bar 65
    raw_next_bar_open = 135.0
    df.loc[66, "open"] = raw_next_bar_open
    df.loc[66, "high"] = 136.0
    df.loc[66, "low"] = 134.0
    df.loc[66, "close"] = 135.0

    # Bar 67: price moves down to TP
    df.loc[67, "open"] = 135.0
    df.loc[67, "high"] = 135.5
    df.loc[67, "low"] = 100.0
    df.loc[67, "close"] = 105.0

    exec_cfg = ExecutionConfig(
        market_type="futures",
        slippage=0.0005,
        taker_fee=0.0004,
        use_taker=True,
        funding_rate=0.0001,
        execution_delay_bars=1,
        initial_capital=1000.0,
        max_leverage=5.0,
    )
    backtest_cfg = BacktestConfig(
        initial_capital=1000.0,
        risk_per_trade=0.01,
        max_open_positions=3,
        max_position_pct=0.50,
        max_leverage=5.0,
        market_type="futures",
        timeframe="1h",
        execution=exec_cfg,
        funding_bars=8,
        honor_take_profit=True,
    )

    engine = BacktestEngine(config=backtest_cfg)
    result = engine.run(strat, df, symbol="BTCUSDT")

    assert len(result.trades) >= 1
    trade = result.trades[0]
    assert trade["direction"] == "short"

    # Exact short fill formula: next_bar_open * (1 - 0.0005)
    expected_short_fill = raw_next_bar_open * (1.0 - 0.0005)
    assert np.isclose(float(trade["entry_price"]), expected_short_fill, atol=1e-5)
    # Entry price must be strictly lower than raw open (adverse for short)
    assert float(trade["entry_price"]) < raw_next_bar_open


def test_body_period_parameter_is_live():
    """Gap 3a: Verify body_period dynamically configures the rolling baseline window."""
    t = 65
    df, _ = _create_bullish_fvg_scenario(t=t, baseline_body=2.0, t1_body=4.0)

    # Mutate bar t-22 (index 43) body to 100.0
    # Under body_period=20, baseline is t-21..t-2 (indices 44..63); index 43 is excluded.
    # Under body_period=21, baseline is t-22..t-2 (indices 43..63); index 43 is included.
    df_perturbed = df.copy()
    df_perturbed.loc[t - 22, "close"] = df_perturbed.loc[t - 22, "open"] + 100.0

    strat_20 = AdaptiveDisplacementTrailingStrategy({"body_period": 20})
    strat_21 = AdaptiveDisplacementTrailingStrategy({"body_period": 21})

    prep_20 = strat_20.setup(df_perturbed)
    prep_21 = strat_21.setup(df_perturbed)

    # 20-period baseline excludes bar 43 -> unaffected (still 2.0)
    assert np.isclose(prep_20["avg_body_20"].iloc[t - 2], 2.0, atol=1e-12)
    # 21-period baseline includes bar 43 -> affected (significantly > 2.0)
    assert prep_21["avg_body_20"].iloc[t - 2] > 6.0

    # Explicit governance assertions: body_period is NOT in research grid, grid size remains 36
    grid = strat_20.param_grid()
    assert "body_period" not in grid
    combos = len(grid["body_mult"]) * len(grid["volume_mult"]) * len(grid["er_threshold"]) * len(grid["atr_multiplier"])
    assert combos == 36


def test_volume_period_parameter_is_live():
    """Gap 3b: Verify volume_period dynamically configures the rolling baseline window."""
    t = 65
    df, _ = _create_bullish_fvg_scenario(t=t, baseline_volume=1000.0, t1_volume=2000.0)

    # Mutate bar t-22 (index 43) volume to 500,000.0
    # Under volume_period=20, baseline is t-21..t-2 (indices 44..63); index 43 is excluded.
    # Under volume_period=21, baseline is t-22..t-2 (indices 43..63); index 43 is included.
    df_perturbed = df.copy()
    df_perturbed.loc[t - 22, "volume"] = 500_000.0

    strat_20 = AdaptiveDisplacementTrailingStrategy({"volume_period": 20})
    strat_21 = AdaptiveDisplacementTrailingStrategy({"volume_period": 21})

    prep_20 = strat_20.setup(df_perturbed)
    prep_21 = strat_21.setup(df_perturbed)

    # 20-period baseline excludes bar 43 -> unaffected (still 1000.0)
    assert np.isclose(prep_20["avg_vol_20"].iloc[t - 2], 1000.0, atol=1e-12)
    # 21-period baseline includes bar 43 -> affected (substantially > 1000.0)
    assert prep_21["avg_vol_20"].iloc[t - 2] > 20_000.0

    # Explicit governance assertions: volume_period is NOT in research grid, grid size remains 36
    grid = strat_20.param_grid()
    assert "volume_period" not in grid
    combos = len(grid["body_mult"]) * len(grid["volume_mult"]) * len(grid["er_threshold"]) * len(grid["atr_multiplier"])
    assert combos == 36


def test_er_threshold_material_signal_flip():
    """Gap 4: Verify er_threshold causes a deterministic signal flip on identical market data."""
    df, _ = _create_bullish_fvg_scenario(t=65, baseline_body=2.0, baseline_volume=1000.0, t1_body=4.0, t1_volume=2000.0)

    # Construct closes across bars 55..65 so that directional change = 3.0 and total vol = 10.0:
    # ER = 3.0 / 10.0 = 0.30 exactly.
    df.loc[55, "close"] = 104.5
    df.loc[56, "close"] = 103.5  # diff = -1.0
    df.loc[57, "close"] = 104.5  # diff = +1.0
    df.loc[58, "close"] = 103.5  # diff = -1.0
    df.loc[59, "close"] = 102.5  # diff = -1.0
    df.loc[60, "close"] = 102.5  # diff = 0.0
    df.loc[61, "close"] = 102.5  # diff = 0.0
    df.loc[62, "close"] = 102.5  # diff = 0.0
    df.loc[62, "high"] = 103.0   # prior structure high
    df.loc[62, "low"] = 102.0
    df.loc[63, "close"] = 102.5
    df.loc[63, "open"] = 102.5
    df.loc[64, "open"] = 104.0
    df.loc[64, "close"] = 108.0  # diff = +5.5 (body = 4.0)
    df.loc[64, "low"] = 103.5
    df.loc[64, "high"] = 109.0
    df.loc[65, "close"] = 107.5  # diff = -0.5

    strat_loose = AdaptiveDisplacementTrailingStrategy({"er_threshold": 0.25})
    strat_strict = AdaptiveDisplacementTrailingStrategy({"er_threshold": 0.35})

    # Run on the 100% IDENTICAL DataFrame
    prep_loose = strat_loose.setup(df)
    prep_strict = strat_strict.setup(df)

    er_value = prep_loose["er"].iloc[65]
    assert np.isclose(er_value, 0.30, atol=1e-12), f"Expected ER=0.30, got {er_value}"

    sig_loose, reason_loose = strat_loose.entry_signal(prep_loose, 65)
    sig_strict, reason_strict = strat_strict.entry_signal(prep_strict, 65)

    # er_threshold=0.25 allows signal (0.30 >= 0.25)
    assert sig_loose == Direction.LONG
    assert reason_loose == "long_fvg_mitigation"

    # er_threshold=0.35 rejects signal (0.30 < 0.35)
    assert sig_strict == Direction.NONE
    assert reason_strict == ""


def test_long_mitigation_boundary_equality():
    """Gap 5a: Verify long mitigation zone boundary equality and breach behavior."""
    df, strat = _create_bullish_fvg_scenario(t=65)
    # Zone is [High_(t-3)=103.0, Low_(t-1)=103.5]

    # 1. Upper boundary exact equality: Low_k == Low_(t-1) == 103.5 -> PASS
    df.loc[65, "low"] = 103.5
    p1 = strat.setup(df)
    sig1, r1 = strat.entry_signal(p1, 65)
    assert sig1 == Direction.LONG and r1 == "long_fvg_mitigation"

    # 2. Lower boundary exact equality: Low_k == High_(t-3) == 103.0 -> PASS
    df.loc[65, "low"] = 103.0
    p2 = strat.setup(df)
    sig2, r2 = strat.entry_signal(p2, 65)
    assert sig2 == Direction.LONG and r2 == "long_fvg_mitigation"

    # 3. Upper boundary breach (+0.01 above zone): Low_k == 103.51 -> REJECT
    df.loc[65, "low"] = 103.51
    p3 = strat.setup(df)
    sig3, _ = strat.entry_signal(p3, 65)
    assert sig3 == Direction.NONE

    # 4. Lower boundary breach (-0.01 below zone): Low_k == 102.99 -> REJECT
    df.loc[65, "low"] = 102.99
    p4 = strat.setup(df)
    sig4, _ = strat.entry_signal(p4, 65)
    assert sig4 == Direction.NONE


def test_short_mitigation_boundary_equality():
    """Gap 5b: Verify short mitigation zone boundary equality and breach behavior."""
    df, strat = _create_bearish_fvg_scenario(t=65)
    # Zone is [High_(t-1)=146.5, Low_(t-3)=147.0]

    # 1. Lower boundary exact equality: High_k == High_(t-1) == 146.5 -> PASS
    df.loc[65, "high"] = 146.5
    p1 = strat.setup(df)
    sig1, r1 = strat.entry_signal(p1, 65)
    assert sig1 == Direction.SHORT and r1 == "short_fvg_mitigation"

    # 2. Upper boundary exact equality: High_k == Low_(t-3) == 147.0 -> PASS
    df.loc[65, "high"] = 147.0
    p2 = strat.setup(df)
    sig2, r2 = strat.entry_signal(p2, 65)
    assert sig2 == Direction.SHORT and r2 == "short_fvg_mitigation"

    # 3. Lower boundary breach (-0.01 below zone): High_k == 146.49 -> REJECT
    df.loc[65, "high"] = 146.49
    p3 = strat.setup(df)
    sig3, _ = strat.entry_signal(p3, 65)
    assert sig3 == Direction.NONE

    # 4. Upper boundary breach (+0.01 above zone): High_k == 147.01 -> REJECT
    df.loc[65, "high"] = 147.01
    p4 = strat.setup(df)
    sig4, _ = strat.entry_signal(p4, 65)
    assert sig4 == Direction.NONE
