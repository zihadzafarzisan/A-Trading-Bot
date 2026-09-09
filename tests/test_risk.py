"""Tests for the risk engine."""

import pytest

from crypto_quant.risk import (
    PositionSizer, RiskLimits, RiskManager, TrailingStop,
)


class TestPositionSizer:
    """Test risk-based position sizing (spec #13)."""

    def test_spec_example(self):
        """$1000, 1% risk, 2% stop -> $500 position notional (exact spec example)."""
        sizer = PositionSizer(RiskLimits(starting_capital=1000.0, risk_per_trade=0.01,
                                         max_position_pct=1.0, max_portfolio_exposure=1.0))
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=98.0, direction="long", market_type="spot")
        assert result.notional == pytest.approx(500.0)
        assert result.quantity == pytest.approx(5.0)
        assert result.risk_amount == pytest.approx(10.0)
        assert result.capped_by == "risk"

    def test_position_pct_cap(self):
        """Notional cannot exceed max_position_pct of equity."""
        sizer = PositionSizer(RiskLimits(risk_per_trade=0.01, max_position_pct=0.2))
        # stop 0.5% away -> risk sizing would give 200% equity -> capped at 20%
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=99.5, direction="long", market_type="spot")
        assert result.notional == pytest.approx(200.0)
        assert result.capped_by == "position_pct"

    def test_leverage_cap_futures(self):
        """Futures notional capped by equity * leverage."""
        sizer = PositionSizer(RiskLimits(risk_per_trade=0.01, max_leverage=5,
                                         max_position_pct=10.0, max_portfolio_exposure=10.0))
        # 0.1% stop -> risk size 1000% of equity -> capped at 5x leverage
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=99.9, direction="long",
                                     market_type="futures", leverage=5)
        assert result.notional == pytest.approx(5000.0)
        assert result.leverage == 5
        assert result.capped_by == "leverage"

    def test_leverage_respected_when_below_max(self):
        sizer = PositionSizer(RiskLimits(risk_per_trade=0.01, max_leverage=5,
                                         max_position_pct=1.0, max_portfolio_exposure=1.0))
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=90.0, direction="long",
                                     market_type="futures", leverage=2)
        assert result.leverage == 2
        # notional = 10 / 0.10 = 100 (well under 2x=2000 cap)
        assert result.notional == pytest.approx(100.0)

    def test_leverage_requested_over_max_clamped(self):
        sizer = PositionSizer(RiskLimits(max_leverage=5))
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=90.0, direction="long",
                                     market_type="futures", leverage=50)
        assert result.leverage == 5

    def test_invalid_stop_long(self):
        sizer = PositionSizer()
        with pytest.raises(ValueError, match="below entry"):
            sizer.size_position(1000.0, 100.0, 105.0, "long", "spot")

    def test_invalid_stop_short(self):
        sizer = PositionSizer()
        with pytest.raises(ValueError, match="above entry"):
            sizer.size_position(1000.0, 100.0, 95.0, "short", "spot")

    def test_short_sizing(self):
        sizer = PositionSizer(RiskLimits(risk_per_trade=0.01, max_position_pct=1.0))
        result = sizer.size_position(equity=1000.0, entry_price=100.0,
                                     stop_price=102.0, direction="short", market_type="spot")
        assert result.notional == pytest.approx(500.0)


class TestTrailingStop:
    """Test trailing stop logic (spec #14)."""

    def test_long_trailing_rises_with_price(self):
        ts = TrailingStop(trail_pct=0.02)
        stop = ts.update(entry=100.0, direction="long", current_price=110.0, current_stop=None)
        # activated at 100 (activation 0), new stop = 110 * 0.98 = 107.8
        assert stop == pytest.approx(107.8)

    def test_long_stop_never_lowers(self):
        ts = TrailingStop(trail_pct=0.02)
        s1 = ts.update(100.0, "long", 110.0, None)
        s2 = ts.update(100.0, "long", 105.0, s1)   # price drops, stop must hold
        assert s2 == s1

    def test_long_activation(self):
        ts = TrailingStop(activation_pct=0.05, trail_pct=0.02)
        # below activation -> initial stop holds
        s0 = ts.update(100.0, "long", 103.0, None)
        assert s0 == pytest.approx(98.0)  # entry * (1 - trail) = 98
        s1 = ts.update(100.0, "long", 107.0, s0)  # above 105 activation
        assert s1 == pytest.approx(104.86)

    def test_short_trailing_falls_with_price(self):
        ts = TrailingStop(trail_pct=0.02)
        stop = ts.update(entry=100.0, direction="short", current_price=90.0, current_stop=None)
        assert stop == pytest.approx(91.8)


class TestRiskManager:
    """Test the pre-trade validation chain (spec #31, #32, #36)."""

    def test_allows_valid_entry(self):
        rm = RiskManager()
        result = rm.check_entry(equity=1000.0, open_positions=0)
        assert result.is_allowed
        assert result.reasons == []

    def test_rejects_max_positions(self):
        rm = RiskManager(RiskLimits(max_open_positions=3))
        result = rm.check_entry(equity=1000.0, open_positions=3)
        assert not result.is_allowed
        assert any("positions" in r for r in result.reasons)
        # Event logged
        assert any(e.event_type == "max_positions" for e in rm.events)

    def test_rejects_daily_loss_limit(self):
        rm = RiskManager(RiskLimits(daily_loss_limit=100.0))
        rm.record_trade_result(-150.0)
        result = rm.check_entry(equity=850.0, open_positions=0)
        assert not result.is_allowed
        assert any("daily loss" in r for r in result.reasons)

    def test_rejects_weekly_loss_limit(self):
        rm = RiskManager(RiskLimits(weekly_loss_limit=300.0))
        rm.record_trade_result(-350.0)
        result = rm.check_entry(equity=650.0, open_positions=0)
        assert not result.is_allowed
        assert any("weekly loss" in r for r in result.reasons)

    def test_rejects_drawdown(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.25))
        rm.update_equity(1000.0)
        rm.update_equity(1200.0)  # peak
        result = rm.check_entry(equity=800.0, open_positions=0)  # 33% drawdown
        assert not result.is_allowed
        assert any("drawdown" in r for r in result.reasons)

    def test_rejects_leverage_over_max(self):
        rm = RiskManager(RiskLimits(max_leverage=5))
        result = rm.check_entry(equity=1000.0, open_positions=0, leverage=10)
        assert not result.is_allowed
        assert any("leverage" in r for r in result.reasons)

    def test_rejects_exposure(self):
        rm = RiskManager(RiskLimits(max_portfolio_exposure=1.0))
        result = rm.check_entry(equity=1000.0, open_positions=0,
                                current_exposure=900.0, proposed_notional=200.0)
        assert not result.is_allowed
        assert any("exposure" in r for r in result.reasons)

    def test_emergency_stop_blocks_all(self):
        rm = RiskManager()
        rm.emergency_stop()
        result = rm.check_entry(equity=1000.0, open_positions=0)
        assert not result.is_allowed
        assert any("EMERGENCY STOP" in r for r in result.reasons)
        assert rm.is_shutdown
        # Events logged
        assert any(e.event_type == "emergency_stop" for e in rm.events)

    def test_release_emergency_stop(self):
        rm = RiskManager()
        rm.emergency_stop()
        rm.release_emergency_stop()
        assert rm.check_entry(equity=1000.0, open_positions=0).is_allowed

    def test_recent_events(self):
        rm = RiskManager(RiskLimits(max_open_positions=1))
        rm.check_entry(equity=1000.0, open_positions=1)
        events = rm.recent_events()
        assert len(events) >= 1
        assert events[0]["event_type"] == "max_positions"
        assert events[0]["severity"] == "warning"

    def test_record_trade_tracking(self):
        rm = RiskManager()
        rm.record_trade_result(50.0)
        assert rm.state.realized_pnl_day == pytest.approx(50.0)
        rm.record_trade_result(-30.0)
        assert rm.state.realized_pnl_day == pytest.approx(20.0)
        rm.reset_daily()
        assert rm.state.realized_pnl_day == pytest.approx(0.0)

    def test_size_position_raises_and_logs(self):
        rm = RiskManager()
        with pytest.raises(ValueError):
            rm.size_position(1000.0, 100.0, 105.0, "long", "spot")
        assert any(e.event_type == "invalid_order" for e in rm.events)