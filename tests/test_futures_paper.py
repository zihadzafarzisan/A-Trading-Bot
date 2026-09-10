"""Paper futures verification tests (STEP 10).

Verify long/short, leverage 1x-5x, margin, funding, and liquidation logic in the
paper futures implementation via the PaperTradingEngine.
"""

import numpy as np
import pandas as pd

from crypto_quant.strategies.base import BaseStrategy, Direction
from crypto_quant.risk.manager import RiskManager
from crypto_quant.risk.limits import RiskLimits
from crypto_quant.execution.paper import PaperBroker
from crypto_quant.execution.engine import (
    PaperTradingEngine, PaperTradingConfig, DataframePriceSource,
)


class EarlyShort(BaseStrategy):
    name = "EarlyShort"
    strategy_type = "test"
    supports_short = True
    def validate_params(self): pass
    def setup(self, df): return df.copy()
    def entry_signal(self, df, i, ctx=None):
        return (Direction.SHORT, "test") if i == 5 else (Direction.NONE, "")
    def compute_stop_loss(self, df, i, direction, entry_price):
        # We MUST provide a stop loss for risking rules/sizing to work.
        # A 5x leverage liquidation is at 20% loss + margin ratio.
        # We put our STOP completely out of the way (e.g. 50% loss)
        # so liquidation is guaranteed to hit *first*.
        return entry_price * 1.50
    def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
        return entry_price * 0.10


def _bars(n=120, start=100.0, drift=1.0):
    # steadily rising data => Short will lose heavily and liquidate
    close = start + np.arange(n) * drift
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3600_000 for i in range(n)],
        "open": close - 0.2, "high": close + 1.0,
        "low": close - 1.0, "close": close,
        "volume": np.full(n, 1_000_000.0),
    })


def test_futures_short_liquidates():
    """A heavily losing short position on futures with leverage should be liquidated."""
    bars = _bars(n=120, start=100.0, drift=5.0)  # fast rise
    limits = RiskLimits(max_leverage=5.0, starting_capital=1000.0)
    broker = PaperBroker(
        price_source=DataframePriceSource(bars),
        starting_capital=1000.0, market_type="futures",
    )
    risk = RiskManager(limits)
    cfg = PaperTradingConfig(
        symbol="BTCUSDT", market_type="futures", timeframe="1h",
        starting_capital=1000.0, leverage=3,
    )
    eng = PaperTradingEngine(
        strategy=EarlyShort(), broker=broker, risk_manager=risk, config=cfg,
    )
    res = eng.run_bars(bars)

    # 1 entry, 1 exit (liquidation)
    assert res.n_trades >= 1
    t = res.trades[0]
    assert t["execution_mode"] == "paper"
    assert t["market_type"] == "futures"
    assert t["direction"] == "short"
    assert t["leverage"] == 3
    # The exit reason should explicitly say liquidation
    assert t["exit_reason"] == "liquidation"
    assert t["net_pnl"] < 0


def test_futures_funding_accrued():
    """Funding should be accrued and subtracted from realized PnL on close."""
    bars = _bars(n=50, start=100.0, drift=0.1)  # slow rise
    limits = RiskLimits(max_leverage=5.0, starting_capital=1000.0)
    broker = PaperBroker(
        price_source=DataframePriceSource(bars),
        starting_capital=1000.0, market_type="futures",
    )
    risk = RiskManager(limits)
    cfg = PaperTradingConfig(
        symbol="BTCUSDT", market_type="futures", timeframe="1h",
        starting_capital=1000.0, leverage=2, funding_bars=8,
    )

    class EarlyShortWithTP(EarlyShort):
        def compute_take_profit(self, df, i, direction, entry_price, stop_loss=None):
            return entry_price * 0.1  # Low TP so it holds until the end

    eng = PaperTradingEngine(
        strategy=EarlyShortWithTP(), broker=broker, risk_manager=risk, config=cfg,
    )
    # Stop the session at the end, which will force-close and realize PnL with funding.
    res = eng.run_bars(bars)

    assert res.n_trades >= 1
    t = res.trades[0]
    assert t["funding"] > 0.0, "Funding should have been accrued"
    assert t["net_pnl"] < t["gross_pnl"], "Net PnL should be gross minus fees minus funding"