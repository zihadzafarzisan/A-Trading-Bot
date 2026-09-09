"""Execution model for backtesting.

Models realistic fills: entry at the next bar's open (respecting an execution
delay), taker/maker fees for spot vs futures, slippage, and funding. Also
models futures liquidation using isolated margin.

The model is intentionally pessimistic where uncertain: when a stop and take
profit are both touched within the same bar we assume the stop is hit first
(safe assumption), and gaps are filled at the gaping open rather than the
limit level.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ExecutionConfig:
    """Execution assumptions (visible in every report)."""

    market_type: str = "spot"          # 'spot' | 'futures'
    slippage: float = 0.0005           # 0.05% adverse per fill
    commission: float = 0.001          # default taker fee for spot (0.1%)
    maker_fee: float = 0.0002          # futures maker
    taker_fee: float = 0.0004          # futures taker
    use_taker: bool = True             # assume taker fills by default
    funding_rate: float = 0.0001       # per 8h funding interval
    execution_delay_bars: int = 1      # bars between signal and fill
    initial_capital: float = 1000.0
    max_leverage: float = 5.0

    @property
    def fee_rate(self) -> float:
        """Effective per-side fee rate for the active assumption."""
        if self.market_type == "futures":
            return self.taker_fee if self.use_taker else self.maker_fee
        return self.commission

    def to_dict(self) -> Dict[str, float]:
        """Record execution assumptions (shown in reports)."""
        return {
            "market_type": self.market_type,
            "slippage": self.slippage,
            "commission": self.commission,
            "maker_fee": self.maker_fee,
            "taker_fee": self.taker_fee,
            "use_taker": self.use_taker,
            "funding_rate": self.funding_rate,
            "execution_delay_bars": self.execution_delay_bars,
            "initial_capital": self.initial_capital,
            "max_leverage": self.max_leverage,
        }


class ExecutionModel:
    """Computes fill prices, costs, and liquidation for trades."""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()

    # ------------------------------------------------------------ fills
    def entry_fill(self, signal_price: float, direction: str, entry_open: float) -> float:
        """Fill price for an entry.

        If execution_delay_bars > 1, fills at entry_open (the bar open at fill
        time) already handled by the engine; here we apply slippage in the
        adverse direction.
        """
        slippage = self.config.slippage
        if direction == "long":
            return entry_open * (1 + slippage)
        return entry_open * (1 - slippage)

    def exit_fill(self, last: float, direction: str) -> float:
        """Fill price for a market/limit exit, with adverse slippage."""
        slippage = self.config.slippage
        if direction == "long":
            return last * (1 - slippage)
        return last * (1 + slippage)

    # ------------------------------------------------------------ costs
    def fee_cost(self, notional: float) -> float:
        """Cost of a single fee-charged fill on notional."""
        return notional * self.config.fee_rate

    def funding_cost(self, notional: float, holding_bars: int, bars_per_funding: int = 8) -> float:
        """Accrued funding for a futures position over holding_bars."""
        if self.config.market_type != "futures":
            return 0.0
        payments = holding_bars / bars_per_funding
        return notional * self.config.funding_rate * payments

    # ------------------------------------------------------------ liquidation
    def liquidation_price(
        self,
        entry: float,
        direction: str,
        leverage: float,
        maintenance_margin_pct: float = 0.005,
    ) -> float:
        """Estimate the isolated-margin liquidation price.

        Simplified model: liquidation occurs when the margin (position value /
        leverage) minus unrealized loss reaches the maintenance margin.
        Used to flag unrealistic results, not exact exchange math.
        """
        if leverage <= 0:
            return 0.0
        # Fraction of entry price that wipes out the margin
        loss_fraction = 1.0 / leverage - maintenance_margin_pct
        if loss_fraction <= 0:
            return 0.0
        if direction == "long":
            return entry * (1 - loss_fraction)
        return entry * (1 + loss_fraction)