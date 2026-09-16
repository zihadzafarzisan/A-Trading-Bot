"""HTML dashboard generator.

Produces a fully self-contained HTML dashboard (no CDN, no JavaScript library,
no network dependency) from backtest, analysis, validation, ML, Monte Carlo,
and risk data. The output is a single .html file that opens in any browser.

Sections (spec #34): overview, equity curve, drawdown, P&L, trade history,
win/loss analysis, strategy comparison, regimes, walk-forward, Monte Carlo,
ML results, risk events.
"""

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .charts import bar_chart, donut_chart, histogram, line_chart, sparkline

logger = get_logger("dashboard")


@dataclass
class DashboardData:
    """All optional data sections a dashboard can show."""

    title: str = "Crypto Quant Research Dashboard"
    subtitle: str = ""
    # Core metrics + equity
    metrics: Dict[str, Any] = field(default_factory=dict)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    # Trades
    trades: List[Dict[str, Any]] = field(default_factory=list)
    # Analysis
    trade_analysis: Dict[str, Any] = field(default_factory=dict)
    regime_breakdown: Dict[str, Any] = field(default_factory=dict)
    # Research
    ranked_strategies: List[Dict[str, Any]] = field(default_factory=list)
    experiment_id: str = ""
    # Validation
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    monte_carlo: Dict[str, Any] = field(default_factory=dict)
    # ML
    ml: Dict[str, Any] = field(default_factory=dict)
    # Risk
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    # Coin comparison
    coin_comparison: Dict[str, Any] = field(default_factory=dict)
    # Spot vs Futures
    spot_vs_futures: Dict[str, Any] = field(default_factory=dict)
    # Warnings
    warnings: List[str] = field(default_factory=list)
    # Cash-and-carry funding strategy ledger
    carry_data: Dict[str, Any] = field(default_factory=dict)


# Palette
C_PRIMARY = "#4f9cf9"
C_GREEN = "#7fd08a"
C_RED = "#f2778a"
C_AMBER = "#f9c74f"
C_PURPLE = "#9b8afb"
C_TEXT = "#c8d2f0"
C_MUTED = "#8a94b8"
C_BG = "#0d1326"
C_CARD = "#151d38"
C_BORDER = "#232e52"


class DashboardGenerator:
    """Renders a DashboardData model into a self-contained HTML file."""

    # ------------------------------------------------------------ builder
    @staticmethod
    def from_backtest(
        result,
        symbol: str = "",
        timeframe: str = "",
        market_type: str = "",
        trade_analysis: Optional[dict] = None,
        walk_forward: Optional[dict] = None,
        monte_carlo: Optional[dict] = None,
        ml: Optional[dict] = None,
        risk_events: Optional[List[dict]] = None,
    ) -> DashboardData:
        """Build DashboardData from a BacktestResult plus optional sections."""
        m = result.metrics.to_dict()
        return DashboardData(
            title=f"Backtest — {symbol or 'Symbol'} / {timeframe or 'TF'} ({market_type or 'spot'})",
            subtitle=f"Experiment: {getattr(result, 'experiment_id', '')}",
            metrics=m,
            equity_curve=result.equity_curve,
            trades=result.trades,
            trade_analysis=trade_analysis or {},
            regime_breakdown=(trade_analysis or {}).get("regime_breakdown", {}),
            ranked_strategies=[],
            walk_forward=walk_forward or {},
            monte_carlo=monte_carlo or {},
            ml=ml or {},
            risk_events=risk_events or [],
            warnings=list(result.warnings),
        )

    @staticmethod
    def from_live(
        account,
        orders: Optional[List[dict]] = None,
        risk_events: Optional[List[dict]] = None,
        environment: str = "TESTNET",
    ) -> DashboardData:
        """Build DashboardData from a LiveAccount snapshot."""
        starting = float(account.initial_capital or 1000.0)
        final_equity = float(account.equity or starting)
        net_return = (final_equity - starting) / starting if starting else 0.0
        pnl = final_equity - starting

        metrics = {
            "total_trades": account.closed_trades,
            "win_rate": 0.0,
            "net_return": net_return,
            "realized_pnl": pnl,
            "final_equity": final_equity,
            "starting_capital": starting,
            "max_drawdown": 0.0,
            "environment": environment.upper(),
        }
        env_label = "BINANCE LIVE TRADING" if environment.upper() == "LIVE" else f"BINANCE {environment.upper()}"
        return DashboardData(
            title=f"{env_label} — {account.symbol} / {account.timeframe} ({account.market_type})",
            subtitle=f"Run: {account.run_id} | Strategy: {account.strategy} | Status: {account.status.upper()}",
            metrics=metrics,
            equity_curve=[{"time": int(account.updated_at.timestamp() * 1000) if account.updated_at else 0, "equity": final_equity}],
            trades=orders or [],
            risk_events=risk_events or [],
            warnings=[],
        )

    @staticmethod
    def from_paper(
        result,
        symbol: str = "",
        timeframe: str = "",
        market_type: str = "",
        risk_events: Optional[List[dict]] = None,
    ) -> DashboardData:
        """Build DashboardData from a PaperSessionResult (paper trading session)."""
        starting = float(result.starting_capital or 0.0)
        final_equity = float(result.final_equity or 0.0)
        net_return = (final_equity - starting) / starting if starting else 0.0
        pnl = float(result.realized_pnl or 0.0)
        # Compute max drawdown from equity curve if available.
        peak = starting
        max_dd = 0.0
        for pt in getattr(result, "equity_curve", []) or []:
            eq = float(pt.get("equity", 0.0))
            if eq > peak:
                peak = eq
            elif peak > 0:
                dd = (peak - eq) / peak
                if dd > max_dd:
                    max_dd = dd
        metrics = {
            "total_trades": result.n_trades,
            "win_rate": result.win_rate,
            "net_return": net_return,
            "realized_pnl": pnl,
            "final_equity": final_equity,
            "starting_capital": starting,
            "max_drawdown": max_dd,
            "orders_placed": result.n_orders,
            "orders_rejected": result.n_rejected,
        }
        return DashboardData(
            title=f"Paper Trading — {symbol or 'Symbol'} / {timeframe or 'TF'} ({market_type or 'spot'})",
            subtitle="SIMULATED EXECUTION — Paper trading environment (no real orders sent to exchange).",
            metrics=metrics,
            equity_curve=list(result.equity_curve),
            trades=list(result.trades),
            risk_events=list(risk_events) if risk_events is not None else list(result.risk_events),
            warnings=list(getattr(result, "warnings", []) or []),
        )

    @staticmethod
    def from_discovery(discovery_result, include_equity=False) -> DashboardData:
        """Build DashboardData from a DiscoveryResult (ranked candidates)."""
        ranked = [r.to_dict() for r in discovery_result.ranked]
        top = discovery_result.top
        metrics = top.metrics if top else {"total_trades": 0}
        return DashboardData(
            title=f"Research Discovery — {discovery_result.symbol} / {discovery_result.timeframe} ({discovery_result.market_type})",
            subtitle=f"Experiment: {discovery_result.experiment_id}",
            metrics=metrics,
            equity_curve=[],
            trades=[],
            ranked_strategies=ranked,
            experiment_id=discovery_result.experiment_id,
            warnings=[],
        )

    @staticmethod
    def from_carry(db_manager) -> DashboardData:
        """Build an offline carry-performance dashboard from the local ledger."""
        from datetime import timezone
        from sqlalchemy import func
        from ..db.models import CarryFundingPaymentRecord, CarryPositionRecord

        def utc_text(value, fallback: str = "-") -> str:
            if value is None:
                return fallback
            try:
                if value.tzinfo is None:
                    value = value.replace(tzinfo=timezone.utc)
                return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M")
            except Exception:
                return fallback

        session = db_manager.get_session()
        try:
            position_rows = session.query(CarryPositionRecord).order_by(CarryPositionRecord.opened_at.asc()).all()
            payment_rows = session.query(CarryFundingPaymentRecord).order_by(CarryFundingPaymentRecord.timestamp.asc()).all()
            total_funding = float(session.query(
                func.coalesce(func.sum(CarryFundingPaymentRecord.funding_payment_usdt), 0.0)
            ).scalar() or 0.0)
        except Exception:
            session.rollback()
            logger.warning("Carry ledger query failed — rendering empty carry section")
            return DashboardData(
                title="Cash-and-Carry Performance & Yield Dashboard",
                subtitle="Offline SQLite carry ledger · confirmed funding settlements only",
                carry_data={
                    "metrics": {},
                    "positions": [],
                    "funding_payments": [],
                    "funding_curve": [],
                    "basis_curve": [],
                },
            )
        finally:
            session.close()

        positions = [{
            "position_id": row.position_id,
            "symbol": row.symbol,
            "quantity": float(row.quantity),
            "spot_fill_price": float(row.spot_fill_price),
            "futures_fill_price": float(row.futures_fill_price),
            "entry_basis_spread_pct": float(row.entry_basis_spread_pct),
            "status": row.status,
            "opened_at": utc_text(row.opened_at),
            "closed_at": utc_text(row.closed_at),
        } for row in position_rows]
        payments_chronological = [{
            "timestamp": utc_text(row.timestamp),
            "position_id": row.position_id,
            "symbol": row.symbol,
            "funding_rate_pct": float(row.funding_rate) * 100.0,
            "mark_price": float(row.mark_price),
            "payment_usdt": float(row.funding_payment_usdt),
        } for row in payment_rows]

        cumulative = 0.0
        funding_curve = []
        for payment in payments_chronological:
            cumulative += payment["payment_usdt"]
            funding_curve.append({"time": payment["timestamp"], "equity": cumulative})
        basis_curve = [
            {"time": position["opened_at"], "equity": position["entry_basis_spread_pct"]}
            for position in positions
        ]
        active = sum(1 for position in positions if position["status"] == "OPEN")
        avg_basis = (
            sum(position["entry_basis_spread_pct"] for position in positions) / len(positions)
            if positions else 0.0
        )
        carry_data = {
            "metrics": {
                "total_funding_usdt": total_funding,
                "active_carries": active,
                "total_positions": len(positions),
                "avg_entry_basis_pct": avg_basis,
            },
            "positions": positions,
            "funding_payments": list(reversed(payments_chronological))[:20],
            "funding_curve": funding_curve,
            "basis_curve": basis_curve,
        }
        return DashboardData(
            title="Cash-and-Carry Performance & Yield Dashboard",
            subtitle="Offline SQLite carry ledger · confirmed funding settlements only",
            carry_data=carry_data,
        )

    # ------------------------------------------------------------ main API

    def generate(self, data: DashboardData, output_path: str | Path) -> Path:
        """Render the dashboard to a file. Returns the output path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sections = []
        sections.append(self._overview(data))
        sections.append(self._performance(data))
        if data.trades:
            sections.append(self._trades(data))
            sections.append(self._trade_analysis(data))
        if data.regime_breakdown:
            sections.append(self._regimes(data))
        if data.ranked_strategies:
            sections.append(self._strategies(data))
        if data.walk_forward:
            sections.append(self._walk_forward(data))
        if data.monte_carlo:
            sections.append(self._monte_carlo(data))
        if data.ml:
            sections.append(self._ml(data))
        if data.coin_comparison:
            sections.append(self._coin_comparison(data))
        if data.spot_vs_futures:
            sections.append(self._spot_vs_futures(data))
        if data.carry_data:
            sections.append(self._carry(data))
        if data.risk_events:
            sections.append(self._risk_events(data))
        if data.warnings:
            sections.append(self._warnings(data))

        body = "\n".join(sections)
        html_doc = self._template(data.title, data.subtitle, body)

        output_path.write_text(html_doc, encoding="utf-8")
        logger.info("Dashboard written to %s", output_path)
        return output_path

    # ------------------------------------------------------------ template
    def _template(self, title: str, subtitle: str, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {C_BG}; color: {C_TEXT}; font-family: 'Segoe UI', system-ui, sans-serif;
         padding: 24px; line-height: 1.45; }}
  .header {{ border-bottom: 1px solid {C_BORDER}; padding-bottom: 14px; margin-bottom: 18px; }}
  .header h1 {{ font-size: 22px; letter-spacing: 0.5px; color: #e8edff; }}
  .header .sub {{ color: {C_MUTED}; font-size: 13px; margin-top: 4px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }}
  .card {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 10px; padding: 14px 16px; }}
  .card h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px;
             color: {C_MUTED}; margin-bottom: 10px; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
  .metric {{ background: {C_CARD}; border: 1px solid {C_BORDER}; border-radius: 8px; padding: 10px 12px; }}
  .metric .label {{ font-size: 11px; color: {C_MUTED}; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric .value {{ font-size: 17px; font-weight: 700; color: #e8edff; margin-top: 3px; }}
  .metric .spark {{ margin-top: 6px; }}
  .pos {{ color: {C_GREEN} !important; }}
  .neg {{ color: {C_RED} !important; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th {{ text-align: left; color: {C_MUTED}; font-weight: 600; padding: 6px 8px;
       border-bottom: 1px solid {C_BORDER}; text-transform: uppercase; font-size: 10px; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #1d2744; }}
  tr:hover td {{ background: #1a2440; }}
  .scroll {{ overflow-x: auto; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 10px;
            font-weight: 700; letter-spacing: 0.4px; }}
  .b-green {{ background: rgba(127,208,138,.15); color: {C_GREEN}; }}
  .b-red {{ background: rgba(242,119,138,.15); color: {C_RED}; }}
  .b-amber {{ background: rgba(249,199,79,.15); color: {C_AMBER}; }}
  .b-gray {{ background: rgba(138,148,184,.15); color: {C_MUTED}; }}
  .warn {{ background: rgba(249,199,79,.08); border: 1px solid rgba(249,199,79,.35);
          color: {C_AMBER}; border-radius: 8px; padding: 10px 14px; margin: 8px 0; font-size: 13px; }}
  .finding {{ background: #121a33; border-left: 3px solid {C_PRIMARY}; border-radius: 0 6px 6px 0;
             padding: 8px 12px; margin: 6px 0; font-size: 13px; }}
  .footer {{ margin-top: 26px; color: {C_MUTED}; font-size: 11px; border-top: 1px solid {C_BORDER}; padding-top: 12px; }}
  svg {{ max-width: 100%; height: auto; display: block; }}
</style>
</head>
<body>
  <div class="header">
    <h1>{html.escape(title)}</h1>
    {f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ''}
  </div>
  {body}
  <div class="footer">
    Generated by Crypto Quant Research Terminal · Historical performance does not guarantee future results ·
    Backtests are simulations · ML predictions are probabilistic · Trading involves substantial risk.
  </div>
</body>
</html>"""

    # ------------------------------------------------------------ sections
    def _overview(self, data: DashboardData) -> str:
        m = data.metrics
        cells = []
        for label, key, fmt, cls in [
            ("Win Rate", "win_rate", "{:.1%}", ""),
            ("Profit Factor", "profit_factor", "{:.2f}", ""),
            ("Net Return", "net_return", "{:.1%}", "pos" if m.get("net_return", 0) >= 0 else "neg"),
            ("Max Drawdown", "max_drawdown", "{:.1%}", "neg"),
            ("Sharpe", "sharpe_ratio", "{:.2f}", ""),
            ("Sortino", "sortino_ratio", "{:.2f}", ""),
            ("Trades", "total_trades", "{:,.0f}", ""),
            ("Expectancy", "expectancy", "${:.2f}", ""),
        ]:
            if key in m:
                cells.append(self._metric_card(label, fmt.format(m[key]), cls))
        return f'<div class="metrics">{"".join(cells)}</div>'

    def _metric_card(self, label: str, value: str, cls: str = "") -> str:
        return f'<div class="metric"><div class="label">{html.escape(label)}</div>' \
               f'<div class="value {cls}">{html.escape(value)}</div></div>'

    def _performance(self, data: DashboardData) -> str:
        eq = data.equity_curve
        cards = []
        if eq:
            xs = [e.get("time") for e in eq]
            labels = [_short_ts(t) for t in xs]
            equities = [e.get("equity", 0) for e in eq]
            cards.append('<div class="card"><h3>Equity Curve</h3>'
                         + line_chart(labels, equities, title="Equity")
                         + "</div>")
            # Drawdown series
            peak = 0.0
            dds = []
            for e in equities:
                peak = max(peak, e)
                dds.append((peak - e) / peak if peak else 0.0)
            cards.append('<div class="card"><h3>Drawdown</h3>'
                         + line_chart(labels, dds, stroke=C_RED, fill=C_RED,
                                      title="Drawdown from peak")
                         + "</div>")
        return f'<div class="grid">{"".join(cards)}</div>' if cards else ""

    def _trades(self, data: DashboardData) -> str:
        trades = data.trades[:500]  # cap table size
        pnls = [t.get("net_pnl") or 0 for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        hist = histogram(pnls, bins=24, title="Net P&L distribution") if pnls else ""
        donut = donut_chart(
            ["Wins", "Losses", "Breakeven"],
            [len(wins), len(losses), sum(1 for p in pnls if p == 0)],
            title="Win / Loss split",
        ) if pnls else ""

        # Collect unique values for filters
        symbols = sorted(set(t.get("symbol", "") for t in trades))
        directions = sorted(set(t.get("direction", "") for t in trades))
        timeframes = sorted(set(t.get("timeframe", "") for t in trades))

        # Filter UI
        filter_ui = f"""
        <div class="trade-filters" style="margin-bottom: 12px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center;">
            <label style="color: {C_MUTED}; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;">Filters:</label>
            <select id="filter-symbol" onchange="filterTrades()" style="background: {C_CARD}; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
                <option value="">All Symbols</option>
                {"".join(f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in symbols)}
            </select>
            <select id="filter-direction" onchange="filterTrades()" style="background: {C_CARD}; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
                <option value="">All Directions</option>
                {"".join(f'<option value="{html.escape(d)}">{html.escape(d)}</option>' for d in directions)}
            </select>
            <select id="filter-timeframe" onchange="filterTrades()" style="background: {C_CARD}; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
                <option value="">All Timeframes</option>
                {"".join(f'<option value="{html.escape(tf)}">{html.escape(tf)}</option>' for tf in timeframes)}
            </select>
            <select id="filter-outcome" onchange="filterTrades()" style="background: {C_CARD}; color: {C_TEXT}; border: 1px solid {C_BORDER}; padding: 4px 8px; border-radius: 4px; font-size: 11px;">
                <option value="">All Outcomes</option>
                <option value="win">Wins Only</option>
                <option value="loss">Losses Only</option>
            </select>
            <button onclick="resetFilters()" style="background: {C_BORDER}; color: {C_TEXT}; border: none; padding: 4px 10px; border-radius: 4px; font-size: 11px; cursor: pointer;">Reset</button>
            <span id="trade-count" style="color: {C_MUTED}; font-size: 11px; margin-left: auto;">Showing {len(trades)} of {len(data.trades)} trades</span>
        </div>
        """

        # JavaScript for filtering
        filter_js = """
        <script>
        function filterTrades() {
            const symbol = document.getElementById('filter-symbol').value;
            const direction = document.getElementById('filter-direction').value;
            const timeframe = document.getElementById('filter-timeframe').value;
            const outcome = document.getElementById('filter-outcome').value;
            const rows = document.querySelectorAll('#trade-table tbody tr');
            let visible = 0;
            rows.forEach(row => {
                const cells = row.querySelectorAll('td');
                const rowSymbol = cells[0]?.textContent || '';
                const rowDir = cells[1]?.textContent || '';
                const rowTF = cells[2]?.textContent || '';
                const rowPnl = parseFloat(cells[7]?.textContent) || 0;

                let show = true;
                if (symbol && rowSymbol !== symbol) show = false;
                if (direction && rowDir !== direction) show = false;
                if (timeframe && rowTF !== timeframe) show = false;
                if (outcome === 'win' && rowPnl <= 0) show = false;
                if (outcome === 'loss' && rowPnl >= 0) show = false;

                row.style.display = show ? '' : 'none';
                if (show) visible++;
            });
            document.getElementById('trade-count').textContent = `Showing ${visible} trades`;
        }

        function resetFilters() {
            document.getElementById('filter-symbol').value = '';
            document.getElementById('filter-direction').value = '';
            document.getElementById('filter-timeframe').value = '';
            document.getElementById('filter-outcome').value = '';
            filterTrades();
        }
        </script>
        """

        rows = []
        for t in trades:
            pnl = t.get("net_pnl")
            cls = "pos" if (pnl or 0) > 0 else ("neg" if (pnl or 0) < 0 else "")
            reason = t.get("exit_reason") or ""
            rows.append(
                f"<tr><td>{html.escape(str(t.get('symbol', '')))}</td>"
                f"<td>{html.escape(t.get('direction', ''))}</td>"
                f"<td>{html.escape(t.get('timeframe', ''))}</td>"
                f"<td>{_short_ts(t.get('entry_time'))}</td>"
                f"<td>{_short_ts(t.get('exit_time'))}</td>"
                f"<td>{t.get('entry_price', 0):.2f}</td>"
                f"<td>{t.get('exit_price', 0) if t.get('exit_price') is not None else '-'}</td>"
                f"<td class='{cls}'>{pnl:+.2f}</td>"
                f"<td>{html.escape(reason)}</td></tr>"
            )
        table = f"""<div class="card"><h3>Trade History ({len(trades)} shown of {len(data.trades)})</h3>
          {filter_ui}
          <div class="scroll"><table id="trade-table"><thead><tr>
            <th>Symbol</th><th>Dir</th><th>TF</th><th>Entry</th><th>Exit</th>
            <th>Entry Px</th><th>Exit Px</th><th>Net P&L</th><th>Exit Reason</th>
          </tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>
          {filter_js}"""
        return f'<div class="grid"><div class="card">{hist}</div><div class="card">{donut}</div></div>{table}'

    def _trade_analysis(self, data: DashboardData) -> str:
        a = data.trade_analysis
        if not a:
            return ""
        cards = []
        # Regime breakdown (win rate per regime)
        rb = data.regime_breakdown or {}
        if rb:
            labels = list(rb.keys())
            wrs = [d["win_rate"] * 100 for d in rb.values()]
            cards.append('<div class="card"><h3>Win Rate by Regime</h3>'
                         + bar_chart(labels, wrs, color=C_GREEN, value_format="{:.0f}%",
                                     title="Win rate %")
                         + "</div>")
        # Feature findings
        findings = a.get("feature_findings", [])
        if findings:
            cards.append(self._findings_card("Winning vs Losing — Feature Patterns", findings))
        # Category findings
        cat = a.get("category_findings", [])
        cat_rows = "".join(
            f"<tr><td>{html.escape(c.get('dimension',''))}</td>"
            f"<td>{html.escape(str(c.get('top_winners','')))}</td>"
            f"<td>{html.escape(str(c.get('top_losers','')))}</td></tr>"
            for c in cat
        )
        if cat_rows:
            cards.append(
                f'<div class="card"><h3>Category Splits (winners vs losers)</h3>'
                f'<div class="scroll"><table><thead><tr><th>Dimension</th><th>Winners</th>'
                f'<th>Losers</th></tr></thead><tbody>{cat_rows}</tbody></table></div></div>'
            )
        return f'<div class="grid">{"".join(cards)}</div>'

    def _findings_card(self, title: str, findings: List[dict]) -> str:
        items = []
        for f in findings:
            eff = f.get("effect", {})
            direction = "higher" if eff.get("winners_higher") else "lower"
            items.append(
                f'<div class="finding">Winners {direction} <b>{html.escape(f.get("label",""))}</b> '
                f'({f.get("winners_mean", 0):.2f} vs {f.get("losers_mean", 0):.2f}, '
                f'd={eff.get("cohens_d", 0):.2f})</div>'
            )
        return f'<div class="card"><h3>{html.escape(title)}</h3>{"".join(items)}</div>'

    def _regimes(self, data: DashboardData) -> str:
        rb = data.regime_breakdown or {}
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{d['win_rate']:.1%}</td>"
            f"<td>{d['trades']}</td></tr>" for k, d in rb.items()
        )
        return (f'<div class="card"><h3>Regime Breakdown</h3>'
                f'<table><thead><tr><th>Regime</th><th>Win Rate</th><th>Trades</th></tr></thead>'
                f'<tbody>{rows}</tbody></table></div>')

    def _strategies(self, data: DashboardData) -> str:
        ranked = data.ranked_strategies[:15]
        rows = []
        for r in ranked:
            m = r.get("metrics", {})
            passed = r.get("passed_filters", False)
            badge = '<span class="badge b-green">QUALIFIED</span>' if passed else \
                    '<span class="badge b-gray">rejected</span>'
            reasons = r.get("filter_reasons", [])
            note = f'<div style="color:{C_MUTED};font-size:10px">{html.escape("; ".join(reasons[:2]))}</div>' if reasons else ""
            rows.append(
                f"<tr><td>#{r.get('rank','')}</td><td>{html.escape(r.get('strategy_type',''))}</td>"
                f"<td>{html.escape(str(r.get('params','')))[:70]}</td>"
                f"<td>{m.get('win_rate',0):.1%}</td><td>{m.get('profit_factor',0):.2f}</td>"
                f"<td>{m.get('net_return',0):.1%}</td><td>{m.get('max_drawdown',0):.1%}</td>"
                f"<td>{m.get('total_trades',0)}</td><td>{badge}</td></tr>{note}"
            )
        exp = f'<div class="sub">Experiment: {html.escape(data.experiment_id)}</div>' if data.experiment_id else ""
        return (f'<div class="card"><h3>Strategy Ranking{exp}</h3>'
                f'<div class="scroll"><table><thead><tr><th>#</th><th>Strategy</th><th>Params</th>'
                f'<th>Win Rate</th><th>P.F.</th><th>Return</th><th>MaxDD</th><th>Trades</th>'
                f'<th>Status</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')

    def _walk_forward(self, data: DashboardData) -> str:
        wf = data.walk_forward
        windows = wf.get("windows", [])
        rows = []
        for w in windows:
            win = w.get("window", {})
            ism = w.get("in_sample", {})
            osm = w.get("out_of_sample", {})
            degrad = w.get("win_rate_degradation", 0)
            cls = "pos" if degrad >= 0 else "neg"
            rows.append(
                f"<tr><td>{win.get('train_start','')[:4]}-{win.get('train_end','')[:4]} "
                f"→ {win.get('test_start','')[:4]}-{win.get('test_end','')[:4]}</td>"
                f"<td>{ism.get('win_rate',0):.1%}</td><td>{ism.get('net_return',0):.1%}</td>"
                f"<td>{osm.get('win_rate',0):.1%}</td><td>{osm.get('net_return',0):.1%}</td>"
                f"<td class='{cls}'>{degrad:+.1%}</td>"
                f"<td>{osm.get('total_trades',0)}</td></tr>"
            )
        mean_wr = wf.get("mean_oos_win_rate")
        summary = f'<p style="margin-top:10px">Mean OOS win rate: <b>{mean_wr:.1%}</b>' if mean_wr is not None else ""
        return (f'<div class="card"><h3>Walk-Forward Validation</h3>{summary}'
                f'<div class="scroll"><table><thead><tr><th>Window</th><th>IS WR</th><th>IS Ret</th>'
                f'<th>OOS WR</th><th>OOS Ret</th><th>WR Degrad</th><th>OOS Trades</th>'
                f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')

    def _monte_carlo(self, data: DashboardData) -> str:
        mc = data.monte_carlo
        labels = ["p5", "p25", "p50", "p75", "p95"]
        finals = [mc.get("final_equity_percentiles", {}).get(p, 0) for p in labels]
        dds = [mc.get("max_drawdown_percentiles", {}).get(p, 0) for p in labels]
        ruin = mc.get("probability_of_ruin", 0)
        cards = [
            f'<div class="card"><h3>Monte Carlo — Final Equity (scenarios)</h3>{bar_chart(labels, finals, color=C_PRIMARY, value_format="${:.0f}", title="Final equity percentiles")}</div>',
            f'<div class="card"><h3>Monte Carlo — Max Drawdown</h3>{bar_chart(labels, dds, color=C_RED, value_format="{:.0%}", title="Drawdown percentiles")}'
            f'<p style="margin-top:8px">Probability of ruin: <b>{ruin:.1%}</b></p>'
            f'<p style="color:{C_MUTED};font-size:11px">Statistical scenario — not a prediction.</p></div>',
        ]
        return f'<div class="grid">{"".join(cards)}</div>'

    def _ml(self, data: DashboardData) -> str:
        ml = data.ml
        cards = []
        ev = ml.get("evaluation", {})
        mets = ev.get("test_metrics", {})
        baseline = ev.get("baseline_win_rate")
        filtered = ev.get("ml_filtered_win_rate")
        improves = ev.get("ml_improves")

        parts = ['<div class="card"><h3>Machine Learning</h3>']
        if baseline is not None:
            parts.append(f'<p>Baseline win rate: <b>{baseline:.1%}</b></p>')
        if filtered is not None:
            parts.append(f'<p>ML-filtered win rate: <b>{filtered:.1%}</b></p>')
            badge = ('<span class="badge b-green">IMPROVES</span>' if improves
                     else '<span class="badge b-red">does not improve</span>')
            parts.append(f'<p>ML {badge} the strategy.</p>')
        else:
            parts.append('<p>Too few takes for an ML verdict.</p>')
        if mets:
            parts.append(f'<p>Test AUC: {mets.get("auc", 0):.3f} · '
                         f'Accuracy: {mets.get("accuracy", 0):.3f}</p>')
        parts.append('</div>')
        cards.append("".join(parts))

        # Feature importance
        imp = ev.get("feature_importance", {})
        if imp:
            top = list(imp.items())[:8]
            cards.append(
                '<div class="card"><h3>Top ML Features</h3>'
                + bar_chart([k for k, _ in top], [v * 100 for _, v in top],
                            color=C_PURPLE, value_format="{:.1f}%", title="Importance")
                + "</div>"
            )
        return f'<div class="grid">{"".join(cards)}</div>'

    def _risk_events(self, data: DashboardData) -> str:
        events = data.risk_events[:50]
        rows = []
        for e in events:
            sev = e.get("severity", "")
            badge = '<span class="badge b-red">CRITICAL</span>' if sev == "critical" else '<span class="badge b-amber">WARNING</span>'
            rows.append(f"<tr><td>{html.escape(str(e.get('timestamp',''))[:19])}</td>"
                        f"<td>{html.escape(e.get('event_type',''))}</td><td>{badge}</td>"
                        f"<td>{html.escape(e.get('message',''))}</td></tr>")
        return (f'<div class="card"><h3>Risk Events</h3>'
                f'<div class="scroll"><table><thead><tr><th>Time</th><th>Type</th>'
                f'<th>Severity</th><th>Message</th></tr></thead><tbody>{"".join(rows)}'
                f'</tbody></table></div></div>')

    def _warnings(self, data: DashboardData) -> str:
        warns = "".join(f'<div class="warn">⚠ {html.escape(w)}</div>' for w in data.warnings)
        return f'<div><h3 style="color:{C_AMBER};font-size:13px;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px">Warnings</h3>{warns}</div>'

    def _coin_comparison(self, data: DashboardData) -> str:
        """Compare performance across different coins."""
        coin_data = data.coin_comparison
        if not coin_data:
            return ""

        # Coin performance table
        coins = coin_data.get("coins", [])
        if not coins:
            return ""

        rows = []
        for coin in coins:
            symbol = coin.get("symbol", "")
            metrics = coin.get("metrics", {})
            win_rate = metrics.get("win_rate", 0)
            profit_factor = metrics.get("profit_factor", 0)
            net_return = metrics.get("net_return", 0)
            max_dd = metrics.get("max_drawdown", 0)
            trades = metrics.get("total_trades", 0)

            return_cls = "pos" if net_return >= 0 else "neg"
            dd_cls = "neg" if max_dd < -0.1 else ""

            rows.append(
                f"<tr>"
                f"<td><b>{html.escape(symbol)}</b></td>"
                f"<td>{win_rate:.1%}</td>"
                f"<td>{profit_factor:.2f}</td>"
                f"<td class='{return_cls}'>{net_return:+.1%}</td>"
                f"<td class='{dd_cls}'>{max_dd:.1%}</td>"
                f"<td>{trades}</td>"
                f"</tr>"
            )

        # Bar chart for top coins by return
        top_coins = sorted(coins, key=lambda x: x.get("metrics", {}).get("net_return", 0), reverse=True)[:10]
        if top_coins:
            labels = [c.get("symbol", "") for c in top_coins]
            returns = [c.get("metrics", {}).get("net_return", 0) * 100 for c in top_coins]
            bar = bar_chart(labels, returns, color=C_PRIMARY, value_format="{:.1f}%", title="Top 10 Coins by Return")
        else:
            bar = ""

        # Win rate chart
        if top_coins:
            win_rates = [c.get("metrics", {}).get("win_rate", 0) * 100 for c in top_coins]
            wr_bar = bar_chart(labels, win_rates, color=C_GREEN, value_format="{:.0f}%", title="Win Rate by Coin")
        else:
            wr_bar = ""

        return (
            f'<div class="card"><h3>Coin Performance Comparison</h3>'
            f'<div class="scroll"><table><thead><tr>'
            f'<th>Coin</th><th>Win Rate</th><th>P.F.</th><th>Return</th><th>Max DD</th><th>Trades</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'
            f'<div class="grid"><div class="card">{bar}</div><div class="card">{wr_bar}</div></div>'
        )

    def _carry(self, data: DashboardData) -> str:
        """Render the fully offline Cash-and-Carry performance/yield section."""
        carry = data.carry_data
        metrics = carry.get("metrics", {})
        positions = carry.get("positions", [])
        payments = carry.get("funding_payments", [])
        if not positions and not payments:
            return ('<div class="card"><p style="color: var(--muted, #8a94b8);">'
                    'No Cash-and-Carry positions or funding payments recorded in this database.</p></div>')

        funding = float(metrics.get("total_funding_usdt", 0.0))
        active = int(metrics.get("active_carries", 0))
        total = int(metrics.get("total_positions", 0))
        avg_basis = float(metrics.get("avg_entry_basis_pct", 0.0))
        kpis = [
            self._metric_card("Confirmed Funding", f"${funding:+.4f} USDT", "pos" if funding >= 0 else "neg"),
            self._metric_card("Active Positions", f"{active} OPEN", "pos" if active > 0 else ""),
            self._metric_card("Total Carry Trades", f"{total}", ""),
            self._metric_card("Average Entry Basis", f"{avg_basis:+.4f}%", "pos" if avg_basis >= 0 else "neg"),
            self._metric_card("Strategy Status", "Delta-Neutral" if active > 0 else "Armed", "pos" if active > 0 else ""),
        ]

        funding_curve = carry.get("funding_curve", [])
        basis_curve = carry.get("basis_curve", [])
        funding_chart = line_chart(
            [point.get("time", "") for point in funding_curve],
            [float(point.get("equity", 0.0)) for point in funding_curve],
            stroke=C_GREEN, fill=C_GREEN,
            title="Cumulative Realized Funding (USDT)",
        )
        basis_chart = line_chart(
            [point.get("time", "") for point in basis_curve],
            [float(point.get("equity", 0.0)) for point in basis_curve],
            stroke=C_AMBER, fill=C_AMBER,
            title="Historical Entry Basis Spread (%)",
        )

        def status_badge(status: str) -> str:
            classes = {"OPEN": "b-green", "CLOSED": "b-gray", "ORPHAN_UNWOUND": "b-red", "UNWINDING": "b-amber"}
            return f'<span class="badge {classes.get(status, "b-gray")}">{html.escape(status)}</span>'

        position_rows = "".join(
            f"<tr><td>{html.escape(str(row.get('position_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('symbol', '')))}</td>"
            f"<td>{float(row.get('quantity', 0.0)):.8f}</td>"
            f"<td>${float(row.get('spot_fill_price', 0.0)):.4f}</td>"
            f"<td>${float(row.get('futures_fill_price', 0.0)):.4f}</td>"
            f"<td class={'pos' if float(row.get('entry_basis_spread_pct', 0.0)) >= 0 else 'neg'}>{float(row.get('entry_basis_spread_pct', 0.0)):+.4f}%</td>"
            f"<td>{status_badge(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('opened_at', '-')))}</td>"
            f"<td>{html.escape(str(row.get('closed_at', '-')))}</td></tr>"
            for row in positions
        )
        payment_rows = "".join(
            f"<tr><td>{html.escape(str(row.get('timestamp', '')))}</td>"
            f"<td>{html.escape(str(row.get('symbol', '')))}</td>"
            f"<td>{html.escape(str(row.get('position_id', '')))}</td>"
            f"<td>{float(row.get('funding_rate_pct', 0.0)):+.4f}%</td>"
            f"<td>${float(row.get('mark_price', 0.0)):.4f}</td>"
            f"<td class={'pos' if float(row.get('payment_usdt', 0.0)) >= 0 else 'neg'}>${float(row.get('payment_usdt', 0.0)):+.4f}</td></tr>"
            for row in payments
        )
        positions_table = (
            '<div class="card"><h3>Active & Historical Carry Positions</h3><div class="scroll"><table><thead><tr>'
            '<th>Position ID</th><th>Symbol</th><th>Quantity</th><th>Spot Entry</th><th>Futures Entry</th>'
            '<th>Entry Basis %</th><th>Status</th><th>Opened (UTC)</th><th>Closed (UTC)</th>'
            f'</tr></thead><tbody>{position_rows}</tbody></table></div></div>'
        )
        payments_table = (
            '<div class="card"><h3>Confirmed 8-Hour Funding Settlements</h3><div class="scroll"><table><thead><tr>'
            '<th>Settlement Time (UTC)</th><th>Symbol</th><th>Position ID</th><th>Funding Rate %</th>'
            '<th>Mark Price</th><th>Income (USDT)</th>'
            f'</tr></thead><tbody>{payment_rows}</tbody></table></div></div>'
        )
        return (
            '<div class="card"><h3>Cash-and-Carry Performance & Yield</h3></div>'
            f'<div class="metrics">{"".join(kpis)}</div>'
            f'<div class="grid"><div class="card">{funding_chart}</div><div class="card">{basis_chart}</div></div>'
            + positions_table + payments_table
        )

    def _spot_vs_futures(self, data: DashboardData) -> str:
        """Compare spot vs futures performance."""
        svf = data.spot_vs_futures
        if not svf:
            return ""

        spot = svf.get("spot", {})
        futures = svf.get("futures", {})

        # Build comparison table
        def _comparison_row(label, spot_val, fmt="{:.1%}"):
            futures_val = futures.get(label.lower().replace(" ", "_"), 0)
            spot_val = spot.get(label.lower().replace(" ", "_"), 0)
            spot_cls = "pos" if spot_val >= 0 else "neg" if spot_val < 0 else ""
            futures_cls = "pos" if futures_val >= 0 else "neg" if futures_val < 0 else ""
            return (
                f"<tr><td><b>{html.escape(label)}</b></td>"
                f"<td class='{spot_cls}'>{fmt.format(spot_val)}</td>"
                f"<td class='{futures_cls}'>{fmt.format(futures_val)}</td></tr>"
            )

        rows = [
            _comparison_row("Win Rate", spot.get("win_rate", 0)),
            _comparison_row("Profit Factor", spot.get("profit_factor", 0), "{:.2f}"),
            _comparison_row("Net Return", spot.get("net_return", 0)),
            _comparison_row("Max Drawdown", spot.get("max_drawdown", 0)),
            _comparison_row("Sharpe Ratio", spot.get("sharpe_ratio", 0), "{:.2f}"),
            _comparison_row("Total Trades", spot.get("total_trades", 0), "{:.0f}"),
        ]

        # Long vs Short breakdown for futures
        long_short = svf.get("long_vs_short", {})
        long_rows = []
        if long_short:
            long_data = long_short.get("long", {})
            short_data = long_short.get("short", {})

            def _long_short_row(label, fmt="{:.1%}"):
                l_val = long_data.get(label.lower().replace(" ", "_"), 0)
                s_val = short_data.get(label.lower().replace(" ", "_"), 0)
                l_cls = "pos" if l_val >= 0 else "neg" if l_val < 0 else ""
                s_cls = "pos" if s_val >= 0 else "neg" if s_val < 0 else ""
                return (
                    f"<tr><td><b>{html.escape(label)}</b></td>"
                    f"<td class='{l_cls}'>{fmt.format(l_val)}</td>"
                    f"<td class='{s_cls}'>{fmt.format(s_val)}</td></tr>"
                )

            long_rows = [
                _long_short_row("Win Rate"),
                _long_short_row("Profit Factor", "{:.2f}"),
                _long_short_row("Net Return"),
                _long_short_row("Total Trades", "{:.0f}"),
            ]

        # Chart: Spot vs Futures returns
        if spot and futures:
            labels = ["Win Rate", "Profit Factor", "Return", "Sharpe"]
            spot_vals = [
                spot.get("win_rate", 0) * 100,
                spot.get("profit_factor", 0),
                spot.get("net_return", 0) * 100,
                spot.get("sharpe_ratio", 0),
            ]
            futures_vals = [
                futures.get("win_rate", 0) * 100,
                futures.get("profit_factor", 0),
                futures.get("net_return", 0) * 100,
                futures.get("sharpe_ratio", 0),
            ]
            # Create grouped bar chart using bar_chart function
            spot_bars = bar_chart(
                labels, spot_vals,
                color=C_PRIMARY,
                value_format="{:.1f}",
                title="Spot Metrics"
            )
            futures_bars = bar_chart(
                labels, futures_vals,
                color=C_PURPLE,
                value_format="{:.1f}",
                title="Futures Metrics"
            )
            chart = f'<div class="grid"><div class="card">{spot_bars}</div><div class="card">{futures_bars}</div></div>'
        else:
            chart = ""

        return (
            f'<div class="card"><h3>Spot vs Futures Comparison</h3>'
            f'<div class="scroll"><table><thead><tr>'
            f'<th>Metric</th><th>Spot</th><th>Futures</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
            f'<div style="margin-top: 12px">{chart}</div></div>'
            + (f'<div class="card"><h3>Long vs Short (Futures)</h3>'
                f'<div class="scroll"><table><thead><tr>'
                f'<th>Metric</th><th>Long</th><th>Short</th>'
                f'</tr></thead><tbody>{"".join(long_rows)}</tbody></table></div></div>'
                if long_rows else "")
        )


def _short_ts(ts) -> str:
    """Format a unix-ms timestamp as YYYY-MM-DD."""
    if ts is None:
        return "-"
    try:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)