"""Paper account & worker-state persistence.

Persists (and reloads) the full paper account — cash, equity, open positions,
realized-trade count, worker state, risk status — so `paper status` can show
accurate live state and a later run can resume instead of silently restarting
the account (STEP 8 of Phase 12).

The trading loop never stores anything here mid-bar; it calls
``save_account_snapshot`` once per completed bar. ``restore_broker_state``
rebuilds a PaperBroker from a saved snapshot for restart recovery.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .broker import PriceSource

logger = get_logger("trading")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_account_snapshot(
    db,
    *,
    run_id: str,
    config,
    broker,
    risk,
    strategy: str = "",
    worker_state: Optional[Dict[str, Any]] = None,
    status: str = "running",
    last_market_ts: Optional[int] = None,
) -> None:
    """Upsert a PaperAccount row from the current broker/risk/worker state."""
    from ..db.models import PaperAccount
    session = db.get_session()
    try:
        row = session.get(PaperAccount, run_id)
        if row is None:
            row = PaperAccount(run_id=run_id, started_at=datetime.now(timezone.utc))
            session.add(row)
        row.symbol = config.symbol
        row.timeframe = config.timeframe
        row.market_type = config.market_type
        row.strategy = strategy
        row.mode = getattr(config, "mode", "replay")
        row.initial_capital = broker.initial_capital
        row.cash = float(broker.cash)
        row.equity = float(broker.equity)
        row.positions = json.dumps(broker.get_positions())
        row.closed_trades = len(broker.closed_trades)
        row.worker_state = json.dumps(worker_state or {})
        row.risk_status = json.dumps({
            "drawdown": _drawdown(broker, risk),
            "exposure": broker.equity if broker.equity else 0.0,
            "kill_switch": bool(risk.is_shutdown),
            "last_events": risk.recent_events(5),
        })
        row.last_market_ts = last_market_ts
        row.status = status
        row.updated_at = datetime.now(timezone.utc)
        session.commit()
    except Exception:
        session.rollback()
        logger.debug("account snapshot save failed (non-fatal)", exc_info=True)
    finally:
        session.close()


def load_account_snapshot(db, run_id: Optional[str] = None, *, any_status: bool = False):
    """Return the latest PaperAccount row, or the one for ``run_id``."""
    from ..db.models import PaperAccount
    session = db.get_session()
    try:
        if run_id is not None:
            return session.get(PaperAccount, run_id)
        q = session.query(PaperAccount)
        if not any_status:
            q = q.filter(PaperAccount.status == "running")
        return q.order_by(PaperAccount.updated_at.desc()).first()
    finally:
        session.close()


def restore_broker_state(broker, row) -> None:
    """Rebuild a PaperBroker's cash / positions / closed_trades from a snapshot.

    Open positions preserve their entry price, side, quantity, leverage and stop
    /take levels so SL/TP and liquidation keep working after a restart.
    """
    if row is None:
        return
    broker.cash = float(row.cash)
    broker.initial_capital = float(row.initial_capital)
    broker.positions = {}
    try:
        positions = json.loads(row.positions or "[]")
    except json.JSONDecodeError:
        positions = []
    for p in positions:
        broker.positions[p["symbol"]] = dict(p)
    # Closed-trade count is preserved so accounting stays consistent across a
    # restart, but the trade list itself is reconstructed from the DB (the
    # closed_trades order also feeds PnL reporting).
    from ..db.models import ExecutionTrade
    session = row._sa_instance_state.session
    if session:
        try:
            trades = (session.query(ExecutionTrade)
                      .filter(ExecutionTrade.execution_mode == "paper")
                      .order_by(ExecutionTrade.created_at.asc()).all())
            # Convert SQLAlchemy models back to dict for the PaperBroker
            broker.closed_trades = [
                {
                    "trade_id": t.id, "symbol": t.symbol, "direction": t.direction,
                    "execution_mode": t.execution_mode, "market_type": t.market_type,
                    "timeframe": getattr(t, "timeframe", None),
                    "entry_price": t.entry_price, "exit_price": t.exit_price,
                    "quantity": t.quantity, "gross_pnl": t.gross_pnl,
                    "net_pnl": t.net_pnl, "fees": getattr(t, "fees", 0),
                    "entry_time": t.entry_time, "exit_time": t.exit_time,
                    "status": t.status, "exit_reason": t.exit_reason,
                } for t in trades
            ]
        except Exception:
            pass

    # Still pad the array to the recorded count in case the trades were archived
    target_len = int(getattr(row, "closed_trades", 0) or 0)
    while len(broker.closed_trades) < target_len:
        broker.closed_trades.append({"net_pnl": 0.0})


def persist_risk_events(db, risk) -> None:
    """Persist RiskEvents not yet written to the risk_events table."""
    from ..db.models import RiskEvent
    session = db.get_session()
    try:
        stored = {r.event_type for r in session.query(RiskEvent).all()}
        events = [e for e in risk.events if e.event_type not in stored]
        for e in events:
            session.add(RiskEvent(
                timestamp=e.timestamp,
                event_type=e.event_type,
                severity=e.severity,
                message=e.message,
                context=json.dumps(e.context or {}),
            ))
        if events:
            session.commit()
            logger.debug("persisted %d risk events", len(events))
    finally:
        session.close()


def _drawdown(broker, risk) -> float:
    equity = float(broker.equity)
    peak = max(float(risk.state.peak_equity), equity) if risk.state.peak_equity else equity
    if peak <= 0:
        return 0.0
    return round((peak - equity) / peak, 6)