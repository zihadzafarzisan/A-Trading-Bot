"""Read-only data adapter for the live trading dashboard.

Provides thread-safe access to:
- SQLite database (read-only, WAL mode) for positions, orders, carry data
- Binance public REST API for candlestick data
- Trading log file tails for event feeds

All SQLite connections use ``?mode=ro`` URIs and ``PRAGMA query_only = ON``
to guarantee zero interference with the live trading daemon.
"""

import json
import sqlite3
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class LiveDataAdapter:
    """Thread-safe, read-only data adapter for live dashboard state."""

    def __init__(self, db_path: str = "data/crypto_quant.db", project_root: Optional[str] = None):
        self.db_path = db_path
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self._binance_base = "https://api1.binance.com"

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------
    def _connect_ro(self) -> sqlite3.Connection:
        """Open a read-only SQLite connection with query_only pragma."""
        db_file = self.project_root / self.db_path
        uri = f"file:{db_file.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.execute("PRAGMA query_only = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a read-only query and return list of dicts."""
        try:
            conn = self._connect_ro()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(row) for row in rows]
            finally:
                conn.close()
        except Exception as e:
            return [{"error": str(e)}]

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute a read-only query and return a single dict or None."""
        try:
            conn = self._connect_ro()
            try:
                row = conn.execute(sql, params).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Live account data
    # ------------------------------------------------------------------
    def get_live_accounts(self) -> List[Dict[str, Any]]:
        """Read all live account snapshots."""
        rows = self._query("""
            SELECT run_id, environment, symbol, timeframe, market_type,
                   strategy, initial_capital, cash, equity, positions,
                   closed_trades, worker_state, risk_status,
                   last_market_ts, status, started_at, updated_at
            FROM live_accounts
            ORDER BY updated_at DESC
        """)
        return self._parse_timestamps(rows, ["started_at", "updated_at"])

    # ------------------------------------------------------------------
    # Live orders
    # ------------------------------------------------------------------
    def get_live_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Read recent live orders with latency info."""
        rows = self._query("""
            SELECT id, exchange_order_id, run_id, environment, strategy_id,
                   symbol, market_type, side, order_type, requested_qty,
                   executed_qty, requested_price, avg_fill_price, fee,
                   status, rejection_reason, latency_ms, created_at, updated_at
            FROM live_orders
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return self._parse_timestamps(rows, ["created_at", "updated_at"])

    # ------------------------------------------------------------------
    # Carry positions
    # ------------------------------------------------------------------
    def get_carry_positions(self) -> List[Dict[str, Any]]:
        """Read all carry position records, active first."""
        rows = self._query("""
            SELECT position_id, symbol, quantity, spot_fill_price,
                   futures_fill_price, entry_basis_spread_pct, leg_gap_ms,
                   status, opened_at, closed_at
            FROM carry_positions
            ORDER BY CASE WHEN status = 'OPEN' THEN 0 ELSE 1 END, opened_at DESC
        """)
        return self._parse_timestamps(rows, ["opened_at", "closed_at"])

    # ------------------------------------------------------------------
    # Carry funding payments
    # ------------------------------------------------------------------
    def get_carry_funding(self, limit: int = 30) -> List[Dict[str, Any]]:
        """Read recent carry funding settlements."""
        rows = self._query("""
            SELECT cf.position_id, cf.symbol, cf.funding_rate,
                   cf.funding_payment_usdt, cf.mark_price, cf.timestamp
            FROM carry_funding_payments cf
            ORDER BY cf.timestamp DESC
            LIMIT ?
        """, (limit,))
        parsed = []
        for row in rows:
            parsed.append({
                "position_id": row["position_id"],
                "symbol": row["symbol"],
                "funding_rate_pct": float(row["funding_rate"] or 0) * 100.0,
                "payment_usdt": float(row["funding_payment_usdt"] or 0),
                "mark_price": float(row["mark_price"] or 0),
                "timestamp": self._fmt_ts(row["timestamp"]),
            })
        return parsed

    # ------------------------------------------------------------------
    # Execution trades (open)
    # ------------------------------------------------------------------
    def get_execution_trades(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Read recent execution trades."""
        rows = self._query("""
            SELECT id, strategy_id, execution_mode, symbol, market_type,
                   timeframe, direction, entry_time, exit_time, entry_price,
                   exit_price, quantity, leverage, stop_loss, take_profit,
                   fees, funding, gross_pnl, status, net_pnl, exit_reason,
                   created_at
            FROM execution_trades
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return self._parse_timestamps(rows, ["created_at"])

    # ------------------------------------------------------------------
    # Closed trade history
    # ------------------------------------------------------------------
    def get_closed_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Read closed execution trades for history / performance panels."""
        rows = self._query("""
            SELECT id, strategy_id, execution_mode, symbol, market_type,
                   timeframe, direction, entry_time, exit_time, entry_price,
                   exit_price, quantity, leverage, stop_loss, take_profit,
                   fees, funding, gross_pnl, net_pnl, exit_reason, created_at
            FROM execution_trades
            WHERE status = 'closed'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        return self._parse_timestamps(rows, ["created_at"])

    # ------------------------------------------------------------------
    # Performance metrics (computed from closed trades)
    # ------------------------------------------------------------------
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Compute aggregate performance metrics from closed execution trades."""
        rows = self._query("""
            SELECT net_pnl, entry_price, exit_price, direction, fees, funding,
                   entry_time, exit_time, created_at
            FROM execution_trades
            WHERE status = 'closed'
            ORDER BY created_at ASC
            LIMIT 500
        """)

        # Filter out error rows
        rows = [r for r in rows if "error" not in r]

        if not rows:
            return {
                "total_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "max_drawdown": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
                "net_pnl": 0.0, "total_fees": 0.0,
                "best_trade": 0.0, "worst_trade": 0.0,
                "avg_trade": 0.0, "consec_wins": 0, "consec_losses": 0,
                "long_count": 0, "short_count": 0,
                "long_win_rate": 0.0, "short_win_rate": 0.0,
                "equity_curve": [],
            }

        pnls = [float(r.get("net_pnl") or 0) for r in rows]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        if profit_factor == float("inf"):
            profit_factor = 9999.0  # cap for JSON serialization

        # Max drawdown from equity curve
        running = 0.0
        peak = 0.0
        max_dd = 0.0
        equity_curve = []
        for i, r in enumerate(rows):
            pnl = float(r.get("net_pnl") or 0)
            running += pnl
            equity_curve.append({"i": i + 1, "equity": round(running, 4)})
            if running > peak:
                peak = running
            if peak > 0:
                dd = (peak - running) / peak
                if dd > max_dd:
                    max_dd = dd

        # Consecutive wins/losses
        max_cw, max_cl, cw, cl = 0, 0, 0, 0
        for p in pnls:
            if p > 0:
                cw += 1; cl = 0
            elif p < 0:
                cl += 1; cw = 0
            max_cw = max(max_cw, cw)
            max_cl = max(max_cl, cl)

        # Long vs short breakdown
        longs = [r for r in rows if (r.get("direction") or "").lower() == "long"]
        shorts = [r for r in rows if (r.get("direction") or "").lower() == "short"]
        long_wins = [r for r in longs if float(r.get("net_pnl") or 0) > 0]
        short_wins = [r for r in shorts if float(r.get("net_pnl") or 0) > 0]

        total_fees = sum(float(r.get("fees") or 0) + float(r.get("funding") or 0) for r in rows)

        return {
            "total_trades": len(rows),
            "win_rate": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "profit_factor": round(profit_factor, 3),
            "expectancy": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
            "avg_win": round(sum(wins) / len(wins), 4) if wins else 0.0,
            "avg_loss": round(sum(losses) / len(losses), 4) if losses else 0.0,
            "max_drawdown": round(max_dd * 100, 2),
            "gross_profit": round(gross_profit, 4),
            "gross_loss": round(gross_loss, 4),
            "net_pnl": round(sum(pnls), 4),
            "total_fees": round(total_fees, 4),
            "best_trade": round(max(pnls), 4) if pnls else 0.0,
            "worst_trade": round(min(pnls), 4) if pnls else 0.0,
            "avg_trade": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
            "consec_wins": max_cw,
            "consec_losses": max_cl,
            "long_count": len(longs),
            "short_count": len(shorts),
            "long_win_rate": round(len(long_wins) / len(longs) * 100, 1) if longs else 0.0,
            "short_win_rate": round(len(short_wins) / len(shorts) * 100, 1) if shorts else 0.0,
            "equity_curve": equity_curve[-200:],  # last 200 points for chart
        }

    # ------------------------------------------------------------------
    # Risk state (fully parsed)
    # ------------------------------------------------------------------
    def get_risk_state(self) -> Dict[str, Any]:
        """Parse the full risk_status JSON blob from the live account."""
        accounts = self.get_live_accounts()
        if not accounts:
            return {"available": False}

        primary = accounts[0]
        risk_raw = primary.get("risk_status")
        risk: Dict[str, Any] = {}
        if risk_raw:
            if isinstance(risk_raw, dict):
                risk = risk_raw
            elif isinstance(risk_raw, str):
                try:
                    risk = json.loads(risk_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            "available": True,
            "kill_switch": bool(risk.get("kill_switch", False)),
            "daily_loss_pct": float(risk.get("daily_loss_pct", 0) or 0),
            "daily_loss_limit_pct": float(risk.get("daily_loss_limit_pct", 5) or 5),
            "drawdown_pct": float(risk.get("drawdown_pct", 0) or 0),
            "max_drawdown_pct": float(risk.get("max_drawdown_pct", 15) or 15),
            "open_positions": int(risk.get("open_positions", 0) or 0),
            "max_positions": int(risk.get("max_positions", 3) or 3),
            "leverage": float(risk.get("leverage", 1) or 1),
            "max_leverage": float(risk.get("max_leverage", 5) or 5),
            "total_harvested_funding": float(risk.get("total_harvested_funding", 0) or 0),
            "unrealized_pnl": float(risk.get("unrealized_pnl", 0) or 0),
            "exposure_pct": float(risk.get("exposure_pct", 0) or 0),
            "max_exposure_pct": float(risk.get("max_exposure_pct", 100) or 100),
            "gates": {
                "position_limit": not bool(risk.get("position_limit_breached", False)),
                "daily_loss": not bool(risk.get("daily_loss_breached", False)),
                "drawdown": not bool(risk.get("drawdown_breached", False)),
                "leverage": not bool(risk.get("leverage_breached", False)),
                "kill_switch": not bool(risk.get("kill_switch", False)),
                "exchange_connected": bool(risk.get("exchange_connected", True)),
            },
            "block_reason": risk.get("block_reason", ""),
            "entries_blocked": bool(risk.get("entries_blocked", False) or risk.get("kill_switch", False)),
        }

    # ------------------------------------------------------------------
    # Worker state (last signal etc.)
    # ------------------------------------------------------------------
    def get_worker_state(self) -> Dict[str, Any]:
        """Parse worker_state JSON from the live account for signal/heartbeat info."""
        accounts = self.get_live_accounts()
        if not accounts:
            return {"available": False}

        primary = accounts[0]
        ws_raw = primary.get("worker_state")
        ws: Dict[str, Any] = {}
        if ws_raw:
            if isinstance(ws_raw, dict):
                ws = ws_raw
            elif isinstance(ws_raw, str):
                try:
                    ws = json.loads(ws_raw)
                except (json.JSONDecodeError, TypeError):
                    pass

        # Last signal (may be nested under 'last_signal' or at top level)
        last_signal = ws.get("last_signal") or {}
        if not last_signal and ws.get("signal_direction"):
            last_signal = ws  # some versions store signal fields at top level

        return {
            "available": True,
            "orders_placed": int(ws.get("orders_placed", 0) or 0),
            "bars_processed": int(ws.get("bars_processed", 0) or 0),
            "error_count": int(ws.get("error_count", 0) or 0),
            "last_signal": {
                "direction": last_signal.get("direction", last_signal.get("signal_direction", "")),
                "symbol": last_signal.get("symbol", primary.get("symbol", "")),
                "strategy": last_signal.get("strategy", primary.get("strategy", "")),
                "entry_price": last_signal.get("entry_price"),
                "stop_loss": last_signal.get("stop_loss"),
                "take_profit": last_signal.get("take_profit"),
                "signal_time": last_signal.get("signal_time", last_signal.get("time", "")),
                "signal_id": last_signal.get("signal_id", ""),
                "conditions": last_signal.get("conditions", []),
            },
            "peak_equity": float(ws.get("peak_equity", 0) or 0),
            "last_reconcile_ts": ws.get("last_reconcile_ts", ""),
        }

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------
    def get_system_health(self) -> Dict[str, Any]:
        """Check process lock, DB file, log file ages, and DB query latency."""
        lock_path = self.project_root / "data" / "live_worker.lock"
        db_file = self.project_root / self.db_path
        trading_log = self.project_root / "logs" / "trading.log"
        risk_log = self.project_root / "logs" / "risk.log"
        now = time.time()

        def file_age(p: Path):
            try:
                return round(now - p.stat().st_mtime, 1) if p.exists() else None
            except OSError:
                return None

        # DB query latency
        t0 = time.time()
        try:
            self._query("SELECT 1")
            db_latency_ms = round((time.time() - t0) * 1000, 1)
            db_ok = True
        except Exception:
            db_latency_ms = None
            db_ok = False

        lock_age = file_age(lock_path)
        db_age = file_age(db_file)
        trading_log_age = file_age(trading_log)
        risk_log_age = file_age(risk_log)

        # Bot process health: lock file must exist and be < 120s old
        bot_running = lock_age is not None and lock_age < 120

        return {
            "bot_process": {
                "status": "running" if bot_running else ("stale" if lock_age else "offline"),
                "lock_age_s": lock_age,
            },
            "database": {
                "status": "ok" if db_ok else "error",
                "latency_ms": db_latency_ms,
                "file_age_s": db_age,
            },
            "trading_log": {
                "status": "active" if (trading_log_age is not None and trading_log_age < 120) else "stale",
                "file_age_s": trading_log_age,
                "exists": trading_log.exists(),
            },
            "risk_log": {
                "status": "active" if (risk_log_age is not None and risk_log_age < 120) else "stale",
                "file_age_s": risk_log_age,
                "exists": risk_log.exists(),
            },
        }

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics, prioritizing carry harvester state."""
        # --- Carry-first branch: if any OPEN carry positions exist, use them ---
        carry_positions = self.get_carry_positions()
        open_carries = [p for p in carry_positions if p.get("status") == "OPEN"]

        if open_carries:
            # Sum confirmed harvested funding from carry_funding_payments
            harvested = 0.0
            try:
                fund_rows = self._query("""
                    SELECT COALESCE(SUM(funding_payment_usdt), 0) AS total
                    FROM carry_funding_payments
                """)
                if fund_rows and fund_rows[0].get("total") is not None:
                    harvested = float(fund_rows[0]["total"])
            except Exception:
                pass

            # Margin used = sum of (quantity * spot_fill_price) for open carries
            margin_used = sum(
                float(p.get("quantity") or 0) * float(p.get("spot_fill_price") or 0)
                for p in open_carries
            )
            total_equity = 5000.0 + harvested
            total_cash = total_equity - margin_used
            active_symbol = open_carries[0].get("symbol", "")

            return {
                "total_equity": round(total_equity, 4),
                "total_cash": round(total_cash, 4),
                "margin_used": round(margin_used, 4),
                "margin_used_pct": round((margin_used / total_equity * 100) if total_equity > 0 else 0, 1),
                "free_collateral_pct": round((total_cash / total_equity * 100) if total_equity > 0 else 0, 1),
                "total_harvested": round(harvested, 4),
                "unrealized_pnl": 0.0,
                "net_pnl": round(harvested, 4),
                "initial_capital": 5000.0,
                "open_positions": len(open_carries),
                "positions_detail": open_carries,
                "environment": "TESTNET",
                "active_symbol": active_symbol,
                "active_strategy": "Twin-Leg Carry",
                "active_timeframe": "",
                "market_type": "",
                "status": "LIVE",
                "run_id": "",
                "last_update": "",
                "closed_trades": 0,
            }

        # --- Fallback: legacy live_accounts path ---
        accounts = self.get_live_accounts()
        if not accounts:
            return {
                "total_equity": 0.0, "total_cash": 0.0,
                "free_collateral_pct": 0.0, "total_harvested": 0.0,
                "open_positions": 0, "environment": "UNKNOWN",
                "active_symbol": "", "active_strategy": "",
                "status": "offline",
            }

        # Use the most recently updated account as primary
        primary = accounts[0]
        equity = float(primary.get("equity") or 0)
        cash = float(primary.get("cash") or 0)
        initial = float(primary.get("initial_capital") or 1000)
        free_pct = (cash / equity * 100) if equity > 0 else 0

        # Parse positions JSON
        positions_raw = primary.get("positions")
        open_count = 0
        positions_detail = []
        if positions_raw:
            try:
                positions = json.loads(positions_raw) if isinstance(positions_raw, str) else positions_raw
                if isinstance(positions, list):
                    open_count = len(positions)
                    positions_detail = positions
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse risk status for harvest info
        risk_raw = primary.get("risk_status")
        harvested = 0.0
        unrealized_pnl = 0.0
        if risk_raw:
            try:
                risk = json.loads(risk_raw) if isinstance(risk_raw, str) else risk_raw
                if isinstance(risk, dict):
                    harvested = float(risk.get("total_harvested_funding", 0) or 0)
                    unrealized_pnl = float(risk.get("unrealized_pnl", 0) or 0)
            except (json.JSONDecodeError, TypeError):
                pass

        net_pnl = equity - initial
        margin_used = equity - cash if equity > cash else 0.0

        return {
            "total_equity": equity,
            "total_cash": cash,
            "margin_used": round(margin_used, 4),
            "margin_used_pct": round((margin_used / equity * 100) if equity > 0 else 0, 1),
            "free_collateral_pct": round(free_pct, 1),
            "total_harvested": harvested,
            "unrealized_pnl": unrealized_pnl,
            "net_pnl": round(net_pnl, 4),
            "initial_capital": initial,
            "open_positions": open_count,
            "positions_detail": positions_detail,
            "environment": primary.get("environment", "UNKNOWN"),
            "active_symbol": primary.get("symbol", ""),
            "active_strategy": primary.get("strategy", ""),
            "active_timeframe": primary.get("timeframe", ""),
            "market_type": primary.get("market_type", ""),
            "status": primary.get("status", "unknown"),
            "run_id": primary.get("run_id", ""),
            "last_update": primary.get("updated_at", ""),
            "closed_trades": int(primary.get("closed_trades") or 0),
        }

    # ------------------------------------------------------------------
    # Aggregate carry metrics
    # ------------------------------------------------------------------
    def get_carry_metrics(self) -> Dict[str, Any]:
        """Compute aggregate carry-specific metrics."""
        positions = self.get_carry_positions()
        funding = self.get_carry_funding(limit=500)  # All for cumulative

        active = sum(1 for p in positions if p.get("status") == "OPEN")
        total = len(positions)
        total_funding = sum(f.get("payment_usdt", 0) for f in funding)

        basis_values = [p.get("entry_basis_spread_pct", 0) for p in positions
                        if p.get("entry_basis_spread_pct") is not None]
        avg_basis = sum(basis_values) / len(basis_values) if basis_values else 0

        return {
            "active_carries": active,
            "total_positions": total,
            "total_funding_usdt": round(total_funding, 4),
            "avg_entry_basis_pct": round(avg_basis, 4),
        }

    # ------------------------------------------------------------------
    # Log tailing
    # ------------------------------------------------------------------
    def tail_log(self, log_name: str = "trading", lines: int = 50) -> List[Dict[str, Any]]:
        """Read the last N lines from a log file, parsed into events."""
        log_path = self.project_root / "logs" / f"{log_name}.log"
        if not log_path.exists():
            return [{"timestamp": "", "level": "INFO", "message": f"Log file not found: {log_name}.log"}]

        try:
            # Read last portion of file efficiently
            file_size = log_path.stat().st_size
            read_size = min(file_size, lines * 300)  # ~300 bytes per line estimate
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                if file_size > read_size:
                    f.seek(file_size - read_size)
                    f.readline()  # Discard partial first line
                raw_lines = f.readlines()[-lines:]
        except Exception as e:
            return [{"timestamp": "", "level": "ERROR", "message": f"Failed to read log: {e}"}]

        events = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            events.append(self._parse_log_line(line))
        return events

    @staticmethod
    def _parse_log_line(line: str) -> Dict[str, Any]:
        """Parse a structured log line into a dict."""
        # Expected format: 2024-01-15 12:34:56 - module.name - LEVEL - message
        parts = line.split(" - ", 3)
        if len(parts) >= 4:
            return {
                "timestamp": parts[0].strip(),
                "module": parts[1].strip(),
                "level": parts[2].strip(),
                "message": parts[3].strip(),
            }
        return {"timestamp": "", "level": "INFO", "message": line}

    # ------------------------------------------------------------------
    # Binance candle proxy
    # ------------------------------------------------------------------
    def fetch_candles(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Fetch candlestick data from Binance public REST API."""
        url = (
            f"{self._binance_base}/api/v3/klines"
            f"?symbol={symbol}&interval={interval}&limit={limit}"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "CryptoQuantDashboard/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode())

            candles = []
            for k in raw:
                candles.append({
                    "time": int(k[0]) // 1000,  # Convert ms -> seconds for TradingView
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
            return candles
        except (urllib.error.URLError, json.JSONDecodeError, IndexError, TimeoutError) as e:
            return [{"error": str(e)}]

    # ------------------------------------------------------------------
    # Full snapshot (called by SSE poller)
    # ------------------------------------------------------------------
    def get_snapshot(self) -> Dict[str, Any]:
        """Build the complete dashboard state snapshot."""
        accounts = self.get_live_accounts()
        metrics = self.get_aggregate_metrics()
        carry_metrics = self.get_carry_metrics()
        risk_state = self.get_risk_state()
        worker_state = self.get_worker_state()
        performance = self.get_performance_metrics()
        health = self.get_system_health()

        return {
            "server_ts": time.time(),
            "metrics": metrics,
            "carry_metrics": carry_metrics,
            "risk_state": risk_state,
            "worker_state": worker_state,
            "performance": performance,
            "health": health,
            "accounts": accounts,
            "carry_positions": self.get_carry_positions(),
            "carry_funding": self.get_carry_funding(),
            "recent_orders": self.get_live_orders(),
            "recent_trades": self.get_execution_trades(),
            "closed_trades": self.get_closed_trades(limit=50),
            "log_events": self.tail_log("trading", 40),
            "risk_events": self.tail_log("risk", 20),
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_timestamps(rows: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        """Convert datetime fields to ISO strings for JSON serialization."""
        for row in rows:
            for field in fields:
                val = row.get(field)
                if val is None:
                    row[field] = None
                elif isinstance(val, datetime):
                    if val.tzinfo is None:
                        val = val.replace(tzinfo=timezone.utc)
                    row[field] = val.isoformat()
                else:
                    row[field] = str(val)
            # Also serialize JSON text fields
            for json_field in ("positions", "worker_state", "risk_status"):
                val = row.get(json_field)
                if val and isinstance(val, str):
                    try:
                        row[json_field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass
        return rows

    @staticmethod
    def _fmt_ts(val) -> str:
        """Format a timestamp to ISO string."""
        if val is None:
            return ""
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=timezone.utc)
            return val.isoformat()
        return str(val)
