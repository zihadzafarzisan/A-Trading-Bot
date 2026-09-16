"""Live dashboard HTTP server with SSE streaming.

Provides a real-time monitoring interface for the live trading daemon:
- Serves the single-page dashboard HTML
- SSE endpoint pushes state snapshots every 1.5s
- REST proxy for Binance candlestick data
- REST endpoints for performance, risk, worker state, and system health
- Thread-safe background polling of read-only SQLite

Launch: ``python -m src.crypto_quant.dashboard.live_server [--port 8050]``
"""

import argparse
import json
import queue
import sqlite3
import sys
import threading
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Set

# ---------------------------------------------------------------------------
# IPC command queue constants
# ---------------------------------------------------------------------------
_ALLOWED_COMMANDS: Set[str] = {
    "RESET_DAILY_LOSS",
    "PAUSE_ENTRIES",
    "RESUME_ENTRIES",
    "KILL_SWITCH_ENGAGE",
    "KILL_SWITCH_DISENGAGE",
}
_COMMAND_FILE = Path("data/bot_commands.json")

from .data_adapter import LiveDataAdapter

# ---------------------------------------------------------------------------
# Trade-stream DB poller
# ---------------------------------------------------------------------------
def _to_iso(ts):
    """Safely convert a DB timestamp to an ISO string.

    SQLite is accessed without ``detect_types``, so ``created_at`` /
    ``opened_at`` / ``updated_at`` come back as *text* strings, not native
    ``datetime`` objects — calling ``.isoformat()`` on them directly raises
    ``'str' object has no attribute 'isoformat'``. This normalises either
    form (plus ``None``) to a single string for serialisation and for the
    lexicographic ``updated_at > ?`` comparisons below.
    """
    if ts is None:
        return ""
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _poll_new_trade_events() -> list:
    """Query new execution trades and active carry fills.

    Opens its own read-only SQLite connection each cycle, so it never
    interferes with the live daemon. Returns a list of normalized event dicts
    (empty if nothing new since the last poll).
    """
    global _last_trade_ts, _last_carry_update_ts

    if _adapter is None:
        return []

    root = Path(_adapter.project_root or Path.cwd())
    db_file = root / _adapter.db_path
    uri = f"file:{db_file.as_posix()}?mode=ro"

    events: list = []
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=5)
        conn.execute("PRAGMA query_only = ON;")
        conn.row_factory = sqlite3.Row
        try:
            # --- New execution trades (fills) ---
            rows = conn.execute("""
                SELECT id, execution_mode, symbol, direction, entry_price,
                       quantity, status, fees, created_at
                FROM execution_trades
                WHERE created_at > ?
                ORDER BY created_at ASC
                LIMIT 50
            """, (_last_trade_ts,)).fetchall()

            for row in rows:
                created = _to_iso(row["created_at"])
                events.append({
                    "type": "trade",
                    "id": str(row["id"]),
                    "symbol": row["symbol"],
                    "side": (row["direction"] or "BUY").upper(),
                    "price": float(row["entry_price"] or 0),
                    "quantity": float(row["quantity"] or 0),
                    "status": row["status"],
                    "mode": row["execution_mode"],
                    "timestamp": created,
                })
                if created > _last_trade_ts:
                    _last_trade_ts = created

            # --- Active/newly-updated carry positions (entries) ---
            carry_rows = conn.execute("""
                SELECT position_id, symbol, quantity, spot_fill_price,
                       futures_fill_price, status, opened_at, updated_at
                FROM carry_positions
                WHERE status IN ('OPEN', 'UNWINDING')
                  AND updated_at > ?
                ORDER BY updated_at ASC
                LIMIT 20
            """, (_last_carry_update_ts,)).fetchall()

            for row in carry_rows:
                updated = _to_iso(row["updated_at"])
                opened = _to_iso(row["opened_at"])
                events.append({
                    "type": "carry",
                    "id": str(row["position_id"]),
                    "symbol": row["symbol"],
                    "side": "CARRY",
                    "spot_fill_price": float(row["spot_fill_price"] or 0),
                    "futures_fill_price": float(row["futures_fill_price"] or 0),
                    "quantity": float(row["quantity"] or 0),
                    "status": row["status"],
                    "mode": "carry",
                    "timestamp": opened,
                })
                if updated > _last_carry_update_ts:
                    _last_carry_update_ts = updated
        finally:
            conn.close()
    except Exception as err:
        # On any transient DB error, log and return nothing new this cycle
        print(f"[TRADE STREAM] Poll error: {err}")

    return events

# Module-level server state
_adapter: LiveDataAdapter | None = None
_clients: Set[queue.Queue] = set()
_clients_lock = threading.Lock()
_last_snapshot: dict = {}
_last_snapshot_ts: float = 0.0
_poller_running = False

# Trade-stream (SSE) state — decoupled from the main snapshot pipeline
_trade_clients: Set[queue.Queue] = set()
_trade_clients_lock = threading.Lock()
_trade_poller_running = False
_last_trade_ts: str = ""        # last seen execution_trades.created_at (ISO)
_last_carry_update_ts: str = ""  # last seen carry_positions.updated_at (ISO)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler serving dashboard HTML, JSON APIs, SSE stream, and static files."""

    @staticmethod
    def _guess_mime(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            '.js': 'application/javascript; charset=utf-8',
            '.css': 'text/css; charset=utf-8',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.svg': 'image/svg+xml',
            '.html': 'text/html; charset=utf-8',
            '.map': 'application/json; charset=utf-8',
        }.get(ext, 'application/octet-stream')

    def _serve_static(self, path: str):
        """Serve vendored frontend static assets.

        Route: GET /static/<relative_path>
        Serves from: src/crypto_quant/dashboard/static/<relative_path>
        """
        # Map /static/foo.js -> <this_dir>/static/foo.js
        rel = path[len('/static/'):]
        static_root = Path(__file__).parent / 'static'
        target = (static_root / rel).resolve()
        if not str(target).startswith(str(static_root.resolve())):
            self._send_json(403, {'error': 'Forbidden'})
            return
        if not target.exists() or not target.is_file():
            self._send_json(404, {'error': 'Static asset not found'})
            return

        content = target.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', self._guess_mime(target.as_posix()))
        # no-cache for fast iteration
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Content-Length', str(len(content)))
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def finish(self):
        try:
            super().finish()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    # Suppress default access log (noisy with SSE)
    def log_message(self, format, *args):
        # Only log non-SSE requests
        if "/api/stream" not in (self.path or ""):
            super().log_message(format, *args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        qs = dict(urllib.parse.parse_qsl(parsed.query))

        if path == "/":
            self._serve_dashboard()
        elif path == "/api/snapshot":
            self._serve_snapshot()
        elif path == "/api/stream":
            self._serve_sse()
        elif path == "/api/candles":
            self._serve_candles(qs)
        elif path == "/api/performance":
            self._serve_performance()
        elif path == "/api/risk":
            self._serve_risk()
        elif path == "/api/worker":
            self._serve_worker()
        elif path == "/api/health":
            self._serve_health()
        elif path == "/api/trades":
            self._serve_trades(qs)
        elif path == "/api/stream/trades":
            self._serve_trade_stream()
        elif path.startswith('/static/'):
            self._serve_static(path)
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        """Handle POST requests for bot control commands."""
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        # Back-compat IPC route (single endpoint) ---------------------------
        if path == "/api/control":
            return self._handle_control_post()

        # Binance-style explicit routes (spec) -------------------------------
        # These map 1:1 to the existing IPC command queue.
        if path == "/api/bot/pause":
            return self._queue_command("PAUSE_ENTRIES", {})
        if path == "/api/bot/resume":
            return self._queue_command("RESUME_ENTRIES", {})
        if path == "/api/bot/kill":
            return self._queue_command("KILL_SWITCH_ENGAGE", {})

        if path != "/api/control":
            self._send_json(404, {"error": "Not found"})
            return

        # Read request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 4096:  # Reasonable limit for command JSON
                self._send_json(400, {"error": "Request too large"})
                return

            body_bytes = self.rfile.read(content_length)
            body_str = body_bytes.decode("utf-8")
            data = json.loads(body_str)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        # Validate command
        command = str(data.get("command", "")).upper()
        if command not in _ALLOWED_COMMANDS:
            self._send_json(400, {
                "error": f"Unknown command: {command}",
                "allowed": sorted(_ALLOWED_COMMANDS)
            })
            return

        params = data.get("params") or {}

        # Write command to file
        try:
            _COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
            command_data = {
                "command": command,
                "params": params,
                "status": "PENDING",
                "timestamp": time.time()
            }
            _COMMAND_FILE.write_text(
                json.dumps(command_data, indent=2),
                encoding="utf-8"
            )
        except Exception as exc:
            self._send_json(500, {"error": f"Failed to queue command: {exc}"})
            return

        self._send_json(200, {
            "status": "queued",
            "command": command,
            "params": params,
            "timestamp": command_data["timestamp"]
        })

    def _handle_control_post(self):
        """Back-compat handler for the legacy POST /api/control endpoint."""
        parsed = urllib.parse.urlparse(self.path)
        _ = parsed  # keep symmetry with do_POST routing
        # Read request body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length > 4096:
                self._send_json(400, {"error": "Request too large"})
                return

            body_bytes = self.rfile.read(content_length)
            body_str = body_bytes.decode("utf-8")
            data = json.loads(body_str) if body_str.strip() else {}
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Invalid JSON: {exc}"})
            return

        command = str(data.get("command", "")).upper()
        if command not in _ALLOWED_COMMANDS:
            self._send_json(400, {"error": f"Unknown command: {command}", "allowed": sorted(_ALLOWED_COMMANDS)})
            return

        params = data.get("params") or {}
        return self._queue_command(command, params)

    def _queue_command(self, command: str, params: dict):
        """Queue one dashboard command into data/bot_commands.json."""
        try:
            _COMMAND_FILE.parent.mkdir(parents=True, exist_ok=True)
            command_data = {
                "command": str(command).upper(),
                "params": params or {},
                "status": "PENDING",
                "timestamp": time.time(),
            }
            _COMMAND_FILE.write_text(json.dumps(command_data, indent=2), encoding="utf-8")
        except Exception as exc:
            self._send_json(500, {"error": f"Failed to queue command: {exc}"})
            return

        self._send_json(200, {
            "status": "queued",
            "command": command_data["command"],
            "params": command_data["params"],
            "timestamp": command_data["timestamp"],
        })

    # ------------------------------------------------------------------
    # Dashboard HTML
    # ------------------------------------------------------------------
    def _serve_dashboard(self):
        html_path = Path(__file__).parent / "live_dashboard.html"
        if not html_path.exists():
            self._send_json(500, {"error": "live_dashboard.html not found"})
            return
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    # ------------------------------------------------------------------
    # JSON snapshot (initial load)
    # ------------------------------------------------------------------
    def _serve_snapshot(self):
        global _last_snapshot, _last_snapshot_ts
        if _last_snapshot:
            self._send_json(200, _last_snapshot)
        else:
            # First request — fetch directly
            try:
                snapshot = _adapter.get_snapshot()
                _last_snapshot = snapshot
                _last_snapshot_ts = time.time()
                self._send_json(200, snapshot)
            except Exception as e:
                self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # SSE stream
    # ------------------------------------------------------------------
    def _serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        # Register this client
        client_queue = queue.Queue(maxsize=10)
        with _clients_lock:
            _clients.add(client_queue)

        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    payload = f"data: {json.dumps(data, default=str)}\n\n"
                    self.wfile.write(payload.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _clients_lock:
                _clients.discard(client_queue)

    # ------------------------------------------------------------------
    # SSE trade stream (execution fills + carry entries)
    # ------------------------------------------------------------------
    def _serve_trade_stream(self):
        """Stream new execution trades and active carry fills over SSE.

        Feed-only endpoint, fully decoupled from the main snapshot pipeline.
        On client connect, immediately yields the current portfolio state
        (open carry legs + recent history), then the background trade poller
        pushes incremental updates at 1s cadence.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Write an immediate comment heartbeat to break TCP buffer delay
        try:
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

        # Register this client
        client_queue = queue.Queue(maxsize=50)
        with _trade_clients_lock:
            _trade_clients.add(client_queue)

        # ------------------------------------------------------------------
        # Immediately send initial portfolio snapshot so the UI renders the
        # current state (e.g. open XLMUSDT / THEUSDT / ARBUSDT carry legs)
        # without waiting for the first poller tick.
        # ------------------------------------------------------------------
        try:
            self._send_initial_trade_state()
        except Exception as err:
            print(f"[TRADE STREAM] Error sending initial state: {err}")

        try:
            while True:
                try:
                    data = client_queue.get(timeout=30)
                    self.wfile.write(self._encode_sse(data))
                    self.wfile.flush()
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _trade_clients_lock:
                _trade_clients.discard(client_queue)

    def _send_initial_trade_state(self):
        """Query and push the current portfolio state to a just-connected SSE client.

        Sends three event batches:
        1. ``carry_open``  – every row in ``carry_positions`` with status OPEN
        2. ``carry_recent`` – the 15 most recent carry positions (any status)
        3. ``trades_recent`` – the 15 most recent execution trades
        Each event carries ``"is_initial": True`` so the frontend can distinguish
        a bootstrap payload from incremental updates.
        """
        if _adapter is None:
            return

        root = Path(_adapter.project_root or Path.cwd())
        db_file = root / _adapter.db_path
        uri = f"file:{db_file.as_posix()}?mode=ro"

        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            conn.execute("PRAGMA query_only = ON;")
            conn.row_factory = sqlite3.Row
            try:
                # 1. All OPEN carry positions ----------------------------------
                try:
                    open_rows = conn.execute("""
                        SELECT position_id, symbol, quantity, spot_fill_price,
                               futures_fill_price, status, opened_at, updated_at
                        FROM carry_positions
                        WHERE status = 'OPEN'
                        ORDER BY opened_at DESC
                    """).fetchall()
                except Exception as err:
                    print(f"[TRADE STREAM] carry_positions OPEN query error: {err}")
                    open_rows = []

                for row in open_rows:
                    opened = _to_iso(row["opened_at"])
                    updated = _to_iso(row["updated_at"])
                    self._write_sse_event({
                        "type": "carry",
                        "subtype": "carry_open",
                        "id": str(row["position_id"]),
                        "symbol": row["symbol"],
                        "side": "CARRY",
                        "spot_fill_price": float(row["spot_fill_price"] or 0),
                        "futures_fill_price": float(row["futures_fill_price"] or 0),
                        "quantity": float(row["quantity"] or 0),
                        "status": row["status"],
                        "mode": "carry",
                        "timestamp": opened,
                        "updated_at": updated,
                        "is_initial": True,
                    })

                # 2. The 15 most recent carry positions ------------------------
                try:
                    recent_carry = conn.execute("""
                        SELECT position_id, symbol, quantity, spot_fill_price,
                               futures_fill_price, status, opened_at, updated_at
                        FROM carry_positions
                        ORDER BY opened_at DESC
                        LIMIT 15
                    """).fetchall()
                except Exception as err:
                    print(f"[TRADE STREAM] carry_positions recent query error: {err}")
                    recent_carry = []

                for row in recent_carry:
                    opened = _to_iso(row["opened_at"])
                    updated = _to_iso(row["updated_at"])
                    self._write_sse_event({
                        "type": "carry",
                        "subtype": "carry_recent",
                        "id": str(row["position_id"]),
                        "symbol": row["symbol"],
                        "side": "CARRY",
                        "spot_fill_price": float(row["spot_fill_price"] or 0),
                        "futures_fill_price": float(row["futures_fill_price"] or 0),
                        "quantity": float(row["quantity"] or 0),
                        "status": row["status"],
                        "mode": "carry",
                        "timestamp": opened,
                        "updated_at": updated,
                        "is_initial": True,
                    })

                # 3. The 15 most recent execution trades -----------------------
                try:
                    recent_trades = conn.execute("""
                        SELECT id, execution_mode, symbol, direction, entry_price,
                               quantity, status, fees, created_at
                        FROM execution_trades
                        ORDER BY created_at DESC
                        LIMIT 15
                    """).fetchall()
                except Exception as err:
                    print(f"[TRADE STREAM] execution_trades query error: {err}")
                    recent_trades = []

                for row in recent_trades:
                    created = _to_iso(row["created_at"])
                    self._write_sse_event({
                        "type": "trade",
                        "subtype": "trades_recent",
                        "id": str(row["id"]),
                        "symbol": row["symbol"],
                        "side": (row["direction"] or "BUY").upper(),
                        "price": float(row["entry_price"] or 0),
                        "quantity": float(row["quantity"] or 0),
                        "status": row["status"],
                        "mode": row["execution_mode"],
                        "timestamp": created,
                        "is_initial": True,
                    })

                # 4. Signal that initial state is complete ---------------------
                self._write_sse_event({
                    "type": "init_complete",
                    "is_initial": True,
                })

            finally:
                conn.close()
        except Exception as err:
            # On transient DB error, skip initial state; the poller will catch up
            print(f"[TRADE STREAM] Initial state error: {err}")

    def _write_sse_event(self, data: dict):
        """Write a single SSE event directly to this client's response stream."""
        self.wfile.write(self._encode_sse(data))
        self.wfile.flush()

    @staticmethod
    def _encode_sse(data: dict) -> bytes:
        """Encode a dict as a named SSE frame.

        Emits ``event: <type>`` for known trade-feed event types, otherwise a
        generic ``message`` frame. The client's LIVE TRADE MONITOR listens for
        the named ``trade``/``carry``/``control`` events and falls back to
        ``onmessage`` for anything else.
        """
        evt = data.get("type")
        name = "message"
        if evt in ("trade", "carry", "control", "init_complete", "initial_state_complete"):
            name = "control" if evt in ("init_complete", "initial_state_complete") else evt
        body = json.dumps(data, default=str)
        return f"event: {name}\ndata: {body}\n\n".encode("utf-8")

    # ------------------------------------------------------------------
    # Candle proxy
    # ------------------------------------------------------------------
    def _serve_candles(self, qs: dict):
        symbol = qs.get("symbol", "BTCUSDT").upper()
        interval = qs.get("interval", "1h")
        limit = min(int(qs.get("limit", "200")), 1000)

        # Validate interval
        valid_intervals = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}
        if interval not in valid_intervals:
            interval = "1h"

        candles = _adapter.fetch_candles(symbol, interval, limit)
        self._send_json(200, candles)

    # ------------------------------------------------------------------
    # Performance metrics endpoint
    # ------------------------------------------------------------------
    def _serve_performance(self):
        try:
            data = _adapter.get_performance_metrics()
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # Risk state endpoint
    # ------------------------------------------------------------------
    def _serve_risk(self):
        try:
            data = _adapter.get_risk_state()
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # Worker state endpoint
    # ------------------------------------------------------------------
    def _serve_worker(self):
        try:
            data = _adapter.get_worker_state()
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # System health endpoint
    # ------------------------------------------------------------------
    def _serve_health(self):
        try:
            data = _adapter.get_system_health()
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # Closed trades endpoint
    # ------------------------------------------------------------------
    def _serve_trades(self, qs: dict):
        limit = min(int(qs.get("limit", "100")), 500)
        try:
            data = _adapter.get_closed_trades(limit=limit)
            self._send_json(200, data)
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _send_json(self, code: int, data):
        payload = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


class ThreadedHTTPServer(HTTPServer):
    """HTTPServer that spawns a thread per request for concurrent SSE."""

    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self._poller_thread: threading.Thread | None = None

    def process_request(self, request, client_address):
        """Override to spawn a thread per request."""
        t = threading.Thread(
            target=self._handle_request_thread,
            args=(request, client_address),
            daemon=True,
        )
        t.start()

    def _handle_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)

    def start_poller(self, interval: float = 1.5):
        """Start the background data poller thread."""
        global _poller_running
        _poller_running = True

        def poll_loop():
            global _last_snapshot, _last_snapshot_ts
            while _poller_running:
                try:
                    snapshot = _adapter.get_snapshot()
                    _last_snapshot = snapshot
                    _last_snapshot_ts = time.time()

                    # Push to all connected SSE clients
                    with _clients_lock:
                        dead_clients = []
                        for q in _clients:
                            try:
                                q.put_nowait(snapshot)
                            except queue.Full:
                                dead_clients.append(q)
                        for d in dead_clients:
                            _clients.discard(d)
                except Exception as e:
                    # On error, still push a heartbeat with error info
                    error_snapshot = {
                        "server_ts": time.time(),
                        "error": str(e),
                        "metrics": {},
                        "carry_metrics": {},
                        "risk_state": {},
                        "worker_state": {},
                        "performance": {},
                        "health": {},
                        "accounts": [],
                        "carry_positions": [],
                        "carry_funding": [],
                        "recent_orders": [],
                        "recent_trades": [],
                        "closed_trades": [],
                        "log_events": [],
                        "risk_events": [],
                    }
                    with _clients_lock:
                        for q in _clients:
                            try:
                                q.put_nowait(error_snapshot)
                            except queue.Full:
                                pass

                time.sleep(interval)

        self._poller_thread = threading.Thread(target=poll_loop, daemon=True, name="dashboard-poller")
        self._poller_thread.start()

    def stop_poller(self):
        """Stop the background poller."""
        global _poller_running
        _poller_running = False

    # ------------------------------------------------------------------
    # Trade-stream poller (execution fills + carry entries)
    # ------------------------------------------------------------------
    def start_trade_poller(self, interval: float = 1.0):
        """Start the background trade-stream poller thread."""
        global _trade_poller_running, _last_trade_ts, _last_carry_update_ts
        _trade_poller_running = True

        self._trade_poller_thread: threading.Thread | None = None

        def poll_loop():
            global _last_trade_ts, _last_carry_update_ts
            while _trade_poller_running:
                try:
                    events = _poll_new_trade_events()
                    if events:
                        with _trade_clients_lock:
                            dead_clients = []
                            for q in _trade_clients:
                                try:
                                    for ev in events:
                                        q.put_nowait(ev)
                                except queue.Full:
                                    dead_clients.append(q)
                            for d in dead_clients:
                                _trade_clients.discard(d)
                except Exception:
                    # Never kill the poller on transient DB errors
                    pass

                time.sleep(interval)

        self._trade_poller_thread = threading.Thread(
            target=poll_loop, daemon=True, name="dashboard-trade-poller"
        )
        self._trade_poller_thread.start()

    def stop_trade_poller(self):
        """Stop the background trade poller."""
        global _trade_poller_running
        _trade_poller_running = False


def run_live_dashboard(
    port: int = 8050,
    db_path: str = "data/crypto_quant.db",
    project_root: str | None = None,
    open_browser: bool = True,
    poll_interval: float = 1.5,
):
    """Start the live trading dashboard server.

    Args:
        port: HTTP port (default 8050)
        db_path: Path to SQLite database (relative to project root)
        project_root: Project root directory (default: cwd)
        open_browser: Auto-open browser on start
        poll_interval: Seconds between data polls (default 1.5)
    """
    global _adapter

    root = Path(project_root) if project_root else Path.cwd()
    _adapter = LiveDataAdapter(db_path=db_path, project_root=str(root))

    server = ThreadedHTTPServer(("127.0.0.1", port), DashboardRequestHandler)
    server.start_poller(interval=poll_interval)
    server.start_trade_poller(interval=1.0)

    url = f"http://127.0.0.1:{port}"
    print(f"\n{'='*60}")
    print(f"  LIVE TRADING DASHBOARD")
    print(f"  Serving at: {url}")
    print(f"  Database:   {db_path} (read-only)")
    print(f"  Poll rate:  {poll_interval}s")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}\n")

    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.stop_poller()
        server.stop_trade_poller()
        server.server_close()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Live Trading Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="HTTP port (default: 8050)")
    parser.add_argument("--db", default="data/crypto_quant.db", help="SQLite database path")
    parser.add_argument("--root", default=None, help="Project root directory")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    parser.add_argument("--poll-interval", type=float, default=1.5, help="Data poll interval in seconds")
    args = parser.parse_args()

    run_live_dashboard(
        port=args.port,
        db_path=args.db,
        project_root=args.root,
        open_browser=not args.no_browser,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
