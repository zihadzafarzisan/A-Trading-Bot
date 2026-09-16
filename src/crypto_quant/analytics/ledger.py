"""Phase 2 Step 2.1 — the local append-only trade ledger.

A small, dependency-light persistence primitive: every closed trade is appended
as a single JSON line to a ``.jsonl`` file so that the reflection layer (regime
analysis, per-strategy PnL attribution, etc.) has a durable, order-stable,
human-inspectable history without touching SQLite. Thread-safe via a per-instance
lock because the carry daemon may call it from multiple worker threads.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# Required keys every closed-trade record must carry. Missing or blank values
# are a signal the payload was misassembled and should never be silently stored.
_REQUIRED_FIELDS: Dict[str, str] = {
    "trade_id": "unique closed-trade identifier",
    "symbol": "e.g. SOLUSDT",
    "strategy": "short strategy name, e.g. CARRY_ARBITRAGE",
    "exit_reason": "why the position was closed",
    "net_pnl_usd": "realized net profit/loss in USDT",
}


class TradeLedger:
    """Append-only JSON-lines store of closed trade records.

    Each call to :meth:`record_closed_trade` appends exactly one line, so the
    file preserves insertion order and is safe to tail/parse incrementally.
    """

    def __init__(self, ledger_path: str = "data/trade_ledger.jsonl") -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_closed_trade(self, trade_record: dict) -> None:
        """Validate, timestamp, and append one closed-trade record.

        Args:
            trade_record: Mapping carrying at least the required fields
                (``trade_id``, ``symbol``, ``strategy``, ``exit_reason``,
                ``net_pnl_usd``). The caller's dict is not mutated.

        Raises:
            ValueError: if a required field is missing or blank.
        """
        for field in _REQUIRED_FIELDS:
            value = trade_record.get(field)
            # Presence-based, not truthiness-based: ``net_pnl_usd == 0.0`` is a
            # perfectly valid (break-even) trade and must not be rejected.
            if value is None or value == "":
                raise ValueError(
                    f"TradeLedger record missing required field '{field}' "
                    f"({_REQUIRED_FIELDS[field]}); got {trade_record!r}"
                )

        record = dict(trade_record)
        if not record.get("recorded_at"):
            record["recorded_at"] = datetime.now(timezone.utc).isoformat()

        line = json.dumps(record, sort_keys=True) + "\n"
        with self._lock:
            with open(self.ledger_path, "a", encoding="utf-8") as fh:
                fh.write(line)

    def load_recent_trades(self, limit: int = 50) -> list[dict]:
        """Return the last ``limit`` records from the ledger, in file order.

        Returns the N most-recent lines (oldest→newest). An empty or missing
        ledger yields an empty list. Malformed lines are skipped rather than
        aborting the whole read.
        """
        if not self.ledger_path.exists():
            return []
        records: List[dict] = []
        with self._lock:
            with open(self.ledger_path, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        for line in lines[-limit:]:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A torn/corrupt tail line should not prevent reading the rest.
                continue
        return records