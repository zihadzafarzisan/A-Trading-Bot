"""Isolated file-based IPC bridge between the live dashboard and daemon processes.

Daemons call ``process_dashboard_commands(handlers)`` at the top of each tick.
The dashboard server writes commands to ``data/bot_commands.json``; this module
consumes them and marks them COMPLETED.  No trading, risk, or exchange imports
live here — the module must stay dependency-free so either daemon can import it
without pulling in the other's dependency tree.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional

logger = logging.getLogger("trading")

# Shared with live_server.py; relative to the project's working directory.
_COMMAND_FILE = Path("data/bot_commands.json")


def process_dashboard_commands(
    handlers: Dict[str, Callable[[dict], None]],
) -> Optional[str]:
    """Read one pending dashboard command and dispatch it.

    Each daemon calls this at the top of its main loop tick.  The function is
    intentionally single-shot (one command per call) and swallows all I/O
    errors so it can never crash the caller's loop.

    Args:
        handlers: mapping of command-name → callable(params).  Unknown commands
            are acknowledged (marked COMPLETED) but not dispatched so the queue
            never stalls on an unrecognised name.

    Returns:
        The command name that was dispatched, or ``None`` if nothing was pending.
    """
    # Fast path: file absent or empty — nothing to do.
    if not _COMMAND_FILE.exists():
        return None

    try:
        raw = _COMMAND_FILE.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
    except Exception as exc:
        logger.debug("[IPC] Failed to read command file: %s", exc)
        return None

    if data.get("status") != "PENDING":
        return None

    command: str = str(data.get("command", "")).upper()
    params: dict = data.get("params") or {}

    dispatched: Optional[str] = None
    try:
        handler = handlers.get(command)
        if handler is not None:
            handler(params)
            dispatched = command
            logger.info(
                "[IPC] Dashboard command dispatched: %s params=%s", command, params
            )
        else:
            logger.warning(
                "[IPC] Dashboard command '%s' received but no handler registered — "
                "acknowledging without action.",
                command,
            )
    except Exception as exc:
        logger.error(
            "[IPC] Handler for command '%s' raised an exception: %s",
            command,
            exc,
            exc_info=True,
        )
    finally:
        # Always mark COMPLETED so the queue clears even on handler errors or
        # unknown commands; a stuck PENDING entry would be re-dispatched on every
        # tick otherwise.
        _mark_completed(data, command)

    return dispatched


def _mark_completed(original: dict, command: str) -> None:
    """Overwrite the command file with status=COMPLETED."""
    try:
        completed = {
            **original,
            "status": "COMPLETED",
            "completed_at": time.time(),
        }
        _COMMAND_FILE.write_text(
            json.dumps(completed, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.warning(
            "[IPC] Failed to mark command '%s' as COMPLETED: %s", command, exc
        )
