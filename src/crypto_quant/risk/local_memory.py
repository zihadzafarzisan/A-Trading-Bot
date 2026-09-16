"""Phase 2 Step 2.2 — the Local Memory Gate.

Adaptive, market-feature-level trade reflection: advisory rules learned from
completed trades (see :class:`~crypto_quant.analytics.ledger.TradeLedger`)
are read back from a JSON file and enforced as an additional admission hurdle
in the carry daemon's pre-flight risk gatekeeper.

A rule file is entirely optional. If ``learnings_path`` does not exist, the
gate admits every candidate (a safe fallback — no memory rules, no blocks).
The file is reloaded lazily only when its ``st_mtime`` changes, so a long-lived
daemon picks up newly-written learnings without a restart and without re-hitting
the disk on every candidate.

Rule schema (a list of objects at the top level, or under a ``"rules"`` key)::

    [
        {
          "type": "BLACKLIST",
          "symbol": "SOLUSDT",           # "*" matches any symbol
          "strategy": "CARRY_ARBITRAGE", # "*" matches any strategy
          "action": "BLOCK",
          "rationale": "why this entry must never reopen",
          "active": true
        },
        {
          "type": "MAX_ENTRY_BASIS_DISCOUNT",
          "symbol": "*",
          "strategy": "CARRY_ARBITRAGE",
          "max_discount_pct": -0.5,      # reject when basis < -0.5%
          "rationale": "..."
        },
        {
          "type": "MIN_APR",
          "symbol": "*",
          "strategy": "CARRY_ARBITRAGE",
          "min_apr": 20.0,               # reject when net APR < 20%
          "rationale": "..."
        }
    ]

A rule is "active" unless it sets ``active: false`` or ``enabled: false``.
``type`` names the guard; ``symbol``/``strategy`` scope it (``"*"`` = any).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..logging_config import get_logger

logger = get_logger("risk")

# Marker for a rule that applies to any symbol / strategy.
_ANY = "*"

# Canonical reason emitted when every rule passes.
_PASS_REASON = "All memory rules passed"


class LocalMemoryGate:
    """Advisory entry gate backed by a reloaded-on-change JSON rule file.

    The gate is *advisory* but authoritative when armed: any matching blocker
    rejects a candidate before the daemon places an order. It never raises on a
    missing, malformed, or transiently unreadable file — it degrades to
    admitting everything, so learning-file issues can never stop legitimate
    entries.
    """

    def __init__(self, learnings_path: str = "config/learnings.json") -> None:
        self.learnings_path = learnings_path
        self._rules: List[Dict[str, Any]] = []
        # st_mtime of the last successfully parsed rule file; None means "not
        # yet loaded" (first call) or "file is currently absent".
        self._mtime: Optional[float] = None

    # ------------------------------------------------------------- persistence
    def _reload_if_stale(self) -> None:
        """(Re)load rules only when the file has changed on disk since last time.

        Lazy: nothing is read until the first :meth:`evaluate_candidate`. A
        missing file leaves the active rules empty (safe fallback); a file whose
        mtime is unchanged from our last read is not re-parsed.
        """
        path = Path(self.learnings_path)
        try:
            stat = path.stat()
        except OSError:
            # File absent (or unreadable enough to stat) -> no rules to enforce.
            if self._rules or self._mtime is not None:
                self._rules = []
                self._mtime = None
            return
        mtime = stat.st_mtime
        if mtime == self._mtime:
            return  # unchanged since our last successful load
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rules = data
            elif isinstance(data, dict) and isinstance(data.get("rules"), list):
                rules = data["rules"]
            else:
                logger.warning(
                    "[MEMORY GATE] %s: unrecognized learnings schema; "
                    "treated as empty rule set.",
                    self.learnings_path,
                )
                rules = []
            self._rules = [r for r in rules if isinstance(r, dict)]
        except Exception as exc:
            logger.warning(
                "[MEMORY GATE] failed to parse %s: %s; treating as empty.",
                self.learnings_path, exc,
            )
            self._rules = []
        self._mtime = mtime

    # ------------------------------------------------------------- rule matching
    @staticmethod
    def _is_active(rule: Dict[str, Any]) -> bool:
        """A rule is enforceable unless explicitly disabled."""
        if rule.get("active", True) is False:
            return False
        if rule.get("enabled", True) is False:
            return False
        return True

    @staticmethod
    def _applies(rule: Dict[str, Any], symbol: str, strategy: str) -> bool:
        """True when the rule scopes include ``symbol``/``strategy``.

        ``"*"`` (or a missing field) means "any". Symbol matching is performed
        case-insensitively so ``SOLUSDT`` and ``solusdt`` address the same rule.
        """
        r_strategy = rule.get("strategy", _ANY)
        if r_strategy != _ANY and r_strategy != strategy:
            return False
        r_symbol = rule.get("symbol", _ANY)
        if (
            r_symbol != _ANY
            and r_symbol.upper() != str(symbol).upper()
        ):
            return False
        return True

    # ------------------------------------------------------------- evaluation
    def evaluate_candidate(
        self,
        symbol: str,
        strategy: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Admit a candidate against every active, matching memory rule.

        ``context`` supplies the live candidate metrics the threshold rules
        compare against — for carry entries the daemon passes
        ``basis_spread_pct`` (percent) and ``net_apr`` (percent).

        Returns ``(True, "All memory rules passed")`` when nothing blocks, or
        ``(False, rule["rationale"])`` on the first matching blocker.
        """
        self._reload_if_stale()
        context = context or {}
        for rule in self._rules:
            if not self._is_active(rule):
                continue
            if not self._applies(rule, symbol, strategy):
                continue
            rule_type = rule.get("type")
            if rule_type == "BLACKLIST":
                # Unconditional block (any candidate matching symbol+strategy).
                action = str(rule.get("action", "BLOCK")).upper()
                if action == "BLOCK":
                    return False, self._rationale(rule, "BLACKLIST")
            elif rule_type == "MAX_ENTRY_BASIS_DISCOUNT":
                # Block when spot is discounted vs futures beyond tolerance, so
                # the (adverse) entry drag can't erode the carry's funding.
                basis = float(context.get("basis_spread_pct", 0.0) or 0.0)
                max_discount = float(rule.get("max_discount_pct", 0.0) or 0.0)
                if basis < max_discount:
                    return False, self._rationale(
                        rule,
                        f"basis {basis:+.4f}% < max_discount {max_discount:+.4f}%",
                    )
            elif rule_type == "MIN_APR":
                # Block carries whose predicted net APR has historically not
                # been worth the capital/risk it ties up.
                net_apr = float(context.get("net_apr", 0.0) or 0.0)
                min_apr = float(rule.get("min_apr", 0.0) or 0.0)
                if net_apr < min_apr:
                    return False, self._rationale(
                        rule,
                        f"net_apr {net_apr:+.2f}% < min_apr {min_apr:+.2f}%",
                    )
        return True, _PASS_REASON

    @staticmethod
    def _rationale(rule: Dict[str, Any], fallback: str) -> str:
        """The rule's operator-written rationale, or a descriptive fallback."""
        rationale = rule.get("rationale")
        return str(rationale).strip() if rationale else fallback