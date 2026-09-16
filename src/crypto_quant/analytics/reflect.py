"""Phase 2 Step 2.3 — the offline reflection loop.

The third and final link in Phase 2's local-memory chain: closed trades written
by :class:`~crypto_quant.analytics.ledger.TradeLedger` are distilled *offline*
into advisory learning rules, then merged (deduplicated) into the same JSON
learnings file that :class:`~crypto_quant.risk.local_memory.LocalMemoryGate`
reads back pre-order. The loop is:
trade ledger --> :meth:`ReflectionEngine.distill_rules` -->
:meth:`ReflectionEngine.apply_rules` --> learnings file --> gate acts on next
entry. Running it is always a dry run unless the operator explicitly applies.

Synthesis heuristics (all scoped per ``(strategy, symbol)``):

1. **Loss Rate Blacklist** — a symbol/strategy pair that closes at net loss in
   ``>= min_trades`` observed trades at a rate ``>= loss_rate_threshold`` is no
   longer worth reopening, so a ``BLACKLIST`` rule is synthesized.
2. **Adverse Basis Tuning** — repeated stopouts where the adverse-basis exit
   fired from an already-negative entry basis suggest entries were taken while
   spot was too discounted vs futures; a ``MAX_ENTRY_BASIS_DISCOUNT`` rule is
   synthesized to reject such entries.
3. **Fee Drag Hurdle** — if funding was captured positive but the trade still
   closed net-negative (fees eating the harvest) on ``>= 2`` trades, a
   ``MIN_APR`` rule is synthesized to demand enough yield to clear the drag.

Rules are compared to what already exists in the learnings file by
``type + symbol + strategy`` so re-running reflection never duplicates a rule.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..logging_config import get_logger

logger = get_logger("analytics")

# Marker used by the memory gate for "applies to any symbol / strategy".
_ANY = "*"

# Field names carried by every ledger record we rely on for synthesis.
_SYMBOL = "symbol"
_STRATEGY = "strategy"
_EXIT_REASON = "exit_reason"
_NET_PNL = "net_pnl_usd"
_ENTRY_BASIS = "entry_basis_spread_pct"
_FUNDING = "funding_on_trade_usdt"
_FEES = "estimated_fees_usdt"

# Exit-reason tag that flags an adverse-basis stopout.
_ADVERSE_TAG = "ADVERSE_BASIS_STOP"

# --- rule shapes (schema consumed by LocalMemoryGate) ------------------------
_RULE_BLACKLIST = "BLACKLIST"
_RULE_BASIS = "MAX_ENTRY_BASIS_DISCOUNT"
_RULE_MIN_APR = "MIN_APR"


class ReflectionEngine:
    """Turn closed-trade history into deduplicated advisory learning rules.

    Pure offline logic: it reads the JSONL ledger and the JSON learnings file,
    never touches an exchange, and never mutates the ledger. Learnings are only
    written when the operator opts in (``dry_run=False``).
    """

    def __init__(
        self,
        ledger_path: str = "data/trade_ledger.jsonl",
        learnings_path: str = "config/learnings.json",
    ) -> None:
        self.ledger_path = str(ledger_path)
        self.learnings_path = str(learnings_path)

    # --------------------------------------------------------------- loading
    def _load_trades(self) -> List[Dict]:
        """Return every valid closed-trade record in the ledger, in file order.

        A missing or empty ledger yields ``[]``; malformed lines are skipped
        (matching the ledger's own resilience: a torn tail line must not abort
        the whole read).
        """
        path = Path(self.ledger_path)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            logger.warning("[REFLECT] could not read ledger %s; no trades.", path)
            return []
        trades: List[Dict] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                trades.append(record)
        return trades

    def _load_existing_rules(self) -> List[Dict]:
        """Return the rules currently persisted in the learnings file.

        Accepts the same two shapes the memory gate reads: a bare JSON list, or
        a ``{"rules": [...]}`` object. A missing / malformed file yields ``[]``
        so a new learnings file can be created from scratch without errors.
        """
        path = Path(self.learnings_path)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("[REFLECT] failed to parse %s: %s; treating as empty.",
                           path, exc)
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("rules"), list):
            return data["rules"]
        logger.warning("[REFLECT] %s: unrecognized learnings schema; empty.", path)
        return []

    # ------------------------------------------------------------- synthesis
    @staticmethod
    def _num(value, default: float = 0.0) -> float:
        """Cooerce a possibly-missing record metric to a number."""
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _distill_heuristics(
        self,
        trades: List[Dict],
        min_trades: int,
        loss_rate_threshold: float,
    ) -> List[Dict]:
        """Synthesize advisory rules from per ``(strategy, symbol)`` metrics."""
        # Aggregate counters per (strategy, symbol).
        groups: Dict[Tuple[str, str], Dict] = defaultdict(
            lambda: {"total": 0, "losses": 0,
                     "adverse_negative_basis": 0, "fee_drag": 0}
        )
        for tr in trades:
            strat = tr.get(_STRATEGY, _ANY)
            sym = tr.get(_SYMBOL, _ANY)
            agg = groups[(strat, sym)]
            agg["total"] += 1
            net = self._num(tr.get(_NET_PNL))
            if net < 0:
                agg["losses"] += 1
            basis = self._num(tr.get(_ENTRY_BASIS))
            exit_reason = str(tr.get(_EXIT_REASON, "") or "")
            if _ADVERSE_TAG in exit_reason and basis < 0:
                agg["adverse_negative_basis"] += 1
            funding = self._num(tr.get(_FUNDING))
            fees = self._num(tr.get(_FEES))
            if funding > 0 and net < 0 and fees > 0:
                agg["fee_drag"] += 1

        rules: List[Dict] = []
        for (strat, sym), agg in groups.items():
            total, losses = agg["total"], agg["losses"]
            # 1. Loss-rate blacklist.
            if total >= min_trades and losses >= 0 and (losses / total) >= loss_rate_threshold:
                rules.append({
                    "type": _RULE_BLACKLIST,
                    "symbol": sym,
                    "strategy": strat,
                    "action": "BLOCK",
                    "rationale": (
                        f"Auto-distilled: {losses}/{total} trades closed at net loss."
                    ),
                    "active": True,
                })
            # 2. Adverse-basis tuning.
            if agg["adverse_negative_basis"] >= 2:
                rules.append({
                    "type": _RULE_BASIS,
                    "symbol": sym,
                    "strategy": strat,
                    "max_discount_pct": -0.02,
                    "action": "BLOCK",
                    "rationale": (
                        "Auto-distilled: multiple adverse basis stopouts from "
                        "negative entry basis."
                    ),
                    "active": True,
                })
            # 3. Fee-drag hurdle.
            if agg["fee_drag"] >= 2:
                rules.append({
                    "type": _RULE_MIN_APR,
                    "symbol": sym,
                    "strategy": strat,
                    "min_apr": 15.0,
                    "action": "BLOCK",
                    "rationale": (
                        "Auto-distilled: fees consumed harvested funding under "
                        "low APR entries."
                    ),
                    "active": True,
                })
        return rules

    # --------------------------------------------------------------- public
    def distill_rules(
        self,
        min_trades: int = 3,
        loss_rate_threshold: float = 0.60,
    ) -> List[Dict]:
        """Distill advisory learning rules from the closed-trade ledger.

        Args:
            min_trades: minimum trades a ``(strategy, symbol)`` group needs
                before its loss-rate blacklists on.
            loss_rate_threshold: fraction of losing trades at/above which the
                group is blacklisted.

        Returns:
            Synthesized rules (always fresh — existing learnings are *not* read
            here, only merged later by :meth:`apply_rules`).
        """
        trades = self._load_trades()
        if not trades:
            return []
        return self._distill_heuristics(trades, min_trades, loss_rate_threshold)

    @staticmethod
    def _rule_key(rule: Dict) -> Tuple[str, str, str]:
        """Dedup identity: ``type + symbol + strategy`` (case-folded symbol)."""
        return (
            str(rule.get("type", "")),
            str(rule.get("symbol", "")).upper(),
            str(rule.get("strategy", "")),
        )

    def apply_rules(self, dry_run: bool = False) -> Tuple[int, List[Dict]]:
        """Merge freshly distilled rules into the learnings file (or preview).

        First distills from the ledger, then compares each candidate against the
        rules already present (by ``type + symbol + strategy``). New rules are
        appended to the existing list; an existing ``{"rules": [...]}`` wrapper
        is preserved. In dry-run mode the file is never touched.

        Returns:
            ``(count_added, new_rules)`` where ``new_rules`` lists only the
            rules that were not already present (and would be written).
        """
        distilled = self.distill_rules()
        existing = self._load_existing_rules()
        present: set = {self._rule_key(r) for r in existing
                        if isinstance(r, dict)}
        new_rules: List[Dict] = []
        for rule in distilled:
            if self._rule_key(rule) not in present:
                present.add(self._rule_key(rule))  # de-dupe within one run
                new_rules.append(rule)

        if new_rules and not dry_run:
            self._persist(existing, new_rules)
        return len(new_rules), new_rules

    def _persist(self, existing: List[Dict], new_rules: List[Dict]) -> None:
        """Write ``existing + new_rules`` back to the learnings file."""
        path = Path(self.learnings_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        merged = existing + new_rules
        # Preserve the top-level shape the gate accepts (list vs {"rules": [...]}).
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            is_wrapped = isinstance(data, dict) and "rules" in data
        except (OSError, json.JSONDecodeError):
            is_wrapped = False
        payload = {"rules": merged} if is_wrapped else merged
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("[REFLECT] wrote %d new rule(s) to %s.", len(new_rules), path)


def _main(argv: List[str] | None = None) -> int:
    """CLI entrypoint: report discovered patterns, apply only on ``--apply``."""
    parser = argparse.ArgumentParser(
        prog="python -m crypto_quant.analytics.reflect",
        description="Distill closed-trade history into local memory rules.",
    )
    parser.add_argument(
        "--ledger", default="data/trade_ledger.jsonl",
        help="JSONL trade ledger to read (default: %(default)s).",
    )
    parser.add_argument(
        "--learnings", default="config/learnings.json",
        help="learnings file to merge into (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="preview without writing (this is the default unless --apply).",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="persist newly distilled rules (overrides --dry-run).",
    )
    parser.add_argument(
        "--min-trades", type=int, default=3,
        help="min trades per (strategy, symbol) before blacklisting.",
    )
    parser.add_argument(
        "--loss-rate", type=float, default=0.60,
        help="losing-trade rate at/above which a pair is blacklisted.",
    )
    args = parser.parse_args(argv)

    engine = ReflectionEngine(args.ledger, args.learnings)
    distilled = engine.distill_rules(args.min_trades, args.loss_rate)
    logger.info(
        "[REFLECT] found %d pattern(s) across the ledger.", len(distilled)
    )
    for rule in distilled:
        logger.info(
            "[REFLECT]   %s %s/%s -> %s",
            rule["type"], rule.get("symbol"), rule.get("strategy"),
            rule.get("rationale"),
        )

    dry_run = not args.apply
    count, _ = engine.apply_rules(dry_run=dry_run)
    if args.apply:
        logger.info(
            "[REFLECT] applied %d new rule(s) to %s.", count, args.learnings
        )
    else:
        logger.info(
            "[REFLECT] dry-run: %d rule(s) would be written; use --apply to persist.",
            count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())