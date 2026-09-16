"""Hermetic unit tests for the Phase 2 Step 2.3 offline reflection loop.

Covers :class:`ReflectionEngine` rule synthesis from the JSONL trade ledger
(loss-rate blacklist, adverse-basis tuning, fee-drag hurdle), the
missing/empty-ledger no-op, and :meth:`apply_rules` dedup + dry-run semantics.
All writes go to ``tmp_path``; no network calls anywhere.
"""

import json

from crypto_quant.analytics.reflect import ReflectionEngine

# ---------------------------------------------------------------------------
# Ledger builders
# ---------------------------------------------------------------------------


def write_ledger(tmp_path, trades) -> str:
    """Persist ``trades`` as a JSONL ledger and return its path (str)."""
    path = tmp_path / "trade_ledger.jsonl"
    path.write_text(
        "".join(json.dumps(t, sort_keys=True) + "\n" for t in trades),
        encoding="utf-8",
    )
    return str(path)


def _trade(
    symbol="SOLUSDT",
    strategy="CARRY_ARBITRAGE",
    net_pnl_usd=-0.04,
    basis=0.2,
    funding=0.0,
    fees=0.04,
    exit_reason="margin safety buffer breached",
    trade_id=None,
) -> dict:
    return {
        "trade_id": trade_id or f"TWIN-{symbol}-{strategy}-x",
        "symbol": symbol,
        "strategy": strategy,
        "exit_reason": exit_reason,
        "net_pnl_usd": net_pnl_usd,
        "entry_basis_spread_pct": basis,
        "funding_on_trade_usdt": funding,
        "estimated_fees_usdt": fees,
        "net_pnl_pct": net_pnl_usd,
        "status": "CLOSED",
    }


def _losing(trade_id, symbol="SOLUSDT"):
    return _trade(trade_id=trade_id, symbol=symbol)


def _profitable(trade_id, symbol="SOLUSDT"):
    t = _trade(trade_id=trade_id, symbol=symbol)
    t["net_pnl_usd"] = 0.08
    return t


# ---------------------------------------------------------------------------
# Test 1: empty / missing ledger
# ---------------------------------------------------------------------------


def test_missing_ledger_yields_no_rules(tmp_path):
    engine = ReflectionEngine(str(tmp_path / "no_ledger.jsonl"), str(tmp_path / "l.json"))
    assert engine.distill_rules() == []
    count, new = engine.apply_rules()
    assert count == 0 and new == []


def test_empty_ledger_yields_no_rules(tmp_path):
    path = write_ledger(tmp_path, [])
    engine = ReflectionEngine(path, str(tmp_path / "l.json"))
    assert engine.distill_rules() == []
    count, new = engine.apply_rules()
    assert count == 0 and new == []


# ---------------------------------------------------------------------------
# Test 2: loss-rate blacklist (>=3 trades, >=60% losses)
# ---------------------------------------------------------------------------


def test_blacklist_synthesized_on_high_loss_rate(tmp_path):
    ledger = write_ledger(tmp_path, [
        _losing("t1"), _losing("t2"), _profitable("t3"),  # 2/3 = 66.7% >= 60%
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))

    rules = engine.distill_rules()
    blacklists = [r for r in rules if r["type"] == "BLACKLIST"]
    assert len(blacklists) == 1
    rule = blacklists[0]
    assert rule["symbol"] == "SOLUSDT"
    assert rule["strategy"] == "CARRY_ARBITRAGE"
    assert rule["action"] == "BLOCK"
    assert rule["active"] is True
    assert rule["rationale"] == "Auto-distilled: 2/3 trades closed at net loss."


def test_blacklist_not_triggered_below_threshold_or_min_trades(tmp_path):
    # 1/3 losing (33%) < 60% -> no blacklist.
    ledger = write_ledger(tmp_path, [
        _losing("t1"), _profitable("t2"), _profitable("t3"),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))
    assert all(r["type"] != "BLACKLIST" for r in engine.distill_rules())

    # Only 2 trades (below min_trades=3) even though 2/2 = 100% losses.
    ledger = write_ledger(tmp_path, [_losing("t4"), _losing("t5")])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))
    assert all(r["type"] != "BLACKLIST" for r in engine.distill_rules(
        min_trades=3))


def test_blacklist_scoped_per_symbol(tmp_path):
    # SOL has 4/4 losses; BTC has 1/4 -> only SOL blacklists.
    ledger = write_ledger(tmp_path, [
        _losing("s1"), _losing("s2"), _losing("s3"), _losing("s4"),
        _losing("b1", symbol="BTCUSDT"),
        _profitable("b2", symbol="BTCUSDT"),
        _profitable("b3", symbol="BTCUSDT"),
        _profitable("b4", symbol="BTCUSDT"),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))
    blacklists = [r for r in engine.distill_rules() if r["type"] == "BLACKLIST"]
    assert [r["symbol"] for r in blacklists] == ["SOLUSDT"]


# ---------------------------------------------------------------------------
# Test 3: adverse-basis tuning (>=2 ADVERSE_BASIS_STOP from negative basis)
# ---------------------------------------------------------------------------


def test_adverse_basis_rule_synthesized(tmp_path):
    ledger = write_ledger(tmp_path, [
        _trade(trade_id="a1", basis=-0.5,
               exit_reason="ADVERSE_BASIS_STOP divergence=+0.8%"),
        _trade(trade_id="a2", basis=-0.4,
               exit_reason="ADVERSE_BASIS_STOP divergence=+0.8%"),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))

    rules = engine.distill_rules()
    basis_rules = [r for r in rules if r["type"] == "MAX_ENTRY_BASIS_DISCOUNT"]
    assert len(basis_rules) == 1
    rule = basis_rules[0]
    assert rule["symbol"] == "SOLUSDT"
    assert rule["max_discount_pct"] == -0.02
    assert rule["action"] == "BLOCK"
    assert rule["active"] is True


def test_adverse_basis_requires_negative_entry_basis(tmp_path):
    # Both stopouts but entry basis is positive -> no basis rule.
    ledger = write_ledger(tmp_path, [
        _trade(trade_id="p1", basis=0.2,
               exit_reason="ADVERSE_BASIS_STOP divergence=+0.8%"),
        _trade(trade_id="p2", basis=0.2,
               exit_reason="ADVERSE_BASIS_STOP divergence=+0.8%"),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))
    assert all(
        r["type"] != "MAX_ENTRY_BASIS_DISCOUNT" for r in engine.distill_rules())


def test_adverse_basis_requires_two_stopouts(tmp_path):
    ledger = write_ledger(tmp_path, [
        _trade(trade_id="o1", basis=-0.3,
               exit_reason="ADVERSE_BASIS_STOP divergence=+0.8%"),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))
    assert all(
        r["type"] != "MAX_ENTRY_BASIS_DISCOUNT" for r in engine.distill_rules())


# ---------------------------------------------------------------------------
# Test 4: fee-drag -> MIN_APR, and dedup on re-run
# ---------------------------------------------------------------------------


def test_min_apr_rule_synthesized_on_fee_drag(tmp_path):
    # Positive funding but net loss on >=2 trades where fees were charged.
    ledger = write_ledger(tmp_path, [
        _trade(trade_id="f1", net_pnl_usd=-0.05, funding=0.12, fees=0.17),
        _trade(trade_id="f2", net_pnl_usd=-0.04, funding=0.13, fees=0.17),
    ])
    engine = ReflectionEngine(ledger, str(tmp_path / "l.json"))

    rules = engine.distill_rules()
    apr_rules = [r for r in rules if r["type"] == "MIN_APR"]
    assert len(apr_rules) == 1
    rule = apr_rules[0]
    assert rule["symbol"] == "SOLUSDT"
    assert rule["min_apr"] == 15.0
    assert rule["action"] == "BLOCK"
    assert rule["active"] is True


def test_apply_rules_dedups_across_runs(tmp_path):
    ledger = write_ledger(tmp_path, [
        _losing("t1"), _losing("t2"), _losing("t3"),  # triggers blacklist
    ])
    learnings = tmp_path / "l.json"
    engine = ReflectionEngine(ledger, str(learnings))

    count1, new1 = engine.apply_rules(dry_run=False)
    assert count1 >= 1 and len(new1) == count1
    # Re-running must not append duplicate (type, symbol, strategy) rules.
    count2, new2 = engine.apply_rules(dry_run=False)
    assert count2 == 0 and new2 == []

    final = json.loads(learnings.read_text(encoding="utf-8"))
    keys = [(r["type"], r["symbol"], r["strategy"]) for r in final]
    assert len(keys) == len(set(keys))


def test_apply_rules_preserves_existing_rules(tmp_path):
    ledger = write_ledger(tmp_path, [_losing("t1"), _losing("t2")])
    learnings = tmp_path / "l.json"
    learnings.write_text(json.dumps([
        {"type": "MIN_APR", "symbol": "BTCUSDT", "strategy": "OTHER",
         "min_apr": 5.0, "rationale": "keep me", "active": True},
    ]), encoding="utf-8")
    engine = ReflectionEngine(ledger, str(learnings))

    engine.apply_rules(dry_run=False)
    final = json.loads(learnings.read_text(encoding="utf-8"))
    rationales = [r.get("rationale") for r in final]
    assert "keep me" in rationales


# ---------------------------------------------------------------------------
# Test 5: dry-run does not modify learnings.json
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write_learnings(tmp_path):
    ledger = write_ledger(tmp_path, [_losing("t1"), _losing("t2"), _losing("t3")])
    learnings = tmp_path / "l.json"
    learnings.write_text(json.dumps([]), encoding="utf-8")
    before = learnings.read_text(encoding="utf-8")
    engine = ReflectionEngine(ledger, str(learnings))

    count, new = engine.apply_rules(dry_run=True)
    assert count >= 1 and new            # patterns were discovered...
    assert learnings.read_text(encoding="utf-8") == before  # ...but not written.


def test_dry_run_does_not_create_missing_learnings(tmp_path):
    ledger = write_ledger(tmp_path, [_losing("t1"), _losing("t2"), _losing("t3")])
    learnings = tmp_path / "never_created.json"
    engine = ReflectionEngine(ledger, str(learnings))

    engine.apply_rules(dry_run=True)
    assert not learnings.exists()