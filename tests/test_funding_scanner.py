"""Tests for the Multi-Asset Funding Yield & Basis Scanner."""

import re
from datetime import datetime, timezone

import pytest
from typer.testing import CliRunner

import crypto_quant.cli.main as main
from crypto_quant.research import CarryOpportunity, FundingScanner
from crypto_quant.research.funding_scanner import SampleProvider, _build_spot_client

runner = CliRunner()


def _plain(output: str) -> str:
    """Strip rich ANSI colour/bold escapes so substring assertions match text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


# ---------------------------------------------------------------------------
# APR formulas
# ---------------------------------------------------------------------------
def test_gross_apr_formula():
    scanner = FundingScanner()
    # rate 0.0001 (0.01%/8h) -> 0.01% * 3 * 365 = 10.95% gross APR
    rate = 0.0001
    gross = rate * 3 * 365 * 100
    assert gross == pytest.approx(10.95)
    # net = gross - fee drag
    assert gross - scanner.fee_drag_pct == pytest.approx(10.95 - 0.14)


# ---------------------------------------------------------------------------
# Pure rank: field correctness, filters, ordering
# ---------------------------------------------------------------------------
def _scaffold():
    scanner = FundingScanner(top_n=None)
    perp = {
        "BTC/USDT:USDT": {
            "fundingRate": 0.0001,
            "markPrice": 67670.0,
            "indexPrice": 67000.0,
            "nextFundingTimestamp": 1_770_000_000_000,
        },
    }
    spot = {
        "BTC/USDT": {"last": 66999.0, "quoteVolume": 28_000_000_000.0},
    }
    active = {"BTC/USDT:USDT"}
    return scanner, perp, spot, active


def test_rank_fields_from_mark_vs_index():
    scanner, perp, spot, active = _scaffold()
    opps = scanner.rank(perp, spot, active)
    assert len(opps) == 1
    o = opps[0]
    assert o.symbol == "BTC/USDT:USDT"
    assert o.spot_price == pytest.approx(67000.0)          # indexPrice (Spot Index)
    assert o.futures_price == pytest.approx(67670.0)       # markPrice
    assert o.basis_spread_pct == pytest.approx((67670 - 67000) / 67000 * 100)
    assert o.predicted_funding_rate == pytest.approx(0.0001)
    assert o.gross_apr_pct == pytest.approx(0.0001 * 3 * 365 * 100)
    assert o.net_apr_pct == pytest.approx(o.gross_apr_pct - scanner.fee_drag_pct)
    assert o.volume_24h_usdt == pytest.approx(28_000_000_000.0)
    assert o.next_funding_time is not None
    assert o.next_funding_time.tzinfo == timezone.utc


def test_rank_filters_low_volume_type_hint():
    scanner = FundingScanner(min_volume_usdt=10_000_000.0)
    perp = {
        "BTC/USDT:USDT": {"fundingRate": 0.0002, "markPrice": 67033.5, "indexPrice": 67000.0},
        "MICRO/USDT:USDT": {"fundingRate": 0.0003, "markPrice": 10.3, "indexPrice": 10.0},
    }
    spot = {
        "BTC/USDT": {"last": 67000.0, "quoteVolume": 50_000_000_000.0},
        "MICRO/USDT": {"last": 10.0, "quoteVolume": 500_000.0},   # < 10M
    }
    active = {"BTC/USDT:USDT", "MICRO/USDT:USDT"}
    opps = scanner.rank(perp, spot, active)
    assert [o.symbol for o in opps] == ["BTC/USDT:USDT"]


def test_rank_excludes_non_quote_and_missing_spot_twin():
    scanner = FundingScanner(quote="USDT")
    perp = {
        "ETH/USDC:USDC": {"fundingRate": 0.0002, "markPrice": 3401.0, "indexPrice": 3400.0},  # non-USDT
        "SOL/USDT:USDT": {"fundingRate": 0.0002, "markPrice": 150.2, "indexPrice": 150.0},     # no spot twin given
    }
    spot = {"ETH/USDC": {"last": 3400.0, "quoteVolume": 1e12}}  # only ETH spot present
    active = {"ETH/USDC:USDC", "SOL/USDT:USDT"}
    opps = scanner.rank(perp, spot, active)
    assert opps == []


def test_rank_excludes_and_records_anomaly():
    scanner = FundingScanner(basis_anomaly_threshold_pct=2.0)
    perp = {
        "BTC/USDT:USDT": {"fundingRate": 0.0001, "markPrice": 67000.0, "indexPrice": 67000.0},
        "TURBO2/USDT:USDT": {"fundingRate": 0.0002, "markPrice": 3.18, "indexPrice": 3.0},   # +6% basis
    }
    spot = {
        "BTC/USDT": {"last": 67000.0, "quoteVolume": 5e10},
        "TURBO2/USDT": {"last": 3.0, "quoteVolume": 2e8},
    }
    active = {"BTC/USDT:USDT", "TURBO2/USDT:USDT"}
    opps = scanner.rank(perp, spot, active)
    assert [o.symbol for o in opps] == ["BTC/USDT:USDT"]
    assert len(scanner.anomalies) == 1
    assert scanner.anomalies[0]["symbol"] == "TURBO2/USDT:USDT"


def test_rank_orders_by_net_apr_desc_and_honors_top_n():
    scanner = FundingScanner()
    perp = {
        "A/USDT:USDT": {"fundingRate": 0.0003, "markPrice": 10.02, "indexPrice": 10.0},
        "B/USDT:USDT": {"fundingRate": 0.0001, "markPrice": 10.01, "indexPrice": 10.0},
        "C/USDT:USDT": {"fundingRate": 0.0002, "markPrice": 10.01, "indexPrice": 10.0},
    }
    spot = {f"{s}/USDT": {"last": 10.0, "quoteVolume": 2e10} for s in ("A", "B", "C")}
    active = {s + "/USDT:USDT" for s in ("A", "B", "C")}
    opps = scanner.rank(perp, spot, active)
    assert [o.symbol for o in opps] == [
        "A/USDT:USDT", "C/USDT:USDT", "B/USDT:USDT",
    ]
    # top_n = 2
    scanner.top_n = 2
    opps2 = scanner.rank(perp, spot, active)
    assert [o.symbol for o in opps2] == ["A/USDT:USDT", "C/USDT:USDT"]


# ---------------------------------------------------------------------------
# scan() end-to-end with an offline SampleProvider
# ---------------------------------------------------------------------------
def test_scan_with_sample_provider_no_network():
    scanner = FundingScanner()
    opps = scanner.scan(SampleProvider())
    # FART (micro-cap) filtered; TURBO2 (anomaly) filtered; SAMPLEDCOIN negative funding survives
    symbols = {o.symbol for o in opps}
    assert "BTC/USDT:USDT" in symbols
    assert "ETH/USDT:USDT" in symbols
    assert "FART/USDT:USDT" not in symbols
    assert "TURBO2/USDT:USDT" not in symbols
    assert scanner.anomalies  # TURBO2 flagged
    # ordered net APR desc
    nets = [o.net_apr_pct for o in opps]
    assert nets == sorted(nets, reverse=True)


def test_scan_top_n_sample():
    scanner = FundingScanner(top_n=3)
    opps = scanner.scan(SampleProvider())
    assert len(opps) <= 3


def test_sample_provider_is_duck_typed_provider():
    prov = SampleProvider()
    assert prov.active_usdm_symbols()
    assert prov.fetch_funding_rates()
    assert prov.fetch_spot_tickers()


# ---------------------------------------------------------------------------
# Spot vs. Futures endpoint pollution (Mission: carry-settlement-spot-futures-pollution)
#
# The scanner's Spot client is built on ccxt's shared ``binance`` class, whose
# ``urls['api']`` table carries every Binance product family (``fapi*`` USD-M,
# ``dapi*`` COIN-M, ``eapi*`` options, ``papi*`` portfolio-margin, demo hosts).
# A pin is required so the spot handle can never resolve to a futures URL or
# carry stale credentials at launch. These guards are fully offline (no network).
# ---------------------------------------------------------------------------
_SPOT_HOST_RE = re.compile(r"https?://api[0-9]*\.binance\.com")


def test_spot_client_urls_are_spot_only():
    spot = _build_spot_client(load=False)
    api = spot.urls["api"]
    assert api, "spot client must retain its URL table"
    # every string-form URL group must resolve to a real Spot host
    for group, url in api.items():
        if isinstance(url, str):
            assert _SPOT_HOST_RE.match(url), f"{group} -> {url} is not a Spot host"
    # no futures/delivery/options/demo family may remain on the spot handle
    for family in ("fapi", "dapi", "eapi", "papi", "testnet", "demo"):
        for group, url in api.items():
            if isinstance(url, str):
                assert family not in url, f"futures URL group {group} -> {url} leaked"


def test_spot_client_carries_no_credentials():
    spot = _build_spot_client(load=False)
    assert not spot.apiKey
    assert not spot.secret


def test_spot_client_default_type_pinned_to_spot():
    spot = _build_spot_client(load=False)
    assert spot.options.get("defaultType") == "spot"


# ---------------------------------------------------------------------------
# CLI: carry scan --sample
# ---------------------------------------------------------------------------
def test_carry_scan_sample_cli(tmp_path):
    result = runner.invoke(main.app, ["carry", "scan", "--sample"])
    assert result.exit_code == 0, result.output
    assert "Funding & Basis Scanner" in _plain(result.output) or "FUNDING & BASIS SCANNER" in _plain(result.output)
    assert "Carry Funding Opportunities" in _plain(result.output)
    # The anomaly flag proves the scan+render pipeline surfaced universe data end-to-end.
    assert "TURBO2/USDT:USDT" in _plain(result.output)
    assert "Flagged basis anomalies" in _plain(result.output)


def test_carry_scan_help_lists_options():
    result = runner.invoke(main.app, ["carry", "scan", "--help"])
    assert result.exit_code == 0
    for opt in ("--top", "--min-volume", "--fee-drag-pct", "--quote", "--anomaly-threshold-pct", "--sample"):
        assert opt in _plain(result.output)