"""Tests for the trade research engine (winning vs losing pattern mining)."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.analysis.trades import TradeAnalyzer


def make_trades(n, entry_times, pnls, directions=None):
    """Build a list of trade dicts."""
    trades = []
    directions = directions or ["long"] * n
    for i in range(n):
        trades.append({
            "entry_time": int(entry_times[i]),
            "exit_time": int(entry_times[i] + 3_600_000),
            "net_pnl": float(pnls[i]),
            "gross_pnl": float(pnls[i]),
            "direction": directions[i],
            "entry_price": 100.0,
            "exit_price": 100.0 + pnls[i],
            "symbol": "BTCUSDT",
            "timeframe": "1h",
        })
    return trades


def features_df_with_regime(closes, base_ts=1_609_459_200_000, step=3_600_000):
    """Build a features DataFrame with rsi_14 + regime columns we control."""
    n = len(closes)
    df = pd.DataFrame({
        "timestamp": [base_ts + i * step for i in range(n)],
        "close": closes,
        "rsi_14": 50.0,
        "volume_ratio": 1.0,
        "atr_ratio": 0.02,
        "hist_vol_20": 0.5,
        "adx_14": 25.0,
        "dist_sma_50": 0.0,
        "roc_12": 0.0,
    })
    # Regime: first half bull, second half bear
    regime = np.where(np.arange(n) < n / 2, "bull", "bear")
    vol = np.where(np.arange(n) < n / 2, "normal_volatility", "low_volatility")
    df["regime"] = regime
    df["vol_regime"] = vol
    return df


class TestTradeAnalyzer:
    """Test the trade research engine."""

    def test_basic_counts(self):
        trades = make_trades(10, np.arange(10) * 3_600_000 + 1_609_459_200_000,
                             [1, 1, 1, 1, 1, 1, -1, -1, -1, -1])
        report = TradeAnalyzer().analyze(trades)
        assert report.n_total == 10
        assert report.n_winning == 6
        assert report.n_losing == 4
        assert report.win_rate == pytest.approx(0.6)

    def test_empty_trades(self):
        report = TradeAnalyzer().analyze([])
        assert report.n_total == 0
        assert report.feature_findings == []

    def test_rsi_finding_detected(self):
        """Winners have high RSI, losers low RSI -> finding should fire."""
        base = 1_609_459_200_000
        n = 100
        rng = np.random.default_rng(0)
        entry_times = np.arange(n) * 3_600_000 + base

        # Winners at entries with rsi ~65, losers at entries with rsi ~35
        # (add small noise so each group has variance for the t-test)
        trades = []
        for i in range(n):
            is_win = i % 2 == 0
            trades.append({
                "entry_time": int(entry_times[i]),
                "net_pnl": 5.0 if is_win else -5.0,
                "direction": "long",
            })

        closes = np.full(n + 20, 100.0)
        fdf = features_df_with_regime(closes)
        rsi_vals = [65.0 + rng.normal(0, 2) if i % 2 == 0 else 35.0 + rng.normal(0, 2)
                    for i in range(n)]
        fdf.loc[:n - 1, "rsi_14"] = rsi_vals

        report = TradeAnalyzer().analyze(trades, fdf)
        rsi_findings = [f for f in report.feature_findings if f.feature == "rsi_14"]
        assert len(rsi_findings) == 1
        f = rsi_findings[0]
        assert f.winners_mean > f.losers_mean
        assert f.effect.meaningful is True

    def test_no_finding_when_identical(self):
        """Identical winner/loser features -> no meaningful finding."""
        base = 1_609_459_200_000
        n = 80
        entry_times = np.arange(n) * 3_600_000 + base
        trades = [
            {"entry_time": int(entry_times[i]), "net_pnl": 5.0 if i % 2 == 0 else -5.0}
            for i in range(n)
        ]
        fdf = features_df_with_regime(np.full(n + 20, 100.0))
        fdf.loc[:n - 1, "rsi_14"] = 50.0  # identical

        report = TradeAnalyzer().analyze(trades, fdf)
        rsi_findings = [f for f in report.feature_findings if f.feature == "rsi_14"]
        assert rsi_findings == []

    def test_direction_distribution(self):
        """If winners are mostly long and losers mostly short, category finding shows it."""
        n = 60
        entry_times = np.arange(n) * 3_600_000 + 1_609_459_200_000
        pnls = [5.0] * 30 + [-5.0] * 30
        dirs = ["long"] * 30 + ["short"] * 30
        trades = make_trades(n, entry_times, pnls, dirs)

        report = TradeAnalyzer().analyze(trades)
        direction_finding = [c for c in report.category_findings if c.dimension == "direction"]
        assert len(direction_finding) == 1
        assert direction_finding[0].top_winners == "long"
        assert direction_finding[0].top_losers == "short"

    def test_regime_breakdown(self):
        """Win rate per regime is computed from context."""
        base = 1_609_459_200_000
        n = 60
        entry_times = np.arange(n) * 3_600_000 + base
        # Build features with EXACTLY n rows: first 30 bull, next 30 bear
        fdf = pd.DataFrame({
            "timestamp": [base + i * 3_600_000 for i in range(n)],
            "close": 100.0,
            "rsi_14": 50.0, "volume_ratio": 1.0, "atr_ratio": 0.02,
            "hist_vol_20": 0.5, "adx_14": 25.0, "dist_sma_50": 0.0, "roc_12": 0.0,
        })
        fdf["regime"] = ["bull"] * 30 + ["bear"] * 30
        fdf["vol_regime"] = ["normal_volatility"] * 30 + ["low_volatility"] * 30
        # First 30 trades win (bull), next 30 lose (bear)
        trades = [
            {"entry_time": int(entry_times[i]), "net_pnl": 5.0 if i < 30 else -5.0}
            for i in range(n)
        ]
        report = TradeAnalyzer().analyze(trades, fdf)
        assert report.regime_breakdown["bull"]["win_rate"] == pytest.approx(1.0)
        assert report.regime_breakdown["bear"]["win_rate"] == pytest.approx(0.0)
        assert report.regime_breakdown["bull"]["trades"] == 30

    def test_summarize(self):
        base = 1_609_459_200_000
        n = 60
        entry_times = np.arange(n) * 3_600_000 + base
        trades = [
            {"entry_time": int(entry_times[i]), "net_pnl": 5.0 if i % 2 == 0 else -5.0}
            for i in range(n)
        ]
        report = TradeAnalyzer().analyze(trades)
        lines = TradeAnalyzer().summarize(report)
        assert isinstance(lines, list)
