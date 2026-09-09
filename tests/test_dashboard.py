"""Tests for the HTML dashboard generator."""

import numpy as np
import pandas as pd
import pytest

from crypto_quant.dashboard import DashboardData, DashboardGenerator
from crypto_quant.dashboard.charts import (
    line_chart, bar_chart, histogram, donut_chart, sparkline,
)
from crypto_quant.strategies import TrendStrategy
from crypto_quant.backtesting import BacktestEngine, BacktestConfig, ExecutionConfig


def make_df(n=500):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    return pd.DataFrame({
        "timestamp": [1609459200000 + i * 3_600_000 for i in range(n)],
        "open": np.roll(close, 1), "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": 1000.0,
    })


class TestCharts:
    """Test SVG chart generators."""

    def test_line_chart_svg(self):
        svg = line_chart([1, 2, 3], [1.0, 2.0, 1.5])
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert 'viewBox="0 0' in svg

    def test_line_chart_empty(self):
        svg = line_chart([], [])
        assert "<svg" in svg

    def test_bar_chart(self):
        svg = bar_chart(["a", "b"], [1.0, 2.0])
        assert "<rect" in svg

    def test_histogram(self):
        svg = histogram([1, 2, 2, 3, 3, 3, 4, 5])
        assert "<rect" in svg

    def test_donut_chart(self):
        svg = donut_chart(["wins", "losses"], [10, 5])
        assert "<path" in svg
        assert "10" in svg

    def test_donut_chart_zero(self):
        svg = donut_chart(["wins", "losses"], [0, 0])
        assert "<svg" in svg

    def test_sparkline(self):
        svg = sparkline([1.0, 2.0, 3.0])
        assert "<polyline" in svg

    def test_escapes_html(self):
        svg = bar_chart(["<script>"], [1.0])
        assert "<script>" not in svg


class TestDashboardGenerator:
    """Test the full dashboard generation."""

    def test_generates_file(self, tmp_path):
        data = DashboardData(title="Test Dashboard", metrics={"win_rate": 0.6})
        out = tmp_path / "dash.html"
        DashboardGenerator().generate(data, out)
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Test Dashboard" in content

    def test_self_contained_no_external(self, tmp_path):
        """The dashboard must be fully local: no http/CDN references."""
        data = DashboardData(metrics={"win_rate": 0.6})
        out = tmp_path / "dash.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "http://" not in content
        assert "https://" not in content
        assert "cdn" not in content.lower()

    def test_from_backtest_full(self, tmp_path):
        df = make_df()
        result = BacktestEngine(BacktestConfig(min_bars=30)).run(TrendStrategy(), df, symbol="BTCUSDT")
        data = DashboardGenerator.from_backtest(result, symbol="BTCUSDT", timeframe="1h")
        out = tmp_path / "full.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        # Sections present
        assert "Win Rate" in content
        assert "Equity Curve" in content
        assert "Trade History" in content
        assert "backtest" in content.lower() or "BTCUSDT" in content

    def test_from_backtest_walk_forward(self, tmp_path):
        data = DashboardData(
            metrics={"win_rate": 0.6},
            walk_forward={
                "windows": [{
                    "window": {"train_start": "2020-01-01", "train_end": "2022-12-31",
                               "test_start": "2023-01-01", "test_end": "2023-12-31"},
                    "in_sample": {"win_rate": 0.5, "net_return": 0.1},
                    "out_of_sample": {"win_rate": 0.4, "net_return": -0.05},
                    "win_rate_degradation": -0.1,
                    "total_trades": 20,
                }],
                "mean_oos_win_rate": 0.4,
            },
        )
        out = tmp_path / "wf.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Walk-Forward Validation" in content
        assert "2020-2022" in content

    def test_ml_section(self, tmp_path):
        data = DashboardData(
            metrics={},
            ml={
                "evaluation": {
                    "baseline_win_rate": 0.55,
                    "ml_filtered_win_rate": 0.68,
                    "ml_improves": True,
                    "test_metrics": {"auc": 0.72, "accuracy": 0.61},
                    "feature_importance": {"rsi_14": 0.4, "volume_ratio": 0.3},
                }
            },
        )
        out = tmp_path / "ml.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Machine Learning" in content
        assert "IMPROVES" in content
        assert "Top ML Features" in content

    def test_risk_events_section(self, tmp_path):
        data = DashboardData(
            metrics={},
            risk_events=[{
                "timestamp": "2026-09-08T00:00:00+00:00",
                "event_type": "max_positions", "severity": "warning",
                "message": "Rejected: max positions reached",
            }],
        )
        out = tmp_path / "risk.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Risk Events" in content
        assert "max_positions" in content

    def test_monte_carlo_section(self, tmp_path):
        data = DashboardData(
            metrics={},
            monte_carlo={
                "final_equity_percentiles": {"p5": 900, "p50": 1100, "p95": 1400},
                "max_drawdown_percentiles": {"p5": 0.1, "p50": 0.2, "p95": 0.4},
                "probability_of_ruin": 0.02,
            },
        )
        out = tmp_path / "mc.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Monte Carlo" in content
        assert "not a prediction" in content

    def test_ranked_strategies(self, tmp_path):
        data = DashboardData(
            metrics={"win_rate": 0.6},
            experiment_id="EXP-TEST-1",
            ranked_strategies=[{
                "rank": 1, "strategy_type": "trend", "params": {"ema_fast": 20},
                "metrics": {"win_rate": 0.6, "profit_factor": 1.5, "net_return": 0.2,
                            "max_drawdown": 0.1, "total_trades": 100},
                "passed_filters": True, "filter_reasons": [],
            }],
        )
        out = tmp_path / "strat.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "Strategy Ranking" in content
        assert "QUALIFIED" in content

    def test_warnings_section(self, tmp_path):
        data = DashboardData(metrics={}, warnings=["HIGH DRAWDOWN", "LOW SAMPLE SIZE"])
        out = tmp_path / "warn.html"
        DashboardGenerator().generate(data, out)
        content = out.read_text(encoding="utf-8")
        assert "HIGH DRAWDOWN" in content
        assert "does not guarantee future results" in content