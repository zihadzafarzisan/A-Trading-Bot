"""Tests for dashboard server and enhanced dashboard features."""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crypto_quant.dashboard import (
    DashboardData,
    DashboardGenerator,
    serve_dashboard,
)
from crypto_quant.dashboard.charts import (
    bar_chart,
    donut_chart,
    histogram,
    line_chart,
    sparkline,
)


class TestDashboardGenerator:
    """Test the enhanced dashboard generator."""

    def test_empty_dashboard(self, tmp_path):
        """Dashboard with no data generates valid HTML."""
        gen = DashboardGenerator()
        data = DashboardData()
        out = gen.generate(data, tmp_path / "empty.html")
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "Crypto Quant Research Dashboard" in html

    def test_overview_metrics(self, tmp_path):
        """Overview section renders all metric cards."""
        gen = DashboardGenerator()
        data = DashboardData(
            metrics={
                "win_rate": 0.62,
                "profit_factor": 1.85,
                "net_return": 0.34,
                "max_drawdown": -0.12,
                "sharpe_ratio": 1.4,
                "sortino_ratio": 2.1,
                "total_trades": 120,
                "expectancy": 45.50,
            }
        )
        out = gen.generate(data, tmp_path / "overview.html")
        html = out.read_text(encoding="utf-8")
        assert "Win Rate" in html
        assert "62.0%" in html
        assert "Profit Factor" in html
        assert "Sharpe" in html
        assert "120" in html

    def test_equity_curve(self, tmp_path):
        """Equity curve and drawdown charts render."""
        gen = DashboardGenerator()
        data = DashboardData(
            equity_curve=[
                {"time": 1700000000000, "equity": 1000},
                {"time": 1700003600000, "equity": 1020},
                {"time": 1700007200000, "equity": 990},
                {"time": 1700010800000, "equity": 1050},
            ]
        )
        out = gen.generate(data, tmp_path / "equity.html")
        html = out.read_text(encoding="utf-8")
        assert "Equity Curve" in html
        assert "Drawdown" in html
        assert "polyline" in html  # SVG chart

    def test_trade_history(self, tmp_path):
        """Trade history table renders."""
        gen = DashboardGenerator()
        trades = [
            {
                "symbol": "BTCUSDT",
                "direction": "long",
                "timeframe": "1h",
                "entry_time": 1700000000000,
                "exit_time": 1700003600000,
                "entry_price": 42000,
                "exit_price": 42500,
                "net_pnl": 50,
                "exit_reason": "take_profit",
            },
            {
                "symbol": "ETHUSDT",
                "direction": "short",
                "timeframe": "4h",
                "entry_time": 1700007200000,
                "exit_time": 1700010800000,
                "entry_price": 2200,
                "exit_price": 2250,
                "net_pnl": -30,
                "exit_reason": "stop_loss",
            },
        ]
        data = DashboardData(trades=trades)
        out = gen.generate(data, tmp_path / "trades.html")
        html = out.read_text(encoding="utf-8")
        assert "Trade History" in html
        assert "BTCUSDT" in html
        assert "ETHUSDT" in html
        assert "filterTrades" in html  # JavaScript filter
        assert "filter-symbol" in html

    def test_trade_filters(self, tmp_path):
        """Trade filters are rendered correctly."""
        gen = DashboardGenerator()
        trades = [
            {"symbol": "BTCUSDT", "direction": "long", "timeframe": "1h",
             "entry_price": 42000, "exit_price": 42500, "net_pnl": 50, "exit_reason": "take_profit"},
            {"symbol": "ETHUSDT", "direction": "short", "timeframe": "4h",
             "entry_price": 2200, "exit_price": 2250, "net_pnl": -30, "exit_reason": "stop_loss"},
        ]
        data = DashboardData(trades=trades)
        out = gen.generate(data, tmp_path / "filters.html")
        html = out.read_text(encoding="utf-8")
        assert "filter-symbol" in html
        assert "filter-direction" in html
        assert "filter-timeframe" in html
        assert "filter-outcome" in html
        assert "resetFilters" in html

    def test_coin_comparison(self, tmp_path):
        """Coin comparison section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            coin_comparison={
                "coins": [
                    {"symbol": "BTCUSDT", "metrics": {
                        "win_rate": 0.65, "profit_factor": 2.0,
                        "net_return": 0.25, "max_drawdown": -0.10, "total_trades": 50
                    }},
                    {"symbol": "ETHUSDT", "metrics": {
                        "win_rate": 0.58, "profit_factor": 1.5,
                        "net_return": 0.15, "max_drawdown": -0.15, "total_trades": 45
                    }},
                ]
            }
        )
        out = gen.generate(data, tmp_path / "coins.html")
        html = out.read_text(encoding="utf-8")
        assert "Coin Performance Comparison" in html
        assert "BTCUSDT" in html
        assert "ETHUSDT" in html

    def test_spot_vs_futures(self, tmp_path):
        """Spot vs futures comparison renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            spot_vs_futures={
                "spot": {
                    "win_rate": 0.62, "profit_factor": 1.8,
                    "net_return": 0.20, "max_drawdown": -0.12,
                    "sharpe_ratio": 1.4, "total_trades": 100
                },
                "futures": {
                    "win_rate": 0.58, "profit_factor": 1.5,
                    "net_return": 0.15, "max_drawdown": -0.18,
                    "sharpe_ratio": 1.1, "total_trades": 80
                },
                "long_vs_short": {
                    "long": {"win_rate": 0.65, "profit_factor": 2.0, "net_return": 0.25, "total_trades": 50},
                    "short": {"win_rate": 0.52, "profit_factor": 1.2, "net_return": 0.05, "total_trades": 30},
                }
            }
        )
        out = gen.generate(data, tmp_path / "svf.html")
        html = out.read_text(encoding="utf-8")
        assert "Spot vs Futures Comparison" in html
        assert "Spot" in html
        assert "Futures" in html
        assert "Long vs Short" in html

    def test_walk_forward(self, tmp_path):
        """Walk-forward validation section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            walk_forward={
                "windows": [
                    {
                        "window": {"train_start": "2020-01-01", "train_end": "2021-01-01",
                                    "test_start": "2021-01-01", "test_end": "2021-06-01"},
                        "in_sample": {"win_rate": 0.65, "net_return": 0.25},
                        "out_of_sample": {"win_rate": 0.58, "net_return": 0.15, "total_trades": 30},
                        "win_rate_degradation": -0.07,
                    }
                ],
                "mean_oos_win_rate": 0.58
            }
        )
        out = gen.generate(data, tmp_path / "wf.html")
        html = out.read_text(encoding="utf-8")
        assert "Walk-Forward Validation" in html
        assert "2020" in html
        assert "58.0%" in html

    def test_monte_carlo(self, tmp_path):
        """Monte Carlo section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            monte_carlo={
                "final_equity_percentiles": {"p5": 800, "p25": 1100, "p50": 1300, "p75": 1600, "p95": 2000},
                "max_drawdown_percentiles": {"p5": -0.05, "p25": -0.10, "p50": -0.15, "p75": -0.20, "p95": -0.30},
                "probability_of_ruin": 0.02
            }
        )
        out = gen.generate(data, tmp_path / "mc.html")
        html = out.read_text(encoding="utf-8")
        assert "Monte Carlo" in html
        assert "Probability of ruin" in html
        assert "2.0%" in html

    def test_ml_section(self, tmp_path):
        """ML section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            ml={
                "evaluation": {
                    "baseline_win_rate": 0.55,
                    "ml_filtered_win_rate": 0.65,
                    "ml_improves": True,
                    "test_metrics": {"auc": 0.72, "accuracy": 0.68},
                    "feature_importance": {"rsi": 0.25, "ema_dist": 0.20, "atr": 0.15}
                }
            }
        )
        out = gen.generate(data, tmp_path / "ml.html")
        html = out.read_text(encoding="utf-8")
        assert "Machine Learning" in html
        assert "55.0%" in html
        assert "65.0%" in html
        assert "IMPROVES" in html

    def test_risk_events(self, tmp_path):
        """Risk events section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            risk_events=[
                {"timestamp": 1700000000000, "event_type": "max_drawdown", "severity": "critical",
                 "message": "Max drawdown exceeded"},
                {"timestamp": 1700003600000, "event_type": "daily_loss", "severity": "warning",
                 "message": "Daily loss limit approaching"},
            ]
        )
        out = gen.generate(data, tmp_path / "risk.html")
        html = out.read_text(encoding="utf-8")
        assert "Risk Events" in html
        assert "max_drawdown" in html
        assert "CRITICAL" in html
        assert "WARNING" in html

    def test_warnings(self, tmp_path):
        """Warnings section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            warnings=["Low trade count", "High drawdown detected"]
        )
        out = gen.generate(data, tmp_path / "warnings.html")
        html = out.read_text(encoding="utf-8")
        assert "Warnings" in html
        assert "Low trade count" in html
        assert "High drawdown detected" in html

    def test_strategy_ranking(self, tmp_path):
        """Strategy ranking table renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            ranked_strategies=[
                {
                    "rank": 1,
                    "strategy_type": "trend",
                    "params": {"ema_fast": 20, "ema_slow": 50},
                    "metrics": {"win_rate": 0.62, "profit_factor": 1.8, "net_return": 0.25,
                                "max_drawdown": -0.12, "total_trades": 50},
                    "passed_filters": True,
                    "filter_reasons": [],
                },
                {
                    "rank": 2,
                    "strategy_type": "momentum",
                    "params": {"rsi_period": 14},
                    "metrics": {"win_rate": 0.55, "profit_factor": 1.3, "net_return": 0.15,
                                "max_drawdown": -0.20, "total_trades": 40},
                    "passed_filters": False,
                    "filter_reasons": ["Max drawdown too high"],
                },
            ]
        )
        out = gen.generate(data, tmp_path / "strategies.html")
        html = out.read_text(encoding="utf-8")
        assert "Strategy Ranking" in html
        assert "trend" in html
        assert "momentum" in html
        assert "QUALIFIED" in html
        assert "rejected" in html

    def test_regime_breakdown(self, tmp_path):
        """Regime breakdown section renders."""
        gen = DashboardGenerator()
        data = DashboardData(
            regime_breakdown={
                "bull_trend": {"win_rate": 0.68, "trades": 45},
                "bear_trend": {"win_rate": 0.45, "trades": 30},
                "sideways": {"win_rate": 0.52, "trades": 25},
            }
        )
        out = gen.generate(data, tmp_path / "regimes.html")
        html = out.read_text(encoding="utf-8")
        assert "Regime Breakdown" in html
        assert "bull_trend" in html
        assert "bear_trend" in html

    def test_self_contained_no_external(self, tmp_path):
        """Dashboard is self-contained with no external dependencies."""
        gen = DashboardGenerator()
        data = DashboardData(
            metrics={"win_rate": 0.6},
            equity_curve=[{"time": 1700000000000, "equity": 1000}],
        )
        out = gen.generate(data, tmp_path / "self_contained.html")
        html = out.read_text(encoding="utf-8")
        # No external CSS/JS (excluding xmlns XML namespace declarations)
        import re
        # Find all http/https URLs, but exclude standard XML namespace declarations
        external_urls = re.findall(r'(https?://[^\s"\'<>]+)', html)
        # Filter out standard XML namespace URLs that are required for SVG
        xml_namespaces = {'http://www.w3.org/2000/svg', 'http://www.w3.org/1999/xlink'}
        external_urls = [url for url in external_urls if url not in xml_namespaces]
        assert len(external_urls) == 0, f"Found external URLs: {external_urls}"
        # Inline SVG
        assert "<svg" in html

    def test_dark_theme(self, tmp_path):
        """Dashboard uses dark theme."""
        gen = DashboardGenerator()
        data = DashboardData()
        out = gen.generate(data, tmp_path / "theme.html")
        html = out.read_text(encoding="utf-8")
        assert "color-scheme: dark" in html
        assert "#0d1326" in html  # Dark background


class TestDashboardCharts:
    """Test chart rendering functions."""

    def test_line_chart_svg(self):
        """Line chart produces valid SVG."""
        svg = line_chart([1, 2, 3], [10, 20, 15])
        assert "<svg" in svg
        assert "polyline" in svg

    def test_line_chart_empty(self):
        """Empty line chart produces placeholder."""
        svg = line_chart([], [])
        assert "<svg" in svg
        assert "No data" in svg

    def test_bar_chart_svg(self):
        """Bar chart produces valid SVG."""
        svg = bar_chart(["A", "B", "C"], [10, 20, 15])
        assert "<svg" in svg
        assert "rect" in svg

    def test_histogram_svg(self):
        """Histogram produces valid SVG."""
        svg = histogram([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], bins=5)
        assert "<svg" in svg

    def test_donut_chart_svg(self):
        """Donut chart produces valid SVG."""
        svg = donut_chart(["A", "B"], [60, 40])
        assert "<svg" in svg
        assert "path" in svg

    def test_donut_chart_zero(self):
        """Zero-value donut chart produces placeholder."""
        svg = donut_chart(["A", "B"], [0, 0])
        assert "<svg" in svg

    def test_sparkline_svg(self):
        """Sparkline produces valid SVG."""
        svg = sparkline([1, 2, 3, 4, 5])
        assert "<svg" in svg
        assert "polyline" in svg

    def test_sparkline_empty(self):
        """Empty sparkline produces empty string."""
        svg = sparkline([])
        assert svg == ""

    def test_html_escaping(self):
        """Chart titles are HTML-escaped."""
        svg = line_chart([1], [1], title="<script>alert('xss')</script>")
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg


class TestDashboardServer:
    """Test the dashboard server module."""

    def test_server_import(self):
        """Server module imports correctly."""
        from crypto_quant.dashboard.server import DashboardServer, serve_dashboard
        assert DashboardServer is not None
        assert serve_dashboard is not None

    def test_server_default_port(self):
        """Server has correct default port."""
        from crypto_quant.dashboard.server import DashboardServer
        server = DashboardServer()
        assert server.port == 8000

    def test_server_custom_port(self):
        """Server accepts custom port."""
        from crypto_quant.dashboard.server import DashboardServer
        server = DashboardServer(port=9000)
        assert server.port == 9000

    def test_serve_dashboard_file(self, tmp_path):
        """serve_dashboard accepts file path."""
        from crypto_quant.dashboard.server import serve_dashboard
        # Create a test HTML file
        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>Test</body></html>", encoding="utf-8")

        # Just verify the function doesn't crash when called
        # We can't actually start a server in tests
        with patch("crypto_quant.dashboard.server.DashboardServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance
            serve_dashboard(html_file, port=8001, open_browser=False)
            mock_server.assert_called_once()
            mock_instance.start.assert_called_once()

    def test_serve_dashboard_directory(self, tmp_path):
        """serve_dashboard accepts directory path."""
        from crypto_quant.dashboard.server import serve_dashboard
        # Create a test directory with HTML
        (tmp_path / "index.html").write_text("<html>Test</html>", encoding="utf-8")

        with patch("crypto_quant.dashboard.server.DashboardServer") as mock_server:
            mock_instance = MagicMock()
            mock_server.return_value = mock_instance
            serve_dashboard(tmp_path, port=8002, open_browser=False)
            mock_server.assert_called_once()
            mock_instance.start.assert_called_once()

    def _request_root(self, handler):
        """Spin up a server with the given handler and GET /, returning (status, body)."""
        import http.server
        import threading

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            import urllib.request
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

    def test_root_serves_fallback_html_when_no_index(self, tmp_path):
        """GET / returns 200 serving an existing report when index.html is absent."""
        from crypto_quant.dashboard.server import DashboardServer
        # Put only report.html in the directory (no index.html) — mimics `serve dashboard`
        report = tmp_path / "report.html"
        report.write_text("<html><body>EXP-FALLBACK-CONTENT</body></html>", encoding="utf-8")

        server = DashboardServer(directory=tmp_path, open_browser=False)
        handler = server._create_handler()
        status, body = self._request_root(handler)
        assert status == 200
        assert "EXP-FALLBACK-CONTENT" in body

    def test_root_prefers_index_html_when_present(self, tmp_path):
        """GET / serves index.html over report.html when both exist."""
        from crypto_quant.dashboard.server import DashboardServer
        (tmp_path / "index.html").write_text("<html><body>INDEX</body></html>", encoding="utf-8")
        (tmp_path / "report.html").write_text("<html><body>REPORT</body></html>", encoding="utf-8")

        server = DashboardServer(directory=tmp_path, open_browser=False)
        handler = server._create_handler()
        status, body = self._request_root(handler)
        assert status == 200
        assert "INDEX" in body
        assert "REPORT" not in body


class TestDashboardIntegration:
    """Integration tests for the dashboard pipeline."""

    def test_from_backtest(self):
        """DashboardData.from_backtest creates valid data."""
        # Create a mock backtest result
        mock_result = MagicMock()
        mock_result.metrics.to_dict.return_value = {
            "win_rate": 0.6,
            "profit_factor": 1.5,
            "net_return": 0.2,
            "max_drawdown": -0.1,
            "sharpe_ratio": 1.2,
            "total_trades": 50,
        }
        mock_result.equity_curve = [{"time": 1700000000000, "equity": 1000}]
        mock_result.trades = [{"symbol": "BTCUSDT", "net_pnl": 100}]
        mock_result.experiment_id = "EXP-001"
        mock_result.warnings = ["Test warning"]

        data = DashboardGenerator.from_backtest(
            mock_result, symbol="BTCUSDT", timeframe="1h", market_type="spot"
        )

        assert data.title == "Backtest — BTCUSDT / 1h (spot)"
        assert data.metrics["win_rate"] == 0.6
        assert len(data.equity_curve) == 1
        assert len(data.trades) == 1
        # Note: from_backtest uses experiment_id from the result, but it's not stored in DashboardData
        # The experiment_id is used in the title instead
        assert "BTCUSDT" in data.title

    def test_full_pipeline(self, tmp_path):
        """Full dashboard generation pipeline."""
        gen = DashboardGenerator()
        data = DashboardData(
            title="Full Pipeline Test",
            subtitle="Integration test",
            metrics={"win_rate": 0.62, "profit_factor": 1.8},
            equity_curve=[
                {"time": 1700000000000, "equity": 1000},
                {"time": 1700003600000, "equity": 1050},
            ],
            trades=[
                {"symbol": "BTCUSDT", "direction": "long", "timeframe": "1h",
                 "entry_price": 42000, "exit_price": 42500, "net_pnl": 50,
                 "exit_reason": "take_profit"}
            ],
            regime_breakdown={"bull": {"win_rate": 0.65, "trades": 20}},
            ranked_strategies=[{"rank": 1, "strategy_type": "trend",
                               "metrics": {"win_rate": 0.6}, "passed_filters": True}],
            walk_forward={"windows": [{"window": {"train_start": "2020-01-01", "train_end": "2021-01-01",
                                                   "test_start": "2021-01-01", "test_end": "2021-06-01"},
                                       "in_sample": {"win_rate": 0.65},
                                       "out_of_sample": {"win_rate": 0.58, "total_trades": 30},
                                       "win_rate_degradation": -0.07}]},
            monte_carlo={"final_equity_percentiles": {"p50": 1300},
                         "max_drawdown_percentiles": {"p50": -0.15},
                         "probability_of_ruin": 0.02},
            ml={"evaluation": {"baseline_win_rate": 0.55, "ml_filtered_win_rate": 0.65,
                               "ml_improves": True}},
            coin_comparison={"coins": [{"symbol": "BTCUSDT",
                                        "metrics": {"win_rate": 0.65, "profit_factor": 2.0,
                                                    "net_return": 0.25, "max_drawdown": -0.10,
                                                    "total_trades": 50}}]},
            spot_vs_futures={"spot": {"win_rate": 0.62}, "futures": {"win_rate": 0.58}},
            risk_events=[{"timestamp": 1700000000000, "event_type": "test", "severity": "warning",
                         "message": "Test event"}],
            warnings=["Test warning"],
        )

        out = gen.generate(data, tmp_path / "full_pipeline.html")
        html = out.read_text(encoding="utf-8")

        # Verify all sections
        assert "Full Pipeline Test" in html
        assert "62.0%" in html  # Win rate
        assert "Equity Curve" in html
        assert "Trade History" in html
        assert "Regime Breakdown" in html
        assert "Strategy Ranking" in html
        assert "Walk-Forward" in html
        assert "Monte Carlo" in html
        assert "Machine Learning" in html
        assert "Coin Performance" in html
        assert "Spot vs Futures" in html
        assert "Risk Events" in html
        assert "Warnings" in html

        # Verify file size is reasonable (not too small or too large)
        file_size = out.stat().st_size
        assert 1000 < file_size < 500000  # 1KB to 500KB
