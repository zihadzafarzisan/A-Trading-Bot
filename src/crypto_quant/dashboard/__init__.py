"""Dashboard generation package."""

from .charts import (
    line_chart, bar_chart, histogram, donut_chart, sparkline,
)
from .generator import DashboardData, DashboardGenerator
from .server import serve_dashboard, DashboardServer


def __getattr__(name: str):
    # Lazy import to avoid runtime warnings when running
    # `python -m src.crypto_quant.dashboard.live_server`.
    if name == "run_live_dashboard":
        from .live_server import run_live_dashboard
        return run_live_dashboard
    raise AttributeError(name)


__all__ = [
    "line_chart", "bar_chart", "histogram", "donut_chart", "sparkline",
    "DashboardData", "DashboardGenerator",
    "serve_dashboard", "DashboardServer",
]