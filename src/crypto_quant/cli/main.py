"""Main CLI application for Crypto Quant Terminal."""

# Entry-point self-hydration: hydrate the root .env BEFORE any crypto_quant.*
# submodule below is imported. discord_dm.py / live_broker.py read
# DISCORD_BOT_TOKEN / DISCORD_USER_ID at notifier-construction time; without
# this the daemon booted with a bare environment and silently disabled Discord
# notifications. Zero-touch: no manual exports, --token flags, or auxiliary
# scripts needed to launch the harvester.
from dotenv import load_dotenv

load_dotenv()

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pathlib import Path

# Ensure UTF-8 output on Windows consoles (Rich / unicode glyphs)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..config.settings import get_config, AppConfig
from ..db.connection import init_database, get_db_manager
from ..logging_config import setup_logging, get_logger
from ..exchange.binance import BinanceAdapter
from ..data.downloader import HistoricalDownloader
from ..data.repository import MarketDataRepository
from ..strategies import create_strategy
from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
from ..research import DiscoveryEngine, DiscoveryConfig
from ..validation import WalkForwardValidator, MonteCarloSimulator
from ..features.engine import FeatureEngine
from ..ml import MLPipeline, TradePredictor, ModelTrainer
from ..dashboard import DashboardData, DashboardGenerator, serve_dashboard
from ..execution.paper import PaperBroker
from ..execution.engine import PaperTradingEngine, PaperTradingConfig, DataframePriceSource
from ..risk.manager import RiskManager
from ..risk.limits import RiskLimits
from ..db.models import ExecutionTrade
from ..utils.constants import DEFAULT_SYMBOLS, HISTORICAL_UNIVERSE

# Create Typer app
app = typer.Typer(
    name="crypto-quant",
    help="Crypto Quant Research Terminal - Quantitative Research & Trading System",
    add_completion=False,
)
carry_app = typer.Typer(help="Cash-and-carry funding harvester controls.", add_completion=False)
app.add_typer(carry_app, name="carry")
notify_app = typer.Typer(help="Discord DM notification controls.", add_completion=False)
app.add_typer(notify_app, name="notify")

console = Console()
logger = get_logger("trading")


def show_header():
    """Display application header."""
    header = """
╔════════════════════════════════════════════════════════════╗
║            CRYPTO QUANT RESEARCH TERMINAL                  ║
║               v0.1.0 - Foundation Phase                   ║
╚════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(header, style="bold cyan"))


def show_main_menu():
    """Display main menu."""
    menu_table = Table(title="Main Menu", show_header=True, header_style="bold magenta")
    menu_table.add_column("Option", style="cyan", width=8)
    menu_table.add_column("Description", style="white")

    menu_table.add_row("1", "Data Manager")
    menu_table.add_row("2", "Research Lab")
    menu_table.add_row("3", "Backtesting")
    menu_table.add_row("4", "Trade Analysis")
    menu_table.add_row("5", "Strategy Optimization")
    menu_table.add_row("6", "Machine Learning")
    menu_table.add_row("7", "Walk-Forward Testing")
    menu_table.add_row("8", "Monte Carlo Simulation")
    menu_table.add_row("9", "Strategy Ranking")
    menu_table.add_row("10", "Portfolio Research")
    menu_table.add_row("11", "Paper Trading")
    menu_table.add_row("12", "Live Trading")
    menu_table.add_row("13", "Generate HTML Dashboard")
    menu_table.add_row("14", "Serve Dashboard Locally")
    menu_table.add_row("15", "Configuration")
    menu_table.add_row("0", "Exit")

    console.print(menu_table)


@app.command()
def main(
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    log_level: str = typer.Option("INFO", "--log-level", "-l", help="Logging level"),
    init_db: bool = typer.Option(False, "--init-db", help="Initialize database"),
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
):
    """Crypto Quant Research Terminal."""
    if version:
        console.print("[bold green]Crypto Quant Terminal v0.1.0[/bold green]")
        raise typer.Exit()

    # Show header
    show_header()

    # Initialize logging
    setup_logging(log_level)
    logger = setup_logging(log_level).get_logger("application")
    logger.info("Starting Crypto Quant Terminal")

    # Load configuration
    config = get_config(config_path)
    logger.info(f"Loaded configuration from {config_path or 'default'}")

    # Initialize database if requested
    if init_db:
        console.print("[bold yellow]Initializing database...[/bold yellow]")
        db = init_database()
        console.print("[bold green]Database initialized successfully![/bold green]")

    # Main loop
    try:
        while True:
            show_main_menu()
            choice = Prompt.ask("\nSelect option", choices=[str(i) for i in range(15)])

            if choice == "0":
                console.print("[bold green]Goodbye![/bold green]")
                break

            handle_menu_choice(choice, config)

    except KeyboardInterrupt:
        console.print("\n[bold yellow]Interrupted by user[/bold yellow]")
        sys.exit(0)


def handle_menu_choice(choice: str, config: AppConfig):
    """Handle menu choice."""
    if choice == "1":
        console.print("[bold cyan]Data Manager - Coming in Phase 2[/bold cyan]")
    elif choice == "2":
        console.print("[bold cyan]Research Lab - Coming in Phase 7[/bold cyan]")
    elif choice == "3":
        console.print("[bold cyan]Backtesting - Coming in Phase 5[/bold cyan]")
    elif choice == "4":
        console.print("[bold cyan]Trade Analysis - Coming in Phase 6[/bold cyan]")
    elif choice == "5":
        console.print("[bold cyan]Strategy Optimization - Coming in Phase 7[/bold cyan]")
    elif choice == "6":
        console.print("[bold cyan]Machine Learning - Coming in Phase 9[/bold cyan]")
    elif choice == "7":
        console.print("[bold cyan]Walk-Forward Testing - Coming in Phase 8[/bold cyan]")
    elif choice == "8":
        console.print("[bold cyan]Monte Carlo Simulation - Coming in Phase 8[/bold cyan]")
    elif choice == "9":
        console.print("[bold cyan]Strategy Ranking - Coming in Phase 7[/bold cyan]")
    elif choice == "10":
        console.print("[bold cyan]Portfolio Research - Coming in Phase 10[/bold cyan]")
    elif choice == "11":
        _paper_menu(config)
    elif choice == "12":
        console.print("[bold cyan]Live Trading - Coming in Phase 13[/bold cyan]")
    elif choice == "13":
        console.print("[bold cyan]HTML Dashboard - Coming in Phase 11[/bold cyan]")
    elif choice == "14":
        serve_dashboard("dashboard", port=8000, open_browser=True)
    elif choice == "15":
        show_config_info(config)
    else:
        console.print("[bold red]Invalid option[/bold red]")


def show_config_info(config: AppConfig):
    """Display current configuration."""
    table = Table(title="Current Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Exchange", config.market.exchange)
    table.add_row("Spot Trading", str(config.market.spot))
    table.add_row("Futures Trading", str(config.market.futures))
    table.add_row("Starting Capital", f"${config.risk.starting_capital:,.2f}")
    table.add_row("Risk Per Trade", f"{config.risk.risk_per_trade*100:.1f}%")
    table.add_row("Max Positions", str(config.risk.max_open_positions))
    table.add_row("Max Leverage", f"{config.risk.max_leverage}x")
    table.add_row("Trading Mode", config.trading.mode)
    table.add_row("Data Start", config.data.start_date)
    table.add_row("Data End", config.data.end_date)

    console.print(table)


@app.command()
def data(
    action: str = typer.Argument(..., help="Action: download, status, validate"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="Symbol (e.g., BTCUSDT)"),
    timeframe: Optional[str] = typer.Option(None, "--timeframe", "-t", help="Timeframe (e.g., 1h)"),
    universe: Optional[str] = typer.Option(None, "--universe", "-u", help="Universe: top_10/top_20/btc_only"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    top_n: int = typer.Option(10, "--top", help="Number of symbols when using top_n universe"),
):
    """Data Manager commands."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                  DATA MANAGER[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    config = get_config()

    if action == "download":
        data_download(
            symbol=symbol, timeframe=timeframe, universe=universe,
            start=start, end=end, market=market, top_n=top_n, config=config,
        )
    elif action == "status":
        data_status(market=market, config=config)
    elif action == "validate":
        data_validate(symbol=symbol, timeframe=timeframe, market=market, config=config)
    else:
        console.print(f"[bold red]Unknown action: {action}[/bold red]")
        console.print("Actions: download, status, validate")


# ---------------------------------------------------------------------------
# Data Manager helpers
# ---------------------------------------------------------------------------
def _resolve_symbols(universe: str, symbol: Optional[str], top_n: int) -> list:
    """Resolve which symbols to operate on."""
    if symbol:
        return [symbol.upper()]
    if universe == "btc_only":
        return ["BTCUSDT"]
    if universe == "eth_only":
        return ["ETHUSDT"]
    if universe and universe.startswith("top_"):
        n = top_n or int(universe.split("_")[1])
        return DEFAULT_SYMBOLS[:n]
    if universe and universe in ("custom", None):
        pass
    return DEFAULT_SYMBOLS[:top_n]


def _parse_date_ms(date_str: Optional[str]) -> Optional[int]:
    """Parse a YYYY-MM-DD date into a UNIX ms timestamp (start of day UTC)."""
    if not date_str:
        return None
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        console.print(f"[bold red]Invalid date: {date_str} (expected YYYY-MM-DD)[/bold red]")
        raise typer.Abort()


def data_download(symbol, timeframe, universe, start, end, market, top_n, config):
    """Download historical market data for configured symbols/timeframes."""
    # Resolve date range (default: full research window)
    start_ms = _parse_date_ms(start) or _parse_date_ms(config.data.start_date)
    end_ms = _parse_date_ms(end) or _parse_date_ms(config.data.end_date)
    if not start_ms or not end_ms:
        console.print("[bold red]Invalid date range[/bold red]")
        raise typer.Abort()

    # Resolve timeframes
    timeframes = [timeframe] if timeframe else config.timeframes.primary

    # Resolve symbols
    symbols = _resolve_symbols(universe, symbol, top_n)

    # Display the ACTUAL requested period, not the config defaults
    from datetime import datetime, timezone
    start_display = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    end_display = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    console.print(f"[bold]Symbols:[/bold] {', '.join(symbols)}")
    console.print(f"[bold]Timeframes:[/bold] {', '.join(timeframes)}")
    console.print(f"[bold]Market:[/bold] {market}")
    console.print(f"[bold]Period:[/bold] {start_display} -> {end_display}")
    console.print(f"[bold]Total symbols:[/bold] {len(symbols)}")
    console.print("")

    # Setup
    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    adapter = BinanceAdapter(market_type=market, max_retries=5, backoff_factor=2.0)
    downloader = HistoricalDownloader(adapter, max_candles_per_request=1000, requests_per_second=10.0)

    logger = get_logger("data")
    logger.info("Starting data download: %d symbols x %d timeframes", len(symbols), len(timeframes))

    total_expected = len(symbols) * len(timeframes)
    done = 0
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading data...", total=total_expected)
            for sym in symbols:
                for tf in timeframes:
                    progress.update(task, description=f"Downloading {sym} {tf}...")
                    try:
                        summary = repo.ensure_data(
                            sym, tf, start_ms, end_ms, downloader, market_type=market
                        )
                        console.print(
                            f"  [green]✓[/green] {sym} {tf}: {summary['n_candles']:,} candles "
                            f"(inserted {summary['inserted']:,})"
                        )
                        if summary["had_errors"]:
                            console.print(f"  [yellow]⚠[/yellow] {sym} {tf}: had errors during download")
                        logger.info(
                            "Downloaded %s %s: %d candles (inserted %d)",
                            sym, tf, summary["n_candles"], summary["inserted"],
                        )
                    except Exception as exc:
                        console.print(f"  [bold red]✗[/bold red] {sym} {tf}: {exc}")
                        logger.error("Download failed %s %s: %s", sym, tf, exc)
                    done += 1
                    progress.update(task, advance=1)
    finally:
        adapter.close()

    console.print(f"\n[bold green]Download complete: {done}/{total_expected} symbol/timeframe combinations handled[/bold green]")


def data_status(market, config):
    """Display stored data summary."""
    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)

    datasets = repo.list_datasets(market_type=market)

    if not datasets:
        console.print("[yellow]No data stored yet. Run: python -m crypto_quant data download[/yellow]")
        return

    table = Table(title=f"Stored Datasets ({market})", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Timeframe", style="magenta")
    table.add_column("Candles", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")

    from datetime import datetime, timezone
    for d in sorted(datasets, key=lambda x: (x["symbol"], x["timeframe"])):
        lo = datetime.fromtimestamp(d["start"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if d["start"] else "-"
        hi = datetime.fromtimestamp(d["end"] / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if d["end"] else "-"
        table.add_row(d["symbol"], d["timeframe"], f"{d['n_candles']:,}", lo, hi)

    console.print(table)
    console.print(f"\n[bold]Total datasets:[/bold] {len(datasets)}")


def data_validate(symbol, timeframe, market, config):
    """Validate stored data and report issues."""
    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)

    datasets = repo.list_datasets(market_type=market)
    if symbol:
        datasets = [d for d in datasets if d["symbol"] == symbol.upper()]
    if timeframe:
        datasets = [d for d in datasets if d["timeframe"] == timeframe]

    if not datasets:
        console.print("[yellow]No datasets to validate.[/yellow]")
        return

    table = Table(title="Data Validation Report", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("TF", style="magenta")
    table.add_column("Candles", justify="right")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Warnings", justify="right", style="yellow")
    table.add_column("Status", style="green")

    for d in datasets:
        report = repo.validate(d["symbol"], d["timeframe"], market_type=market)
        status = "✓ CLEAN" if report.is_clean else "✗ ISSUES"
        style = "green" if report.is_clean else "red"
        table.add_row(
            d["symbol"], d["timeframe"], f"{d['n_candles']:,}",
            str(report.error_count), str(report.warning_count),
            f"[{style}]{status}[/{style}]",
        )

    console.print(table)


@app.command()
def research(
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Strategy types (comma-sep): trend,momentum,mean_reversion,breakout"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    search: str = typer.Option("grid", "--search", help="grid, random, or adaptive"),
    max_combinations: int = typer.Option(50, "--max-combinations", help="Hard cap on evaluations"),
    min_trades: int = typer.Option(50, "--min-trades", help="Minimum trades quality filter"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
):
    """Run the research lab (strategy discovery) on stored data."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                 RESEARCH LAB[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    strategy_types = [s.strip() for s in (strategy or "trend,momentum,breakout,mean_reversion").split(",")]

    # Load data
    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    config = get_config()
    start_ms = _parse_date_ms(start) or _parse_date_ms(config.data.start_date)
    end_ms = _parse_date_ms(end) or _parse_date_ms(config.data.end_date)
    df = repo.load(symbol, timeframe, start_ms, end_ms, market_type=market)
    if df is None or len(df) < 60:
        console.print(f"[bold red]Not enough data for {symbol} {timeframe}. Run data download first.[/bold red]")
        raise typer.Abort()

    console.print(f"[bold]Symbol:[/bold] {symbol} | [bold]TF:[/bold] {timeframe} | [bold]Market:[/bold] {market}")
    console.print(f"[bold]Strategies:[/bold] {', '.join(strategy_types)} | [bold]Search:[/bold] {search} (max {max_combinations}/type)")
    console.print(f"[bold]Loaded:[/bold] {len(df):,} candles\n")

    cfg = DiscoveryConfig(
        strategy_types=strategy_types,
        search_type=search,
        max_combinations=max_combinations,
        min_trades=min_trades,
        market_type=market,
        timeframe=timeframe,
    )
    engine = DiscoveryEngine(config=cfg)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Running discovery...", total=None)
        result = engine.discover(df, symbol=symbol, db=db, start_date=config.data.start_date, end_date=config.data.end_date)
        progress.update(task, completed=1, description="Discovery complete")

    console.print(f"\n[bold green]Experiment:[/bold green] {result.experiment_id}")
    console.print(f"[bold]Candidates tested:[/bold] {result.n_tested} | [bold]Passing filters:[/bold] {result.n_passed} | [bold]Duration:[/bold] {result.duration_seconds:.1f}s")

    table = Table(title="Top Qualified Strategies")
    table.add_column("Rank", justify="right", style="cyan")
    table.add_column("Strategy", style="magenta")
    table.add_column("Parameters", style="white")
    table.add_column("Win Rate", justify="right")
    table.add_column("P.F.", justify="right")
    table.add_column("Net Return", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Trades", justify="right")

    shown = 0
    for r in result.ranked:
        if not r.passed_filters:
            break
        m = r.metrics
        table.add_row(str(r.rank), r.strategy_type, str(r.params),
                      f"{m['win_rate']:.1%}", f"{m['profit_factor']:.2f}",
                      f"{m['net_return']:.1%}", f"{m['max_drawdown']:.1%}",
                      str(m['total_trades']))
        shown += 1
        if shown >= 10:
            break
    if shown:
        console.print(table)
    else:
        console.print("[yellow]No strategies passed the quality filters.[/yellow]")
        console.print("[yellow]Try looser filters (--min-trades, --max-drawdown via config) or different data.[/yellow]")


@app.command()
def backtest(
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type: trend/momentum/mean_reversion/breakout"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
):
    """Run a backtest on stored data."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                   BACKTESTING[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    try:
        strat_obj = create_strategy(strategy)
    except KeyError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Abort()

    console.print(f"[bold]Strategy:[/bold] {strat_obj.name} ({strategy})")
    console.print(f"[bold]Symbol:[/bold] {symbol}")
    console.print(f"[bold]Timeframe:[/bold] {timeframe}")
    console.print(f"[bold]Market:[/bold] {market}")

    # Load data from repository
    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)

    config = get_config()
    start_ms = _parse_date_ms(start) or _parse_date_ms(config.data.start_date)
    end_ms = _parse_date_ms(end) or _parse_date_ms(config.data.end_date)

    df = repo.load(symbol, timeframe, start_ms, end_ms, market_type=market)
    if df is None or len(df) < 30:
        console.print(
            f"[bold red]Not enough data for {symbol} {timeframe}. "
            f"Run: python -m crypto_quant data download --symbol {symbol} --timeframe {timeframe}[/bold red]"
        )
        raise typer.Abort()

    console.print(f"Loaded {len(df):,} candles from stored data\n")

    # Run backtest
    bc = BacktestConfig(
        initial_capital=config.risk.starting_capital,
        risk_per_trade=config.risk.risk_per_trade,
        max_open_positions=config.risk.max_open_positions,
        max_leverage=config.risk.max_leverage,
        market_type=market,
        timeframe=timeframe,
        execution=ExecutionConfig(market_type=market, slippage=config.fees.slippage),
    )
    engine = BacktestEngine(bc)
    result = engine.run(strat_obj, df, symbol=symbol)

    m = result.metrics
    t = m.trades

    # Results table
    table = Table(title=f"Backtest Results — {strat_obj.name} / {symbol} / {timeframe} ({market})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")
    table.add_row("Total Trades", str(t.total_trades))
    table.add_row("Win Rate", f"{t.win_rate:.1%}")
    table.add_row("Profit Factor", f"{t.profit_factor:.2f}")
    table.add_row("Net Return", f"{m.net_return:.2%}")
    table.add_row("Max Drawdown", f"{m.max_drawdown:.1%}")
    table.add_row("Sharpe Ratio", f"{m.sharpe_ratio:.2f}")
    table.add_row("Sortino Ratio", f"{m.sortino_ratio:.2f}")
    table.add_row("Expectancy", f"${t.expectancy:.2f}")
    table.add_row("Total Fees", f"${t.total_fees:.2f}")
    table.add_row("Total Funding", f"${t.total_funding:.2f}")
    table.add_row("Liquidations", str(t.liquidations))
    console.print(table)

    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow] " + ", ".join(f"[bold yellow]{w}[/bold yellow]" for w in result.warnings))


@app.command()
def validate(
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    ema_fast: int = typer.Option(20, "--ema-fast", help="EMA fast parameter"),
    ema_slow: int = typer.Option(50, "--ema-slow", help="EMA slow parameter"),
    reoptimize: bool = typer.Option(False, "--reoptimize", help="Re-optimize params per window"),
    monte_carlo: int = typer.Option(0, "--monte-carlo", help="Run N Monte Carlo simulations on the OOS trades"),
):
    """Run walk-forward validation (and optional Monte Carlo) on stored data."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]             WALK-FORWARD VALIDATION[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    config = get_config()
    start_ms = _parse_date_ms(config.data.start_date)
    end_ms = _parse_date_ms(config.data.end_date)
    df = repo.load(symbol, timeframe, start_ms, end_ms, market_type=market)
    if df is None or len(df) < 60:
        console.print(f"[bold red]Not enough data for {symbol} {timeframe}.[/bold red]")
        raise typer.Abort()

    console.print(f"Loaded {len(df):,} candles ({symbol} {timeframe})\n")

    params = {"ema_fast": ema_fast, "ema_slow": ema_slow} if strategy == "trend" else {}
    validator = WalkForwardValidator(reoptimize=reoptimize)
    report = validator.validate(df, strategy, params, symbol=symbol, market_type=market, timeframe=timeframe)

    table = Table(title=f"Walk-Forward Results — {strategy} / {symbol} / {timeframe}")
    table.add_column("Window", style="cyan")
    table.add_column("IS Win Rate", justify="right")
    table.add_column("OOS Win Rate", justify="right")
    table.add_column("IS Return", justify="right")
    table.add_column("OOS Return", justify="right")
    table.add_column("WR Degrad.", justify="right")
    table.add_column("OOS Trades", justify="right")

    for w in report.windows:
        win = w.window
        table.add_row(
            f"{win.train_start[:4]}-{win.train_end[:4]} -> {win.test_start[:4]}-{win.test_end[:4]}",
            f"{w.is_metrics.get('win_rate', 0):.1%}",
            f"{w.oos_metrics.get('win_rate', 0):.1%}",
            f"{w.is_metrics.get('net_return', 0):.1%}",
            f"{w.oos_metrics.get('net_return', 0):.1%}",
            f"{w.win_rate_degradation:+.1%}",
            str(w.oos_metrics.get('total_trades', 0)),
        )
    console.print(table)

    if report.mean_oos_win_rate is not None:
        console.print(f"\n[bold]Mean OOS win rate:[/bold] {report.mean_oos_win_rate:.1%}")
        console.print(f"[bold]Mean OOS return:[/bold] {report.mean_oos_return:.1%}")
        console.print(f"[bold]Mean win-rate degradation:[/bold] {report.mean_degradation:+.1%}")

    for warn in report.warnings():
        console.print(f"[bold yellow]⚠ {warn}[/bold yellow]")

    # Optional Monte Carlo on the last window's OOS trades
    if monte_carlo > 0 and report.windows:
        last = report.windows[-1]
        trades = last.oos_metrics
        # Re-run the backtest on the test slice to get per-trade returns
        test_df = df  # already the full series; recompute from trades is complex,
        # so run a direct backtest on the final test window
        from ..backtesting import BacktestConfig, BacktestEngine, ExecutionConfig
        from ..strategies import create_strategy
        from datetime import datetime, timezone
        w = report.windows[-1].window
        t = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        lo = pd.Timestamp(w.test_start, tz="UTC"); hi = pd.Timestamp(w.test_end, tz="UTC")
        test_slice = df[(t >= lo) & (t <= hi)]
        strat = create_strategy(strategy, params=params)
        result = BacktestEngine(BacktestConfig(market_type=market, timeframe=timeframe,
            execution=ExecutionConfig(market_type=market))).run(strat, test_slice, symbol=symbol)
        trade_returns = [tr["return_pct"] for tr in result.trades if tr["return_pct"] is not None]
        if len(trade_returns) >= 10:
            sim = MonteCarloSimulator(n_simulations=monte_carlo)
            mc = sim.simulate(trade_returns, initial_capital=config.risk.starting_capital)
            console.print(f"\n[bold]Monte Carlo ({mc.n_simulations:,} scenarios) on OOS trade returns:[/bold]")
            console.print(f"  Final equity p5/p50/p95: ${mc.final_equity_pctiles['p5']:,.0f} / "
                          f"${mc.final_equity_pctiles['p50']:,.0f} / ${mc.final_equity_pctiles['p95']:,.0f}")
            console.print(f"  Max drawdown p50: {mc.max_drawdown_pctiles['p50']:.1%} "
                          f"(p95: {mc.max_drawdown_pctiles['p95']:.1%})")
            console.print(f"  Probability of ruin: {mc.probability_of_ruin:.1%}")
            console.print("  [italic]Statistical scenario — not a prediction.[/italic]")
        else:
            console.print("[yellow]Too few OOS trades for Monte Carlo.[/yellow]")


@app.command()
def ml(
    action: str = typer.Argument(..., help="Action: train, predict"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    threshold: float = typer.Option(0.60, "--threshold", help="TAKE/SKIP threshold"),
    model_dir: str = typer.Option("models", "--model-dir", help="Model save directory"),
):
    """Run the ML pipeline on a backtest's trades."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]              MACHINE LEARNING[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    config = get_config()
    start_ms = _parse_date_ms(config.data.start_date)
    end_ms = _parse_date_ms(config.data.end_date)
    df = repo.load(symbol, timeframe, start_ms, end_ms, market_type=market)
    if df is None or len(df) < 60:
        console.print(f"[bold red]Not enough data for {symbol} {timeframe}.[/bold red]")
        raise typer.Abort()

    # Run the underlying backtest to generate trades
    strat_obj = create_strategy(strategy)
    result = BacktestEngine(BacktestConfig(
        market_type=market, timeframe=timeframe,
        execution=ExecutionConfig(market_type=market),
    )).run(strat_obj, df, symbol=symbol)
    if not result.trades:
        console.print("[bold red]No trades to train on.[/bold red]")
        raise typer.Abort()

    console.print(f"Backtest produced {len(result.trades)} trades\n")

    # Features for ML
    feats = FeatureEngine().compute(df)

    if action == "train":
        report = MLPipeline(threshold=threshold, save_dir=model_dir).run(
            result.trades, feats, symbol=symbol, timeframe=timeframe, strategy_type=strategy,
        )
        ev = report.evaluation
        console.print(f"[bold]Samples:[/bold] {report.n_samples} | [bold]Baseline win rate:[/bold] {report.baseline_win_rate:.1%}")
        console.print(f"[bold]Best model:[/bold] {report.training.best_model_name}")
        table = Table(title="Model Comparison (validation)")
        table.add_column("Model", style="cyan")
        table.add_column("Val AUC", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Precision", justify="right")
        table.add_column("Recall", justify="right")
        for name, m in report.training.val_metrics.items():
            table.add_row(name, f"{m['auc']:.3f}", f"{m['accuracy']:.3f}",
                          f"{m['precision']:.3f}", f"{m['recall']:.3f}")
        console.print(table)

        console.print(f"\n[bold]Test set:[/bold] AUC {ev.test_metrics['auc']:.3f} | Acc {ev.test_metrics['accuracy']:.3f}")
        if ev.ml_filtered_win_rate is not None:
            verdict = "[green]IMPROVES[/green]" if ev.ml_improves else "[red]does NOT improve[/red]"
            console.print(f"ML-filtered win rate: {ev.ml_filtered_win_rate:.1%} vs baseline {ev.baseline_win_rate:.1%} "
                          f"-> ML {verdict} the strategy")
        else:
            console.print(f"[yellow]Too few takes at threshold {ev.threshold} — no improvement claim.[/yellow]")
        for w in report.warnings:
            console.print(f"[bold yellow]⚠ {w}[/bold yellow]")
    elif action == "predict":
        # Load the best saved model
        import glob
        saved = sorted(glob.glob(f"{model_dir}/*.joblib"))
        if not saved:
            console.print("[bold red]No trained model found. Run: python -m crypto_quant ml train[/bold red]")
            raise typer.Abort()
        loaded = ModelTrainer.load(saved[0])
        predictor = TradePredictor(loaded, default_threshold=threshold)
        # Predict on the last bar's features
        last = feats.iloc[-1]
        feats_dict = {c: (float(last[c]) if c in feats.columns else 0.0) for c in loaded.feature_names}
        pred = predictor.predict(feats_dict, direction="long")
        console.print(f"\n[bold]Trade Signal:[/bold] LONG")
        console.print(f"[bold]ML Probability:[/bold] {pred.probability:.3f}")
        color = "green" if pred.decision == "TAKE" else "red"
        console.print(f"[bold]Decision:[/bold] [{color}]{pred.decision}[/{color}] (threshold {pred.threshold})")
    else:
        console.print(f"[bold red]Unknown action: {action} (train, predict)[/bold red]")


@app.command()
def dashboard(
    action: str = typer.Argument(..., help="Action: generate, carry"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    experiment_id: Optional[str] = typer.Option(None, "--experiment", "-e", help="Experiment ID (from research run)"),
    output: str = typer.Option("dashboard/report.html", "--output", "-o", help="Output HTML path"),
    open_in_browser: bool = typer.Option(False, "--open", help="Open in browser after generating"),
    carry: bool = typer.Option(False, "--carry", "-c", help="Generate Cash-and-Carry strategy dashboard"),
):
    """Generate the local HTML dashboard."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                 HTML DASHBOARD[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    config = get_config()

    if carry:
        data = DashboardGenerator.from_carry(db)
        path = DashboardGenerator().generate(data, output)
        console.print(f"[green]✓[/green] Cash-and-Carry Dashboard written to {path}")
        if open_in_browser or action == "open":
            import webbrowser
            webbrowser.open(Path(path).resolve().as_uri())
        return

    if action == "carry":
        data = DashboardGenerator.from_carry(db)
        path = DashboardGenerator().generate(data, output)
        console.print(f"[green]✓[/green] Cash-and-carry dashboard written to {path}")
    # If an experiment ID is given, render its ranked strategies
    elif experiment_id:
        from ..research import ExperimentTracker
        tracker = ExperimentTracker(db)
        exp = tracker.get_experiment(experiment_id)
        if exp is None:
            console.print(f"[bold red]Experiment {experiment_id} not found.[/bold red]")
            raise typer.Abort()
        top = tracker.top_candidates(experiment_id, limit=15)
        ranked = []
        for i, c in enumerate(top):
            ranked.append({
                "rank": c["rank"], "strategy_type": c["strategy_type"],
                "params": c["params"], "metrics": c["metrics"],
                "passed_filters": c["passed_filters"], "filter_reasons": [],
            })
        data = DashboardData(
            title=f"Research Discovery — {experiment_id}",
            subtitle=f"Experiment {experiment_id}",
            metrics=top[0]["metrics"] if top else {},
            ranked_strategies=ranked,
            experiment_id=experiment_id,
        )
        path = DashboardGenerator().generate(data, output)
        console.print(f"[green]✓[/green] Dashboard written to {path}")
    else:
        # Full single-strategy dashboard: backtest + analysis + MC + risk
        start_ms = _parse_date_ms(config.data.start_date)
        end_ms = _parse_date_ms(config.data.end_date)
        df = repo.load(symbol, timeframe, start_ms, end_ms, market_type=market)
        if df is None or len(df) < 60:
            console.print(f"[bold red]Not enough data for {symbol} {timeframe}.[/bold red]")
            raise typer.Abort()

        # Backtest
        strat_obj = create_strategy(strategy)
        result = BacktestEngine(BacktestConfig(
            market_type=market, timeframe=timeframe,
            execution=ExecutionConfig(market_type=market),
        )).run(strat_obj, df, symbol=symbol)

        # Trade analysis
        feats = FeatureEngine().compute(df)
        from ..analysis.regimes import RegimeClassifier
        from ..analysis.trades import TradeAnalyzer
        classified = RegimeClassifier().classify(df)
        feats["regime"] = classified["regime"].values
        feats["vol_regime"] = classified["vol_regime"].values
        analysis = TradeAnalyzer().analyze(result.trades, feats).to_dict()

        # Monte Carlo on trade returns
        mc = {}
        returns = [tr["return_pct"] for tr in result.trades if tr["return_pct"] is not None]
        if len(returns) >= 10:
            mc = MonteCarloSimulator(n_simulations=1000).simulate(
                returns, initial_capital=config.risk.starting_capital
            ).to_dict()

        data = DashboardGenerator.from_backtest(
            result, symbol=symbol, timeframe=timeframe, market_type=market,
            trade_analysis=analysis, monte_carlo=mc,
        )
        path = DashboardGenerator().generate(data, output)
        console.print(f"[green]✓[/green] Dashboard written to {path}")

    if open_in_browser:
        import webbrowser
        webbrowser.open(path.resolve().as_uri())


@app.command()
def serve(
    path: str = typer.Argument("dashboard", help="Path to dashboard HTML file or directory"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to serve on"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="Open browser automatically"),
    auto_refresh: bool = typer.Option(False, "--auto-refresh", help="Auto-refresh every 30 seconds"),
):
    """Serve the dashboard locally."""
    from pathlib import Path as P
    target = P(path)
    if not target.exists():
        console.print(f"[bold red]Path does not exist: {path}[/bold red]")
        raise typer.Abort()

    console.print(f"[bold cyan]Serving dashboard at http://127.0.0.1:{port}[/bold cyan]")
    if auto_refresh:
        console.print("[dim]Auto-refresh enabled (every 30 seconds)[/dim]")
    console.print("Press Ctrl+C to stop\n")

    serve_dashboard(
        path=target,
        port=port,
        open_browser=open_browser,
        auto_refresh=auto_refresh,
    )


@app.command()
def paper(
    action: str = typer.Argument(..., help="Action: start, stop, status, report"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Strategy type: trend, momentum, mean_reversion, breakout"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="HTML report path (report action)"),
    live: bool = typer.Option(False, "--live", help="Run a live polling loop instead of replay"),
    mode: Optional[str] = typer.Option(None, "--mode", help="Data mode: replay or realtime (defaults to replay unless --live)"),
    poll: Optional[float] = typer.Option(None, "--poll", help="Poll interval in seconds (live)"),
    window: int = typer.Option(300, "--window", help="Warmup bars held for live signal generation"),
    capital: Optional[float] = typer.Option(None, "--capital", help="Starting capital override ($)"),
    leverage: Optional[int] = typer.Option(None, "--leverage", "-l", help="Leverage (futures only, 1-5x)"),
    risk_per_trade: Optional[float] = typer.Option(None, "--risk-per-trade", help="Risk fraction per trade (e.g., 0.01 = 1%)"),
    max_positions: Optional[int] = typer.Option(None, "--max-positions", help="Max open positions (e.g., 3)"),
    detach: bool = typer.Option(False, "--detach", "-d", help="Run realtime worker in background (detached process)"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume paper account from run ID"),
):
    """Paper trading commands (simulated execution, never a real order)."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                  PAPER TRADING[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    config = get_config()
    resolved_mode = mode or ("realtime" if live else "replay")

    if action == "start":
        paper_start(
            strategy=strategy, symbol=symbol, timeframe=timeframe, market=market,
            start=start, end=end, config=config, live=(resolved_mode == "realtime"),
            poll=poll, window=window, capital=capital, leverage=leverage,
            risk_per_trade=risk_per_trade, max_positions=max_positions,
            detach=detach, resume=resume,
        )
    elif action == "status":
        paper_status(config)
    elif action == "stop":
        paper_stop()
    elif action == "report":
        paper_report(strategy=strategy, symbol=symbol, timeframe=timeframe,
                     market=market, start=start, end=end, output=output, config=config)
    else:
        console.print("[bold red]Unknown action:[/bold red] start, stop, status, report")


# ---------------------------------------------------------------------------
# Paper Trading helpers
# ---------------------------------------------------------------------------
SESSION_FILE = "data/paper_session.json"
_LIVE_WAIT_S = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400}


def _paper_session_path() -> Path:
    return Path(SESSION_FILE)


def _fee_rate_for(config, market: str) -> float:
    if market == "futures":
        return config.fees.futures_taker
    return config.fees.spot_taker


def _run_one_session(strategy, symbol, timeframe, market, start, end, config,
                     persist: bool = True):
    """Run a deterministic replay paper session via the worker.

    Args:
        persist: When True, write the session's trades to the database (used by
            ``paper start``). ``paper report`` passes False so rendering a report
            does not accumulate duplicate session rows.
    """
    db = get_db_manager()
    db.create_tables()

    strat_types = [s.strip() for s in (strategy or "trend").split(",")]
    if len(strat_types) != 1:
        console.print("[bold red]Paper trading uses exactly one strategy type.[/bold red]")
        raise typer.Abort()
    strat = strat_types[0]

    from ..execution.worker import LiveTradingWorker, WorkerConfig
    wcfg = WorkerConfig(
        strategy=strat, symbol=symbol, timeframe=timeframe,
        market_type=market, mode="replay",
        start=start, end=end,
        starting_capital=float(config.risk.starting_capital),
        risk_per_trade=float(config.risk.risk_per_trade),
        max_open_positions=int(config.risk.max_open_positions),
        leverage=int(config.risk.max_leverage) if market == "futures" else 1,
        persist=persist,
    )
    try:
        worker = LiveTradingWorker(config=wcfg, db=db)
    except ValueError as exc:
        console.print(f"[bold red]Not enough data for {symbol} {timeframe}. Run `data download` first.[/bold red]")
        raise typer.Abort()

    console.print(f"[bold]Symbol:[/bold] {symbol} | [bold]TF:[/bold] {timeframe} | "
                  f"[bold]Market:[/bold] {market} | [bold]Strategy:[/bold] {strat}")
    result = worker.run_replay()
    return result, worker.strategy, worker.broker, worker.risk


def paper_start(strategy, symbol, timeframe, market, start, end, config,
                live: bool = False, poll: Optional[float] = None, window: int = 300,
                capital: Optional[float] = None, leverage: Optional[int] = None,
                risk_per_trade: Optional[float] = None, max_positions: Optional[int] = None,
                detach: bool = False, resume: Optional[str] = None):
    """Start a paper trading session: deterministic replay, or a realtime loop."""
    import subprocess
    db = get_db_manager()
    db.create_tables()

    strat = (strategy or "trend").strip()
    cap = capital if capital is not None else float(config.risk.starting_capital)
    rpt = risk_per_trade if risk_per_trade is not None else float(config.risk.risk_per_trade)
    mpos = max_positions if max_positions is not None else int(config.risk.max_open_positions)
    lev = leverage if leverage is not None else (int(config.risk.max_leverage) if market == "futures" else 1)
    poll_s = int(poll or _LIVE_WAIT_S.get(timeframe, 60))

    if detach and live:
        # Run worker as a detached background process (STEP 5).
        args = [
            sys.executable, "-m", "crypto_quant.execution.worker",
            "--strategy", strat, "--symbol", symbol, "--timeframe", timeframe,
            "--market", market, "--mode", "realtime",
            "--capital", str(cap), "--risk-per-trade", str(rpt),
            "--max-positions", str(mpos), "--leverage", str(lev),
            "--poll", str(poll_s), "--window", str(window),
        ]
        if resume:
            args += ["--resume", resume]
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        console.print(f"[bold green]✓[/bold green] Realtime paper worker detached (pid={proc.pid}).")
        console.print("Run `paper status` to view live state, `paper stop` to shut down.")
        return

    from ..execution.worker import LiveTradingWorker, WorkerConfig
    wcfg = WorkerConfig(
        strategy=strat, symbol=symbol, timeframe=timeframe,
        market_type=market, mode="realtime" if live else "replay",
        start=start, end=end, starting_capital=cap,
        risk_per_trade=rpt, max_open_positions=mpos, leverage=lev,
        poll_interval=poll_s, window=window, resume=resume,
    )
    try:
        worker = LiveTradingWorker(config=wcfg, db=db)
    except ValueError:
        console.print(f"[bold red]Not enough data for {symbol} {timeframe}. Run `data download` first.[/bold red]")
        raise typer.Abort()

    if not live:
        console.print(f"[bold]Symbol:[/bold] {symbol} | [bold]TF:[/bold] {timeframe} | "
                      f"[bold]Market:[/bold] {market} | [bold]Strategy:[/bold] {strat} | "
                      f"[bold]Mode:[/bold] REPLAY")
        result = worker.run_replay()
        _print_paper_result(result, strat)
    else:
        console.print(f"[bold cyan]Realtime paper session started ({wcfg.run_id})[/bold cyan]")
        console.print(f"[bold]Symbol:[/bold] {symbol} | [bold]TF:[/bold] {timeframe} | "
                      f"[bold]Market:[/bold] {market} | [bold]Poll:[/bold] {poll_s}s")
        console.print("[dim]Run `paper stop` from another shell or press Ctrl+C to stop.[/dim]")
        worker.run_loop()


@app.command("research")
def research_cmd(
    strategy: str = typer.Option("all", "--strategy", "-s", help="Strategy: mtf_trend_pullback, breakout_retest, trend_filtered_rsi, regime_adaptive, vwap_bollinger_mr, or all"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol: BTCUSDT, ETHUSDT, or all"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe: 5m, 15m, 1h, 4h, 1d, or all"),
    market: str = typer.Option("futures", "--market", "-m", help="Market type: futures or spot"),
    matrix: bool = typer.Option(False, "--matrix", help="Run full 2x5x5 research matrix"),
    output: Optional[str] = typer.Option("data/multi_strategy_research_report.json", "--output", "-o", help="Report output JSON path"),
):
    """Run systematic strategy research and multi-objective ranking."""
    from ..research.multi_strategy_engine import MultiStrategyResearchEngine, STRATEGY_FAMILIES, SYMBOLS, TIMEFRAMES

    engine = MultiStrategyResearchEngine(market_type=market)

    target_syms = SYMBOLS if (symbol.lower() == "all" or matrix) else [symbol]
    target_tfs = TIMEFRAMES if (timeframe.lower() == "all" or matrix) else [timeframe]
    target_strats = STRATEGY_FAMILIES if (strategy.lower() == "all" or matrix) else [strategy]

    console.print(f"[bold cyan]Running Quantitative Strategy Research[/bold cyan]")
    console.print(f"Symbols: {target_syms} | Timeframes: {target_tfs} | Strategies: {target_strats} | Market: {market}\n")

    results = engine.run_matrix(
        symbols=target_syms,
        timeframes=target_tfs,
        strategies=target_strats,
        market_type=market,
    )

    # Print summary table
    table = Table(title="Strategy Research Ranking Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Rank", style="cyan", width=5)
    table.add_column("Strategy", style="white", width=20)
    table.add_column("Symbol", style="yellow", width=8)
    table.add_column("TF", style="green", width=5)
    table.add_column("Trades", justify="right", width=7)
    table.add_column("Win Rate", justify="right", width=9)
    table.add_column("PF", justify="right", width=6)
    table.add_column("Expectancy", justify="right", width=10)
    table.add_column("Net Ret", justify="right", width=8)
    table.add_column("MaxDD", justify="right", width=7)
    table.add_column("Sharpe", justify="right", width=7)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Status", width=8)

    for i, r in enumerate(results):
        m = r.metrics
        status = "[green]PASS[/green]" if r.passed_filters else "[red]REJECT[/red]"
        table.add_row(
            str(i + 1),
            r.strategy,
            r.symbol,
            r.timeframe,
            str(m.get("total_trades", 0)),
            f"{m.get('win_rate', 0)*100:.1f}%",
            f"{m.get('profit_factor', 0):.2f}",
            f"${m.get('expectancy', 0):.2f}",
            f"{m.get('net_return', 0)*100:.1f}%",
            f"{m.get('max_drawdown', 0)*100:.1f}%",
            f"{m.get('sharpe_ratio', 0):.2f}",
            f"{r.score:.4f}",
            status,
        )

    console.print(table)

    if output:
        out_data = [r.to_dict(include_trades=False) for r in results]
        with open(output, "w") as f:
            json.dump(out_data, f, indent=2)
        console.print(f"\n[bold green]✓[/bold green] Research report saved to {output}")



def _print_paper_result(result, strat_type: str) -> None:
    from datetime import datetime as _dt, timezone as _tz
    starting = float(result.starting_capital or 0.0)
    net_return = (result.final_equity - starting) / starting if starting else 0.0
    table = Table(title="Paper Trading Session Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Strategy", strat_type)
    table.add_row("Starting Capital", f"${starting:,.2f}")
    table.add_row("Final Equity", f"${result.final_equity:,.2f}")
    table.add_row("Realized PnL", f"${result.realized_pnl:,.2f}")
    table.add_row("Net Return", f"{net_return:+.2%}")
    table.add_row("Trades", str(result.n_trades))
    table.add_row("Orders", str(result.n_orders))
    table.add_row("Rejected", str(result.n_rejected))
    table.add_row("Win Rate", f"{result.win_rate:.1%}")
    console.print(table)
    for ev in (result.risk_events or [])[:10]:
        console.print(f"  [yellow]{ev.get('severity', 'risk')} • {ev.get('event_type', '')}:[/yellow] {ev.get('message', '')}")
    console.print(f"\n[green]✓[/green] Session persisted (execution_mode=paper). Run `paper status` to review.")


def paper_status(config) -> None:
    """Show the full live account surface: balance, equity, unrealized/realized PnL,
    open positions, trades, win rate, drawdown, exposure, risk status, last feed ts.
    """
    from datetime import datetime as _dt, timezone as _tz
    import json
    db = get_db_manager()
    db.create_tables()

    from ..execution.persistence import load_account_snapshot
    from ..execution.worker import WorkerLock

    lock_info = WorkerLock.current()
    acct = load_account_snapshot(db, any_status=True)

    # 1. Worker process status
    state_table = Table(title="Worker Status", show_header=True)
    state_table.add_column("Property", style="cyan")
    state_table.add_column("Value", style="green")
    if lock_info:
        state_table.add_row("Process State", "[bold green]RUNNING[/bold green]")
        state_table.add_row("PID", str(lock_info.get("pid", "-")))
        state_table.add_row("Run ID", str(lock_info.get("run_id", "-")))
        state_table.add_row("Started At", str(lock_info.get("started_at", "-")))
    else:
        state_table.add_row("Process State", "[yellow]STOPPED[/yellow]")
    if acct is not None:
        state_table.add_row("Last Run ID", acct.run_id)
        state_table.add_row("Market / Mode", f"{acct.symbol} {acct.timeframe} ({acct.market_type}) • {acct.mode.upper()}")
        state_table.add_row("Strategy", acct.strategy)
        if acct.last_market_ts:
            ts_str = _dt.fromtimestamp(acct.last_market_ts / 1000, tz=_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            state_table.add_row("Last Market Feed", ts_str)
    console.print(state_table)

    # 2. Account summary (from PaperAccount or ExecutionTrade aggregation)
    s = db.get_session()
    try:
        rows = (s.query(ExecutionTrade)
                .filter(ExecutionTrade.execution_mode == "paper")
                .order_by(ExecutionTrade.created_at.desc())
                .limit(500).all())
    finally:
        s.close()

    pnls = [r.net_pnl or 0.0 for r in rows]
    realized = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    capital = float(acct.initial_capital) if acct else float(config.risk.starting_capital)
    cash = float(acct.cash) if acct else capital + realized
    equity = float(acct.equity) if acct else capital + realized

    positions = []
    risk_info = {}
    if acct and acct.positions:
        try:
            positions = json.loads(acct.positions)
        except Exception:
            positions = []
    if acct and acct.risk_status:
        try:
            risk_info = json.loads(acct.risk_status)
        except Exception:
            risk_info = {}

    unrealized = equity - cash
    drawdown = float(risk_info.get("drawdown", 0.0))
    exposure = sum(abs(float(p.get("entry_price", 0)) * float(p.get("quantity", 0))) for p in positions)
    kill_switch = risk_info.get("kill_switch", False)

    summary = Table(title="Paper Account Summary", show_header=True)
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")
    summary.add_row("Starting Capital", f"${capital:,.2f}")
    summary.add_row("Balance (Cash)", f"${cash:,.2f}")
    summary.add_row("Unrealized PnL", f"${unrealized:+,.2f}")
    summary.add_row("Realized PnL", f"${realized:+,.2f}")
    summary.add_row("Total Equity", f"${equity:,.2f}")
    summary.add_row("Current Drawdown", f"{drawdown:.2%}")
    summary.add_row("Current Exposure", f"${exposure:,.2f}")
    summary.add_row("Open Positions", str(len(positions)))
    summary.add_row("Closed Trades", str(len(rows)))
    summary.add_row("Win Rate", f"{(wins / len(rows)):.1%}" if rows else "n/a")
    summary.add_row("Risk Status", "[red]KILL SWITCH ACTIVE[/red]" if kill_switch else "[green]NORMAL[/green]")
    console.print(summary)

    # 3. Open positions
    if positions:
        pos_table = Table(title="Open Positions", show_header=True)
        pos_table.add_column("Symbol", style="magenta")
        pos_table.add_column("Side", style="white")
        pos_table.add_column("Qty", justify="right")
        pos_table.add_column("Entry Price", justify="right")
        pos_table.add_column("Stop Loss", justify="right")
        pos_table.add_column("Take Profit", justify="right")
        pos_table.add_column("Leverage", justify="right")
        for p in positions:
            pos_table.add_row(
                p.get("symbol", "-"), p.get("side", "-"),
                f"{float(p.get('quantity', 0)):.4f}",
                f"{float(p.get('entry_price', 0)):.4f}",
                f"{float(p.get('stop_loss', 0)):.4f}" if p.get("stop_loss") else "-",
                f"{float(p.get('take_profit', 0)):.4f}" if p.get("take_profit") else "-",
                f"{p.get('leverage', 1)}x",
            )
        console.print(pos_table)

    # 4. Recent closed trades
    if not rows:
        console.print("[yellow]No paper trades yet. Run: python -m crypto_quant paper start[/yellow]")
        return

    table = Table(title="Recent Paper Trades", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Symbol", style="magenta")
    table.add_column("Side", style="white")
    table.add_column("Entry", justify="right")
    table.add_column("Exit", justify="right")
    table.add_column("Qty", justify="right")
    table.add_column("Net PnL", justify="right")
    table.add_column("Reason", style="yellow")
    table.add_column("Exit Time")
    for r in rows[:20]:
        color = "green" if (r.net_pnl or 0) >= 0 else "red"
        reason = r.exit_reason or "-"
        table.add_row(
            r.id, r.symbol, r.direction,
            f"{r.entry_price:.4f}", f"{r.exit_price:.4f}" if r.exit_price else "-",
            f"{r.quantity:.4f}", f"[{color}]{r.net_pnl:.4f}[/{color}]",
            reason,
            _dt.fromtimestamp((r.exit_time or 0) / 1000, tz=_tz.utc).strftime("%Y-%m-%d %H:%M") if r.exit_time else "-",
        )
    console.print(table)


def paper_stop() -> None:
    """Safely stop any active paper worker process and persist state."""
    import json
    session_path = _paper_session_path()
    if session_path.exists():
        try:
            state = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
        if state.get("phase") == "running" and not state.get("stopped"):
            state["stop_requested"] = True
            session_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            console.print("[yellow]Stop requested. The live session will shut down on its next tick and persist.[/yellow]")
            return

    from ..execution.worker import LiveTradingWorker, WorkerLock
    lock = WorkerLock.current()
    if lock:
        worker = LiveTradingWorker()
        worker.stop()
        console.print(f"[yellow]Stop signaled to worker pid={lock.get('pid')} (run={lock.get('run_id')}).[/yellow]")
        console.print("[green]The worker will finalize, close positions, persist, and release its lock.[/green]")
        return

    console.print("[yellow]No active live paper session to stop. Replay sessions close & persist automatically.[/yellow]")


def paper_report(strategy, symbol, timeframe, market, start, end, output, config) -> None:
    """Generate a dashboard report for a paper session (manual, offline replay)."""
    out = Path(output) if output else Path("dashboard") / "paper_report.html"
    result, strat_obj, broker, risk = _run_one_session(
        strategy, symbol, timeframe, market, start, end, config, persist=False)
    data = DashboardGenerator.from_paper(
        result, symbol=symbol, timeframe=timeframe, market_type=market)
    path = DashboardGenerator().generate(data, out)
    console.print(f"[green]✓[/green] Paper report written to {path}")


def _run_live_paper(strategy, symbol, timeframe, market, config, db,
                    poll: Optional[float] = None, window: int = 300) -> None:
    """Live paper loop: poll completed bars, risk-gate signals, paper-execute.

    Runs until the session state file has stop_requested=true (set by `paper stop`
    from another process) or KeyboardInterrupt.
    """
    import json
    import time
    from datetime import datetime, timezone

    strat_types = [s.strip() for s in (strategy or "trend").split(",")]
    if len(strat_types) != 1:
        console.print("[bold red]Paper trading uses exactly one strategy type.[/bold red]")
        raise typer.Abort()
    strat_obj = create_strategy(strat_types[0])
    fee_rate = _fee_rate_for(config, market)

    adapter = BinanceAdapter(market_type=market)
    interval = poll or _LIVE_WAIT_S.get(timeframe, 60)
    run_id = f"PAPER-{int(time.time() * 1000)}"
    session_path = _paper_session_path()

    def _write_state(**updates):
        state = {
            "run_id": run_id, "symbol": symbol, "timeframe": timeframe,
            "market_type": market, "strategy": strat_obj.strategy_type,
            "started_at": started_at, "last_seen": datetime.now(timezone.utc).isoformat(),
            "phase": "running", "stop_requested": False, "stopped": False,
            "equity": None, "n_trades": 0,
        }
        state.update(updates)
        session_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _stop_requested() -> bool:
        try:
            return bool(json.loads(session_path.read_text(encoding="utf-8")).get("stop_requested"))
        except Exception:
            return False

    started_at = datetime.now(timezone.utc).isoformat()
    session_path.parent.mkdir(parents=True, exist_ok=True)

    broker = PaperBroker(
        price_source=DataframePriceSource(_empty_frame()),
        starting_capital=config.risk.starting_capital,
        fee_rate=fee_rate, slippage=config.fees.slippage, market_type=market,
    )
    risk = RiskManager(RiskLimits.from_config(config.risk))
    cfg = PaperTradingConfig(
        symbol=symbol, market_type=market, timeframe=timeframe,
        starting_capital=config.risk.starting_capital,
        fee_rate=fee_rate, slippage=config.fees.slippage,
    )
    engine = PaperTradingEngine(strategy=strat_obj, broker=broker,
                                risk_manager=risk, config=cfg, db=db)
    _write_state()

    console.print(f"[bold cyan]Live paper session started ({run_id})[/bold cyan]")
    console.print(f"[bold]Symbol:[/bold] {symbol} | [bold]TF:[/bold] {timeframe} | "
                  f"[bold]Market:[/bold] {market} | [bold]Poll:[/bold] {interval}s")
    console.print("[dim]Run `paper stop` from another shell to stop gracefully. Ctrl+C also stops.[/dim]")

    try:
        df = adapter.get_ohlcv_as_dataframe(symbol, timeframe, limit=window)
        if df is None or df.empty:
            console.print("[bold red]No live data returned. Check network / symbol availability.[/bold red]")
            return
        df = df.reset_index(drop=True)
        broker.price_source = DataframePriceSource(df)
        broker.price_source.set_index(len(df) - 1)
        last_ts = int(df["timestamp"].iloc[-1])

        while True:
            time.sleep(interval)
            try:
                ndf = adapter.get_ohlcv_as_dataframe(symbol, timeframe, limit=window)
            except Exception as exc:
                console.print(f"[yellow]fetch error: {exc}[/yellow]")
                continue
            if ndf is None or ndf.empty:
                continue
            ndf = ndf.reset_index(drop=True)
            ts = int(ndf["timestamp"].iloc[-1])
            if ts == last_ts:
                continue
            last_ts = ts

            df = ndf
            broker.price_source = DataframePriceSource(df)
            broker.price_source.set_index(len(df) - 1)
            prepared = strat_obj.setup(df)
            signal = strat_obj.generate_signal(prepared, len(df) - 1)
            bar = {
                "open": float(df["open"].iloc[-1]),
                "high": float(df["high"].iloc[-1]),
                "low": float(df["low"].iloc[-1]),
                "close": float(df["close"].iloc[-1]),
                "timestamp": int(ts),
            }
            engine.set_live_signal(signal)
            engine.on_bar(bar)
            _write_state(equity=round(broker.equity, 2),
                         n_trades=len(broker.closed_trades))
            console.print(f"[dim][{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                          f"equity={broker.equity:.2f} closed={len(broker.closed_trades)}[/dim]")

            if _stop_requested():
                console.print("[yellow]Stop requested — closing positions and persisting.[/yellow]")
                break

        # Graceful close + persist
        now_ms = int(time.time() * 1000)
        broker.close_all(reason="live_stop")
        broker.mark_positions(now_ms)
        result = engine._result()
        if engine.db:
            engine._persist(result)
        _write_state(phase="stopped", equity=round(broker.equity, 2),
                     n_trades=len(broker.closed_trades), stopped=True)
        console.print("[green]✓[/green] Live paper session stopped and persisted.")
        _print_paper_result(result, strat_obj.strategy_type)
    except KeyboardInterrupt:
        now_ms = int(time.time() * 1000)
        broker.close_all(reason="manual_stop")
        broker.mark_positions(now_ms)
        result = engine._result()
        if engine.db:
            engine._persist(result)
        _write_state(phase="stopped", equity=round(broker.equity, 2),
                     n_trades=len(broker.closed_trades), stopped=True)
        console.print("\n[yellow]Interrupted — session stopped and persisted.[/yellow]")
        _print_paper_result(result, strat_obj.strategy_type)
    finally:
        adapter.close()


def _empty_frame():
    import pandas as pd
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _paper_menu(config) -> None:
    """Interactive paper trading flow from the main menu (replay by default)."""
    strategy = Prompt.ask("Strategy type", default="trend")
    symbol = Prompt.ask("Symbol", default="BTCUSDT")
    timeframe = Prompt.ask("Timeframe", default="1h")
    market = Prompt.ask("Market", choices=["spot", "futures"], default="spot")
    try:
        paper_start(strategy=strategy, symbol=symbol, timeframe=timeframe,
                    market=market, start=None, end=None, config=config)
    except typer.Abort:
        console.print("[yellow]Paper trading cancelled.[/yellow]")


@app.command()
def worker(
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    mode: str = typer.Option("replay", "--mode", help="replay or realtime"),
    start: Optional[str] = typer.Option(None, "--start", help="Start date"),
    end: Optional[str] = typer.Option(None, "--end", help="End date"),
    capital: float = typer.Option(1000.0, "--capital", help="Starting capital"),
    risk_per_trade: float = typer.Option(0.01, "--risk-per-trade", help="Risk fraction"),
    max_positions: int = typer.Option(3, "--max-positions", help="Max positions"),
    leverage: int = typer.Option(1, "--leverage", "-l", help="Leverage"),
    poll: int = typer.Option(60, "--poll", help="Poll interval"),
    run_id: str = typer.Option("", "--run-id", help="Session ID"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume run ID"),
):
    """Direct worker daemon entry point (subprocesses / detached runs)."""
    from ..execution.worker import LiveTradingWorker, WorkerConfig
    db = get_db_manager()
    db.create_tables()
    cfg = WorkerConfig(
        strategy=strategy, symbol=symbol, timeframe=timeframe,
        market_type=market, mode=mode, start=start, end=end,
        starting_capital=capital, risk_per_trade=risk_per_trade,
        max_open_positions=max_positions, leverage=leverage,
        poll_interval=poll, run_id=run_id, resume=resume,
    )
    w = LiveTradingWorker(config=cfg, db=db)
    w.run()


@app.command()
def live(
    action: str = typer.Argument(..., help="Action: start, stop, status, pause, resume, kill, reconcile, orders, positions"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy: trend, momentum, mean_reversion, breakout"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Trading pair symbol (e.g. BTCUSDT)"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Bar timeframe (e.g. 1h, 15m)"),
    market: str = typer.Option("spot", "--market", "-m", help="Market type: spot or futures"),
    testnet: bool = typer.Option(True, "--testnet/--live-net", help="Use Binance Testnet (True) or Live Real Money (False)"),
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run", help="Dry-run simulation mode (no real orders)"),
    confirm: Optional[str] = typer.Option(None, "--confirm", help="Confirmation string for non-interactive live execution"),
    capital: Optional[float] = typer.Option(None, "--capital", help="Starting capital ($)"),
    risk_per_trade: Optional[float] = typer.Option(0.01, "--risk-per-trade", help="Risk fraction (0.01 = 1%)"),
    max_positions: Optional[int] = typer.Option(3, "--max-positions", help="Maximum simultaneous open positions"),
    leverage: Optional[int] = typer.Option(1, "--leverage", "-l", help="Futures leverage (1-5x)"),
    poll: Optional[int] = typer.Option(60, "--poll", help="Poll interval in seconds"),
):
    """Production live trading execution commands (Binance Spot & Futures)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from ..exchange.binance_live import BinanceLiveConnector
    from ..execution.live_broker import BinanceLiveBroker
    from ..execution.live_worker import LiveExecutionWorker, LiveWorkerConfig
    from ..execution.safety import LiveSafetyGateKeeper, PositionReconciler
    from ..risk.limits import RiskLimits
    from ..risk.manager import RiskManager

    config = get_config()
    db = get_db_manager()
    db.create_tables()

    env_label = "DRY_RUN (Simulation)" if dry_run else ("BINANCE TESTNET" if testnet else "BINANCE LIVE (REAL MONEY)")
    console.print(f"[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print(f"[bold cyan]       BINANCE LIVE TRADING: {env_label}[/bold cyan]")
    console.print(f"[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    connector = BinanceLiveConnector(
        api_key=api_key,
        api_secret=api_secret,
        market_type=market,
        testnet=testnet,
    )
    broker = BinanceLiveBroker(
        connector=connector,
        dry_run=dry_run,
        default_leverage=leverage or 1,
    )
    limits = RiskLimits(
        starting_capital=capital or float(config.risk.starting_capital),
        risk_per_trade=risk_per_trade or float(config.risk.risk_per_trade),
        max_open_positions=max_positions or int(config.risk.max_open_positions),
        max_leverage=leverage or 1,
    )
    risk = RiskManager(limits)

    if action == "start":
        # DOUBLE CONFIRMATION FOR REAL MONEY
        if not dry_run and not testnet:
            console.print("\n[bold red]╔════════════════════════════════════════════════════════════╗[/bold red]")
            console.print("[bold red]║  CRITICAL WARNING: REAL MONEY LIVE TRADING IS REQUESTED   ║[/bold red]")
            console.print("[bold red]║  REAL CAPITAL IS AT RISK ON BINANCE PRODUCTION EXCHANGE    ║[/bold red]")
            console.print("[bold red]╚════════════════════════════════════════════════════════════╝[/bold red]\n")

            required_phrase = "START LIVE TRADING"
            if confirm != required_phrase:
                user_input = Prompt.ask(f"To confirm real-money live trading, type exactly '[bold red]{required_phrase}[/bold red]'")
                if user_input.strip() != required_phrase:
                    console.print("[bold yellow]Live trading start ABORTED: confirmation mismatch.[/bold yellow]")
                    raise typer.Abort()

        # Gate Verification
        gate_res = LiveSafetyGateKeeper.verify_all_gates(
            connector=connector,
            broker=broker,
            risk=risk,
            db=db,
            strategy_name=strategy,
            symbol=symbol,
            dry_run=dry_run,
            is_live_flag=(not dry_run and not testnet),
        )
        if not gate_res.all_passed:
            console.print("[bold red]LIVE TRADING START REFUSED! Safety gates failed:[/bold red]")
            for f in gate_res.failure_reasons:
                console.print(f"  [red]✗[/red] {f}")
            raise typer.Abort()

        console.print("[bold green]✓ All 15 Safety Gates Passed Successfully.[/bold green]")
        console.print(f"Starting live execution worker on [bold]{symbol}[/bold] ({market.upper()}) using [bold]{strategy}[/bold]...")

        wcfg = LiveWorkerConfig(
            strategy=strategy,
            symbol=symbol,
            timeframe=timeframe,
            market_type=market,
            testnet=testnet,
            dry_run=dry_run,
            is_live_flag=(not dry_run and not testnet),
            starting_capital=capital or float(config.risk.starting_capital),
            risk_per_trade=risk_per_trade or float(config.risk.risk_per_trade),
            max_open_positions=max_positions or int(config.risk.max_open_positions),
            leverage=leverage or 1,
            poll_interval=poll or 60,
        )
        worker = LiveExecutionWorker(
            config=wcfg,
            connector=connector,
            broker=broker,
            risk=risk,
            db=db,
        )
        worker.run()

    elif action == "status":
        from ..db.models import LiveAccount, LiveOrder
        s = db.get_session()
        try:
            acct = s.query(LiveAccount).order_by(LiveAccount.updated_at.desc()).first()
            orders = s.query(LiveOrder).order_by(LiveOrder.created_at.desc()).limit(10).all()
        finally:
            s.close()

        table = Table(title="Live Trading Operational Status", show_header=True)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        if acct:
            table.add_row("Run ID", acct.run_id)
            table.add_row("Environment", acct.environment.upper())
            table.add_row("Status", f"[bold green]{acct.status.upper()}[/bold green]" if acct.status == "running" else f"[yellow]{acct.status.upper()}[/yellow]")
            table.add_row("Symbol / TF", f"{acct.symbol} {acct.timeframe} ({acct.market_type})")
            table.add_row("Strategy", acct.strategy)
            table.add_row("Equity", f"${acct.equity:,.2f}")
            table.add_row("Cash Balance", f"${acct.cash:,.2f}")
            table.add_row("Closed Trades", str(acct.closed_trades))
        else:
            table.add_row("Status", "[yellow]NO ACTIVE SESSION[/yellow]")
        console.print(table)

        if orders:
            o_table = Table(title="Recent Live Orders", show_header=True)
            o_table.add_column("Order ID", style="cyan")
            o_table.add_column("Symbol", style="magenta")
            o_table.add_column("Side", style="white")
            o_table.add_column("Qty", justify="right")
            o_table.add_column("Fill Price", justify="right")
            o_table.add_column("Status", style="yellow")
            for o in orders:
                o_table.add_row(
                    o.id[:18], o.symbol, o.side.upper(),
                    f"{o.requested_qty:.4f}",
                    f"${o.avg_fill_price:.4f}" if o.avg_fill_price else "-",
                    f"[green]{o.status}[/green]" if o.status == "filled" else o.status,
                )
            console.print(o_table)

    elif action == "stop":
        session_path = Path("data/live_session.json")
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps({"stop_requested": True}, indent=2), encoding="utf-8")
        console.print("[yellow]Stop signal dispatched. Live worker will gracefully finalize on next tick.[/yellow]")

    elif action == "kill":
        risk.emergency_stop(close_positions=True)
        session_path = Path("data/live_session.json")
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(json.dumps({"stop_requested": True, "kill_switch": True}, indent=2), encoding="utf-8")
        console.print("[bold red]EMERGENCY KILL SWITCH ENGAGED! New trades halted and positions flagged for closure.[/bold red]")

    elif action == "reconcile":
        reconciler = PositionReconciler(connector, broker)
        is_clean, discrepancies = reconciler.reconcile()
        if is_clean:
            console.print("[bold green]✓ Local state matches Binance exchange state perfectly (0 discrepancies).[/bold green]")
        else:
            console.print(f"[bold red]✗ {len(discrepancies)} Discrepancies detected:[/bold red]")
            for d in discrepancies:
                console.print(f"  [red]•[/red] [{d.category}] {d.message}")

    elif action == "positions":
        # Check database for bot's active trading session positions
        from ..db.models import LiveAccount
        s = db.get_session()
        positions = []
        acct_symbol = symbol
        try:
            acct = s.query(LiveAccount).filter(LiveAccount.status == "running").order_by(LiveAccount.updated_at.desc()).first()
            if not acct:
                acct = s.query(LiveAccount).order_by(LiveAccount.updated_at.desc()).first()
            if acct:
                acct_symbol = acct.symbol
                if acct.positions:
                    try:
                        all_pos = json.loads(acct.positions)
                        # Filter to the bot's target trading symbol (e.g. BTCUSDT)
                        positions = [p for p in all_pos if p.get("symbol") == acct.symbol]
                        if not positions and all_pos:
                            # If no target symbol position is open, let user know
                            positions = []
                    except Exception:
                        positions = []
        finally:
            s.close()

        if not positions:
            console.print(f"[yellow]No active strategy position open for {acct_symbol}.[/yellow]")
            console.print("[dim]The bot is waiting for a valid strategy entry signal (trend / breakout / momentum) to enter a trade.[/dim]")
        else:
            pos_table = Table(title=f"Open Strategy Position ({acct_symbol})", show_header=True)
            pos_table.add_column("Symbol", style="cyan")
            pos_table.add_column("Side", style="white")
            pos_table.add_column("Quantity", justify="right")
            pos_table.add_column("Entry Price", justify="right")
            pos_table.add_column("Notional", justify="right")
            pos_table.add_column("Leverage", justify="right")
            for p in positions:
                notional = float(p.get("notional", 0.0)) or (float(p.get("quantity", 0.0)) * float(p.get("entry_price", 0.0)))
                pos_table.add_row(
                    p.get("symbol", "-"), p.get("side", "-"),
                    f"{float(p.get('quantity', 0)):.4f}",
                    f"${float(p.get('entry_price', 0)):.4f}",
                    f"${notional:.2f}",
                    f"{p.get('leverage', 1)}x",
                )
            console.print(pos_table)

    elif action == "orders":
        from ..db.models import LiveOrder
        s = db.get_session()
        try:
            orders = s.query(LiveOrder).order_by(LiveOrder.created_at.desc()).limit(20).all()
        finally:
            s.close()

        if not orders:
            console.print("[yellow]No orders recorded yet in database.[/yellow]")
        else:
            o_table = Table(title="Live Trading Orders (Database)", show_header=True)
            o_table.add_column("ID", style="cyan")
            o_table.add_column("Symbol", style="magenta")
            o_table.add_column("Side", style="white")
            o_table.add_column("Qty", justify="right")
            o_table.add_column("Status", style="yellow")
            o_table.add_column("Fill Price", justify="right")
            o_table.add_column("Created At")
            for o in orders:
                o_table.add_row(
                    o.id[:18], o.symbol, o.side.upper(),
                    f"{o.requested_qty:.4f}",
                    f"[green]{o.status}[/green]" if o.status == "filled" else o.status,
                    f"${o.avg_fill_price:.4f}" if o.avg_fill_price else "-",
                    o.created_at.strftime("%Y-%m-%d %H:%M:%S") if o.created_at else "-",
                )
            console.print(o_table)
    else:
        console.print(f"[bold red]Unknown live action:[/bold red] {action}. Expected: start, stop, status, kill, reconcile, orders, positions")


# ---------------------------------------------------------------------------
# Discord DM notifications
# ---------------------------------------------------------------------------
@notify_app.command("test-dm")
def notify_test_dm(
    token: Optional[str] = typer.Option(None, "--token", help="Discord Bot Token (overrides DISCORD_BOT_TOKEN env var)"),
    user_id: Optional[str] = typer.Option(None, "--user-id", help="Discord User Id (overrides DISCORD_USER_ID env var)"),
):
    """Send a test Discord DM to verify the notification pipeline."""
    from ..notifications.discord_dm import DiscordDMNotifier

    # Load .env if present (same convention used in `live` command).
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    notifier = DiscordDMNotifier(token=token, user_id=user_id)
    if not notifier._enabled:
        console.print("[bold yellow]Discord notifier is disabled — DISCORD_BOT_TOKEN and DISCORD_USER_ID must be set (via --token/--user-id or env vars).[/bold yellow]")
        raise typer.Abort()

    # Build a test embed (GREEN) with system info
    import platform
    payload = {
        "title": "🟢 Discord DM Pipeline Test",
        "color": DiscordDMNotifier.COLOR_GREEN,
        "fields": [
            {"name": "System", "value": f"{platform.system()} {platform.release()}", "inline": True},
            {"name": "Python", "value": platform.python_version(), "inline": True},
            {"name": "Hostname", "value": platform.node() or "unknown", "inline": True},
            {"name": "Status", "value": "✅ Notification pipeline is operational", "inline": False},
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    # Fire synchronously (blocking is OK in a one-off CLI test)
    try:
        channel_id = notifier._ensure_dm_channel()
        notifier._request("POST", f"https://discord.com/api/v10/channels/{channel_id}/messages", json={"embeds": [payload]})
        console.print("[bold green]✓ Discord DM test embed sent successfully.[/bold green]")
        console.print(f"  DM channel: {channel_id}")
    except Exception as exc:
        console.print(f"[bold red]✗ Discord DM test failed: {exc}[/bold red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Cash-and-carry terminal operations
# ---------------------------------------------------------------------------
def _carry_build_broker(dry_run: bool, testnet: bool = True):
    """Build the production twin-leg broker from environment credentials."""
    from dotenv import load_dotenv
    from ..exchange.binance_live import BinanceLiveConnector
    from ..execution.live_broker import BinanceLiveBroker, TwinLegCarryBroker

    load_dotenv(override=True)
    spot = BinanceLiveConnector(
        api_key=os.getenv("BINANCE_SPOT_TESTNET_KEY", ""),
        api_secret=os.getenv("BINANCE_SPOT_TESTNET_SECRET", ""),
        market_type="spot",
        testnet=testnet,
        recv_window=60000,
    )
    futures = BinanceLiveConnector(
        api_key=os.getenv("BINANCE_FUTURES_TESTNET_KEY", ""),
        api_secret=os.getenv("BINANCE_FUTURES_TESTNET_SECRET", ""),
        market_type="futures",
        testnet=testnet,
        recv_window=60000,
    )
    return TwinLegCarryBroker(
        BinanceLiveBroker(spot, dry_run=dry_run),
        BinanceLiveBroker(futures, dry_run=dry_run),
    )


def _carry_position_from_record(record):
    """Reconstruct the minimum broker position object for a manual unwind."""
    from ..execution.live_broker import TwinLegPosition

    opened = record.opened_at.replace(tzinfo=timezone.utc).timestamp()
    closed = record.closed_at.replace(tzinfo=timezone.utc).timestamp() if record.closed_at else None
    base = record.symbol[:-4] if record.symbol.endswith("USDT") else record.symbol
    return TwinLegPosition(
        position_id=record.position_id,
        base_asset=base,
        quote_asset="USDT",
        quantity=float(record.quantity),
        spot_order_id="",
        spot_fill_price=float(record.spot_fill_price),
        futures_order_id="",
        futures_fill_price=float(record.futures_fill_price),
        entry_basis_spread_pct=float(record.entry_basis_spread_pct),
        execution_gap_ms=float(record.leg_gap_ms or 0.0),
        status=record.status,
        opened_at=opened,
        closed_at=closed,
    )


def _carry_duration_str(seconds: float) -> str:
    """Render a duration in seconds as a compact human string."""
    hours = max(0.0, seconds / 3600.0)
    if hours < 1.0:
        return f"{hours * 60:.0f}m"
    if hours < 24.0:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def _carry_live_metrics(record, funding_total: float) -> Dict[str, Any]:
    """Fetch current carry metrics; callers render unavailability safely."""
    broker = _carry_build_broker(dry_run=True)
    symbol = record.symbol
    mark = broker.futures.connector.get_mark_price(symbol)
    spot = broker.spot.connector.get_ticker_price(symbol)
    account = broker.futures.connector.get_account_info()
    margin_balance = float(account.get("totalMarginBalance", account.get("totalWalletBalance", 0.0)) or 0.0)
    maintenance = float(account.get("totalMaintMargin", 0.0) or 0.0)
    buffer_pct = ((margin_balance - maintenance) / margin_balance * 100) if margin_balance > 0 else None
    qty = float(record.quantity)
    spot_pnl = (spot - float(record.spot_fill_price)) * qty
    futures_pnl = (float(record.futures_fill_price) - mark) * qty
    return {
        "mark": mark,
        "spot": spot,
        "basis_pct": ((mark - spot) / spot * 100) if spot else None,
        "margin_balance": margin_balance,
        "margin_buffer_pct": buffer_pct,
        "funding": funding_total,
        "net_pnl": spot_pnl + futures_pnl + funding_total,
    }


def _carry_funding_rate_map(broker, symbol: str, start_time: int, end_time: int) -> Dict[int, float]:
    """Map settlement-time (ms) -> settled funding rate for a symbol's window."""
    rate_by_time: Dict[int, float] = {}
    try:
        history = broker.futures.connector.get_funding_rate_history(
            symbol, start_time=start_time, limit=1000
        )
        for item in history:
            ft = int(item.get("fundingTime", item.get("time", 0)) or 0)
            rate_by_time[ft] = float(item.get("fundingRate", 0.0) or 0.0)
    except Exception as exc:
        logger.warning("Funding rate history fetch failed for %s: %s", symbol, exc)
    return rate_by_time


def _carry_mark_price(broker, symbol: str) -> float:
    """Current futures mark price; 0.0 fallback when unavailable."""
    try:
        return float(broker.futures.connector.get_mark_price(symbol))
    except Exception:
        return 0.0


@carry_app.command("status")
def carry_status():
    """Show persisted carry positions and live metrics for currently open carries."""
    from ..db.models import CarryFundingPaymentRecord, CarryPositionRecord

    db = get_db_manager()
    db.create_tables()
    session = db.get_session()
    try:
        positions = session.query(CarryPositionRecord).order_by(CarryPositionRecord.opened_at.desc()).all()
        funding_by_position = {
            position_id: float(total or 0.0)
            for position_id, total in session.query(
                CarryFundingPaymentRecord.position_id,
                __import__("sqlalchemy").func.sum(CarryFundingPaymentRecord.funding_payment_usdt),
            ).group_by(CarryFundingPaymentRecord.position_id).all()
        }
    finally:
        session.close()

    if not positions:
        console.print("[yellow]No carry positions recorded yet.[/yellow]")
        return

    now = datetime.now(timezone.utc)
    table = Table(title="Cash-and-Carry Position Ledger", show_header=True, header_style="bold cyan")
    for column in ("Position ID", "Symbol", "Qty", "Spot Price", "Futures Price", "Basis Spread %", "Leg Gap (ms)", "Status", "Duration (Hours)"):
        table.add_column(column)
    for row in positions:
        opened = row.opened_at.replace(tzinfo=timezone.utc)
        end = row.closed_at.replace(tzinfo=timezone.utc) if row.closed_at else now
        duration = max(0.0, (end - opened).total_seconds() / 3600)
        table.add_row(
            row.position_id, row.symbol, f"{row.quantity:.8f}", f"${row.spot_fill_price:.4f}",
            f"${row.futures_fill_price:.4f}", f"{row.entry_basis_spread_pct:+.4f}%",
            f"{row.leg_gap_ms:.1f}" if row.leg_gap_ms is not None else "-",
            row.status, f"{duration:.2f}",
        )
    console.print(table)

    for row in positions:
        if row.status != "OPEN":
            continue
        funding = funding_by_position.get(row.position_id, 0.0)
        try:
            metrics = _carry_live_metrics(row, funding)
            live = Table(title=f"Live Carry Health: {row.position_id}", show_header=True)
            for name in ("Mark Price", "Spot Price", "Current Basis", "Margin Balance", "Safety Buffer", "Confirmed Funding", "Net Unrealized PnL"):
                live.add_column(name)
            live.add_row(
                f"${metrics['mark']:.4f}", f"${metrics['spot']:.4f}",
                f"{metrics['basis_pct']:+.4f}%" if metrics["basis_pct"] is not None else "N/A",
                f"${metrics['margin_balance']:.2f}",
                f"{metrics['margin_buffer_pct']:.2f}%" if metrics["margin_buffer_pct"] is not None else "N/A",
                f"${metrics['funding']:+.6f}", f"${metrics['net_pnl']:+.6f}",
            )
            console.print(live)
        except Exception as exc:
            logger.warning("Carry live metrics unavailable for %s: %s", row.position_id, exc)
            console.print(f"[yellow]Live metrics unavailable for {row.position_id}: {exc}[/yellow]")


@carry_app.command("payments")
def carry_payments(limit: int = typer.Option(20, "--limit", min=1, max=500, help="Maximum ledger rows to display")):
    """Show settled funding credits/debits and cumulative realized funding."""
    from sqlalchemy import func
    from ..db.models import CarryFundingPaymentRecord

    db = get_db_manager()
    db.create_tables()
    session = db.get_session()
    try:
        rows = session.query(CarryFundingPaymentRecord).order_by(CarryFundingPaymentRecord.timestamp.desc()).limit(limit).all()
        cumulative = float(session.query(func.coalesce(func.sum(CarryFundingPaymentRecord.funding_payment_usdt), 0.0)).scalar() or 0.0)
    finally:
        session.close()
    if not rows:
        console.print("[yellow]No confirmed carry funding payments recorded yet.[/yellow]")
        return
    table = Table(title="Carry Funding Payment Ledger", show_header=True, header_style="bold cyan")
    for column in ("Timestamp (UTC)", "Position ID", "Symbol", "Funding Rate %", "Mark Price", "Payment (USDT)"):
        table.add_column(column)
    for row in rows:
        timestamp = row.timestamp.replace(tzinfo=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        table.add_row(timestamp, row.position_id, row.symbol, f"{row.funding_rate * 100:+.6f}%", f"${row.mark_price:.4f}", f"${row.funding_payment_usdt:+.6f}")
    console.print(table)
    console.print(f"[bold green]Cumulative confirmed funding income: ${cumulative:+.6f}[/bold green]")


@carry_app.command("unwind")
def carry_unwind(position_id: str = typer.Argument(..., help="Persisted carry position id"), dry: bool = typer.Option(False, "--dry", help="Simulate broker unwind without real orders")):
    """Manually and symmetrically close one persisted OPEN carry position."""
    from ..db.models import CarryFundingPaymentRecord, CarryPositionRecord

    db = get_db_manager()
    db.create_tables()
    session = db.get_session()
    try:
        record = session.get(CarryPositionRecord, position_id)
    finally:
        session.close()
    if record is None:
        console.print(f"[red]Carry position not found: {position_id}[/red]")
        raise typer.Exit(1)
    if record.status != "OPEN":
        console.print(f"[yellow]Carry {position_id} is {record.status}; only OPEN positions can be unwound.[/yellow]")
        raise typer.Exit(1)

    position = _carry_position_from_record(record)
    broker = _carry_build_broker(dry_run=dry)
    try:
        broker.unwind_twin_leg_carry(position)
    except Exception as exc:
        console.print(f"[red]Unwind could not be confirmed for {position_id}: {exc}[/red]")
        raise typer.Exit(1)

    session = db.get_session()
    try:
        persisted = session.get(CarryPositionRecord, position_id)
        persisted.status = position.status
        persisted.closed_at = datetime.fromtimestamp(position.closed_at, tz=timezone.utc)
        session.commit()
    finally:
        session.close()
    console.print(f"[bold green]Carry {position_id} closed successfully.[/bold green]")

    # Discord DM (Blue) carry-unwind embed on a confirmed clean close.
    from ..notifications import get_notifier
    try:
        funds_session = db.get_session()
        try:
            total_funding = float(funds_session.query(
                __import__("sqlalchemy").func.coalesce(
                    __import__("sqlalchemy").func.sum(CarryFundingPaymentRecord.funding_payment_usdt), 0.0,
                )
            ).filter(CarryFundingPaymentRecord.position_id == position_id).scalar() or 0.0)
        finally:
            funds_session.close()
        opened = record.opened_at.replace(tzinfo=timezone.utc).timestamp()
        closed = position.closed_at or time.time()
        duration = _carry_duration_str(closed - opened)
        get_notifier().notify_trade_close(
            symbol=record.symbol,
            side="CARRY UNWIND",
            qty=float(record.quantity),
            fill_price=float(record.spot_fill_price),
            duration=duration,
            is_carry=True,
            extra={
                "Status": "CLOSED",
                "Total Funding Harvested (USDT)": f"{total_funding:+.6f}",
            },
        )
    except Exception as exc:
        logger.warning("Carry-unwind DM embed not sent: %s", exc)
    try:
        futures = broker.futures.connector.get_positions(record.symbol)
        spot = broker.spot.connector.get_balances().get(position.base_asset, {})
        console.print(f"Post-unwind reconciliation: futures positions={len(futures)}, spot {position.base_asset}={float(spot.get('total', 0.0)):.8f}")
    except Exception as exc:
        console.print(f"[yellow]Unwind confirmed, but post-unwind reconciliation unavailable: {exc}[/yellow]")


@carry_app.command("start")
def carry_start(
    max_pairs: int = typer.Option(3, "--max-pairs", min=1, help="Maximum concurrent twin-leg positions"),
    total_usdt: float = typer.Option(1000.0, "--total-usdt", min=1.0, help="Max portfolio capital deployed across all pairs (USDT)"),
    per_pair_usdt: float = typer.Option(None, "--per-pair-usdt", help="Per-pair allocation (USDT); defaults to total / max-pairs"),
    scan_interval: float = typer.Option(300.0, "--scan-interval", min=1.0, help="Cadence (seconds) to re-run FundingScanner discovery"),
    min_apr: float = typer.Option(0.08, "--min-apr", help="Entry annualized-funding hurdle (fraction, e.g. 0.08 = 8%)"),
    exit_apr: float = typer.Option(0.02, "--exit-apr", help="Unwind hurdle if funding collapses below (fraction)"),
    quote: str = typer.Option("USDT", "--quote", help="Quote currency of the pairs to scan"),
    poll: float = typer.Option(30.0, "--poll", help="Health-check cadence (seconds)"),
    margin_buffer: float = typer.Option(0.20, "--margin-buffer", help="Minimum account margin safety ratio"),
    dry: bool = typer.Option(False, "--dry", help="Simulate fills and never submit orders"),
):
    """Launch the autonomous multi-pair funding-harvester allocator on Binance Testnet."""
    from ..execution.carry_daemon import CarryDaemonConfig, CarryHarvesterDaemon
    from ..research.funding_scanner import FundingScanner

    db = get_db_manager()
    db.create_tables()
    broker = _carry_build_broker(dry_run=dry)
    config = CarryDaemonConfig(
        quote_asset=quote,
        max_active_pairs=max_pairs,
        total_allocation_usdt=total_usdt,
        allocation_per_pair_usdt=per_pair_usdt,
        scan_interval_seconds=scan_interval,
        min_apr_threshold=min_apr,
        exit_apr_threshold=exit_apr,
        poll_interval_seconds=poll,
        min_margin_ratio=margin_buffer,
        dry_run=dry,
    )
    per_pair = per_pair_usdt if per_pair_usdt is not None else total_usdt / max_pairs
    console.print(
        f"[bold cyan]Starting carry allocator | up to {max_pairs} pairs | "
        f"${per_pair:,.2f}/pair | min APR={min_apr:.2%} (exit {exit_apr:.2%}) | "
        f"{'DRY' if dry else 'TESTNET'}[/bold cyan]"
    )
    CarryHarvesterDaemon(config, broker, db, scanner=FundingScanner(quote=quote)).run()


@carry_app.command("scan")
def carry_scan(
    top: Optional[int] = typer.Option(None, "--top", "-n", min=1, help="Limit to top-N opportunities by net APR"),
    min_volume: float = typer.Option(10_000_000.0, "--min-volume", help="Minimum 24h USDT volume to include a pair"),
    fee_drag_pct: float = typer.Option(0.14, "--fee-drag-pct", help="Round-trip both-leg fee drag, in percentage points"),
    quote: str = typer.Option("USDT", "--quote", help="Quote currency to scan"),
    anomaly_threshold_pct: float = typer.Option(2.0, "--anomaly-threshold-pct", help="Flag/exclude basis anomalies above this |basis|%"),
    sample: bool = typer.Option(False, "--sample", help="Use representative offline sample data (no network)"),
):
    """Scan the Binance USDⓈ-M universe and rank carry/funding opportunities by net APR."""
    from ..research.funding_scanner import FundingScanner, sample_provider

    scanner = FundingScanner(
        min_volume_usdt=min_volume,
        fee_drag_pct=fee_drag_pct,
        basis_anomaly_threshold_pct=anomaly_threshold_pct,
        quote=quote,
        top_n=top,
    )

    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]            FUNDING & BASIS SCANNER[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    try:
        if sample:
            opportunities = scanner.scan(sample_provider(quote=quote))
            source_tag = "SAMPLE DATA (offline)"
        else:
            opportunities = scanner.scan()
            source_tag = "LIVE BINANCE USDⓈ-M"
    except Exception as exc:
        console.print(f"[red]Funding scan failed: {exc}[/red]")
        if not sample:
            console.print("[yellow]Hint: many networks block production fapi.binance.com. "
                          "Use --sample for an offline demo, or check connectivity.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[dim]{source_tag} · min volume ${min_volume:,.0f} · fee drag {fee_drag_pct:.2f}pp[/dim]")

    if not opportunities:
        console.print("[yellow]No carry opportunities met the filters.[/yellow]")
        return

    table = Table(title=f"Carry Funding Opportunities ({len(opportunities)})",
                  show_header=True, header_style="bold cyan")
    for column in ("Symbol", "Spot", "Futures", "Basis %", "Funding %/8h", "Gross APR %", "Net APR %", "Next Funding", "24h Vol ($)"):
        table.add_column(column)
    for o in opportunities:
        table.add_row(
            o.symbol,
            f"${o.spot_price:,.4f}",
            f"${o.futures_price:,.4f}",
            f"{o.basis_spread_pct:+.3f}%",
            f"{o.predicted_funding_rate * 100:+.4f}%",
            f"{o.gross_apr_pct:+.2f}%",
            f"{o.net_apr_pct:+.2f}%",
            o.next_funding_time.strftime("%Y-%m-%d %H:%M UTC") if o.next_funding_time else "-",
            f"{o.volume_24h_usdt:,.0f}",
        )
    console.print(table)

    if scanner.anomalies:
        console.print("[yellow]Flagged basis anomalies (excluded from ranking):[/yellow]")
        for a in scanner.anomalies:
            console.print(f"[yellow]  ⚠ {a['symbol']}: basis {a['basis_spread_pct']:+.3f}%[/yellow]")


@carry_app.command("reconcile-funding")
def carry_reconcile_funding(
    lookback_hours: int = typer.Option(168, "--lookback", min=1, help="Hours of income history to scan for open or recently closed positions"),
    dry: bool = typer.Option(False, "--dry", help="Report rows that would be inserted without writing to the ledger"),
):
    """Backfill missing settled-funding credits into the funding ledger.

    Scans Binance USDⓈ-M /fapi/v1/income for each open (or recently closed)
    carry position, matches FUNDING_FEE events to its ``CarryPositionRecord``,
    and inserts any settlement whose ``(position_id, timestamp)`` is not already
    recorded — recovering settlements the daemon's race missed (Defect 1).
    Uses the exact ``CarryFundingPaymentRecord`` schema — no invented fields.
    """
    from ..db.models import CarryFundingPaymentRecord, CarryPositionRecord

    db = get_db_manager()
    db.create_tables()
    broker = _carry_build_broker(dry_run=dry)

    session = db.get_session()
    try:
        positions = session.query(CarryPositionRecord).all()
        existing: Dict[str, set] = {}
        for position_id, timestamp in session.query(
            CarryFundingPaymentRecord.position_id, CarryFundingPaymentRecord.timestamp
        ).all():
            ms = int(timestamp.replace(tzinfo=timezone.utc).timestamp() * 1000)
            existing.setdefault(position_id, set()).add(ms)
    finally:
        session.close()

    if not positions:
        console.print("[yellow]No carry positions recorded; nothing to reconcile.[/yellow]")
        return

    cutoff_ms = int((time.time() - lookback_hours * 3600) * 1000)
    inserted = 0
    dup_skipped = 0
    no_income = 0
    mode = "DRY-RUN (no writes)" if dry else "LIVE (writing to ledger)"
    console.print(f"[bold cyan]Reconciling carry funding income | {mode}[/bold cyan]")
    console.print(f"[dim]lookback {lookback_hours}h · {len(positions)} position(s)[/dim]")

    for record in positions:
        opened_ms = int(record.opened_at.replace(tzinfo=timezone.utc).timestamp() * 1000)
        closed_ms = (
            int(record.closed_at.replace(tzinfo=timezone.utc).timestamp() * 1000)
            if record.closed_at else int(time.time() * 1000)
        )
        # Only scan positions that could still have unsettled income: open ones,
        # or closed ones still within the lookback window.
        if record.status != "OPEN" and closed_ms < cutoff_ms:
            continue
        start_time = max(opened_ms, cutoff_ms)
        try:
            events = broker.futures.connector.get_funding_income_history(
                record.symbol, start_time=start_time, limit=1000
            )
        except Exception as exc:
            console.print(f"[yellow]  income fetch failed for {record.symbol}: {exc}[/yellow]")
            no_income += 1
            continue
        if not events:
            continue

        rate_by_time = _carry_funding_rate_map(broker, record.symbol, start_time, closed_ms)
        mark = _carry_mark_price(broker, record.symbol)
        have = existing.get(record.position_id, set())
        for event in events:
            ev_time = int(event.get("time", 0) or 0)
            income = float(event.get("income", 0.0) or 0.0)
            if ev_time <= 0 or ev_time < start_time or ev_time > closed_ms or ev_time in have:
                continue
            ts = datetime.fromtimestamp(ev_time / 1000, tz=timezone.utc)
            row = CarryFundingPaymentRecord(
                position_id=record.position_id,
                symbol=record.symbol,
                funding_rate=rate_by_time.get(ev_time, 0.0),
                funding_payment_usdt=income,
                mark_price=mark,
                timestamp=ts,
            )
            if dry:
                console.print(
                    f"  [dim]would insert[/dim] {record.symbol} {record.position_id} "
                    f"{income:+.6f} USDT @ {ts:%Y-%m-%d %H:%M:%S}"
                )
                have.add(ev_time)
                continue
            w = db.get_session()
            try:
                w.add(row)
                w.commit()
                have.add(ev_time)
                inserted += 1
                console.print(
                    f"  [green]✓[/green] {record.symbol} {record.position_id} "
                    f"funding {income:+.6f} USDT recorded @ {ts:%Y-%m-%d %H:%M:%S} "
                    f"(rate {rate_by_time.get(ev_time, 0.0) * 100:+.6f}%)"
                )
            except Exception as exc:
                w.rollback()
                dup_skipped += 1
                logger.warning("Reconcile insert skipped for %s @ %s: %s", record.symbol, ts, exc)
            finally:
                w.close()

    if dry:
        console.print(f"\n[bold]Dry-run summary:[/bold] {len(existing)} already recorded; no ledger writes performed.")
        return
    console.print(
        f"\n[bold green]Reconcile complete:[/bold green] {inserted} inserted · "
        f"{dup_skipped} duplicate-skips · {no_income} symbols with no income fetch."
    )
    if inserted:
        console.print("[dim]Run `carry payments` to view the updated funding ledger.[/dim]")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinitialize"),
):
    """Initialize the system."""
    console.print("[bold cyan]Initializing Crypto Quant Terminal[/bold cyan]")

    # Create directories
    dirs = ["data", "logs", "models", "reports", "dashboard"]
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
        console.print(f"  [green]✓[/green] Created directory: {d}")

    # Initialize database
    db = init_database()
    console.print("  [green]✓[/green] Database initialized")

    # Show config info
    config = get_config()
    show_config_info(config)

    console.print("\n[bold green]Initialization complete![/bold green]")
    console.print("\n[bold yellow]Note: This is Phase 1 (Foundation).[/bold yellow]")
    console.print("[bold yellow]Full functionality will be available in subsequent phases.[/bold yellow]")


if __name__ == "__main__":
    app()
