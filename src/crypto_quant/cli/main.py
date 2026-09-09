"""Main CLI application for Crypto Quant Terminal."""

import sys
from typing import Optional
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
from ..utils.constants import DEFAULT_SYMBOLS, HISTORICAL_UNIVERSE

# Create Typer app
app = typer.Typer(
    name="crypto-quant",
    help="Crypto Quant Research Terminal - Quantitative Research & Trading System",
    add_completion=False,
)

console = Console()


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
        console.print("[bold cyan]Paper Trading - Coming in Phase 12[/bold cyan]")
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
    action: str = typer.Argument(..., help="Action: generate, open"),
    symbol: str = typer.Option("BTCUSDT", "--symbol", help="Symbol"),
    timeframe: str = typer.Option("1h", "--timeframe", "-t", help="Timeframe"),
    strategy: str = typer.Option("trend", "--strategy", "-s", help="Strategy type"),
    market: str = typer.Option("spot", "--market", "-m", help="spot or futures"),
    experiment_id: Optional[str] = typer.Option(None, "--experiment", "-e", help="Experiment ID (from research run)"),
    output: str = typer.Option("dashboard/report.html", "--output", "-o", help="Output HTML path"),
    open_in_browser: bool = typer.Option(False, "--open", help="Open in browser after generating"),
):
    """Generate the local HTML dashboard."""
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")
    console.print("[bold cyan]                 HTML DASHBOARD[/bold cyan]")
    console.print("[bold cyan]──────────────────────────────────────────────[/bold cyan]")

    db = get_db_manager()
    db.create_tables()
    repo = MarketDataRepository(db)
    config = get_config()

    # If an experiment ID is given, render its ranked strategies
    if experiment_id:
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
    action: str = typer.Argument(..., help="Action: start, stop, status"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Strategy ID"),
):
    """Paper trading commands."""
    console.print("[bold cyan]Paper Trading[/bold cyan]")
    console.print(f"Action: {action}")
    if strategy:
        console.print(f"Strategy: {strategy}")
    console.print("[yellow]Coming in Phase 12[/yellow]")


@app.command()
def live(
    action: str = typer.Argument(..., help="Action: start, stop, status"),
    strategy: Optional[str] = typer.Option(None, "--strategy", "-s", help="Strategy ID"),
    dry_run: bool = typer.Option(True, "--dry-run", help="Enable dry run"),
):
    """Live trading commands."""
    console.print("[bold cyan]Live Trading[/bold cyan]")
    console.print(f"Action: {action}")
    if strategy:
        console.print(f"Strategy: {strategy}")
    console.print(f"Dry Run: {dry_run}")
    console.print("[yellow]Coming in Phase 13[/yellow]")


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
