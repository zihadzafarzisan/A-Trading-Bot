"""Global constants for the crypto quant terminal."""

# Timeframe constants
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Timeframe labels
TIMEFRAME_LABELS = {
    "1m": "1 Minute",
    "5m": "5 Minutes",
    "15m": "15 Minutes",
    "30m": "30 Minutes",
    "1h": "1 Hour",
    "4h": "4 Hours",
    "1d": "1 Day",
}

# Market types
MARKET_SPOT = "spot"
MARKET_FUTURES = "futures"

# Trade directions
DIRECTION_LONG = "long"
DIRECTION_SHORT = "short"

# Trade status
TRADE_OPEN = "open"
TRADE_CLOSED = "closed"
TRADE_CANCELLED = "cancelled"

# Exit reasons
EXIT_TP = "tp"
EXIT_SL = "sl"
EXIT_SIGNAL = "signal"
EXIT_TIME = "time"
EXIT_LIQUIDATION = "liquidation"

# Risk event types
RISK_MAX_POSITIONS = "max_positions"
RISK_DAILY_LOSS = "daily_loss"
RISK_WEEKLY_LOSS = "weekly_loss"
RISK_MAX_DRAWDOWN = "max_drawdown"
RISK_LEVERAGE = "leverage"
RISK_EXPOSURE = "exposure"

# Risk severity
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

# Experiment status
EXPERIMENT_PENDING = "pending"
EXPERIMENT_RUNNING = "running"
EXPERIMENT_COMPLETED = "completed"
EXPERIMENT_FAILED = "failed"

# Strategy types
STRATEGY_TREND = "trend"
STRATEGY_MOMENTUM = "momentum"
STRATEGY_MEAN_REVERSION = "mean_reversion"
STRATEGY_BREAKOUT = "breakout"
STRATEGY_COMBINED = "combined"

# Optimization objectives
OBJECTIVE_MAX_WIN_RATE = "max_win_rate"
OBJECTIVE_MAX_PROFIT_FACTOR = "max_profit_factor"
OBJECTIVE_MAX_SHARPE = "max_sharpe"
OBJECTIVE_MAX_RETURN = "max_return"
OBJECTIVE_MIN_DRAWDOWN = "min_drawdown"
OBJECTIVE_BEST_OVERALL = "best_overall"

# Search types
SEARCH_GRID = "grid"
SEARCH_RANDOM = "random"
SEARCH_BAYESIAN = "bayesian"

# Execution modes
MODE_SIGNAL_ONLY = "signal_only"
MODE_PAPER = "paper"
MODE_LIVE = "live"

# Trading style
STYLE_SCALPING = "scalping"
STYLE_DAY_TRADING = "day_trading"
STYLE_SWING = "swing_trading"
STYLE_POSITION = "position_trading"

# Market regimes
REGIME_BULL = "bull"
REGIME_BEAR = "bear"
REGIME_SIDEWAYS = "sideways"
REGIME_HIGH_VOL = "high_volatility"
REGIME_LOW_VOL = "low_volatility"

# Default Binance symbols for testing
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "LINKUSDT",
]

# Historical top 20 (approximation, documented as limitation)
HISTORICAL_UNIVERSE = {
    "2020": [
        "BTCUSDT", "ETHUSDT", "XRPUSDT", "BCHUSDT", "LTCUSDT",
        "EOSUSDT", "BNBUSDT", "ADAUSDT", "XLMUSDT", "TRXUSDT",
        "BSVUSDT", "MIOTAUSDT", "DASHUSDT", "NEOUSDT", "ATOMUSDT",
        "ETCUSDT", "XMRUSDT", "XTZUSDT", "ATOMUSDT", "VETUSDT",
    ],
    "2021": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT",
        "LUNAUSDT", "UNIUSDT", "ATOMUSDT", "LTCUSDT", "ALGOUSDT",
        "FILUSDT", "NEARUSDT", "ICPUSDT", "MATICUSDT", "ETCUSDT",
    ],
    "2022": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT",
        "SHIBUSDT", "TRXUSDT", "LTCUSDT", "UNIUSDT", "LINKUSDT",
        "ATOMUSDT", "XLMUSDT", "ALGOUSDT", "NEARUSDT", "APTUSDT",
    ],
    "2023": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
        "SOLUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT",
        "TRXUSDT", "LINKUSDT", "SHIBUSDT", "LTCUSDT", "UNIUSDT",
        "ATOMUSDT", "XLMUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT",
    ],
    "2024": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "SHIBUSDT",
        "TRXUSDT", "LINKUSDT", "MATICUSDT", "LTCUSDT", "UNIUSDT",
        "NEARUSDT", "APTUSDT", "SUIUSDT", "ICPUSDT", "FILUSDT",
    ],
    "2025": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "TRXUSDT", "SHIBUSDT", "LTCUSDT", "NEARUSDT", "APTUSDT",
        "SUIUSDT", "ICPUSDT", "UNIUSDT", "MATICUSDT", "FILUSDT",
    ],
    "2026": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT",
        "TRXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "ICPUSDT",
        "SHIBUSDT", "LTCUSDT", "UNIUSDT", "FILUSDT", "MATICUSDT",
    ],
}
