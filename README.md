# Crypto Quant Research Terminal

A production-grade quantitative research and trading system for cryptocurrency markets.

## Overview

Crypto Quant Terminal is a comprehensive quantitative research laboratory, backtesting engine, strategy discovery system, risk engine, and paper/live trading framework. It is designed for serious quantitative research with emphasis on correctness, reproducibility, and robustness.

### Key Features

- **Historical Data Engine**: Download and manage 6+ years of cryptocurrency market data
- **Feature Engineering**: 50+ technical indicators with anti-look-ahead bias protection
- **Strategy Research**: Automated strategy discovery and optimization
- **Backtesting Engine**: Event-driven backtesting with realistic execution assumptions
- **Risk Management**: Dynamic position sizing and comprehensive risk controls
- **Machine Learning**: ML-enhanced trade probability prediction
- **Walk-Forward Validation**: Rigorous out-of-sample testing
- **Monte Carlo Simulation**: Statistical risk assessment
- **Paper Trading**: Live simulation without real money
- **Live Trading**: Binance integration with safety controls (disabled by default)

## Installation

### Prerequisites

- Python 3.12+
- pip or poetry

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd crypto_quant_terminal

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env

# Initialize the system
python -m crypto_quant init
```

### Database Setup

```bash
# Initialize database
python -m crypto_quant init --init-db
```

## Configuration

### Main Configuration

Edit `config/settings.yaml` to configure:

- **Market**: Exchange, spot/futures settings
- **Data**: Date ranges, API endpoints
- **Universe**: Asset selection (top N, custom, etc.)
- **Risk**: Capital, risk per trade, position limits
- **Trading**: Mode (paper/live), execution settings
- **Fees**: Trading fees, slippage assumptions

### Environment Variables

Create `.env` file with:

```bash
# Binance API (for live trading)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Trading mode
TRADING_MODE=paper
DRY_RUN=true

# Logging
LOG_LEVEL=INFO
```

## Usage

### Terminal Interface

```bash
# Launch the terminal
python -m crypto_quant

# Or use the short command
cq
```

### CLI Commands

```bash
# Initialize system
python -m crypto_quant init

# Data management
python -m crypto_quant data download --symbol BTCUSDT --timeframe 1h

# Research
python -m crypto_quant research --strategy trend --timeframe 1h

# Backtesting
python -m crypto_quant backtest --strategy STRAT-001

# Machine Learning
python -m crypto_quant ml train --model xgboost

# Dashboard
python -m crypto_quant dashboard generate --experiment EXP-001

# Paper Trading
python -m crypto_quant paper start --strategy STRAT-001
```

## Architecture

```
crypto_quant_terminal/
├── config/                    # Configuration files
├── src/crypto_quant/          # Main source code
│   ├── cli/                   # Terminal interface
│   ├── data/                  # Data engine
│   ├── indicators/            # Technical indicators
│   ├── features/              # Feature engineering
│   ├── strategies/            # Strategy framework
│   ├── backtesting/           # Backtesting engine
│   ├── research/              # Research & optimization
│   ├── analysis/              # Trade analysis
│   ├── ml/                    # Machine learning
│   ├── validation/            # Train/test, walk-forward
│   ├── risk/                  # Risk management
│   ├── execution/             # Paper & live brokers
│   ├── exchange/              # Exchange adapters
│   ├── dashboard/             # Dashboard generation
│   ├── db/                    # Database models
│   ├── config/                # Configuration system
│   └── utils/                 # Utilities
├── tests/                     # Test suite
├── data/                      # Market data
├── models/                    # ML models
├── reports/                   # Generated reports
├── logs/                      # Application logs
└── dashboard/                 # Generated dashboards
```

## Data Management

### Downloading Data

```bash
# Download specific symbol
python -m crypto_quant data download --symbol BTCUSDT

# Download all top 20
python -m crypto_quant data download --universe top_20

# Download specific timeframes
python -m crypto_quant data download --timeframes 1h,4h,1d
```

### Data Validation

```bash
# Validate data quality
python -m crypto_quant data validate

# Check for gaps
python -m crypto_quant data validate --check-gaps
```

## Research Workflow

### 1. Download Historical Data

```bash
python -m crypto_quant data download --universe top_20 --start 2020-01-01
```

### 2. Run Research

```bash
python -m crypto_quant research --full-research
```

### 3. Analyze Results

```bash
python -m crypto_quant dashboard generate --experiment EXP-001
```

### 4. Validate Best Strategies

```bash
python -m crypto_quant validate walk-forward --strategy STRAT-001
```

### 5. Paper Trade

```bash
python -m crypto_quant paper start --strategy STRAT-001
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=crypto_quant

# Run specific test file
pytest tests/test_config.py

# Run slow tests
pytest -m slow
```

## Risk Warnings

**IMPORTANT**: This system is for research and educational purposes.

- Historical performance does NOT guarantee future results
- Backtests are simulations and may not reflect real trading
- ML predictions are probabilistic, not certain
- Trading involves substantial risk of financial loss
- Never trade with money you cannot afford to lose
- Past performance is not indicative of future results

## Development

### Adding New Strategies

1. Create strategy class in `src/crypto_quant/strategies/`
2. Inherit from `BaseStrategy`
3. Implement `generate_signal()` method
4. Register in strategy registry

### Adding New Indicators

1. Create indicator function in `src/crypto_quant/indicators/`
2. Implement anti-look-ahead bias protection
3. Add tests in `tests/`

### Contributing

1. Fork the repository
2. Create feature branch
3. Write tests
4. Submit pull request

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
- Check documentation
- Review test files for examples
- Open GitHub issue
