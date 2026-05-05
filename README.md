# Quant Trading System

Quantitative trading system for B3 futures (WDO/WIN) with regime-based setup classification using machine learning.

## Quick Start

```bash
# Install dependencies
make install

# Run linter
make lint

# Run tests
make test

# Ingest OHLCV data
make ingest

# Format code
make format
```

## Project Structure

```
quant-trading-system/
├── configs/              # YAML configuration files
│   └── default.yaml      # Default asset/timeframe/ingestion settings
├── data/
│   ├── raw/              # Raw OHLCV parquet files from ingestion
│   └── processed/        # Cleaned and transformed datasets
├── notebooks/            # Exploratory analysis (never production)
├── src/
│   ├── data/             # Data ingestion, cleaning, storage
│   │   └── ingestion.py  # OHLCV downloader via yfinance
│   ├── features/         # Feature engineering (point-in-time)
│   ├── models/           # Training, evaluation, serialization
│   └── utils/            # Helpers, logging, configuration
├── tests/                # pytest test suite
├── Makefile              # Convenience commands
└── pyproject.toml        # Project metadata and dependencies
```

## Supported Assets

| Symbol | Description              | yfinance Ticker | Fallback     |
|--------|--------------------------|-----------------|--------------|
| WDO    | Mini dolar futuro (B3)   | `WDO=F`         | `USDBRL=X`   |
| WIN    | Mini indice futuro (B3)  | `^BVSP`         | `^BVSP`      |

## Supported Timeframes

- **1 minute** (`1m`) -- up to 7 days of history
- **5 minutes** (`5m`) -- up to 60 days of history

## Requirements

- Python >= 3.10
- Dependencies managed via `pyproject.toml` (install with `pip install -e ".[dev]"`)
