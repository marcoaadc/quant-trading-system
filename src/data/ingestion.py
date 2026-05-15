"""OHLCV data ingestion module for B3 futures (WDO/WIN).

Downloads OHLCV data via yfinance for configured assets and timeframes,
saving results as Parquet files in the raw data directory.

Notes on B3 futures via yfinance:
    - WDO (mini dolar futuro): Primary ticker ``WDO=F``. If unavailable,
      falls back to ``USDBRL=X`` (USD/BRL spot) as a price-action proxy.
    - WIN (mini indice futuro): Uses ``^BVSP`` (Ibovespa index) as proxy
      since individual mini-index contracts have limited history.
    - Intraday data limits: yfinance provides at most 7 days of 1-minute
      data and 60 days of 5-minute data.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import yfinance as yf
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"

_PERIOD_BY_TIMEFRAME: dict[str, str] = {
    "1m": "7d",
    "5m": "60d",
}

_REQUIRED_OHLCV_COLUMNS: list[str] = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config. Defaults to ``configs/default.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh)

    logger.info("Loaded configuration from {}", path)
    return config


# ---------------------------------------------------------------------------
# OHLCVIngestor
# ---------------------------------------------------------------------------


class OHLCVIngestor:
    """Download and persist OHLCV data from yfinance.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).
        output_dir: Override for the raw data output directory. When *None*,
            uses the path specified in ``config["data"]["raw_dir"]``.

    Example::

        config = load_config()
        ingestor = OHLCVIngestor(config)
        results = ingestor.run()
    """

    def __init__(self, config: dict[str, Any], output_dir: Path | None = None) -> None:
        self._config = config
        self._assets: list[dict[str, str]] = config.get("assets", [])
        self._timeframes: list[str] = config.get("timeframes", [])
        self._output_dir = output_dir or Path(config.get("data", {}).get("raw_dir", "data/raw"))
        self._max_retries: int = config.get("ingestion", {}).get("max_retries", 3)
        self._retry_delay: int = config.get("ingestion", {}).get("retry_delay_seconds", 5)

        self._validate_config()
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def run(self) -> list[Path]:
        """Execute ingestion for all configured assets and timeframes.

        Returns:
            List of paths to successfully written Parquet files.
        """
        written_files: list[Path] = []

        for asset in self._assets:
            symbol = asset["symbol"]
            primary_ticker = asset["yfinance_ticker"]
            fallback_ticker = asset.get("fallback_ticker")

            for timeframe in self._timeframes:
                logger.info("Ingesting {} @ {} ...", symbol, timeframe)
                df = self._download_with_fallback(
                    symbol=symbol,
                    primary_ticker=primary_ticker,
                    fallback_ticker=fallback_ticker,
                    timeframe=timeframe,
                )
                if df is None or df.empty:
                    logger.warning("No data returned for {} @ {}. Skipping.", symbol, timeframe)
                    continue

                output_path = self._save_parquet(df, symbol, timeframe)
                written_files.append(output_path)
                logger.info(
                    "Saved {} rows for {} @ {} -> {}",
                    len(df),
                    symbol,
                    timeframe,
                    output_path,
                )

        logger.info("Ingestion complete. {} files written.", len(written_files))
        return written_files

    def download_single(
        self,
        ticker: str,
        timeframe: str,
        period: str | None = None,
    ) -> pd.DataFrame | None:
        """Download OHLCV data for a single ticker/timeframe.

        Args:
            ticker: yfinance ticker symbol.
            timeframe: Candle interval (``"1m"``, ``"5m"``).
            period: Lookback period string (e.g. ``"7d"``). Defaults to the
                maximum period allowed for the given *timeframe*.

        Returns:
            DataFrame with OHLCV columns, or *None* if download fails.
        """
        resolved_period = period or _PERIOD_BY_TIMEFRAME.get(timeframe, "7d")
        return self._download_with_retry(ticker, timeframe, resolved_period)

    # -- private helpers ----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate minimum required configuration fields."""
        if not self._assets:
            raise ValueError("Configuration must include at least one asset in 'assets'.")
        if not self._timeframes:
            raise ValueError("Configuration must include at least one entry in 'timeframes'.")

        for asset in self._assets:
            if "symbol" not in asset or "yfinance_ticker" not in asset:
                raise ValueError(f"Each asset must have 'symbol' and 'yfinance_ticker' keys. Got: {asset}")

        for tf in self._timeframes:
            if tf not in _PERIOD_BY_TIMEFRAME:
                raise ValueError(f"Unsupported timeframe '{tf}'. Supported: {list(_PERIOD_BY_TIMEFRAME.keys())}")

    def _download_with_fallback(
        self,
        symbol: str,
        primary_ticker: str,
        fallback_ticker: str | None,
        timeframe: str,
    ) -> pd.DataFrame | None:
        """Try primary ticker first; fall back to secondary if it fails."""
        period = _PERIOD_BY_TIMEFRAME.get(timeframe, "7d")

        df = self._download_with_retry(primary_ticker, timeframe, period)
        if df is not None and not df.empty:
            return df

        if fallback_ticker and fallback_ticker != primary_ticker:
            logger.warning(
                "Primary ticker '{}' failed for {}. Trying fallback '{}'.",
                primary_ticker,
                symbol,
                fallback_ticker,
            )
            df = self._download_with_retry(fallback_ticker, timeframe, period)
            if df is not None and not df.empty:
                return df

        return None

    def _download_with_retry(
        self,
        ticker: str,
        timeframe: str,
        period: str,
    ) -> pd.DataFrame | None:
        """Download data with retry logic.

        Args:
            ticker: yfinance ticker symbol.
            timeframe: Candle interval.
            period: Lookback period.

        Returns:
            Validated DataFrame or *None* on failure.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                logger.debug(
                    "Attempt {}/{}: downloading {} (interval={}, period={})",
                    attempt,
                    self._max_retries,
                    ticker,
                    timeframe,
                    period,
                )
                ticker_obj = yf.Ticker(ticker)
                df: pd.DataFrame = ticker_obj.history(period=period, interval=timeframe)

                if df.empty:
                    logger.debug("Empty DataFrame returned for {}", ticker)
                    return None

                df = self._clean_dataframe(df)
                return df

            except (ConnectionError, TimeoutError, OSError) as exc:
                logger.warning(
                    "Attempt {}/{} failed for ticker '{}': {}",
                    attempt,
                    self._max_retries,
                    ticker,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay)

        logger.error("All {} attempts exhausted for ticker '{}'", self._max_retries, ticker)
        return None

    @staticmethod
    def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and validate OHLCV schema.

        Args:
            df: Raw DataFrame from yfinance.

        Returns:
            Cleaned DataFrame with standardized columns.

        Raises:
            ValueError: If required OHLCV columns are missing.
        """
        # yfinance may return extra columns (Dividends, Stock Splits).
        # Keep only OHLCV and rename to lowercase.
        available = [c for c in _REQUIRED_OHLCV_COLUMNS if c in df.columns]
        if len(available) < len(_REQUIRED_OHLCV_COLUMNS):
            missing = set(_REQUIRED_OHLCV_COLUMNS) - set(available)
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        df = df[_REQUIRED_OHLCV_COLUMNS].copy()
        df.columns = [c.lower() for c in df.columns]

        # Drop rows where all OHLCV values are NaN
        df = df.dropna(how="all")

        # Ensure index is tz-aware UTC datetime
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df.index.name = "datetime"
        return df

    def _save_parquet(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
        """Persist DataFrame as a Parquet file with a deterministic name.

        Naming convention: ``{symbol}_{timeframe}_{YYYYMMDD}.parquet``

        Args:
            df: Cleaned OHLCV DataFrame.
            symbol: Asset symbol (e.g. ``"WDO"``).
            timeframe: Candle interval (e.g. ``"1m"``).

        Returns:
            Path to the written Parquet file.
        """
        date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
        filename = f"{symbol}_{timeframe}_{date_str}.parquet"
        output_path = self._output_dir / filename
        df.to_parquet(output_path, engine="pyarrow", index=True)
        return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Ingest OHLCV data for B3 futures via yfinance.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for raw data",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Subset of symbols to ingest (default: all configured)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=None,
        help="Subset of timeframes to ingest (default: all configured)",
    )
    return parser


def main() -> None:
    """CLI entry point for OHLCV ingestion."""
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)

    # Filter assets/timeframes if CLI overrides were provided
    if args.symbols:
        config["assets"] = [a for a in config["assets"] if a["symbol"] in args.symbols]
        if not config["assets"]:
            logger.error("No matching assets found for symbols: {}", args.symbols)
            return

    if args.timeframes:
        config["timeframes"] = [t for t in config["timeframes"] if t in args.timeframes]
        if not config["timeframes"]:
            logger.error("No matching timeframes found: {}", args.timeframes)
            return

    ingestor = OHLCVIngestor(config, output_dir=args.output_dir)
    results = ingestor.run()

    if not results:
        logger.error("No data was ingested. Exiting with error.")
        sys.exit(1)


if __name__ == "__main__":
    main()
