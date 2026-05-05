"""Unit tests for src.data.ingestion module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.ingestion import OHLCVIngestor, load_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_config() -> dict:
    """Minimal valid configuration for testing."""
    return {
        "assets": [
            {
                "symbol": "WDO",
                "yfinance_ticker": "WDO=F",
                "fallback_ticker": "USDBRL=X",
                "exchange": "B3",
                "type": "futures",
            },
            {
                "symbol": "WIN",
                "yfinance_ticker": "^BVSP",
                "fallback_ticker": "^BVSP",
                "exchange": "B3",
                "type": "futures",
            },
        ],
        "timeframes": ["1m", "5m"],
        "data": {"raw_dir": "data/raw", "processed_dir": "data/processed"},
        "ingestion": {"max_retries": 2, "retry_delay_seconds": 0},
    }


@pytest.fixture()
def sample_ohlcv_df() -> pd.DataFrame:
    """Synthetic OHLCV DataFrame mimicking yfinance output."""
    rng = np.random.default_rng(seed=42)
    n = 100
    dates = pd.date_range("2024-01-02 10:00", periods=n, freq="1min", tz="America/Sao_Paulo")
    close = 5000.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Open": close + rng.uniform(-5, 5, n),
            "High": close + rng.uniform(0, 10, n),
            "Low": close - rng.uniform(0, 10, n),
            "Close": close,
            "Volume": rng.integers(100, 10000, n),
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# Tests: load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_default_config(self) -> None:
        """Default config file should load without errors."""
        config = load_config()
        assert "assets" in config
        assert "timeframes" in config
        assert len(config["assets"]) >= 1

    def test_load_missing_config_raises(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")


# ---------------------------------------------------------------------------
# Tests: OHLCVIngestor instantiation
# ---------------------------------------------------------------------------


class TestOHLCVIngestorInit:
    """Tests for OHLCVIngestor initialization and validation."""

    def test_instantiation_with_valid_config(self, sample_config: dict, tmp_path: Path) -> None:
        """Ingestor should instantiate without errors given valid config."""
        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        assert ingestor._output_dir == tmp_path
        assert len(ingestor._assets) == 2
        assert len(ingestor._timeframes) == 2

    def test_empty_assets_raises(self, sample_config: dict, tmp_path: Path) -> None:
        """Should raise ValueError when assets list is empty."""
        sample_config["assets"] = []
        with pytest.raises(ValueError, match="at least one asset"):
            OHLCVIngestor(sample_config, output_dir=tmp_path)

    def test_empty_timeframes_raises(self, sample_config: dict, tmp_path: Path) -> None:
        """Should raise ValueError when timeframes list is empty."""
        sample_config["timeframes"] = []
        with pytest.raises(ValueError, match="at least one entry in 'timeframes'"):
            OHLCVIngestor(sample_config, output_dir=tmp_path)

    def test_missing_symbol_key_raises(self, sample_config: dict, tmp_path: Path) -> None:
        """Should raise ValueError when an asset lacks 'symbol'."""
        sample_config["assets"] = [{"yfinance_ticker": "FOO"}]
        with pytest.raises(ValueError, match="'symbol' and 'yfinance_ticker'"):
            OHLCVIngestor(sample_config, output_dir=tmp_path)

    def test_unsupported_timeframe_raises(self, sample_config: dict, tmp_path: Path) -> None:
        """Should raise ValueError for unsupported timeframe."""
        sample_config["timeframes"] = ["15m"]
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            OHLCVIngestor(sample_config, output_dir=tmp_path)

    def test_output_dir_created(self, sample_config: dict, tmp_path: Path) -> None:
        """Output directory should be created if it does not exist."""
        out = tmp_path / "nested" / "output"
        OHLCVIngestor(sample_config, output_dir=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: DataFrame cleaning
# ---------------------------------------------------------------------------


class TestCleanDataframe:
    """Tests for OHLCVIngestor._clean_dataframe."""

    def test_columns_lowercased(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Output columns should be lowercase OHLCV."""
        cleaned = OHLCVIngestor._clean_dataframe(sample_ohlcv_df)
        assert list(cleaned.columns) == ["open", "high", "low", "close", "volume"]

    def test_extra_columns_dropped(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Extra yfinance columns (Dividends, Stock Splits) should be removed."""
        cleaned = OHLCVIngestor._clean_dataframe(sample_ohlcv_df)
        assert "dividends" not in cleaned.columns
        assert "stock splits" not in cleaned.columns

    def test_timezone_converted_to_utc(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Index timezone should be converted to UTC."""
        cleaned = OHLCVIngestor._clean_dataframe(sample_ohlcv_df)
        assert str(cleaned.index.tz) == "UTC"

    def test_index_name_is_datetime(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Index should be named 'datetime'."""
        cleaned = OHLCVIngestor._clean_dataframe(sample_ohlcv_df)
        assert cleaned.index.name == "datetime"

    def test_missing_columns_raises(self) -> None:
        """Should raise ValueError when required columns are missing."""
        df = pd.DataFrame({"Open": [1], "High": [2]})
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            OHLCVIngestor._clean_dataframe(df)

    def test_nan_rows_dropped(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Rows where all OHLCV values are NaN should be dropped."""
        sample_ohlcv_df.iloc[0] = np.nan
        cleaned = OHLCVIngestor._clean_dataframe(sample_ohlcv_df)
        assert len(cleaned) == len(sample_ohlcv_df) - 1


# ---------------------------------------------------------------------------
# Tests: download with mocked yfinance
# ---------------------------------------------------------------------------


class TestDownload:
    """Tests for download logic with mocked yfinance."""

    @patch("src.data.ingestion.yf.Ticker")
    def test_download_single_returns_dataframe(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        sample_ohlcv_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """download_single should return a cleaned DataFrame."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_ohlcv_df
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result = ingestor.download_single("WDO=F", "1m")

        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["open", "high", "low", "close", "volume"]

    @patch("src.data.ingestion.yf.Ticker")
    def test_download_returns_none_on_empty(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        tmp_path: Path,
    ) -> None:
        """download_single should return None when yfinance returns empty DF."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result = ingestor.download_single("INVALID", "1m")

        assert result is None

    @patch("src.data.ingestion.yf.Ticker")
    def test_retry_on_exception(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        sample_ohlcv_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Should retry on exception and succeed on second attempt."""
        mock_instance = MagicMock()
        mock_instance.history.side_effect = [
            ConnectionError("Network error"),
            sample_ohlcv_df,
        ]
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result = ingestor.download_single("WDO=F", "1m")

        assert result is not None
        assert mock_instance.history.call_count == 2


# ---------------------------------------------------------------------------
# Tests: full run with mocked yfinance
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the full ingestion run."""

    @patch("src.data.ingestion.yf.Ticker")
    def test_run_writes_parquet_files(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        sample_ohlcv_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """run() should write Parquet files for each asset/timeframe pair."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_ohlcv_df
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result_paths = ingestor.run()

        # 2 assets * 2 timeframes = 4 files
        assert len(result_paths) == 4
        for path in result_paths:
            assert path.exists()
            assert path.suffix == ".parquet"
            # Verify the parquet is readable
            df = pd.read_parquet(path)
            assert len(df) > 0
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    @patch("src.data.ingestion.yf.Ticker")
    def test_run_with_all_failures_returns_empty(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        tmp_path: Path,
    ) -> None:
        """run() should return empty list when all downloads fail."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result_paths = ingestor.run()

        assert result_paths == []

    @patch("src.data.ingestion.yf.Ticker")
    def test_parquet_naming_convention(
        self,
        mock_ticker_cls: MagicMock,
        sample_config: dict,
        sample_ohlcv_df: pd.DataFrame,
        tmp_path: Path,
    ) -> None:
        """Parquet filenames should follow {symbol}_{timeframe}_{date} pattern."""
        # Use only one asset and one timeframe for simpler assertion
        sample_config["assets"] = [sample_config["assets"][0]]
        sample_config["timeframes"] = ["1m"]

        mock_instance = MagicMock()
        mock_instance.history.return_value = sample_ohlcv_df
        mock_ticker_cls.return_value = mock_instance

        ingestor = OHLCVIngestor(sample_config, output_dir=tmp_path)
        result_paths = ingestor.run()

        assert len(result_paths) == 1
        filename = result_paths[0].name
        assert filename.startswith("WDO_1m_")
        assert filename.endswith(".parquet")
