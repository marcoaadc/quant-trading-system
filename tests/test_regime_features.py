"""Tests for RegimeFeatureEngineer -- feature correctness and data leakage.

Each test uses synthetic OHLCV data with a fixed RNG seed for
reproducibility. The key invariants tested are:
    1. Features are computed correctly (numerical spot-checks).
    2. Warm-up rows are removed (no NaN in output).
    3. Z-scores are clipped to [-5, +5].
    4. Volume relative uses shift(1) -- zero look-ahead bias.
    5. Stationarity validation runs without errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.regime_features import (
    FEATURE_COLUMNS,
    RegimeFeatureEngineer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_ohlcv(
    n_rows: int = 500,
    seed: int = 42,
    n_days: int = 5,
) -> pd.DataFrame:
    """Create synthetic OHLCV data spanning multiple trading days.

    Args:
        n_rows: Total number of candles.
        seed: RNG seed for reproducibility.
        n_days: Number of trading days to simulate.

    Returns:
        DataFrame with columns open, high, low, close, volume and a
        UTC DatetimeIndex.
    """
    rng = np.random.default_rng(seed)

    # Simulate a random walk for close prices
    log_returns = rng.normal(0, 0.001, size=n_rows)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    # Generate high/low with realistic spread around close
    spread = np.abs(rng.normal(0, 0.002, size=n_rows)) * close
    high = close + spread
    low = close - spread
    open_prices = close + rng.normal(0, 0.0005, size=n_rows) * close

    # Volume: base level with some randomness
    volume = rng.integers(1000, 10000, size=n_rows).astype(float)

    # Create multi-day datetime index (78 candles per day for 5min bars)
    candles_per_day = n_rows // n_days
    dates = []
    for d in range(n_days):
        base = pd.Timestamp(f"2026-05-0{d + 1} 09:00:00", tz="UTC")
        day_dates = pd.date_range(
            start=base, periods=candles_per_day, freq="5min"
        )
        dates.extend(day_dates)

    # Trim or pad to match n_rows
    dates = dates[:n_rows]
    if len(dates) < n_rows:
        extra = pd.date_range(
            start=dates[-1] + pd.Timedelta(minutes=5),
            periods=n_rows - len(dates),
            freq="5min",
        )
        dates.extend(extra)

    index = pd.DatetimeIndex(dates, name="datetime")

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Standard synthetic OHLCV DataFrame for tests."""
    return _make_synthetic_ohlcv(n_rows=500, seed=42, n_days=5)


@pytest.fixture
def engineer() -> RegimeFeatureEngineer:
    """Standard RegimeFeatureEngineer with default parameters."""
    return RegimeFeatureEngineer()


# ---------------------------------------------------------------------------
# Tests: individual feature computation
# ---------------------------------------------------------------------------


class TestLogReturns:
    """Tests for log return computation."""

    def test_first_value_is_nan(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        assert np.isnan(lr.iloc[0])

    def test_second_value_correct(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        expected = np.log(
            synthetic_ohlcv["close"].iloc[1]
            / synthetic_ohlcv["close"].iloc[0]
        )
        assert np.isclose(lr.iloc[1], expected, atol=1e-12)

    def test_series_name(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        assert lr.name == "log_return"

    def test_length_matches_input(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        assert len(lr) == len(synthetic_ohlcv)


class TestLogRealizedVolatility:
    """Tests for log realized volatility computation."""

    def test_first_window_values_are_nan(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        lrv = engineer.compute_log_realized_volatility(lr)
        # First vol_window values of log_returns are needed (plus 1 NaN
        # from log_returns itself), so first vol_window values should be NaN
        assert lrv.iloc[: engineer.vol_window].isna().all()

    def test_values_after_window_are_finite(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        lrv = engineer.compute_log_realized_volatility(lr)
        valid = lrv.iloc[engineer.vol_window + 1 :]
        assert valid.notna().all()
        assert np.isfinite(valid).all()

    def test_manual_computation(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        lrv = engineer.compute_log_realized_volatility(lr)

        # Check at a specific index (vol_window + 5)
        idx = engineer.vol_window + 5
        window_returns = lr.iloc[idx - engineer.vol_window + 1 : idx + 1]
        expected_vol = np.sqrt((window_returns**2).sum())
        expected = np.log(expected_vol + engineer.epsilon)
        assert np.isclose(lrv.iloc[idx], expected, atol=1e-10)


class TestMomentumZscore:
    """Tests for momentum z-score computation."""

    def test_clipping_bounds(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        mz = engineer.compute_momentum_zscore(lr)
        valid = mz.dropna()
        assert valid.max() <= engineer.clip_range
        assert valid.min() >= -engineer.clip_range

    def test_series_name(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        lr = engineer.compute_log_returns(synthetic_ohlcv)
        mz = engineer.compute_momentum_zscore(lr)
        assert mz.name == "momentum_zscore"


class TestLogRelativeVolume:
    """Tests for log relative volume -- including look-ahead check."""

    def test_shift_prevents_lookahead(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Verify that volume at time t is NOT used in its own baseline.

        Strategy: modify volume at a single timestamp and check that the
        feature value at that same timestamp does NOT change (only future
        values should change).
        """
        df_original = synthetic_ohlcv.copy()
        df_modified = synthetic_ohlcv.copy()

        # Pick an index well past warm-up
        test_idx = 150

        # Dramatically increase volume at test_idx
        df_modified.iloc[test_idx, df_modified.columns.get_loc("volume")] = (
            999999.0
        )

        feat_orig = engineer.compute_log_relative_volume(df_original)
        feat_mod = engineer.compute_log_relative_volume(df_modified)

        # The baseline mean at test_idx should NOT be affected by the
        # volume change at test_idx (because of shift(1))
        # But the feature value itself WILL change because the numerator
        # V_t changes. The key check is that the DENOMINATOR doesn't use V_t.

        # Check the value at test_idx + 1: the baseline should now include
        # the modified volume, so it SHOULD differ
        # But at test_idx, the baseline should be identical
        # We can verify by checking that the relative volume at test_idx
        # changes proportionally to the numerator change only
        vol_orig = df_original["volume"].iloc[test_idx]
        vol_mod = df_modified["volume"].iloc[test_idx]
        ratio = (vol_mod + engineer.epsilon) / (vol_orig + engineer.epsilon)
        expected_diff = np.log(ratio)

        actual_diff = feat_mod.iloc[test_idx] - feat_orig.iloc[test_idx]
        assert np.isclose(actual_diff, expected_diff, atol=1e-8), (
            f"Volume baseline at t should not include V_t. "
            f"Expected diff={expected_diff:.6f}, got={actual_diff:.6f}"
        )

    def test_values_are_finite_after_warmup(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        feat = engineer.compute_log_relative_volume(synthetic_ohlcv)
        # After vol_rel_window + 1 (shift) values should be valid
        valid = feat.iloc[engineer.vol_rel_window + 1 :]
        assert valid.notna().all()
        assert np.isfinite(valid).all()


class TestRangeZscore:
    """Tests for range z-score computation."""

    def test_clipping_bounds(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        rz = engineer.compute_range_zscore(synthetic_ohlcv)
        valid = rz.dropna()
        assert valid.max() <= engineer.clip_range
        assert valid.min() >= -engineer.clip_range

    def test_positive_range(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Normalized range ln(H/L) should be non-negative before z-score."""
        raw_range = np.log(
            synthetic_ohlcv["high"] / synthetic_ohlcv["low"]
        )
        assert (raw_range >= 0).all()


# ---------------------------------------------------------------------------
# Tests: composite feature computation
# ---------------------------------------------------------------------------


class TestComputeAllFeatures:
    """Tests for the composite compute_all_features method."""

    def test_output_columns(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        assert list(features.columns) == FEATURE_COLUMNS

    def test_no_nan_after_warmup(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        assert features.isna().sum().sum() == 0

    def test_warmup_removed(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        # Output should have fewer rows than input (warm-up + session starts)
        assert len(features) < len(synthetic_ohlcv)

    def test_session_starts_removed(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Verify that the original first candle of each day is removed.

        After warm-up removal, each day's first remaining candle is NOT
        the original 09:00 candle (which was dropped). We check that
        the original session-start timestamps are absent from the output.
        """
        features = engineer.compute_all_features(synthetic_ohlcv)
        if hasattr(features.index, "date"):
            # Identify original session-start timestamps
            orig_dates = pd.Series(
                synthetic_ohlcv.index.date, index=synthetic_ohlcv.index
            )
            orig_first_mask = orig_dates != orig_dates.shift(1)
            session_start_timestamps = synthetic_ohlcv.index[orig_first_mask]

            # None of the original session starts should be in features
            overlap = features.index.isin(session_start_timestamps)
            assert overlap.sum() == 0, (
                "Original session-start candles should be removed"
            )

    def test_zscore_features_are_clipped(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        for col in ["momentum_zscore", "range_zscore"]:
            assert features[col].max() <= engineer.clip_range
            assert features[col].min() >= -engineer.clip_range

    def test_missing_columns_raises(
        self, engineer: RegimeFeatureEngineer
    ) -> None:
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="missing columns"):
            engineer.compute_all_features(df)

    def test_sufficient_rows_for_hmm(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        # Spec requires >= 100 rows after warm-up for HMM training
        assert len(features) >= 100


# ---------------------------------------------------------------------------
# Tests: stationarity validation
# ---------------------------------------------------------------------------


class TestStationarityValidation:
    """Tests for ADF + KPSS stationarity validation."""

    def test_returns_all_features(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        results = engineer.validate_stationarity(features)
        for col in FEATURE_COLUMNS:
            assert col in results

    def test_result_fields(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        features = engineer.compute_all_features(synthetic_ohlcv)
        results = engineer.validate_stationarity(features)
        for result in results.values():
            assert hasattr(result, "adf_statistic")
            assert hasattr(result, "adf_pvalue")
            assert hasattr(result, "kpss_statistic")
            assert hasattr(result, "kpss_pvalue")
            assert result.conclusion in {
                "stationary",
                "non-stationary",
                "stationary_with_trend",
                "inconclusive",
            }

    def test_log_returns_are_stationary(
        self, engineer: RegimeFeatureEngineer, synthetic_ohlcv: pd.DataFrame
    ) -> None:
        """Log returns of a random walk should be stationary."""
        features = engineer.compute_all_features(synthetic_ohlcv)
        results = engineer.validate_stationarity(features)
        lr_result = results["log_return"]
        # ADF should reject H0 (series is stationary)
        assert lr_result.adf_reject_h0, (
            f"Log returns should be stationary. ADF p-value={lr_result.adf_pvalue}"
        )
