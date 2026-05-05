"""Regime feature engineering with strict point-in-time guarantees.

Computes a 5-dimensional observation vector for HMM regime detection:
    1. Log returns
    2. Log realized volatility
    3. Momentum z-score (clipped)
    4. Log relative volume
    5. Range z-score (clipped)

All rolling windows are backward-looking (``center=False``, ``min_periods``
equal to window size). The volume baseline explicitly uses ``.shift(1)``
to exclude the current candle -- zero look-ahead bias.

Reference: docs/specs/sprint2_hmm_regime_detection.md
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.tsa.stattools import adfuller, kpss


# ---------------------------------------------------------------------------
# Feature column names (canonical order)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "log_return",
    "log_realized_vol",
    "momentum_zscore",
    "log_rel_volume",
    "range_zscore",
]


# ---------------------------------------------------------------------------
# Stationarity result container
# ---------------------------------------------------------------------------


@dataclass
class StationarityResult:
    """Result of ADF + KPSS stationarity tests for a single feature."""

    feature: str
    adf_statistic: float
    adf_pvalue: float
    adf_reject_h0: bool
    kpss_statistic: float
    kpss_pvalue: float
    kpss_reject_h0: bool
    conclusion: str


# ---------------------------------------------------------------------------
# RegimeFeatureEngineer
# ---------------------------------------------------------------------------


class RegimeFeatureEngineer:
    """Compute regime-detection features from OHLCV data.

    All features are strictly causal (backward-looking only). The volume
    baseline uses ``.shift(1)`` to prevent look-ahead bias.

    Args:
        vol_window: Window size *N* for realized volatility (default 20).
        mom_window: Window size *M* for momentum accumulation (default 10).
        vol_rel_window: Window size *K* for volume baseline (default 60).
        zscore_window: Window size *L* for rolling z-score (default 60).
        clip_range: Symmetric clipping bound for z-scores (default 5.0).
        epsilon: Small constant to avoid log(0) (default 1e-10).
    """

    def __init__(
        self,
        vol_window: int = 20,
        mom_window: int = 10,
        vol_rel_window: int = 60,
        zscore_window: int = 60,
        clip_range: float = 5.0,
        epsilon: float = 1e-10,
    ) -> None:
        self.vol_window = vol_window
        self.mom_window = mom_window
        self.vol_rel_window = vol_rel_window
        self.zscore_window = zscore_window
        self.clip_range = clip_range
        self.epsilon = epsilon

        # Warm-up = largest window required before features are valid
        self.warmup_period: int = max(
            vol_window, mom_window, vol_rel_window, zscore_window
        )

        logger.debug(
            "RegimeFeatureEngineer initialized: vol_window={}, mom_window={}, "
            "vol_rel_window={}, zscore_window={}, clip_range={}, epsilon={}, "
            "warmup_period={}",
            vol_window,
            mom_window,
            vol_rel_window,
            zscore_window,
            clip_range,
            epsilon,
            self.warmup_period,
        )

    # -- Individual feature methods -----------------------------------------

    def compute_log_returns(self, df: pd.DataFrame) -> pd.Series:
        """Compute log returns: r_t = ln(C_t / C_{t-1}).

        Args:
            df: OHLCV DataFrame with a ``close`` column.

        Returns:
            Series of log returns (first value is NaN).
        """
        log_returns: pd.Series = np.log(df["close"] / df["close"].shift(1))
        log_returns.name = "log_return"
        return log_returns

    def compute_log_realized_volatility(
        self, log_returns: pd.Series
    ) -> pd.Series:
        """Compute log realized volatility.

        sigma_t = sqrt(sum(r^2, N=vol_window))
        result  = ln(sigma_t + epsilon)

        Args:
            log_returns: Series of log returns.

        Returns:
            Series of log realized volatility values.
        """
        squared_returns = log_returns**2
        realized_var = squared_returns.rolling(
            window=self.vol_window, min_periods=self.vol_window
        ).sum()
        realized_vol = np.sqrt(realized_var)
        log_rv: pd.Series = np.log(realized_vol + self.epsilon)
        log_rv.name = "log_realized_vol"
        return log_rv

    def compute_momentum_zscore(self, log_returns: pd.Series) -> pd.Series:
        """Compute momentum z-score (clipped).

        Step 1: mom_t = sum(r, M=mom_window)
        Step 2: z_t   = (mom_t - rolling_mean) / rolling_std, L=zscore_window
        Step 3: clip to [-clip_range, +clip_range]

        Args:
            log_returns: Series of log returns.

        Returns:
            Series of clipped momentum z-scores.
        """
        momentum = log_returns.rolling(
            window=self.mom_window, min_periods=self.mom_window
        ).sum()

        rolling_mean = momentum.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).mean()
        rolling_std = momentum.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).std()

        zscore = (momentum - rolling_mean) / rolling_std
        zscore_clipped: pd.Series = zscore.clip(
            lower=-self.clip_range, upper=self.clip_range
        )
        zscore_clipped.name = "momentum_zscore"
        return zscore_clipped

    def compute_log_relative_volume(self, df: pd.DataFrame) -> pd.Series:
        """Compute log relative volume.

        vrel_t = V_t / mean(V_{t-1} ... V_{t-K})
        result = ln(vrel_t + epsilon)

        CRITICAL: The volume baseline uses ``.shift(1)`` to exclude the
        current candle, preventing look-ahead bias.

        When volume data is unavailable (all zeros, as happens with
        yfinance proxy tickers like USDBRL=X or ^BVSP), the feature
        is set to 0.0 (neutral) to avoid NaN propagation.

        Args:
            df: OHLCV DataFrame with a ``volume`` column.

        Returns:
            Series of log relative volume values.
        """
        volume = df["volume"].astype(float)

        # If all volume is zero (proxy data without volume), return
        # neutral feature (0.0 = ln(1) = no relative change)
        if (volume == 0).all() or volume.sum() == 0:
            logger.warning(
                "Volume is all zeros -- likely a proxy ticker without "
                "volume data. Setting log_rel_volume to 0.0 (neutral)."
            )
            neutral = pd.Series(0.0, index=df.index, name="log_rel_volume")
            return neutral

        # .shift(1) ensures we use V_{t-1}..V_{t-K} (excludes current candle)
        volume_mean = (
            volume.shift(1)
            .rolling(
                window=self.vol_rel_window,
                min_periods=self.vol_rel_window,
            )
            .mean()
        )

        # Guard against zero denominator (sparse volume periods)
        volume_mean = volume_mean.replace(0.0, np.nan)

        relative_volume = volume / volume_mean
        log_rv: pd.Series = np.log(relative_volume + self.epsilon)
        log_rv.name = "log_rel_volume"
        return log_rv

    def compute_range_zscore(self, df: pd.DataFrame) -> pd.Series:
        """Compute range z-score (clipped).

        NR_t  = ln(H_t / L_t)
        z_t   = (NR_t - rolling_mean) / rolling_std, L=zscore_window
        clip to [-clip_range, +clip_range]

        Args:
            df: OHLCV DataFrame with ``high`` and ``low`` columns.

        Returns:
            Series of clipped range z-scores.
        """
        normalized_range = np.log(df["high"] / df["low"])

        rolling_mean = normalized_range.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).mean()
        rolling_std = normalized_range.rolling(
            window=self.zscore_window, min_periods=self.zscore_window
        ).std()

        zscore = (normalized_range - rolling_mean) / rolling_std
        zscore_clipped: pd.Series = zscore.clip(
            lower=-self.clip_range, upper=self.clip_range
        )
        zscore_clipped.name = "range_zscore"
        return zscore_clipped

    # -- Composite method ---------------------------------------------------

    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all 5 regime features, remove warm-up and session gaps.

        Processing steps:
            1. Compute each feature individually.
            2. Remove the first candle of each trading day/session
               (overnight gap contaminates log returns).
            3. Remove warm-up rows (first ``warmup_period`` valid rows).
            4. Drop any remaining NaN rows.

        Args:
            df: OHLCV DataFrame with columns ``open``, ``high``, ``low``,
                ``close``, ``volume`` and a DatetimeIndex.

        Returns:
            DataFrame with columns matching ``FEATURE_COLUMNS`` and no NaN.

        Raises:
            ValueError: If the input DataFrame lacks required columns.
        """
        required_cols = {"open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Input DataFrame is missing columns: {missing}"
            )

        logger.info(
            "Computing regime features for {} rows", len(df)
        )

        # Step 1: compute individual features
        log_returns = self.compute_log_returns(df)
        log_realized_vol = self.compute_log_realized_volatility(log_returns)
        momentum_zscore = self.compute_momentum_zscore(log_returns)
        log_rel_volume = self.compute_log_relative_volume(df)
        range_zscore = self.compute_range_zscore(df)

        features = pd.DataFrame(
            {
                "log_return": log_returns,
                "log_realized_vol": log_realized_vol,
                "momentum_zscore": momentum_zscore,
                "log_rel_volume": log_rel_volume,
                "range_zscore": range_zscore,
            },
            index=df.index,
        )

        # Step 2: remove first candle of each session/day
        if hasattr(features.index, "date"):
            dates = pd.Series(features.index.date, index=features.index)
            first_candle_mask = dates != dates.shift(1)
            n_session_starts = first_candle_mask.sum()
            features = features[~first_candle_mask]
            logger.debug(
                "Removed {} session-start candles (overnight gaps)",
                n_session_starts,
            )

        # Step 3: remove warm-up period (first warmup_period rows)
        if len(features) > self.warmup_period:
            features = features.iloc[self.warmup_period :]
            logger.debug(
                "Removed warm-up period ({} rows)", self.warmup_period
            )

        # Step 4: drop remaining NaN
        n_before = len(features)
        features = features.dropna()
        n_dropped = n_before - len(features)
        if n_dropped > 0:
            logger.debug(
                "Dropped {} rows with NaN after warm-up removal", n_dropped
            )

        logger.info(
            "Feature computation complete: {} valid rows, {} features",
            len(features),
            len(FEATURE_COLUMNS),
        )

        return features

    # -- Stationarity validation -------------------------------------------

    def validate_stationarity(
        self, features_df: pd.DataFrame
    ) -> dict[str, StationarityResult]:
        """Run ADF and KPSS stationarity tests on each feature.

        Uses the confirmation strategy from the spec:
        - ADF H0: series has unit root (non-stationary)
        - KPSS H0: series is stationary

        Args:
            features_df: DataFrame with feature columns (post warm-up).

        Returns:
            Dictionary mapping feature name to ``StationarityResult``.
        """
        results: dict[str, StationarityResult] = {}

        for col in FEATURE_COLUMNS:
            if col not in features_df.columns:
                logger.warning(
                    "Feature '{}' not found in DataFrame, skipping "
                    "stationarity test",
                    col,
                )
                continue

            series = features_df[col].dropna()
            if len(series) < 20:
                logger.warning(
                    "Feature '{}' has only {} observations, skipping "
                    "stationarity test",
                    col,
                    len(series),
                )
                continue

            # Skip constant series (e.g., volume=0 proxy data)
            if series.max() == series.min():
                logger.warning(
                    "Feature '{}' is constant (value={}), marking as "
                    "stationary by definition",
                    col,
                    series.iloc[0],
                )
                results[col] = StationarityResult(
                    feature=col,
                    adf_statistic=0.0,
                    adf_pvalue=0.0,
                    adf_reject_h0=True,
                    kpss_statistic=0.0,
                    kpss_pvalue=1.0,
                    kpss_reject_h0=False,
                    conclusion="stationary",
                )
                continue

            # ADF test
            adf_stat, adf_pval, *_ = adfuller(
                series, autolag="AIC", regression="c"
            )
            adf_reject = adf_pval < 0.05

            # KPSS test (suppress the interpolation warning)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_stat, kpss_pval, *_ = kpss(
                    series, regression="c", nlags="auto"
                )
            kpss_reject = kpss_pval < 0.05

            # Conclusion per spec table
            if adf_reject and not kpss_reject:
                conclusion = "stationary"
            elif not adf_reject and kpss_reject:
                conclusion = "non-stationary"
            elif adf_reject and kpss_reject:
                conclusion = "stationary_with_trend"
            else:
                conclusion = "inconclusive"

            result = StationarityResult(
                feature=col,
                adf_statistic=float(adf_stat),
                adf_pvalue=float(adf_pval),
                adf_reject_h0=adf_reject,
                kpss_statistic=float(kpss_stat),
                kpss_pvalue=float(kpss_pval),
                kpss_reject_h0=kpss_reject,
                conclusion=conclusion,
            )
            results[col] = result

            logger.info(
                "Stationarity test [{}]: ADF stat={:.4f} p={:.4f} "
                "(reject={}), KPSS stat={:.4f} p={:.4f} (reject={}) "
                "-> {}",
                col,
                adf_stat,
                adf_pval,
                adf_reject,
                kpss_stat,
                kpss_pval,
                kpss_reject,
                conclusion,
            )

        return results
