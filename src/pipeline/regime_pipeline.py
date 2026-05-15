"""Regime detection pipeline orchestrator.

Connects ingestion, feature engineering, model training, and validation
into a single reproducible pipeline with run tracking.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.data.ingestion import OHLCVIngestor, load_config
from src.features.regime_features import RegimeFeatureEngineer
from src.models.hmm_regime import HMMRegimeDetector
from src.validation.walk_forward import WalkForwardResult, WalkForwardValidator


class RegimePipeline:
    """Orchestrates the full regime detection pipeline.

    Args:
        config: Parsed configuration dictionary.
        run_id: Unique identifier for this pipeline run. Auto-generated
            if not provided.
        output_dir: Base directory for pipeline artifacts.
    """

    def __init__(
        self,
        config: dict[str, Any],
        run_id: str | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.output_dir = output_dir or Path(
            config.get("data", {}).get("processed_dir", "data/processed")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline initialized: run_id={}", self.run_id)

    @classmethod
    def from_config(cls, config_path: str | Path) -> RegimePipeline:
        """Create a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML config.

        Returns:
            Configured ``RegimePipeline`` instance.
        """
        config = load_config(Path(config_path))
        return cls(config)

    def run_ingestion(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
    ) -> dict[str, Path]:
        """Download OHLCV data for configured assets.

        Args:
            symbols: Subset of symbols to ingest (default: all).
            timeframes: Subset of timeframes (default: all).

        Returns:
            Mapping of ``"{symbol}_{timeframe}"`` to Parquet file paths.
        """
        logger.info("[{}] Starting ingestion", self.run_id)

        config = dict(self.config)
        if symbols:
            config["assets"] = [
                a for a in config.get("assets", [])
                if a["symbol"] in symbols
            ]
        if timeframes:
            config["timeframes"] = [
                t for t in config.get("timeframes", [])
                if t in timeframes
            ]

        ingestor = OHLCVIngestor(config)
        written = ingestor.run()

        result: dict[str, Path] = {}
        for path in written:
            stem = path.stem
            parts = stem.rsplit("_", 1)[0]
            result[parts] = path

        logger.info("[{}] Ingestion complete: {} files", self.run_id, len(result))
        return result

    def run_features(
        self,
        data_paths: dict[str, Path],
        include_volume: bool = False,
    ) -> dict[str, Path]:
        """Compute features from OHLCV Parquet files.

        Args:
            data_paths: Mapping from key to Parquet path.
            include_volume: Whether to include volume feature.

        Returns:
            Mapping of key to feature Parquet file paths.
        """
        logger.info("[{}] Starting feature engineering", self.run_id)

        engineer = RegimeFeatureEngineer(include_volume=include_volume)
        result: dict[str, Path] = {}

        for key, parquet_path in data_paths.items():
            df = pd.read_parquet(parquet_path)
            features_df = engineer.compute_all_features(df)

            out_path = self.output_dir / f"{key}_{self.run_id}_features.parquet"
            features_df.to_parquet(out_path, engine="pyarrow", index=True)
            result[key] = out_path

            logger.info(
                "[{}] Features for {}: {} rows, {} cols -> {}",
                self.run_id,
                key,
                len(features_df),
                len(features_df.columns),
                out_path,
            )

        return result

    def run_model(
        self,
        feature_paths: dict[str, Path],
        n_states: int = 3,
        covariance_type: str = "full",
    ) -> dict[str, dict[str, Any]]:
        """Fit HMM and save model artifacts.

        Args:
            feature_paths: Mapping from key to feature Parquet path.
            n_states: Number of HMM states.
            covariance_type: Covariance parametrization.

        Returns:
            Mapping of key to result dict with model_path, bic, etc.
        """
        logger.info("[{}] Starting model training", self.run_id)

        results: dict[str, dict[str, Any]] = {}

        for key, feat_path in feature_paths.items():
            features_df = pd.read_parquet(feat_path)
            features_array = features_df.values

            detector = HMMRegimeDetector(
                n_states=n_states,
                covariance_type=covariance_type,
            )
            detector.fit(features_array)

            model_path = self.output_dir / f"{key}_{self.run_id}_model.json"
            detector.save(model_path)

            bic = detector.compute_bic(features_array)
            aic = detector.compute_aic(features_array)
            ll = detector.score(features_array)
            quality_warnings = detector.validate_regime_quality(features_array)

            results[key] = {
                "model_path": model_path,
                "bic": bic,
                "aic": aic,
                "log_likelihood": ll,
                "n_states": n_states,
                "covariance_type": covariance_type,
                "quality_warnings": quality_warnings,
                "regime_stats": detector.get_regime_statistics(),
            }

            logger.info(
                "[{}] Model for {}: BIC={:.2f}, AIC={:.2f}, LL={:.2f}, "
                "warnings={}",
                self.run_id,
                key,
                bic,
                aic,
                ll,
                len(quality_warnings),
            )

        return results

    def run_validation(
        self,
        feature_paths: dict[str, Path],
        n_states: int = 3,
        covariance_type: str = "full",
        min_train_size: int = 500,
        test_size: int = 100,
        step_size: int = 50,
    ) -> dict[str, WalkForwardResult]:
        """Run walk-forward validation on feature data.

        Args:
            feature_paths: Mapping from key to feature Parquet path.
            n_states: Number of HMM states.
            covariance_type: Covariance parametrization.
            min_train_size: Minimum training window size.
            test_size: Test window size.
            step_size: Step between folds.

        Returns:
            Mapping of key to ``WalkForwardResult``.
        """
        logger.info("[{}] Starting walk-forward validation", self.run_id)

        validator = WalkForwardValidator(
            min_train_size=min_train_size,
            test_size=test_size,
            step_size=step_size,
            n_states=n_states,
            covariance_type=covariance_type,
        )

        results: dict[str, WalkForwardResult] = {}

        for key, feat_path in feature_paths.items():
            features_df = pd.read_parquet(feat_path)
            features_array = features_df.values

            try:
                wf_result = validator.validate(features_array)
                results[key] = wf_result
                logger.info(
                    "[{}] Validation for {}:\n{}",
                    self.run_id,
                    key,
                    wf_result.summary(),
                )
            except ValueError as exc:
                logger.warning(
                    "[{}] Validation skipped for {}: {}",
                    self.run_id,
                    key,
                    exc,
                )

        return results

    def run_full(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        n_states: int = 3,
        include_volume: bool = False,
        run_validation: bool = True,
    ) -> dict[str, Any]:
        """Execute full pipeline: ingest -> features -> model -> validate.

        Args:
            symbols: Subset of symbols to process.
            timeframes: Subset of timeframes to process.
            n_states: Number of HMM states.
            include_volume: Whether to include volume feature.
            run_validation: Whether to run walk-forward validation.

        Returns:
            Dictionary with all pipeline results.
        """
        logger.info("[{}] Running full pipeline", self.run_id)

        data_paths = self.run_ingestion(symbols, timeframes)
        feature_paths = self.run_features(data_paths, include_volume=include_volume)
        model_results = self.run_model(feature_paths, n_states=n_states)

        validation_results: dict[str, WalkForwardResult] = {}
        if run_validation:
            validation_results = self.run_validation(
                feature_paths, n_states=n_states
            )

        return {
            "run_id": self.run_id,
            "data_paths": data_paths,
            "feature_paths": feature_paths,
            "model_results": model_results,
            "validation_results": validation_results,
        }
