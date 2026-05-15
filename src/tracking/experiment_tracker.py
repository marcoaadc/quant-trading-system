"""MLflow experiment tracking for regime detection.

Wraps MLflow to provide structured logging of feature parameters,
model hyperparameters, metrics, and artifacts.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
from loguru import logger

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.features.regime_features import RegimeFeatureEngineer
    from src.models.hmm_regime import HMMRegimeDetector
    from src.validation.walk_forward import WalkForwardResult


class ExperimentTracker:
    """Wrapper around MLflow for regime detection experiments.

    Args:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI. Falls back to
            ``MLFLOW_TRACKING_URI`` env var or local ``mlruns/``.
    """

    def __init__(
        self,
        experiment_name: str = "regime-detection",
        tracking_uri: str | None = None,
    ) -> None:
        if mlflow is None:
            raise ImportError(
                "mlflow is required for experiment tracking. "
                "Install with: pip install mlflow"
            )

        uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        )
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

        self._experiment_name = experiment_name
        logger.info(
            "ExperimentTracker initialized: experiment='{}', uri='{}'",
            experiment_name,
            uri,
        )

    @contextmanager
    def start_run(self, run_name: str) -> Generator[Any, None, None]:
        """Context manager for an MLflow run.

        Args:
            run_name: Human-readable name for this run.

        Yields:
            The active MLflow run object.
        """
        with mlflow.start_run(run_name=run_name) as run:
            logger.info("MLflow run started: {}", run_name)
            yield run
        logger.info("MLflow run ended: {}", run_name)

    def log_feature_params(self, engineer: RegimeFeatureEngineer) -> None:
        """Log feature engineering parameters.

        Args:
            engineer: Configured ``RegimeFeatureEngineer`` instance.
        """
        mlflow.log_params(
            {
                "vol_window": engineer.vol_window,
                "mom_window": engineer.mom_window,
                "vol_rel_window": engineer.vol_rel_window,
                "zscore_window": engineer.zscore_window,
                "clip_range": engineer.clip_range,
                "include_volume": engineer.include_volume,
            }
        )

    def log_model_params(self, detector: HMMRegimeDetector) -> None:
        """Log HMM model hyperparameters.

        Args:
            detector: Configured ``HMMRegimeDetector`` instance.
        """
        mlflow.log_params(
            {
                "n_states": detector.n_states,
                "covariance_type": detector.covariance_type,
                "n_iter": detector.n_iter,
                "tol": detector.tol,
                "n_init": detector.n_init,
                "random_state": detector.random_state,
            }
        )

    def log_model_metrics(
        self,
        detector: HMMRegimeDetector,
        features: np.ndarray,
    ) -> None:
        """Log model performance metrics.

        Args:
            detector: Fitted ``HMMRegimeDetector``.
            features: Feature array used for evaluation.
        """
        ll = detector.score(features)
        bic = detector.compute_bic(features)
        aic = detector.compute_aic(features)
        n_params = detector._count_free_params()

        mlflow.log_metrics(
            {
                "log_likelihood": ll,
                "bic": bic,
                "aic": aic,
                "n_params": n_params,
                "n_samples": features.shape[0],
            }
        )

        labels = detector.predict(features)
        regime_labels = detector.label_regimes()
        n_total = len(labels)
        for state_id, label in regime_labels.items():
            fraction = float((labels == state_id).sum()) / n_total
            mlflow.log_metric(f"regime_pct_{label}", fraction)

        stats = detector.get_regime_statistics()
        for stat in stats:
            mlflow.log_metric(
                f"expected_duration_{stat.label}",
                stat.expected_duration,
            )

        converged = detector.model.monitor_.converged
        mlflow.log_metric("converged", int(converged))

    def log_validation_metrics(
        self,
        result: WalkForwardResult,
    ) -> None:
        """Log walk-forward validation metrics.

        Args:
            result: Walk-forward validation result.
        """
        mlflow.log_metrics(
            {
                "wf_mean_test_ll": result.mean_test_log_likelihood,
                "wf_std_test_ll": result.std_test_log_likelihood,
                "wf_mean_test_bic": result.mean_test_bic,
                "wf_mean_stability": result.mean_regime_stability,
                "wf_n_converged": result.n_converged,
                "wf_n_folds": result.n_total,
            }
        )

    def log_model_artifact(self, model_path: Path) -> None:
        """Log a serialized model file as an MLflow artifact.

        Args:
            model_path: Path to the model JSON file.
        """
        mlflow.log_artifact(str(model_path))
        logger.info("Logged model artifact: {}", model_path)
