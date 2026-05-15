"""Expanding-window walk-forward validation for HMM regime detection.

Splits the feature time series into expanding training windows with a
fixed-size test window. Each fold trains on ``[0, train_end)`` and
tests on ``[train_end, train_end + test_size)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.models.hmm_regime import HMMRegimeDetector


@dataclass
class WalkForwardFold:
    """Results from a single walk-forward fold."""

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_size: int
    test_size: int
    n_states: int
    covariance_type: str
    train_log_likelihood: float
    test_log_likelihood: float
    train_bic: float
    test_bic: float
    regime_distribution: dict[str, float]
    regime_stability: float
    converged: bool


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward validation."""

    folds: list[WalkForwardFold] = field(default_factory=list)
    mean_test_log_likelihood: float = 0.0
    std_test_log_likelihood: float = 0.0
    mean_test_bic: float = 0.0
    mean_regime_stability: float = 0.0
    n_converged: int = 0
    n_total: int = 0

    def summary(self) -> str:
        """Human-readable summary of walk-forward results."""
        lines = [
            f"Walk-Forward Validation: {self.n_total} folds "
            f"({self.n_converged} converged)",
            f"  Test LL:  {self.mean_test_log_likelihood:.2f} "
            f"+/- {self.std_test_log_likelihood:.2f}",
            f"  Test BIC: {self.mean_test_bic:.2f}",
            f"  Regime stability: {self.mean_regime_stability:.1%}",
        ]
        return "\n".join(lines)


class WalkForwardValidator:
    """Expanding-window walk-forward validation for HMM regime detection.

    Args:
        min_train_size: Minimum observations in the first training window.
        test_size: Fixed number of observations per test window.
        step_size: How many observations to advance between folds.
        n_states: Number of HMM states.
        covariance_type: HMM covariance parametrization.
        random_state: RNG seed for reproducibility.
    """

    def __init__(
        self,
        min_train_size: int = 500,
        test_size: int = 100,
        step_size: int = 50,
        n_states: int = 3,
        covariance_type: str = "full",
        random_state: int = 42,
    ) -> None:
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.random_state = random_state

    def validate(self, features: np.ndarray) -> WalkForwardResult:
        """Run walk-forward validation on a feature matrix.

        Args:
            features: Array of shape ``(n_samples, n_features)``
                from ``RegimeFeatureEngineer`` (warm-up already removed).

        Returns:
            ``WalkForwardResult`` with fold-level and aggregate metrics.

        Raises:
            ValueError: If the dataset is too small for even one fold.
        """
        n_samples = features.shape[0]
        min_required = self.min_train_size + self.test_size

        if n_samples < min_required:
            raise ValueError(
                f"Dataset has {n_samples} samples but needs at least "
                f"{min_required} (min_train={self.min_train_size} + "
                f"test={self.test_size})"
            )

        folds: list[WalkForwardFold] = []
        fold_id = 0
        train_end = self.min_train_size

        while train_end + self.test_size <= n_samples:
            fold = self._run_fold(features, fold_id, train_end)
            folds.append(fold)

            logger.debug(
                "Fold {}: train=[0,{}), test=[{},{}), "
                "test_ll={:.2f}, stability={:.1%}, converged={}",
                fold_id,
                train_end,
                train_end,
                train_end + self.test_size,
                fold.test_log_likelihood,
                fold.regime_stability,
                fold.converged,
            )

            fold_id += 1
            train_end += self.step_size

        result = self._aggregate(folds)

        logger.info(
            "Walk-forward complete: {} folds, mean test LL={:.2f}, "
            "stability={:.1%}",
            result.n_total,
            result.mean_test_log_likelihood,
            result.mean_regime_stability,
        )

        return result

    def _run_fold(
        self,
        features: np.ndarray,
        fold_id: int,
        train_end: int,
    ) -> WalkForwardFold:
        """Execute a single walk-forward fold."""
        test_end = train_end + self.test_size
        train_data = features[:train_end]
        test_data = features[train_end:test_end]

        detector = HMMRegimeDetector(
            n_states=self.n_states,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )

        converged = False
        train_ll = float("-inf")
        test_ll = float("-inf")
        train_bic = float("inf")
        test_bic = float("inf")
        regime_dist: dict[str, float] = {}
        stability = 0.0

        try:
            detector.fit(train_data)
            converged = detector.model.monitor_.converged

            train_ll = detector.score(train_data)
            test_ll = detector.score(test_data)
            train_bic = detector.compute_bic(train_data)
            test_bic = self._compute_test_bic(detector, test_data)

            test_labels = detector.predict(test_data)
            regime_dist = self._compute_regime_distribution(
                test_labels, detector
            )
            stability = self._compute_regime_stability(test_labels)

        except Exception:
            logger.exception("Fold {} failed during training", fold_id)

        return WalkForwardFold(
            fold_id=fold_id,
            train_start=0,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            train_size=train_end,
            test_size=self.test_size,
            n_states=self.n_states,
            covariance_type=self.covariance_type,
            train_log_likelihood=train_ll,
            test_log_likelihood=test_ll,
            train_bic=train_bic,
            test_bic=test_bic,
            regime_distribution=regime_dist,
            regime_stability=stability,
            converged=converged,
        )

    @staticmethod
    def _compute_regime_stability(labels: np.ndarray) -> float:
        """Fraction of consecutive observations staying in the same regime."""
        if len(labels) < 2:
            return 1.0
        same = int(np.sum(labels[1:] == labels[:-1]))
        return same / (len(labels) - 1)

    @staticmethod
    def _compute_test_bic(
        detector: HMMRegimeDetector,
        test_features: np.ndarray,
    ) -> float:
        """Compute BIC on test set using test-set size for penalty."""
        ll = detector.score(test_features)
        n_params = detector._count_free_params()
        n_obs = test_features.shape[0]
        return float(-2.0 * ll + n_params * np.log(n_obs))

    @staticmethod
    def _compute_regime_distribution(
        labels: np.ndarray,
        detector: HMMRegimeDetector,
    ) -> dict[str, float]:
        """Compute fraction of time spent in each regime."""
        n_total = len(labels)
        if n_total == 0:
            return {}

        regime_labels = detector.label_regimes()
        dist: dict[str, float] = {}
        for state_id, label in regime_labels.items():
            count = int((labels == state_id).sum())
            dist[label] = count / n_total
        return dist

    @staticmethod
    def _aggregate(folds: list[WalkForwardFold]) -> WalkForwardResult:
        """Aggregate fold-level metrics into a summary."""
        if not folds:
            return WalkForwardResult()

        converged_folds = [f for f in folds if f.converged]
        test_lls = [f.test_log_likelihood for f in converged_folds]
        test_bics = [f.test_bic for f in converged_folds]
        stabilities = [f.regime_stability for f in converged_folds]

        return WalkForwardResult(
            folds=folds,
            mean_test_log_likelihood=float(np.mean(test_lls)) if test_lls else float("-inf"),
            std_test_log_likelihood=float(np.std(test_lls)) if test_lls else 0.0,
            mean_test_bic=float(np.mean(test_bics)) if test_bics else float("inf"),
            mean_regime_stability=float(np.mean(stabilities)) if stabilities else 0.0,
            n_converged=len(converged_folds),
            n_total=len(folds),
        )
