"""Tests for walk-forward validation module."""

from __future__ import annotations

import numpy as np
import pytest

from src.validation.walk_forward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardValidator,
)


def _make_synthetic_features(
    n_samples: int = 800,
    n_features: int = 4,
    n_states: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """Create well-separated synthetic features for HMM testing."""
    rng = np.random.default_rng(seed)

    means = [
        np.array([0.0, -5.0, 0.0, 0.0]),
        np.array([0.01, -3.0, 1.0, 0.5]),
        np.array([-0.01, -1.0, -1.0, 1.0]),
    ][:n_states]

    samples_per_state = n_samples // n_states
    parts = []
    for mean in means:
        data = rng.normal(loc=mean, scale=0.3, size=(samples_per_state, n_features))
        parts.append(data)

    return np.vstack(parts)[:n_samples]


@pytest.fixture
def features() -> np.ndarray:
    return _make_synthetic_features(n_samples=800)


class TestWalkForwardValidator:
    def test_validate_returns_result(self, features: np.ndarray) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=50
        )
        result = validator.validate(features)
        assert isinstance(result, WalkForwardResult)
        assert result.n_total > 0

    def test_folds_have_correct_structure(self, features: np.ndarray) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=100
        )
        result = validator.validate(features)

        for fold in result.folds:
            assert isinstance(fold, WalkForwardFold)
            assert fold.train_start == 0
            assert fold.test_start == fold.train_end
            assert fold.test_end == fold.test_start + fold.test_size
            assert fold.train_size == fold.train_end

    def test_expanding_window(self, features: np.ndarray) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=50
        )
        result = validator.validate(features)

        train_sizes = [f.train_size for f in result.folds]
        assert train_sizes == sorted(train_sizes)
        for i in range(1, len(train_sizes)):
            assert train_sizes[i] - train_sizes[i - 1] == 50

    def test_dataset_too_small_raises(self) -> None:
        small = np.random.randn(100, 4)
        validator = WalkForwardValidator(min_train_size=500, test_size=100)
        with pytest.raises(ValueError, match="needs at least"):
            validator.validate(small)

    def test_regime_stability_range(self, features: np.ndarray) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=100
        )
        result = validator.validate(features)

        for fold in result.folds:
            assert 0.0 <= fold.regime_stability <= 1.0

    def test_converged_folds_have_finite_metrics(
        self, features: np.ndarray
    ) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=100
        )
        result = validator.validate(features)

        converged = [f for f in result.folds if f.converged]
        assert len(converged) > 0

        for fold in converged:
            assert np.isfinite(fold.train_log_likelihood)
            assert np.isfinite(fold.test_log_likelihood)
            assert np.isfinite(fold.train_bic)
            assert np.isfinite(fold.test_bic)

    def test_summary_is_string(self, features: np.ndarray) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=100
        )
        result = validator.validate(features)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Walk-Forward" in summary

    def test_regime_distribution_sums_to_one(
        self, features: np.ndarray
    ) -> None:
        validator = WalkForwardValidator(
            min_train_size=300, test_size=50, step_size=100
        )
        result = validator.validate(features)

        for fold in result.folds:
            if fold.regime_distribution:
                total = sum(fold.regime_distribution.values())
                assert abs(total - 1.0) < 1e-6


class TestRegimeStability:
    def test_constant_labels(self) -> None:
        labels = np.array([0, 0, 0, 0, 0])
        stability = WalkForwardValidator._compute_regime_stability(labels)
        assert stability == 1.0

    def test_alternating_labels(self) -> None:
        labels = np.array([0, 1, 0, 1, 0])
        stability = WalkForwardValidator._compute_regime_stability(labels)
        assert stability == 0.0

    def test_single_label(self) -> None:
        labels = np.array([0])
        stability = WalkForwardValidator._compute_regime_stability(labels)
        assert stability == 1.0
