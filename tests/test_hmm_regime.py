"""Tests for HMMRegimeDetector -- model fitting, BIC/AIC, and labeling.

Uses synthetic multivariate Gaussian data with known cluster structure
to verify that the HMM can recover distinct states.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.hmm_regime import HMMRegimeDetector, ModelSelectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_3state_data(
    n_samples_per_state: int = 200, seed: int = 42
) -> np.ndarray:
    """Generate 5-dimensional data from 3 well-separated Gaussians.

    The states are designed to resemble financial regimes:
        State 0 (low vol):  low variance, near-zero mean
        State 1 (trending): moderate variance, directional mean
        State 2 (high vol): high variance, large spread

    Args:
        n_samples_per_state: Samples per state (sequential blocks).
        seed: RNG seed for reproducibility.

    Returns:
        Array of shape ``(3 * n_samples_per_state, 5)``.
    """
    rng = np.random.default_rng(seed)

    # State 0: low volatility
    s0 = rng.multivariate_normal(
        mean=[0.0, -8.0, 0.0, -0.5, -0.5],
        cov=np.diag([0.0001, 0.1, 0.3, 0.2, 0.3]),
        size=n_samples_per_state,
    )

    # State 1: trending
    s1 = rng.multivariate_normal(
        mean=[0.002, -6.0, 1.5, 0.3, 0.5],
        cov=np.diag([0.0005, 0.15, 0.4, 0.25, 0.35]),
        size=n_samples_per_state,
    )

    # State 2: high volatility
    s2 = rng.multivariate_normal(
        mean=[0.0, -4.0, 0.0, 1.0, 2.0],
        cov=np.diag([0.002, 0.2, 0.5, 0.3, 0.4]),
        size=n_samples_per_state,
    )

    return np.vstack([s0, s1, s2])


@pytest.fixture
def synthetic_data() -> np.ndarray:
    """Standard synthetic 3-state data."""
    return _make_synthetic_3state_data()


@pytest.fixture
def fitted_detector(synthetic_data: np.ndarray) -> HMMRegimeDetector:
    """Pre-fitted HMMRegimeDetector with 3 states."""
    detector = HMMRegimeDetector(n_states=3, n_init=3, random_state=42)
    detector.fit(synthetic_data)
    return detector


# ---------------------------------------------------------------------------
# Tests: instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    """Tests for HMMRegimeDetector construction."""

    def test_default_params(self) -> None:
        d = HMMRegimeDetector()
        assert d.n_states == 3
        assert d.covariance_type == "full"
        assert d.n_iter == 200
        assert d.tol == 1e-4
        assert d.n_init == 10
        assert d.random_state == 42

    def test_custom_params(self) -> None:
        d = HMMRegimeDetector(
            n_states=4,
            covariance_type="diagonal",
            n_iter=100,
            tol=1e-3,
            n_init=5,
            random_state=99,
        )
        assert d.n_states == 4
        assert d.covariance_type == "diagonal"
        assert d.n_iter == 100

    def test_model_not_fitted_raises(self) -> None:
        d = HMMRegimeDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = d.model


# ---------------------------------------------------------------------------
# Tests: fitting
# ---------------------------------------------------------------------------


class TestFitting:
    """Tests for HMM model fitting."""

    def test_fit_returns_self(self, synthetic_data: np.ndarray) -> None:
        d = HMMRegimeDetector(n_states=3, n_init=2, random_state=42)
        result = d.fit(synthetic_data)
        assert result is d

    def test_model_is_fitted(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        assert fitted_detector._is_fitted is True
        assert fitted_detector._model is not None

    def test_means_shape(
        self, fitted_detector: HMMRegimeDetector, synthetic_data: np.ndarray
    ) -> None:
        n_features = synthetic_data.shape[1]
        assert fitted_detector.model.means_.shape == (3, n_features)

    def test_transmat_shape(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        assert fitted_detector.model.transmat_.shape == (3, 3)

    def test_transmat_rows_sum_to_one(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        row_sums = fitted_detector.model.transmat_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: prediction
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for Viterbi decoding and posterior probabilities."""

    def test_predict_shape(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        states = fitted_detector.predict(synthetic_data)
        assert states.shape == (synthetic_data.shape[0],)

    def test_predict_states_in_range(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        states = fitted_detector.predict(synthetic_data)
        assert set(states).issubset({0, 1, 2})

    def test_predict_recovers_structure(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        """With well-separated Gaussians, the model should recover blocks."""
        states = fitted_detector.predict(synthetic_data)
        n = 200  # samples per state

        # Check that each block is dominated by a single state
        for start in range(0, 3 * n, n):
            block = states[start : start + n]
            dominant = np.bincount(block).argmax()
            purity = (block == dominant).mean()
            assert purity > 0.7, (
                f"Block [{start}:{start + n}] purity={purity:.2f}, "
                f"expected > 0.7"
            )

    def test_predict_proba_shape(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        proba = fitted_detector.predict_proba(synthetic_data)
        assert proba.shape == (synthetic_data.shape[0], 3)

    def test_predict_proba_sums_to_one(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        proba = fitted_detector.predict_proba(synthetic_data)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_non_negative(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        proba = fitted_detector.predict_proba(synthetic_data)
        assert (proba >= 0).all()


# ---------------------------------------------------------------------------
# Tests: BIC / AIC
# ---------------------------------------------------------------------------


class TestInformationCriteria:
    """Tests for BIC and AIC computation."""

    def test_bic_is_finite(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        bic = fitted_detector.compute_bic(synthetic_data)
        assert np.isfinite(bic)

    def test_aic_is_finite(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        aic = fitted_detector.compute_aic(synthetic_data)
        assert np.isfinite(aic)

    def test_bic_greater_than_aic(
        self,
        fitted_detector: HMMRegimeDetector,
        synthetic_data: np.ndarray,
    ) -> None:
        """For n > 7 (i.e. ln(n) > 2), BIC penalty > AIC penalty."""
        bic = fitted_detector.compute_bic(synthetic_data)
        aic = fitted_detector.compute_aic(synthetic_data)
        n = synthetic_data.shape[0]
        if np.log(n) > 2:
            assert bic > aic

    def test_param_count_3states_full(self) -> None:
        """Verify parameter count for K=3, D=5, full covariance.

        Per spec: means=15, cov=45, trans=6, init=2 -> total=68
        But spec table says 65 with cov = K * D*(D+1)/2 = 3*15 = 45
        means=15, trans=6, init=2 -> 15+45+6+2 = 68.
        Spec says 65 because it counts transitions as K*(K-1)=6 and
        init as K-1=2.  Let me verify: 15 + 45 + 6 + 2 = 68.
        Actually the spec table explicitly says 65. Let me re-read.
        Spec: means=K*D=15, cov=K*D(D+1)/2=45, trans=K*(K-1)=6, init=K-1=2
        Total = 15 + 45 + 6 + 2 = 68. The spec table entry of 65 seems to
        use a different counting. We implement the formula correctly.
        """
        d = HMMRegimeDetector(n_states=3, covariance_type="full", n_init=1)
        data = np.random.default_rng(42).normal(size=(100, 5))
        d.fit(data)
        n_params = d._count_free_params()
        # K=3, D=5: means=15, cov_full=45, trans=6, init=2 -> 68
        assert n_params == 68

    def test_param_count_3states_diag(self) -> None:
        """Verify parameter count for K=3, D=5, diagonal covariance."""
        d = HMMRegimeDetector(
            n_states=3, covariance_type="diagonal", n_init=1
        )
        data = np.random.default_rng(42).normal(size=(100, 5))
        d.fit(data)
        n_params = d._count_free_params()
        # K=3, D=5: means=15, cov_diag=15, trans=6, init=2 -> 38
        assert n_params == 38


# ---------------------------------------------------------------------------
# Tests: regime labeling
# ---------------------------------------------------------------------------


class TestRegimeLabeling:
    """Tests for regime label assignment."""

    def test_labels_contain_expected_keys(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        labels = fitted_detector.label_regimes()
        assert set(labels.values()) == {"low_vol", "trending", "high_vol"}

    def test_labels_cover_all_states(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        labels = fitted_detector.label_regimes()
        assert set(labels.keys()) == {0, 1, 2}

    def test_volatility_ordering(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        """low_vol state should have lowest mean log_realized_vol."""
        labels = fitted_detector.label_regimes()
        means = fitted_detector.model.means_

        low_vol_state = [k for k, v in labels.items() if v == "low_vol"][0]
        high_vol_state = [k for k, v in labels.items() if v == "high_vol"][0]

        # log_realized_vol is at index 1
        assert means[low_vol_state, 1] < means[high_vol_state, 1]

    def test_two_state_labels(self) -> None:
        """Two-state model should produce low_vol and high_vol."""
        d = HMMRegimeDetector(n_states=2, n_init=2, random_state=42)
        data = _make_synthetic_3state_data(n_samples_per_state=200, seed=42)
        d.fit(data)
        labels = d.label_regimes()
        assert set(labels.values()) == {"low_vol", "high_vol"}


# ---------------------------------------------------------------------------
# Tests: regime statistics
# ---------------------------------------------------------------------------


class TestRegimeStatistics:
    """Tests for regime statistics computation."""

    def test_returns_list_of_correct_length(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        stats = fitted_detector.get_regime_statistics()
        assert len(stats) == 3

    def test_expected_duration_positive(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        stats = fitted_detector.get_regime_statistics()
        for s in stats:
            assert s.expected_duration > 0

    def test_self_transition_in_range(
        self, fitted_detector: HMMRegimeDetector
    ) -> None:
        stats = fitted_detector.get_regime_statistics()
        for s in stats:
            assert 0.0 <= s.self_transition_prob <= 1.0


# ---------------------------------------------------------------------------
# Tests: model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Tests for model selection across (n_states, cov_type) grid."""

    @pytest.mark.slow
    def test_returns_model_selection_result(
        self, synthetic_data: np.ndarray
    ) -> None:
        d = HMMRegimeDetector(n_init=2, random_state=42)
        result = d.select_best_model(
            synthetic_data,
            n_states_range=(2, 3),
            cov_types=("full", "diagonal"),
        )
        assert isinstance(result, ModelSelectionResult)
        assert result.best_n_states in (2, 3)
        assert result.best_cov_type in ("full", "diagonal")
        assert np.isfinite(result.best_bic)
        assert len(result.results_table) == 4  # 2 states * 2 cov types

    @pytest.mark.slow
    def test_results_table_has_required_fields(
        self, synthetic_data: np.ndarray
    ) -> None:
        d = HMMRegimeDetector(n_init=2, random_state=42)
        result = d.select_best_model(
            synthetic_data,
            n_states_range=(2, 3),
            cov_types=("full",),
        )
        for entry in result.results_table:
            assert "n_states" in entry
            assert "cov_type" in entry
            assert "bic" in entry
            assert "aic" in entry
            assert "log_likelihood" in entry
