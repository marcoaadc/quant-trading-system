"""Tests for HMM model save/load serialization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.hmm_regime import HMMRegimeDetector


def _make_synthetic_data(
    n_samples: int = 600,
    n_features: int = 4,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = [
        np.zeros(n_features),
        np.ones(n_features) * 2.0,
        np.ones(n_features) * -2.0,
    ]
    parts = []
    for mean in means:
        parts.append(rng.normal(loc=mean, scale=0.5, size=(n_samples // 3, n_features)))
    return np.vstack(parts)


@pytest.fixture
def fitted_detector() -> tuple[HMMRegimeDetector, np.ndarray]:
    data = _make_synthetic_data()
    detector = HMMRegimeDetector(n_states=3, covariance_type="full")
    detector.fit(data)
    return detector, data


class TestSaveLoad:
    def test_save_creates_file(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray], tmp_path: Path
    ) -> None:
        detector, _ = fitted_detector
        path = detector.save(tmp_path / "model.json")
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_is_valid_json(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray], tmp_path: Path
    ) -> None:
        detector, _ = fitted_detector
        path = detector.save(tmp_path / "model.json")
        data = json.loads(path.read_text())
        assert data["version"] == "1.0.0"
        assert data["n_states"] == 3
        assert data["covariance_type"] == "full"
        assert "means" in data
        assert "covars" in data
        assert "transmat" in data
        assert "startprob" in data

    def test_load_recovers_predictions(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray], tmp_path: Path
    ) -> None:
        detector, data = fitted_detector
        original_predictions = detector.predict(data)
        original_score = detector.score(data)

        path = detector.save(tmp_path / "model.json")
        loaded = HMMRegimeDetector.load(path)

        loaded_predictions = loaded.predict(data)
        loaded_score = loaded.score(data)

        np.testing.assert_array_equal(original_predictions, loaded_predictions)
        np.testing.assert_almost_equal(original_score, loaded_score, decimal=5)

    def test_save_unfitted_raises(self, tmp_path: Path) -> None:
        detector = HMMRegimeDetector(n_states=3)
        with pytest.raises(RuntimeError, match="not fitted"):
            detector.save(tmp_path / "model.json")

    def test_loaded_model_is_fitted(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray], tmp_path: Path
    ) -> None:
        detector, _ = fitted_detector
        path = detector.save(tmp_path / "model.json")
        loaded = HMMRegimeDetector.load(path)
        assert loaded._is_fitted is True
        assert loaded.n_states == detector.n_states


class TestCovarianceValidation:
    def test_valid_covariance_types(self) -> None:
        for cov in ("full", "diagonal", "diag"):
            d = HMMRegimeDetector(covariance_type=cov)
            assert d.covariance_type == cov

    def test_invalid_covariance_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported covariance_type"):
            HMMRegimeDetector(covariance_type="spherical")


class TestRegimeQualityValidation:
    def test_quality_returns_list(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray]
    ) -> None:
        detector, data = fitted_detector
        warnings = detector.validate_regime_quality(data)
        assert isinstance(warnings, list)

    def test_quality_detects_degenerate_with_strict_threshold(
        self, fitted_detector: tuple[HMMRegimeDetector, np.ndarray]
    ) -> None:
        detector, data = fitted_detector
        warnings = detector.validate_regime_quality(
            data, min_fraction=0.99, min_duration=1000.0
        )
        assert len(warnings) > 0
