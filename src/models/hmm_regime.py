"""HMM-based regime detection for financial time series.

Wraps ``hmmlearn.GaussianHMM`` with:
    - K-Means-informed initialization
    - BIC/AIC model selection across state counts and covariance types
    - Regime labeling by volatility ordering
    - Regime statistics (mean, covariance, expected duration)

Reference: docs/specs/sprint2_hmm_regime_detection.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from hmmlearn.hmm import GaussianHMM
from loguru import logger
from sklearn.cluster import KMeans

from src.features.regime_features import FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class RegimeStatistics:
    """Statistics for a single HMM regime/state."""

    state_id: int
    label: str
    mean: np.ndarray
    covariance: np.ndarray
    self_transition_prob: float
    expected_duration: float


@dataclass
class ModelSelectionResult:
    """Result of model selection across (n_states, cov_type) grid."""

    best_n_states: int
    best_cov_type: str
    best_bic: float
    best_model: HMMRegimeDetector
    results_table: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HMMRegimeDetector
# ---------------------------------------------------------------------------


class HMMRegimeDetector:
    """Gaussian HMM wrapper for financial regime detection.

    Args:
        n_states: Number of hidden states *K* (default 3).
        covariance_type: Covariance parametrization -- ``'full'`` or
            ``'diagonal'`` (default ``'full'``).
        n_iter: Maximum EM iterations (default 200).
        tol: EM convergence tolerance (default 1e-4).
        n_init: Number of random restarts (default 10).
        random_state: RNG seed for reproducibility (default 42).
    """

    _COV_TYPE_MAP: ClassVar[dict[str, str]] = {
        "full": "full",
        "diagonal": "diag",
        "diag": "diag",
    }

    _VALID_COV_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"full", "diagonal", "diag"}
    )

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int = 42,
    ) -> None:
        if covariance_type not in self._VALID_COV_TYPES:
            raise ValueError(
                f"Unsupported covariance_type '{covariance_type}'. "
                f"Valid: {sorted(self._VALID_COV_TYPES)}"
            )

        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self._model: GaussianHMM | None = None
        self._is_fitted: bool = False

        logger.debug(
            "HMMRegimeDetector initialized: n_states={}, cov_type={}, "
            "n_iter={}, tol={}, n_init={}, seed={}",
            n_states,
            covariance_type,
            n_iter,
            tol,
            n_init,
            random_state,
        )

    # -- Properties ---------------------------------------------------------

    @property
    def model(self) -> GaussianHMM:
        """Return the underlying hmmlearn model.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None or not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call .fit() first."
            )
        return self._model

    # -- Core API -----------------------------------------------------------

    def fit(self, features: np.ndarray) -> HMMRegimeDetector:
        """Train the Gaussian HMM on feature data.

        Uses K-Means initialization followed by EM (Baum-Welch).
        Runs ``n_init`` random restarts and keeps the model with
        the highest log-likelihood.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Self (for method chaining).
        """
        n_samples, n_features = features.shape
        cov_type = self._COV_TYPE_MAP.get(
            self.covariance_type, self.covariance_type
        )

        logger.info(
            "Fitting HMM: n_states={}, cov_type={}, samples={}, features={}",
            self.n_states,
            cov_type,
            n_samples,
            n_features,
        )

        # K-Means initialization
        kmeans_means, kmeans_covars, kmeans_transmat, kmeans_startprob = (
            self._kmeans_init(features, n_features, cov_type)
        )

        best_model: GaussianHMM | None = None
        best_score = -np.inf

        # Run with K-Means init + n_init random restarts
        for i in range(self.n_init + 1):
            try:
                hmm = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=cov_type,
                    n_iter=self.n_iter,
                    tol=self.tol,
                    random_state=self.random_state + i,
                    init_params="" if i == 0 else "stmc",
                    params="stmc",
                )

                if i == 0:
                    # Use K-Means initialization
                    hmm.means_ = kmeans_means
                    hmm.covars_ = kmeans_covars
                    hmm.transmat_ = kmeans_transmat
                    hmm.startprob_ = kmeans_startprob

                hmm.fit(features)
                score = hmm.score(features)

                if score > best_score:
                    best_score = score
                    best_model = hmm

                logger.debug(
                    "Init {}/{}: log-likelihood={:.2f} (best={:.2f})",
                    i + 1,
                    self.n_init + 1,
                    score,
                    best_score,
                )

            except Exception:
                logger.exception(
                    "Init {}/{} failed, skipping", i + 1, self.n_init + 1
                )
                continue

        if best_model is None:
            raise RuntimeError(
                "All HMM initializations failed. Check input data."
            )

        self._model = best_model
        self._is_fitted = True

        logger.info(
            "HMM fitted successfully. Best log-likelihood: {:.2f}",
            best_score,
        )
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Decode most likely state sequence via Viterbi algorithm.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Array of state labels of shape ``(n_samples,)``.
        """
        states: np.ndarray = self.model.predict(features)
        return states

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities via Forward-Backward.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, n_states)`` with probabilities
            summing to 1.0 per row.
        """
        proba: np.ndarray = self.model.predict_proba(features)
        return proba

    def score(self, features: np.ndarray) -> float:
        """Compute log-likelihood of the data under the fitted model.

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            Total log-likelihood.
        """
        ll: float = float(self.model.score(features))
        return ll

    # -- Information criteria -----------------------------------------------

    def _count_free_params(self) -> int:
        """Count free parameters of the fitted HMM.

        For K states and D features:
            - Means:       K * D
            - Covariances: K * D*(D+1)/2  (full) or K * D (diagonal)
            - Transitions: K * (K - 1)
            - Initial:     K - 1

        Returns:
            Total number of free parameters.
        """
        k = self.n_states
        d = self.model.n_features
        cov_type = self.model.covariance_type

        n_mean = k * d
        if cov_type == "full":
            n_cov = k * d * (d + 1) // 2
        elif cov_type == "diag":
            n_cov = k * d
        else:
            # Fallback for tied/spherical (not used but safe)
            n_cov = k * d

        n_trans = k * (k - 1)
        n_init = k - 1

        total = n_mean + n_cov + n_trans + n_init
        return total

    def compute_bic(self, features: np.ndarray) -> float:
        """Compute Bayesian Information Criterion.

        BIC = -2 * log_likelihood + n_params * ln(n_observations)

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            BIC value (lower is better).
        """
        ll = self.score(features)
        n_params = self._count_free_params()
        n_obs = features.shape[0]
        bic = -2.0 * ll + n_params * np.log(n_obs)
        return float(bic)

    def compute_aic(self, features: np.ndarray) -> float:
        """Compute Akaike Information Criterion.

        AIC = -2 * log_likelihood + 2 * n_params

        Args:
            features: Array of shape ``(n_samples, n_features)``.

        Returns:
            AIC value (lower is better).
        """
        ll = self.score(features)
        n_params = self._count_free_params()
        aic = -2.0 * ll + 2.0 * n_params
        return float(aic)

    # -- Regime interpretation ----------------------------------------------

    def get_regime_statistics(self) -> list[RegimeStatistics]:
        """Compute statistics for each regime.

        For each state: mean vector, covariance matrix, self-transition
        probability, and expected duration (1 / (1 - a_jj)).

        Returns:
            List of ``RegimeStatistics``, one per state.
        """
        labels = self.label_regimes()
        stats: list[RegimeStatistics] = []

        for j in range(self.n_states):
            a_jj = float(self.model.transmat_[j, j])
            expected_dur = 1.0 / (1.0 - a_jj) if a_jj < 1.0 else np.inf

            cov_type = self.model.covariance_type
            if cov_type == "full":
                cov_matrix = self.model.covars_[j]
            elif cov_type == "diag":
                cov_matrix = np.diag(self.model.covars_[j])
            else:
                cov_matrix = self.model.covars_[j]

            stats.append(
                RegimeStatistics(
                    state_id=j,
                    label=labels.get(j, f"state_{j}"),
                    mean=self.model.means_[j].copy(),
                    covariance=cov_matrix.copy(),
                    self_transition_prob=a_jj,
                    expected_duration=expected_dur,
                )
            )

        return stats

    def label_regimes(self) -> dict[int, str]:
        """Assign semantic labels to HMM states by volatility ordering.

        States are sorted by their mean log_realized_vol value
        (index 1 in the feature vector matching FEATURE_COLUMNS ordering).

        For 3 states: lowest vol -> 'low_vol', middle -> 'trending',
        highest vol -> 'high_vol'.

        For 2 states: lowest -> 'low_vol', highest -> 'high_vol'.

        For 4 states: lowest -> 'low_vol', two middle -> 'trending_1',
        'trending_2', highest -> 'high_vol'.

        Returns:
            Mapping from state index to label string.
        """
        # log_realized_vol is at index 1 in FEATURE_COLUMNS
        vol_index = FEATURE_COLUMNS.index("log_realized_vol")
        vol_means = self.model.means_[:, vol_index]
        sorted_indices = np.argsort(vol_means)

        labels: dict[int, str] = {}

        if self.n_states == 2:
            labels[int(sorted_indices[0])] = "low_vol"
            labels[int(sorted_indices[1])] = "high_vol"
        elif self.n_states == 3:
            labels[int(sorted_indices[0])] = "low_vol"
            labels[int(sorted_indices[1])] = "trending"
            labels[int(sorted_indices[2])] = "high_vol"
        elif self.n_states == 4:
            labels[int(sorted_indices[0])] = "low_vol"
            labels[int(sorted_indices[1])] = "trending_1"
            labels[int(sorted_indices[2])] = "trending_2"
            labels[int(sorted_indices[3])] = "high_vol"
        else:
            for i, idx in enumerate(sorted_indices):
                labels[int(idx)] = f"state_{i}"

        logger.debug("Regime labels: {}", labels)
        return labels

    # -- Serialization ------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Export fitted model parameters to JSON (safe, no pickle).

        Args:
            path: Output file path.

        Returns:
            Path to the written JSON file.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        artifact = {
            "version": "1.0.0",
            "n_states": self.n_states,
            "covariance_type": self.covariance_type,
            "means": self._model.means_.tolist(),  # type: ignore[union-attr]
            "covars": self._model.covars_.tolist(),  # type: ignore[union-attr]
            "transmat": self._model.transmat_.tolist(),  # type: ignore[union-attr]
            "startprob": self._model.startprob_.tolist(),  # type: ignore[union-attr]
        }
        out = Path(path)
        out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        logger.info("Model saved to {}", out)
        return out

    @classmethod
    def load(cls, path: str | Path) -> HMMRegimeDetector:
        """Load model from JSON parameters (safe, no pickle).

        Args:
            path: Path to JSON model file.

        Returns:
            Fitted ``HMMRegimeDetector`` instance.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        detector = cls(
            n_states=data["n_states"],
            covariance_type=data["covariance_type"],
        )
        cov_type = cls._COV_TYPE_MAP.get(
            data["covariance_type"], data["covariance_type"]
        )
        model = GaussianHMM(
            n_components=data["n_states"],
            covariance_type=cov_type,
        )
        model.means_ = np.array(data["means"])
        model.covars_ = np.array(data["covars"])
        model.transmat_ = np.array(data["transmat"])
        model.startprob_ = np.array(data["startprob"])

        detector._model = model
        detector._is_fitted = True
        logger.info("Model loaded from {}", path)
        return detector

    # -- Regime validation --------------------------------------------------

    def validate_regime_quality(
        self,
        features: np.ndarray,
        min_fraction: float = 0.05,
        min_duration: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Check for degenerate regimes after fitting.

        A regime is degenerate if it occupies less than ``min_fraction``
        of total time or has expected duration below ``min_duration``.

        Args:
            features: Array used for prediction.
            min_fraction: Minimum fraction of time in a regime (default 5%).
            min_duration: Minimum expected duration in observations.

        Returns:
            List of warnings for degenerate regimes (empty if all OK).
        """
        labels = self.predict(features)
        n_total = len(labels)
        warnings_list: list[dict[str, Any]] = []

        regime_labels = self.label_regimes()
        stats = self.get_regime_statistics()

        for stat in stats:
            count = int((labels == stat.state_id).sum())
            fraction = count / n_total if n_total > 0 else 0.0

            if fraction < min_fraction:
                msg = (
                    f"Regime '{stat.label}' (state {stat.state_id}) occupies "
                    f"only {fraction:.1%} of time (min {min_fraction:.0%})"
                )
                logger.warning(msg)
                warnings_list.append({"state": stat.state_id, "issue": "low_fraction", "value": fraction})

            if stat.expected_duration < min_duration:
                msg = (
                    f"Regime '{stat.label}' (state {stat.state_id}) has expected "
                    f"duration {stat.expected_duration:.1f} (min {min_duration:.0f})"
                )
                logger.warning(msg)
                warnings_list.append({"state": stat.state_id, "issue": "short_duration", "value": stat.expected_duration})

        if not warnings_list:
            logger.info("All regimes pass quality checks")

        return warnings_list

    # -- Model selection ----------------------------------------------------

    def select_best_model(
        self,
        features: np.ndarray,
        n_states_range: tuple[int, ...] = (2, 3),
        cov_types: tuple[str, ...] = ("full", "diagonal"),
    ) -> ModelSelectionResult:
        """Train models across a grid and select by BIC.

        Args:
            features: Array of shape ``(n_samples, n_features)``.
            n_states_range: State counts to evaluate.
            cov_types: Covariance types to evaluate.

        Returns:
            ``ModelSelectionResult`` with the best model and comparison table.
        """
        logger.info(
            "Model selection: n_states={}, cov_types={}",
            n_states_range,
            cov_types,
        )

        results: list[dict[str, Any]] = []
        best_bic = np.inf
        best_model: HMMRegimeDetector | None = None
        best_k = n_states_range[0]
        best_cov = cov_types[0]

        for k in n_states_range:
            for cov in cov_types:
                try:
                    detector = HMMRegimeDetector(
                        n_states=k,
                        covariance_type=cov,
                        n_iter=self.n_iter,
                        tol=self.tol,
                        n_init=self.n_init,
                        random_state=self.random_state,
                    )
                    detector.fit(features)

                    bic = detector.compute_bic(features)
                    aic = detector.compute_aic(features)
                    ll = detector.score(features)
                    n_params = detector._count_free_params()

                    result_entry = {
                        "n_states": k,
                        "cov_type": cov,
                        "log_likelihood": ll,
                        "n_params": n_params,
                        "bic": bic,
                        "aic": aic,
                        "converged": detector.model.monitor_.converged,
                    }
                    results.append(result_entry)

                    logger.info(
                        "K={}, cov={}: LL={:.2f}, BIC={:.2f}, AIC={:.2f}, "
                        "params={}, converged={}",
                        k,
                        cov,
                        ll,
                        bic,
                        aic,
                        n_params,
                        result_entry["converged"],
                    )

                    if bic < best_bic:
                        # If BIC difference < 10, prefer simpler model
                        if (
                            best_model is not None
                            and (best_bic - bic) < 10
                            and k > best_k
                        ):
                            logger.debug(
                                "BIC difference < 10 (delta={:.2f}), "
                                "keeping simpler model K={}",
                                best_bic - bic,
                                best_k,
                            )
                        else:
                            best_bic = bic
                            best_model = detector
                            best_k = k
                            best_cov = cov

                except Exception:
                    logger.exception(
                        "Model selection failed for K={}, cov={}", k, cov
                    )
                    results.append(
                        {
                            "n_states": k,
                            "cov_type": cov,
                            "log_likelihood": np.nan,
                            "n_params": 0,
                            "bic": np.inf,
                            "aic": np.inf,
                            "converged": False,
                        }
                    )

        if best_model is None:
            raise RuntimeError("All model configurations failed.")

        logger.info(
            "Best model: K={}, cov={}, BIC={:.2f}",
            best_k,
            best_cov,
            best_bic,
        )

        return ModelSelectionResult(
            best_n_states=best_k,
            best_cov_type=best_cov,
            best_bic=best_bic,
            best_model=best_model,
            results_table=results,
        )

    # -- Private helpers ----------------------------------------------------

    def _kmeans_init(
        self,
        features: np.ndarray,
        n_features: int,
        cov_type: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute K-Means-based initialization for HMM parameters.

        Args:
            features: Training data array.
            n_features: Dimensionality of feature vectors.
            cov_type: Covariance type string for hmmlearn.

        Returns:
            Tuple of (means, covars, transmat, startprob).
        """
        kmeans = KMeans(
            n_clusters=self.n_states,
            random_state=self.random_state,
            n_init=10,
        )
        labels = kmeans.fit_predict(features)

        # Means: cluster centroids
        means = kmeans.cluster_centers_.copy()

        # Covariances: per-cluster empirical covariance
        covars_list = []
        for j in range(self.n_states):
            cluster_data = features[labels == j]
            if len(cluster_data) < 2:
                # Fallback to identity if cluster is too small
                if cov_type == "full":
                    covars_list.append(np.eye(n_features) * 0.1)
                else:
                    covars_list.append(np.ones(n_features) * 0.1)
                continue

            cov = np.cov(cluster_data, rowvar=False)
            if cov_type == "full":
                # Regularize to ensure positive definiteness
                cov += np.eye(n_features) * 1e-6
                covars_list.append(cov)
            else:
                covars_list.append(np.diag(cov) + 1e-6)

        covars = np.array(covars_list)

        # Transition matrix: empirical from K-Means label sequence
        transmat = np.zeros((self.n_states, self.n_states))
        np.add.at(transmat, (labels[:-1], labels[1:]), 1)

        # Normalize rows (add small epsilon to avoid zero rows)
        row_sums = transmat.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        transmat = transmat / row_sums

        # Start probabilities: proportional to cluster frequency
        startprob = np.zeros(self.n_states)
        for j in range(self.n_states):
            startprob[j] = (labels == j).sum()
        startprob = startprob / startprob.sum()

        return means, covars, transmat, startprob
