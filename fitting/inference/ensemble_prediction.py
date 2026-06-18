from __future__ import annotations

import enum
import logging
from typing import Any

import jax
import jax.numpy as jnp
from jax import random

from ..core.data import BinnedData
from ..core.transforms import DataTransformation

logger = logging.getLogger(__name__)


class EnsembleMode(enum.Enum):
    """How to combine ensemble predictions."""

    AVERAGE = "average"
    BEST_MLL = "best_mll"
    MEDIAN = "median"


def computeEnsemblePrediction(
    posteriors: list[Any],
    datasets: list[Any],
    test_data: BinnedData,
    transform: DataTransformation,
    mode: EnsembleMode,
    rng_key: jax.Array,
    samples_list: list[dict[str, jnp.ndarray] | None] | None = None,
    losses: list[float] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute ensemble-averaged prediction in real space.

    For each seed, computes the posterior mean and covariance in real
    space, then combines using BMA decomposition.

    Args:
        posteriors: List of trained posterior objects, one per seed.
        datasets: List of training datasets (one per seed, or a single
            dataset replicated).
        test_data: BinnedData for prediction points.
        transform: DataTransformation for converting to real space.
        mode: How to combine predictions (average, median, best_mll).
        rng_key: RNG key for predictions.
        samples_list: Optional MCMC samples per seed (usually None).
        losses: Final MLL losses per seed (needed for best_mll mode).

    Returns:
        (ensemble_mean, ensemble_cov, seed_means_arr) — the combined
        mean and covariance in real space, plus all per-seed means
        for downstream diagnostics.
    """
    from .prediction import predictInRealSpace

    num_seeds = len(posteriors)

    seed_means = []
    seed_covs = []

    for i in range(num_seeds):
        pred_key, rng_key = random.split(rng_key)

        ds = datasets[i] if len(datasets) > 1 else datasets[0]
        samples = samples_list[i] if samples_list else None

        mean_i, cov_i = predictInRealSpace(
            posterior=posteriors[i],
            dataset_train=ds,
            test_data=test_data,
            transform=transform,
            samples=samples,
            rng_key=pred_key,
        )
        seed_means.append(mean_i)
        seed_covs.append(cov_i)

    seed_means_arr = jnp.stack(seed_means, axis=0)  # (num_seeds, n_bins)
    seed_covs_arr = jnp.stack(seed_covs, axis=0)  # (num_seeds, n_bins, n_bins)

    if mode == EnsembleMode.AVERAGE:
        ensemble_mean = jnp.mean(seed_means_arr, axis=0)
        within_cov = jnp.mean(seed_covs_arr, axis=0)
        mean_deviations = seed_means_arr - ensemble_mean[None, :]
        between_cov = (mean_deviations.T @ mean_deviations) / num_seeds
        ensemble_cov = within_cov + between_cov

    elif mode == EnsembleMode.MEDIAN:
        ensemble_mean = jnp.median(seed_means_arr, axis=0)
        within_cov = jnp.mean(seed_covs_arr, axis=0)
        mean_deviations = seed_means_arr - ensemble_mean[None, :]
        between_cov = (mean_deviations.T @ mean_deviations) / num_seeds
        ensemble_cov = within_cov + between_cov

    elif mode == EnsembleMode.BEST_MLL:
        if losses is None:
            raise ValueError("best_mll mode requires losses")
        best_idx = int(jnp.argmin(jnp.array(losses)))
        ensemble_mean = seed_means_arr[best_idx]
        ensemble_cov = seed_covs_arr[best_idx]

    else:
        raise ValueError(f"Unknown ensemble mode: {mode}")

    seed_std = jnp.std(seed_means_arr, axis=0)
    within_std = jnp.sqrt(jnp.maximum(jnp.diag(jnp.mean(seed_covs_arr, axis=0)), 0.0))
    ensemble_std = jnp.sqrt(jnp.maximum(jnp.diag(ensemble_cov), 0.0))

    logger.info(
        f"Ensemble prediction ({mode.value}, {num_seeds} seeds): "
        f"between-seed std [{float(jnp.min(seed_std)):.2f}, {float(jnp.max(seed_std)):.2f}], "
        f"within-model std [{float(jnp.min(within_std)):.2f}, {float(jnp.max(within_std)):.2f}], "
        f"total std [{float(jnp.min(ensemble_std)):.2f}, {float(jnp.max(ensemble_std)):.2f}]"
    )

    return ensemble_mean, ensemble_cov, seed_means_arr
