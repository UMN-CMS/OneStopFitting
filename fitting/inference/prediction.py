"""Prediction utilities.

Compute GP predictions at test points and back-transform
the resulting multivariate normal to real (un-normalized) space.
Supports sample-based prediction for MCMC using NumPyro Predictive.
"""

from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import Predictive

from ..core.data import BinnedData
from ..core.transforms import DataTransformation

logger = logging.getLogger(__name__)


def computePrediction(
    posterior: Any,
    dataset_train: Any,
    test_X: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GP predictive mean and covariance at test points.

    Args:
        posterior: Optimized gpjax posterior.
        dataset_train: gpjax.Dataset used for training.
        test_X: Test input locations, shape (N_test, D).

    Returns:
        Tuple of (mean, covariance) in normalized space.
        mean has shape (N_test,), cov has shape (N_test, N_test).
    """
    latent_dist = posterior.predict(test_X, train_data=dataset_train)
    mean = latent_dist.mean.ravel()
    cov = latent_dist.covariance()
    return mean, cov


def predictInRealSpace(
    posterior: Any,
    dataset_train: Any,
    test_data: BinnedData,
    transform: DataTransformation,
    samples: dict[str, jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Predict and back-transform to real (un-normalized) bin counts.

    This is the key function that bridges normalized GP inference
    with the real-space quantities needed for physics analysis.

    Supports both point-estimate (MLE/MAP) and sample-based (MCMC) prediction.

    Args:
        posterior: GP posterior.
        dataset_train: gpjax.Dataset used for training.
        test_data: Test data in ORIGINAL (un-normalized) space.
        transform: The normalization transform applied during training.
        samples: Optional dict of MCMC samples.

    Returns:
        Tuple of (real_mean, real_cov) in original bin count space.
    """
    # Transform test points to normalized space
    norm_test_X = transform.applyX(test_data.X)

    if samples is not None and len(samples) > 0:
        norm_mean, norm_cov = predictWithSamples(posterior, dataset_train, norm_test_X, samples)
    else:
        # Predict in normalized space using current posterior parameters (MLE/MAP)
        norm_mean, norm_cov = computePrediction(posterior, dataset_train, norm_test_X)

    # Back-transform to real space
    real_mean, real_cov = transform.invertMVN(norm_mean, norm_cov)

    logger.info(
        f"Prediction complete: {len(real_mean)} bins, "
        f"mean range [{float(jnp.min(real_mean)):.1f}, {float(jnp.max(real_mean)):.1f}]"
    )

    return real_mean, real_cov


def predictWithSamples(
    posterior: Any,
    dataset_train: Any,
    test_X: jnp.ndarray,
    samples: dict[str, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute predictive mean and covariance averaged over parameter samples.

    Uses numpyro.infer.Predictive to aggregate across MCMC samples.

    Args:
        posterior: GP posterior module.
        dataset_train: Training data.
        test_X: Test points.
        samples: Dict of sampled parameters.

    Returns:
        Averaged (mean, cov) in normalized space.
    """
    from .optimization import mcmc_model_fn

    num_samples = len(next(iter(samples.values())))
    logger.info(f"Aggregating prediction across {num_samples} MCMC samples using NumPyro Predictive")

    predictive = Predictive(mcmc_model_fn, posterior_samples=samples)
    
    # Run predictive to get samples of 'f_new' (the latent function)
    rng_key = jax.random.PRNGKey(0xBEEF)
    predictions = predictive(rng_key, posterior, dataset_train.X, dataset_train.y, test_X)
    
    # f_samples shape: (num_samples, num_test_points)
    f_samples = predictions["f_new"]
    
    # Total mean: E[f] = E_theta[ E[f|theta] ]
    avg_mean = jnp.mean(f_samples, axis=0)
    
    # Total covariance: Cov(f) = E_theta[ Cov(f|theta) ] + Cov_theta( E[f|theta] )
    # This is exactly what the variance of sampling from Predictive captures.
    if num_samples > 1:
        avg_cov = jnp.cov(f_samples, rowvar=False)
    else:
        # If only one sample, we can only return the conditional variance from that theta
        # But MCMC should usually have many samples.
        avg_cov = jnp.zeros((test_X.shape[0], test_X.shape[0]))

    return avg_mean, avg_cov


def fixCovarianceMatrix(cov: jnp.ndarray) -> jnp.ndarray:
    """Ensure a covariance matrix is positive semi-definite.

    Performs eigendecomposition and clamps negative eigenvalues to zero,
    then reconstructs. This is needed when numerical issues produce
    slightly non-PSD matrices.

    Args:
        cov: Covariance matrix, shape (N, N).

    Returns:
        Fixed PSD covariance matrix.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    # Clamp negative eigenvalues
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    fixed_cov = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    # Symmetrize
    fixed_cov = (fixed_cov + fixed_cov.T) / 2
    return fixed_cov


def computeScaledEigenvectors(
    cov: jnp.ndarray,
    threshold_fraction: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute eigendecomposition of the posterior covariance.

    Returns eigenvalues and eigenvectors scaled by sqrt(eigenvalue),
    sorted by descending eigenvalue. Used for constructing shape
    variations for Higgs Combine.

    Args:
        cov: Covariance matrix, shape (N, N).
        threshold_fraction: Keep eigenvariations with eigenvalue
            >= threshold_fraction * max_eigenvalue.

    Returns:
        Tuple of (eigenvalues, scaled_eigenvectors) where eigenvalues
        are sorted descending and eigenvectors are scaled.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    # Sort descending
    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Filter by threshold
    max_eval = eigenvalues[0]
    mask = eigenvalues >= threshold_fraction * max_eval
    n_kept = int(jnp.sum(mask))

    eigenvalues = eigenvalues[:n_kept]
    eigenvectors = eigenvectors[:, :n_kept]

    # Scale by sqrt(eigenvalue)
    scales = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    scaled_vecs = eigenvectors * scales[None, :]

    logger.info(
        f"Eigendecomposition: kept {n_kept} of {cov.shape[0]} "
        f"variations (threshold={threshold_fraction})"
    )

    return eigenvalues, scaled_vecs
