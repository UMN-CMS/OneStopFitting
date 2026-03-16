from __future__ import annotations

import logging
from typing import Any

import jax
import jax.numpy as jnp
from numpyro.infer import Predictive

from ..core.data import BinnedData
from ..core.transforms import DataTransformation

logger = logging.getLogger(__name__)


def computePrediction(
    posterior: Any,
    dataset_train: Any,
    test_X: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
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
    rng_key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    norm_test_X = transform.applyX(test_data.X)

    if samples is not None and len(samples) > 0:
        norm_mean, norm_cov = predictWithSamples(
            posterior, dataset_train, norm_test_X, samples, rng_key
        )
    else:
        norm_mean, norm_cov = computePrediction(posterior, dataset_train, norm_test_X)
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
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from .optimization import modelFunc

    num_samples = len(next(iter(samples.values())))
    logger.info(
        f"Aggregating prediction across {num_samples} MCMC samples using NumPyro Predictive"
    )

    predictive = Predictive(modelFunc, posterior_samples=samples)
    predictions = predictive(
        rng_key, posterior, dataset_train.X, dataset_train.y, test_X
    )
    f_samples = predictions["f_new"]
    avg_mean = jnp.mean(f_samples, axis=0)
    if num_samples > 1:
        avg_cov = jnp.cov(f_samples, rowvar=False)
    else:
        avg_cov = jnp.zeros((test_X.shape[0], test_X.shape[0]))

    return avg_mean, avg_cov


def fixCovarianceMatrix(cov: jnp.ndarray) -> jnp.ndarray:
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
    eigenvalues = jnp.maximum(eigenvalues, 0.0)
    fixed_cov = eigenvectors @ jnp.diag(eigenvalues) @ eigenvectors.T
    fixed_cov = (fixed_cov + fixed_cov.T) / 2
    return fixed_cov


def computeScaledEigenvectors(
    cov: jnp.ndarray,
    threshold_fraction: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(cov)

    idx = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    max_eval = eigenvalues[0]
    mask = eigenvalues >= threshold_fraction * max_eval
    n_kept = int(jnp.sum(mask))

    eigenvalues = eigenvalues[:n_kept]
    eigenvectors = eigenvectors[:, :n_kept]

    scales = jnp.sqrt(jnp.maximum(eigenvalues, 0.0))
    scaled_vecs = eigenvectors * scales[None, :]

    logger.info(
        f"Eigendecomposition: kept {n_kept} of {cov.shape[0]} "
        f"variations (threshold={threshold_fraction})"
    )

    return eigenvalues, scaled_vecs


def getPriorMeanInRealSpace(
    posterior: Any,
    test_data: BinnedData,
    transform: DataTransformation,
    samples: dict[str, jnp.ndarray] | None = None,
    rng_key: jax.Array | None = None,
) -> jnp.ndarray:
    norm_test_X = transform.applyX(test_data.X)
    mean_fn = posterior.prior.mean_function

    norm_prior_mean = mean_fn(norm_test_X).ravel()
    real_prior_mean, _ = transform.invertMVN(
        norm_prior_mean, jnp.zeros((len(norm_prior_mean), len(norm_prior_mean)))
    )

    return real_prior_mean


def drawPoissonSamples(
    rng_key: jax.Array,
    mean: jnp.ndarray,
    cov: jnp.ndarray | None = None,
    num_samples: int = 1,
) -> jnp.ndarray:
    key1, key2 = jax.random.split(rng_key)

    if cov is not None:
        cov = fixCovarianceMatrix(cov)
        f_samples = jax.random.multivariate_normal(
            key1, mean, cov, shape=(num_samples,)
        )
    else:
        f_samples = jnp.broadcast_to(mean, (num_samples, len(mean)))

    f_samples = jnp.maximum(f_samples, 0.0)

    samples = jax.random.poisson(key2, f_samples)
    return samples
