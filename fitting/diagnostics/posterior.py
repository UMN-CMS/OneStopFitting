"""Bayesian posterior predictive checks.

Implements posterior predictive checking: for each posterior sample,
generate synthetic data from the observation model and compare
against the actual observations. This assesses whether the
fitted model could plausibly have generated the observed data.

The procedure is:
1. Draw function samples f* from the posterior GP at test points
2. For each f*, generate synthetic "observed" data y_rep
   by drawing from the observation model p(y | f*)
3. Compute test statistics T(y_rep) and T(y_obs)
4. p-value = fraction of samples where T(y_rep) >= T(y_obs)

A well-calibrated model yields p-values near 0.5.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import jax
from jax import random
import jax.numpy as jnp

from ..core.data import BinnedData
from ..core.transforms import DataTransformation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in test statistics
# ---------------------------------------------------------------------------


def chi2TestStatistic(
    y_obs: jnp.ndarray,
    y_pred: jnp.ndarray,
    variance: jnp.ndarray,
) -> float:
    """Chi-squared test statistic.

    T(y) = sum((y_obs - y_pred)^2 / variance)
    Only includes bins with variance > 0.
    """
    mask = variance > 0
    return float(jnp.sum((y_obs[mask] - y_pred[mask]) ** 2 / variance[mask]))


def sumDiffTestStatistic(
    y_obs: jnp.ndarray,
    y_pred: jnp.ndarray,
    variance: jnp.ndarray,
) -> float:
    mask = variance > 0

    return float(jnp.sum(jnp.abs(y_obs[mask] - y_pred[mask])))


def maxDeviationTestStatistic(
    y_obs: jnp.ndarray,
    y_pred: jnp.ndarray,
    variance: jnp.ndarray,
) -> float:
    """Maximum normalized deviation test statistic.

    T(y) = max(|y_obs - y_pred| / sqrt(variance))
    Only includes bins with variance > 0.
    """
    mask = variance > 0
    return float(
        jnp.max(jnp.abs(y_obs[mask] - y_pred[mask]) / jnp.sqrt(variance[mask]))
    )


# Type alias for test statistic functions
TestStatisticFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]


# ---------------------------------------------------------------------------
# Core posterior predictive functions
# ---------------------------------------------------------------------------


def generateReplicatedData(
    posterior_samples: jnp.ndarray,
    obs_variance: jnp.ndarray,
    rng_key: jax.Array | None = None,
    likelihood: str = "gaussian",
) -> jnp.ndarray:
    """Generate replicated observations from posterior predictive samples.

    For each posterior function sample f*, draws y_rep ~ N(f*, obs_variance).
    This represents what new data would look like if the model is correct.

    Args:
        posterior_samples: GP function samples, shape (S, N_bins).
        obs_variance: Per-bin observation variances, shape (N_bins,).
        key: JAX random key.

    Returns:
        Replicated data, shape (S, N_bins).
    """
    if rng_key is None:
        rng_key = random.key(1)

    if likelihood == "gaussian":
        noise = random.normal(rng_key, shape=posterior_samples.shape)
        return posterior_samples + noise * jnp.sqrt(obs_variance)
    elif likelihood == "poisson":
        rates = jnp.clip(posterior_samples, a_min=0.0)
        return jax.random.poisson(rng_key, rates).astype(posterior_samples.dtype)
    else:
        raise ValueError(f"Unsupported likelihood: {likelihood}")


def posteriorPredictiveCheck(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    test_data: BinnedData,
    test_statistics: dict[str, TestStatisticFn] | None = None,
    num_samples: int = 200,
    rng_key: jax.Array | None = None,
    likelihood: str = "gaussian",
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Run a full Bayesian posterior predictive check.

    Procedure:
    1. Sample f* from the posterior at test points
    2. Back-transform samples to real space if needed
    3. Generate replicated data y_rep ~ p(y | f*)
    4. Compute test statistics T(y_rep) and T(y_obs)
    5. Compute Bayesian p-values

    Args:
        pred_mean: Predicted real-space mean at test points.
        pred_cov: Predicted real-space covariance at test points.
        test_data: Test data with observed Y and V in real space.
        test_statistics: Dict of {name: fn(y_obs, y_pred, variance) -> float}.
            Defaults to {"chi2": chi2TestStatistic}.
        num_samples: Number of posterior predictive draws.
        key: JAX random key.
        blind_mask: Boolean mask indicating blinded bins. Use to calculate
            separate test stats for all/blinded/unblinded regions.

    Returns:
        Dict with:
        - 'samples': Posterior predictive samples in real space (S, N)
        - 'replicated': Replicated observations (S, N)
        - 'test_stats': Dict per test statistic with:
            - 'obs': T(y_obs)
            - 'rep': T(y_rep) for each replicate (S,)
            - 'pvalue': Bayesian p-value P(T(y_rep) >= T(y_obs))
        - 'summary': Summary statistics (mean, std, q05, q95)
    """
    if test_statistics is None:
        test_statistics = {"chi2": chi2TestStatistic}

    if rng_key is None:
        rng_key = random.key(0)

    # 1. Sample f* from real-space MVN
    rng_key, sample_key, rep_key = random.split(rng_key, 3)

    # Add small jitter to diagonal for numerical stability of Cholesky
    jitter = 1e-6 * jnp.eye(pred_cov.shape[0])
    samples = jax.random.multivariate_normal(
        sample_key, mean=pred_mean, cov=pred_cov + jitter, shape=(num_samples,)
    )
    pred_variance = jnp.diag(pred_cov)

    # 2. Generate replicated data
    replicated = generateReplicatedData(
        samples, test_data.V, rep_key, likelihood=likelihood
    )

    # 4. Compute test statistics
    obs_Y = test_data.Y
    obs_V = test_data.V
    pred_mean = jnp.mean(samples, axis=0)

    test_stat_results: dict[str, dict[str, dict[str, Any]]] = {}

    # Pre-calculate region masks
    masks = {"all": jnp.ones_like(obs_Y, dtype=bool)}
    if blind_mask is not None and jnp.any(blind_mask):
        masks["blinded"] = blind_mask
        masks["unblinded"] = ~blind_mask

    for name, stat_fn in test_statistics.items():
        test_stat_results[name] = {}
        for region_name, r_mask in masks.items():
            # Skip if region has no valid bins
            if not jnp.any(r_mask) or not jnp.any(obs_V[r_mask] > 0):
                continue

            y_obs_masked = obs_Y[r_mask]
            v_obs_masked = obs_V[r_mask]
            pred_mean_masked = pred_mean[r_mask]
            pred_variance_masked = pred_variance[r_mask]

            obs_val = stat_fn(y_obs_masked, pred_mean_masked, pred_variance_masked)

            # Mask replicated data as well
            rep_vals = jnp.array(
                [
                    stat_fn(y_rep[r_mask], pred_mean_masked, pred_variance_masked)
                    for y_rep in replicated
                ]
            )

            pvalue = float(jnp.mean(rep_vals <= obs_val))

            test_stat_results[name][region_name] = {
                "obs": obs_val,
                "rep": rep_vals,
                "pvalue": pvalue,
            }

            logger.info(
                f"PPC [{name} - {region_name}]: p-value = {pvalue:.3f} "
                f"(obs={obs_val:.1f}, median_rep={float(jnp.median(rep_vals)):.1f})"
            )

    # 5. Summary statistics
    summary = {
        "mean": jnp.mean(samples, axis=0),
        "std": jnp.std(samples, axis=0),
        "q05": jnp.percentile(samples, 5, axis=0),
        "q95": jnp.percentile(samples, 95, axis=0),
        "median": jnp.median(samples, axis=0),
    }

    return {
        "samples": samples,
        "replicated": replicated,
        "test_stats": test_stat_results,
        "summary": summary,
    }
