"""Eigenvariation computation from posterior covariance.

Provides utilities for decomposing the posterior covariance matrix
into eigenvariations for use as shape systematics in Combine.
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


def computeEigenvariations(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    threshold_fraction: float = 0.01,
    max_variations: int | None = None,
) -> list[dict[str, jnp.ndarray]]:
    """Compute eigenvariations for Combine shape systematics.

    Each eigenvariation produces an Up and Down shape variation
    corresponding to ±1σ along the eigenvector direction.

    Args:
        pred_mean: Predicted mean in real space, shape (N,).
        pred_cov: Predicted covariance in real space, shape (N, N).
        threshold_fraction: Keep eigenvectors with eigenvalue
            >= threshold_fraction * max_eigenvalue.
        max_variations: Maximum number of variations. None = keep all above threshold.

    Returns:
        List of dicts, each with keys:
        - 'nominal': pred_mean
        - 'up': pred_mean + scaled_eigenvector
        - 'down': pred_mean - scaled_eigenvector
        - 'eigenvalue': the eigenvalue
        - 'index': variation index
    """
    from ..inference.prediction import computeScaledEigenvectors

    eigenvalues, scaled_vecs = computeScaledEigenvectors(
        pred_cov, threshold_fraction=threshold_fraction
    )

    n_vars = scaled_vecs.shape[1]
    if max_variations is not None:
        n_vars = min(n_vars, max_variations)

    variations = []
    for i in range(n_vars):
        variation_vec = scaled_vecs[:, i]
        variations.append(
            {
                "nominal": pred_mean,
                "up": jnp.maximum(pred_mean + variation_vec, 0.0),
                "down": jnp.maximum(pred_mean - variation_vec, 0.0),
                "eigenvalue": eigenvalues[i],
                "index": i,
            }
        )

    logger.info(
        f"Computed {n_vars} eigenvariations "
        f"(threshold={threshold_fraction}, "
        f"max eigenvalue={float(eigenvalues[0]):.4g})"
    )

    return variations
