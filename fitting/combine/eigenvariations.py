from __future__ import annotations
import logging
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def computeEigenvariations(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    threshold_fraction: float = 0.00,
) -> list[dict[str, jnp.ndarray]]:
    from ..inference.prediction import computeScaledEigenvectors

    eigenvalues, scaled_vecs = computeScaledEigenvectors(
        pred_cov, threshold_fraction=threshold_fraction
    )

    n_vars = scaled_vecs.shape[1]

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
