from __future__ import annotations
import logging
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def computeEigenvariations(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    threshold_fraction: float = 0.00,
    signal_size: float | None = None,
) -> list[dict[str, jnp.ndarray]]:
    from ..inference.prediction import computeScaledEigenvectors

    if pred_cov.shape[0] == 0:
        return []

    eigenvalues, scaled_vecs = computeScaledEigenvectors(
        pred_cov, threshold_fraction=threshold_fraction, signal_size=signal_size
    )

    n_vars = scaled_vecs.shape[1]

    variations = []
    for i in range(n_vars):
        variation_vec = scaled_vecs[:, i]
        # max_disp = jnp.maximum(pred_mean, 0.0)
        # clipped_var = jnp.clip(variation_vec, -max_disp, max_disp)
        variations.append(
            {
                "nominal": pred_mean,
                "up": pred_mean + variation_vec,
                "down": pred_mean - variation_vec,
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
