"""Statistical metrics for evaluating GP regression quality."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def chi2PerBin(
    obs: jnp.ndarray,
    exp: jnp.ndarray,
    var: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> float:
    """Compute chi-squared per bin.

    Args:
        obs: Observed bin values.
        exp: Expected (predicted) bin values.
        var: Variances for the chi2 denominator.
        mask: Optional boolean mask to select bins.

    Returns:
        Chi-squared sum divided by number of bins.
    """
    if mask is not None:
        obs, exp, var = obs[mask], exp[mask], var[mask]
    n = obs.shape[0]
    if n == 0:
        return float("nan")
    return float(jnp.sum((obs - exp) ** 2 / var) / n)


def pullDistribution(
    obs: jnp.ndarray,
    exp: jnp.ndarray,
    var: jnp.ndarray,
) -> jnp.ndarray:
    """Compute pull values: (obs - exp) / sqrt(var).

    Args:
        obs: Observed bin values.
        exp: Expected (predicted) bin values.
        var: Variances (can be observed or predicted).

    Returns:
        Array of pull values, same shape as inputs.
    """
    return (obs - exp) / jnp.sqrt(var)


def computeDiagnosticMetrics(
    test_data_Y: jnp.ndarray,
    test_data_V: jnp.ndarray,
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    blind_mask: jnp.ndarray | None = None,
    min_count_mask: jnp.ndarray | None = None,
) -> dict[str, float]:
    """Compute a suite of diagnostic metrics.

    Args:
        test_data_Y: Observed bin values in real space.
        test_data_V: Observed bin variances in real space.
        pred_mean: Predicted mean in real space.
        pred_var: Predicted variances in real space (diagonal of covariance).
        blind_mask: Boolean mask for blinded region bins.
        min_count_mask: Optional mask for bins with sufficient counts.

    Returns:
        Dict of metric names to values.
    """
    metrics = {}

    # Global chi2 (using observed variance)
    metrics["global_chi2_per_bin"] = chi2PerBin(test_data_Y, pred_mean, test_data_V)

    # Global chi2 (using predicted variance)
    metrics["global_chi2_pred_var"] = chi2PerBin(test_data_Y, pred_mean, pred_var)

    # Blinded region chi2
    if blind_mask is not None and jnp.any(blind_mask):
        metrics["blinded_chi2_per_bin"] = chi2PerBin(
            test_data_Y, pred_mean, test_data_V, mask=blind_mask
        )

    # Chi2 for high-stat bins only
    if min_count_mask is not None and jnp.any(min_count_mask):
        metrics["highstat_chi2_per_bin"] = chi2PerBin(
            test_data_Y, pred_mean, test_data_V, mask=min_count_mask
        )

    return metrics
