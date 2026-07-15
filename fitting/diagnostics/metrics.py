from __future__ import annotations

import jax.numpy as jnp


def chi2PerBin(
    obs: jnp.ndarray,
    exp: jnp.ndarray,
    var: jnp.ndarray,
    mask: jnp.ndarray | None = None,
) -> float:
    if mask is not None:
        obs, exp, var = obs[mask], exp[mask], var[mask]
    zero_mask = var == 0
    obs, exp, var = obs[~zero_mask], exp[~zero_mask], var[~zero_mask]
    n = obs.shape[0]
    if n == 0:
        return float("nan")
    return float(jnp.sum((obs - exp) ** 2 / var) / n)


def pullDistribution(
    obs: jnp.ndarray,
    exp: jnp.ndarray,
    var: jnp.ndarray,
) -> jnp.ndarray:
    return (obs - exp) / jnp.sqrt(var)


def totalPullDistribution(
    obs: jnp.ndarray,
    exp: jnp.ndarray,
    obs_var: jnp.ndarray,
    pred_var: jnp.ndarray,
) -> jnp.ndarray:
    return (obs - exp) / jnp.sqrt(obs_var + pred_var)


def computeDiagnosticMetrics(
    test_data_Y: jnp.ndarray,
    test_data_V: jnp.ndarray,
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    blind_mask: jnp.ndarray | None = None,
    min_count_mask: jnp.ndarray | None = None,
) -> dict[str, float]:
    metrics = {}

    metrics["global_chi2_per_bin"] = chi2PerBin(test_data_Y, pred_mean, test_data_V)

    metrics["global_chi2_pred_var"] = chi2PerBin(test_data_Y, pred_mean, pred_var)

    if blind_mask is not None and jnp.any(blind_mask):
        metrics["blinded_chi2_per_bin"] = chi2PerBin(
            test_data_Y, pred_mean, test_data_V, mask=blind_mask
        )
        metrics["blinded_chi2_pred_per_bin"] = chi2PerBin(
            test_data_Y, pred_mean, pred_var, mask=blind_mask
        )

    if min_count_mask is not None and jnp.any(min_count_mask):
        metrics["highstat_chi2_per_bin"] = chi2PerBin(
            test_data_Y, pred_mean, test_data_V, mask=min_count_mask
        )

    return metrics
