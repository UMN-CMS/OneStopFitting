from __future__ import annotations

import logging

import attrs
import jax
import jax.numpy as jnp
from jax import random

from ..core.data import AnalysisState
from ..diagnostics.metrics import computeDiagnosticMetrics
from ..inference.prediction import predictInRealSpace
from ..diagnostics.posterior import posteriorPredictiveCheck

logger = logging.getLogger(__name__)


def runDiagnostics(state: AnalysisState, rng_key: jax.Array) -> AnalysisState:
    """Predict in real space and compute diagnostic metrics."""

    if state.training_result is None or state.dataset is None:
        raise ValueError("Cannot run diagnostics without training result and dataset.")

    pred_key, ppc_key = random.split(rng_key)

    # Check if ensemble posteriors are available
    tr = state.training_result
    if tr.all_posteriors is not None and len(tr.all_posteriors) > 1:
        pred_mean, pred_cov = _ensemblePrediction(
            state, pred_key, tr.all_posteriors, tr.all_losses, tr.ensemble_mode
        )
    else:
        pred_mean, pred_cov = predictInRealSpace(
            posterior=tr.posterior,
            dataset_train=state.dataset,
            test_data=state.test_data,
            transform=state.transform,
            samples=tr.samples,
            rng_key=pred_key,
        )

    inflation = state.config.window_variance_inflation
    if inflation != 1.0 and state.blind_mask is not None:
        diag_idx = jnp.arange(pred_cov.shape[0])
        inflation_factors = jnp.where(state.blind_mask, inflation, 1.0)
        logger.info(f"Inflating variance by factor {inflation} in blind regions.")

        pred_cov = pred_cov.at[diag_idx, diag_idx].multiply(inflation_factors)

    flat_variance_scale = state.config.window_variance_increase
    if flat_variance_scale is not None and state.blind_mask is not None:
        diag_idx = jnp.arange(pred_cov.shape[0])
        increase = jnp.where(state.blind_mask, flat_variance_scale, 0.0)
        logger.info(
            f"Adding a constant variance in blinded window of {flat_variance_scale}"
        )

        pred_cov = pred_cov.at[diag_idx, diag_idx].add(increase)

    pred_var = jnp.diag(pred_cov)
    state = attrs.evolve(state, pred_mean=pred_mean, pred_cov=pred_cov)

    metrics = computeDiagnosticMetrics(
        test_data_Y=state.test_data.Y,
        test_data_V=state.test_data.V,
        pred_mean=pred_mean,
        pred_var=pred_var,
        blind_mask=state.blind_mask,
    )
    logger.info(f"Diagnostic metrics: {metrics}")
    state = attrs.evolve(state, diagnostic_metrics=metrics)

    # Posterior predictive checks
    # likelihood_type = "gaussian"
    # likelihood_name = state.config.model.likelihood.__class__.__name__.lower()
    # if "poisson" in likelihood_name:
    #     likelihood_type = "poisson"
    likelihood_type = "poisson"

    ppc_results = posteriorPredictiveCheck(
        pred_mean=pred_mean,
        pred_cov=pred_cov,
        test_data=state.test_data,
        num_samples=200,
        rng_key=ppc_key,
        likelihood=likelihood_type,
        blind_mask=state.blind_mask,
    )

    state = attrs.evolve(state, ppc_results=ppc_results)
    return state


def _ensemblePrediction(
    state: AnalysisState,
    rng_key: jax.Array,
    posteriors: list,
    losses: list[float] | None,
    mode: str | None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    from ..inference.ensemble_prediction import (
        computeEnsemblePrediction,
        EnsembleMode,
    )

    ensemble_mode = EnsembleMode(mode or "average")
    num_seeds = len(posteriors)
    logger.info(
        f"=== Ensemble prediction: {num_seeds} seeds, mode={ensemble_mode.value} ==="
    )

    ensemble_mean, ensemble_cov, seed_means = computeEnsemblePrediction(
        posteriors=posteriors,
        datasets=[state.dataset],  # same dataset for all seeds
        test_data=state.test_data,
        transform=state.transform,
        mode=ensemble_mode,
        rng_key=rng_key,
        losses=losses,
    )

    # Log per-seed loss spread
    if losses:
        losses_arr = jnp.array(losses)
        logger.info(
            f"  Per-seed MLL: mean={float(jnp.mean(losses_arr)):.4f}, "
            f"std={float(jnp.std(losses_arr)):.4f}, "
            f"range=[{float(jnp.min(losses_arr)):.4f}, {float(jnp.max(losses_arr)):.4f}]"
        )

    return ensemble_mean, ensemble_cov
