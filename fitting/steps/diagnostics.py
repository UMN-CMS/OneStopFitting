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
    pred_mean, pred_cov = predictInRealSpace(
        posterior=state.training_result.posterior,
        dataset_train=state.dataset,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=pred_key,
    )

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
    likelihood_type = "gaussian"
    likelihood_name = state.config.model.likelihood.__class__.__name__.lower()
    if "poisson" in likelihood_name:
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
