from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from jax import random
import json

from ..core.data import AnalysisState, BinnedData
from ..diagnostics.plot_utils import savePlots
from ..diagnostics.plots import makeDiagnosticPlots, makePosteriorPredictivePlots
from ..inference.prediction import getPriorMeanInRealSpace
from gpjax.variational_families import VariationalGaussian
from ..core.serialization import getSummary, limitedSummary

logger = logging.getLogger(__name__)


def generatePlots(state: AnalysisState, rng_key: jax.Array) -> None:
    """Generate and save diagnostic plots."""

    if (
        state.training_result is None
        or state.pred_mean is None
        or state.pred_cov is None
    ):
        raise ValueError("Cannot generate plots without prediction results.")

    # Prepare signal for overlay if injected
    signal_plot_data = None
    signal_template = None
    if state.signal is not None:
        # Match test_data domain by applying same mask
        signal_template = state.signal.masked(state.domain_mask)

        if state.injection_rate > 0:
            # Scale by injection rate for the "Injected Signal" plot/overlay
            signal_plot_data = BinnedData(
                X=signal_template.X,
                Y=signal_template.Y * state.injection_rate,
                V=signal_template.V * (state.injection_rate**2),
                edges=signal_template.edges,
                axis_names=signal_template.axis_names,
            )

    prior_key, diag_key = random.split(rng_key)
    prior_mean = getPriorMeanInRealSpace(
        posterior=state.training_result.posterior,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=prior_key,
    )

    pred_var = jnp.diag(state.pred_cov)

    posterior = state.training_result.posterior
    if isinstance(posterior, VariationalGaussian):
        posterior = posterior.posterior

    plots = makeDiagnosticPlots(
        pred_mean=state.pred_mean,
        pred_var=pred_var,
        test_data=state.test_data,
        train_data=state.train_data,
        blind_mask=state.blind_mask,
        signal_data=signal_plot_data,
        signal_template=signal_template,
        prior_mean=prior_mean,
        kernel=posterior.prior.kernel,
        transform=state.transform,
        pred_cov=state.pred_cov,
    )

    if state.ppc_results is not None:
        ppc_plots = makePosteriorPredictivePlots(
            ppc_results=state.ppc_results,
            test_data=state.test_data,
            blind_mask=state.blind_mask,
        )
        plots.update(ppc_plots)

    plot_dir = state.getRealOutPath() / "diagnostics"
    savePlots(plots, plot_dir, formats=state.config.image_formats)
    with open(plot_dir / "ALL.json", "w") as f:
        json.dump(limitedSummary(state), f)
    logger.info(f"Pipeline complete. Plots saved to {plot_dir}")
