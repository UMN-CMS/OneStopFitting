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
from ..core.serialization import limitedSummary

logger = logging.getLogger(__name__)


def generatePlots(state: AnalysisState, rng_key: jax.Array) -> None:
    """Generate and save diagnostic plots."""

    if (
        state.training_result is None
        or state.pred_mean is None
        or state.pred_cov is None
    ):
        raise ValueError("Cannot generate plots without prediction results.")

    # Prepare signals for overlay if injected
    signals_plot_data = {}
    signals_template = {}
    for lbl, sig in state.signals.items():
        # Match test_data domain by applying same mask
        sig_template = sig.masked(state.domain_mask)
        signals_template[lbl] = sig_template

        if state.injection_rate > 0:
            # Scale by injection rate for the "Injected Signal" plot/overlay
            signals_plot_data[lbl] = BinnedData(
                X=sig_template.X,
                Y=sig_template.Y * state.injection_rate,
                V=sig_template.V * (state.injection_rate**2),
                edges=sig_template.edges,
                axis_names=sig_template.axis_names,
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
        signal_data=signals_plot_data if signals_plot_data else None,
        signal_template=signals_template if signals_template else None,
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
    savePlots(plots, plot_dir, [state.metadata], formats=state.config.image_formats)
    with open(plot_dir / "ALL.json", "w") as f:
        json.dump(limitedSummary(state), f)
    logger.info(f"Pipeline complete. Plots saved to {plot_dir}")
