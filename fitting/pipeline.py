"""Pipeline configuration and execution.

Defines PipelineConfig and runPipeline — the top-level orchestration
of the background estimation workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import gpjax
import jax.numpy as jnp
import mplhep
import flax.nnx as nnx

from .core.data import AnalysisState, BinnedData
from .core.serialization import save
from .core.transforms import (
    StandardizationConfig,
    TransformConfig,
    computeNormalization,
)
from .data.preprocessing import preprocess
from .diagnostics.metrics import computeDiagnosticMetrics
from .diagnostics.plot_utils import savePlots
from .diagnostics.plots import makeDiagnosticPlots
from .inference.models import ExactGPConfig, GPModelConfig
from .inference.optimization import OptimizationConfig, train
from .inference.prediction import predictInRealSpace

logger = logging.getLogger(__name__)


@attrs.define
class PipelineConfig:
    """Complete pipeline configuration.

    Fully serializable via cattrs. All types are real Python types,
    not strings.

    Attributes:
        background_path: Path to background histogram file.
        signal_path: Path to signal histogram file (optional).
        signal_name: Label/key for the signal in the file.
        signal_selection: Selection key to extract signal from file dict.
        injection_rate: Signal injection strength (0 = no injection).
        fit_region: Tuple of (lower, upper) per axis for domain cut.
        rebin: Rebinning factor applied to histograms on load.
        min_counts: Minimum bin count to include in fit.
        window_spread: Gaussian window spread in sigma.
        transform: Data normalization config.
        model: GP model config.
        optimization: Training loop config.
        output_dir: Output directory for results.
        metadata: Flexible bookkeeping dict.
    """

    # Data
    background_path: Path
    signal_path: Path | None = None
    signal_name: str | None = None
    signal_selection: str | None = None
    injection_rate: float = 0.0
    rebin: int = 1
    min_counts: float = 10.0

    rng_seed: int = 0xBEEFBEEF

    # Windowing
    window_spread: float = 2.0

    # Transform
    transform: TransformConfig = attrs.Factory(StandardizationConfig)

    # Model
    model: GPModelConfig = attrs.Factory(ExactGPConfig)

    # Optimization
    optimization: OptimizationConfig = attrs.Factory(OptimizationConfig)

    # Output
    output_dir: Path = Path("output")
    image_formats: list[str] = attrs.Factory(lambda: ["png"])

    # Flexible bookkeeping — not forced into any specific schema
    metadata: dict[str, Any] = attrs.Factory(dict)


def runPipeline(config: PipelineConfig) -> AnalysisState:
    """Execute the full background estimation pipeline.

    Steps:
    1. Load background (and optionally signal) histograms
    2. Create blinding window (from signal or pre-configured)
    3. Preprocess: domain mask, signal injection, train/test split
    4. Normalize training data
    5. Train GP model
    6. Predict in real space (back-transform MVN)
    7. Compute and save diagnostics
    8. Save full state for resume

    Args:
        config: Pipeline configuration.

    Returns:
        AnalysisState with all fields populated.
    """
    from .data.loading import (
        FileLoader,
        extractHistogram,
        extractMetadata,
        histToBinnedData,
    )
    import jax

    jax.config.update("jax_enable_x64", True)

    # --- 1. Load data ---
    logger.info(f"Loading background from {config.background_path}")
    loader = FileLoader.forPath(config.background_path)
    bkg_raw = loader.load(config.background_path)
    bkg_hist = extractHistogram(bkg_raw)
    background = histToBinnedData(bkg_hist, rebin=config.rebin, variation="central")

    # Extract and merge file-level metadata
    file_metadata = extractMetadata(bkg_raw)
    combined_metadata = {**config.metadata, **file_metadata}

    signal = None
    signal_hist = None
    if config.signal_path is not None:
        logger.info(f"Loading signal from {config.signal_path}")
        sig_loader = FileLoader.forPath(config.signal_path)
        sig_raw = sig_loader.load(config.signal_path)
        sig_hist_full = extractHistogram(sig_raw)
        signal = histToBinnedData(
            sig_hist_full, rebin=config.rebin, variation="central"
        )
        signal_hist = sig_hist_full  # keep all variations for downstream

        # Merge signal metadata (overrides background metadata on conflict)
        sig_metadata = extractMetadata(sig_raw)
        combined_metadata = {**combined_metadata, **sig_metadata}

    state = AnalysisState(
        config=config,
        background=background,
        signal=signal,
        signal_name=config.signal_name,
        injection_rate=config.injection_rate,
        background_hist=bkg_hist,
        signal_hist=signal_hist,
        metadata=combined_metadata,
    )

    # --- 2-3. Preprocess ---
    state = preprocess(state, min_counts=config.min_counts)

    # --- 4. Normalize ---
    transform = computeNormalization(state.train_data, config=config.transform)
    norm_train = transform.applyToBinnedData(state.train_data)
    state = attrs.evolve(state, transform=transform)

    # --- 5. Train GP ---
    dataset = gpjax.Dataset(
        X=norm_train.X,
        y=norm_train.Y.reshape(-1, 1),
    )

    rngs = nnx.Rngs(123)
    posterior, likelihood, prior = config.model.buildModel(
        dataset=dataset,
        ndim=background.ndim,
        obs_variance=norm_train.V.reshape(-1, 1) if norm_train.V is not None else None,
        rngs=rngs,
    )
    training_result = train(
        posterior=posterior,
        likelihood=likelihood,
        dataset=dataset,
        config=config.optimization,
    )

    params_dict = {}
    st = nnx.state(training_result.posterior)
    for k, v in dict(st.flat_state()).items():
        val = getattr(v, "value", v)
        import numpy as np

        if hasattr(val, "ndim") and val.ndim > 0:
            params_dict[str(".".join(str(ki) for ki in k))] = np.asarray(val).tolist()
        else:
            try:
                params_dict[str(".".join(str(ki) for ki in k))] = float(val)
            except Exception as e:
                pass

    state = attrs.evolve(
        state,
        loss_history=training_result.loss_history,
        trained_params=params_dict,
        samples=training_result.samples,
    )

    # --- 6. Predict in real space ---
    pred_mean, pred_cov = predictInRealSpace(
        posterior=training_result.posterior,
        dataset_train=dataset,
        test_data=state.test_data,
        transform=transform,
        samples=training_result.samples,
    )

    pred_var = jnp.diag(pred_cov)
    state = attrs.evolve(state, pred_mean=pred_mean, pred_cov=pred_cov)

    # --- 7. Diagnostics ---
    metrics = computeDiagnosticMetrics(
        test_data_Y=state.test_data.Y,
        test_data_V=state.test_data.V,
        pred_mean=pred_mean,
        pred_var=pred_var,
        blind_mask=state.blind_mask,
    )
    logger.info(f"Diagnostic metrics: {metrics}")

    # --- 7.5 Posterior predictive checks ---
    from .diagnostics.posterior import posteriorPredictiveCheck
    from .diagnostics.plots import makePosteriorPredictivePlots

    likelihood_type = "gaussian"
    likelihood_name = config.model.likelihood.__class__.__name__.lower()
    if "poisson" in likelihood_name:
        likelihood_type = "poisson"

    ppc_results = posteriorPredictiveCheck(
        pred_mean=pred_mean,
        pred_cov=pred_cov,
        test_data=state.test_data,
        num_samples=200,
        likelihood=likelihood_type,
        blind_mask=state.blind_mask,
    )

    state = attrs.evolve(state, ppc_results=ppc_results)

    # --- 8. Save state ---
    # Save before plotting so that if plotting fails, we still have the results
    save(state, config.output_dir)
    logger.info(f"Pipeline computation complete. State saved to {config.output_dir}")

    # --- 9. Plot Diagnostics ---
    mplhep.style.use("CMS")

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

    plots = makeDiagnosticPlots(
        pred_mean=pred_mean,
        pred_var=pred_var,
        test_data=state.test_data,
        train_data=state.train_data,
        blind_mask=state.blind_mask,
        signal_data=signal_plot_data,
        signal_template=signal_template,
    )

    try:
        ppc_plots = makePosteriorPredictivePlots(
            ppc_results=ppc_results,
            test_data=state.test_data,
            blind_mask=state.blind_mask,
        )
        plots.update(ppc_plots)
    except Exception as e:
        logger.warning(f"Failed to create posterior predictive plots: {e}")

    plot_dir = config.output_dir / "diagnostics"
    savePlots(plots, plot_dir, formats=config.image_formats)

    logger.info(f"Pipeline complete. Plots saved to {plot_dir}")

    return state
