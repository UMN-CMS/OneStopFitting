"""Pipeline configuration and execution.

Defines PipelineConfig and runPipeline — the top-level orchestration
of the background estimation workflow.
"""

from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Any

import attrs
import jax
import jax.numpy as jnp
from jax import random
import gpjax
import flax.nnx as nnx
import mplhep
from .data.windowing import Window

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
from .inference.prediction import (
    predictInRealSpace,
    getPriorMeanInRealSpace,
    drawPoissonSamples,
)
from .data.loading import (
    FileLoader,
    extractHistogram,
    extractMetadata,
    histToBinnedData,
)
from .diagnostics.posterior import posteriorPredictiveCheck

logger = logging.getLogger(__name__)

mplhep.style.use("CMS")


@attrs.define
class PipelineConfig:
    background_path: Path
    signal_path: Path | None = None
    signal_selection: str | None = None
    injection_rate: float = 0.0
    rebin: int = 1
    min_counts: float = 0.0
    rng_seed: int = 0xBEEFBEEF
    domain_window: Window | None = None
    window_spread: float = 2.0
    transform: TransformConfig = attrs.Factory(StandardizationConfig)
    model: GPModelConfig = attrs.Factory(ExactGPConfig)
    optimization: OptimizationConfig = attrs.Factory(OptimizationConfig)
    output_dir_format: str = "output"
    image_formats: list[str] = attrs.Factory(lambda: ["png"])
    metadata: dict[str, Any] = attrs.Factory(dict)


def loadData(config: PipelineConfig) -> AnalysisState:
    """Load background and optional signal data."""

    logger.info(f"Loading background from {config.background_path}")
    loader = FileLoader.forPath(config.background_path)
    bkg_raw = loader.load(config.background_path)
    bkg_hist = extractHistogram(bkg_raw)
    background = histToBinnedData(bkg_hist, rebin=config.rebin, variation="central")

    # Extract and merge file-level metadata
    file_metadata = extractMetadata(bkg_raw)
    combined_metadata = {
        **config.metadata,
        **file_metadata,
        "injection_rate": config.injection_rate,
    }

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

    return AnalysisState(
        config=config,
        background=background,
        signal=signal,
        injection_rate=config.injection_rate,
        background_hist=bkg_hist,
        signal_hist=signal_hist,
        metadata=combined_metadata,
    )


def trainModel(state: AnalysisState, rng_key: jax.Array) -> AnalysisState:
    """Preprocess, normalize, and train the GP model."""

    # 1. Preprocess (masking, blinding)
    state = preprocess(state, min_counts=state.config.min_counts)

    # 2. Normalize
    transform = computeNormalization(state.train_data, config=state.config.transform)
    norm_train = transform.applyToBinnedData(state.train_data)
    state = attrs.evolve(state, transform=transform)

    # 3. Train GP
    dataset = gpjax.Dataset(
        X=norm_train.X,
        y=norm_train.Y.reshape(-1, 1),
    )

    build_key, train_key = random.split(rng_key)
    posterior, likelihood, prior = state.config.model.buildModel(
        dataset=dataset,
        ndim=state.background.ndim,
        obs_variance=norm_train.V.reshape(-1, 1) if norm_train.V is not None else None,
        rngs=nnx.Rngs(build_key),
    )

    training_result = train(
        posterior=posterior,
        likelihood=likelihood,
        dataset=dataset,
        config=state.config.optimization,
        rng_key=train_key,
    )

    state = attrs.evolve(state, training_result=training_result, dataset=dataset)

    # Log specific parameters for visibility
    logger.info("Trained Hyperparameters:")
    # (Note: we could extract this to a helper if it gets complex again)
    # For now, keeping it simple as per user's latest manual change
    logger.info(f"Final Loss: {training_result.final_loss}")

    return state


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


def generatePlots(state: AnalysisState, rng_key: jax.Array) -> None:
    """Generate and save diagnostic plots."""
    from .diagnostics.plots import makePosteriorPredictivePlots

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
    plots = makeDiagnosticPlots(
        pred_mean=state.pred_mean,
        pred_var=pred_var,
        test_data=state.test_data,
        train_data=state.train_data,
        blind_mask=state.blind_mask,
        signal_data=signal_plot_data,
        signal_template=signal_template,
        prior_mean=prior_mean,
        kernel=state.training_result.posterior.prior.kernel,
        transform=state.transform,
        pred_cov=state.pred_cov,
    )

    if state.ppc_results is not None:
        try:
            ppc_plots = makePosteriorPredictivePlots(
                ppc_results=state.ppc_results,
                test_data=state.test_data,
                blind_mask=state.blind_mask,
            )
            plots.update(ppc_plots)
        except Exception as e:
            logger.warning(f"Failed to create posterior predictive plots: {e}")

    plot_dir = state.getRealOutPath() / "diagnostics"
    savePlots(plots, plot_dir, formats=state.config.image_formats)
    logger.info(f"Pipeline complete. Plots saved to {plot_dir}")


class PipelineStep(IntEnum):
    LOAD = 0
    TRAIN = 1
    DIAGNOSTICS = 2
    PLOT = 3
    COMBINE = 4

    @classmethod
    def fromStr(cls, s: str | None) -> PipelineStep | None:
        if s is None:
            return None
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid step: {s}. Valid steps are: {[step.name.lower() for step in cls]}"
            )


def prepareCombine(state: AnalysisState, rng_key: jax.Array) -> None:
    from .combine.histograms import exportCombineData
    from .combine.datacard import Process, Channel, Systematic, DataCard
    from .combine.histograms import normalizeVarName
    from .data.loading import variationNames

    out_dir = state.getRealOutPath() / "combine"
    shapes_file = "shapes.root"
    shapes_path = out_dir / shapes_file
    datacard_path = out_dir / "datacard.txt"

    logger.info(f"Preparing Combine inputs in {out_dir}")

    n_eigen = exportCombineData(state=state, output_path=shapes_path)

    channels = []

    def doMask(x):
        return x[state.blind_mask]

    ch_name = state.metadata.get("channel", "ch1")
    observation = float(jnp.sum(doMask(state.test_data.Y)))
    processes = []
    bg_rate = float(jnp.sum(doMask(state.pred_mean)))
    processes.append(Process(name="background", rate=bg_rate, index=1))
    if state.signal is not None:
        sig_name = "signal"
        sig_rate = float(jnp.sum(doMask(state.signal.Y[state.domain_mask])))
        processes.append(Process(name=sig_name, rate=sig_rate, index=0))

    channels.append(
        Channel(
            name=ch_name,
            observation=observation,
            processes=processes,
            shapes_file=shapes_file,
        )
    )

    systematics = []
    for i in range(n_eigen):
        syst_values = {"background": "1"}
        systematics.append(
            Systematic(
                name=f"gpr_eigen{i}",
                distribution="shape",
                values=syst_values,
            )
        )

    if state.signal_hist is not None:
        sig_name = "signal"
        all_vars = variationNames(state.signal_hist)
        sig_systs = set()
        for v in all_vars:
            if v == "central" or v.endswith("_disabled"):
                continue

            base, direction = normalizeVarName(v)
            sig_systs.add(base)

        for syst_name in sorted(list(sig_systs)):
            systematics.append(
                Systematic(
                    name=syst_name,
                    distribution="shape",
                    values={sig_name: "1"},
                )
            )
    systematics.append(
        Systematic(
            name="lumi",
            distribution="lnN",
            values={p.name: "1.02" for p in processes if p.name != "background"},
        )
    )

    card = DataCard(channels=channels, systematics=systematics)
    card.write(datacard_path)

    # Combine Diagnostics
    from .diagnostics.combine import plotCombineInputs, verifyEigenvariations

    diag_dir = state.getRealOutPath() / "diagnostics" / "combine"
    plotCombineInputs(state, diag_dir)
    verifyEigenvariations(state, diag_dir)

    logger.info(f"Combine preparation complete. Datacard: {datacard_path}")


STEP_FUNCS = {
    PipelineStep.LOAD: loadData,
    PipelineStep.TRAIN: trainModel,
    PipelineStep.DIAGNOSTICS: runDiagnostics,
    PipelineStep.PLOT: generatePlots,
    PipelineStep.COMBINE: prepareCombine,
}


def runPipeline(
    config: PipelineConfig,
    single_step: PipelineStep | None = None,
    start_from: PipelineStep = PipelineStep.LOAD,
) -> AnalysisState:
    from .core.serialization import load

    jax.config.update("jax_enable_x64", True)
    rng_key = random.key(config.rng_seed)

    if single_step:
        to_run = [single_step]
    else:
        to_run = [s for s in PipelineStep if s >= start_from]

    if PipelineStep.LOAD in to_run:
        state = loadData(config)
        logger.info(f"Output directory: {state.getRealOutPath()}")
        save(state, state.getRealOutPath())
    else:
        dummy_state = loadData(config)
        out_path = dummy_state.getRealOutPath()
        logger.info(f"Resuming from state at {out_path}")
        state = load(out_path)
        state = attrs.evolve(state, config=config)

    for s in to_run:
        if s == PipelineStep.LOAD:
            continue
        func = STEP_FUNCS[s]
        rng_key, key = random.split(rng_key)
        if s in [PipelineStep.TRAIN, PipelineStep.DIAGNOSTICS]:
            state = func(state, key)
            save(state, state.getRealOutPath())
        elif s == PipelineStep.PLOT:
            func(state, key)
        else:
            func(state, key)

    return state


def generateSmoothedBackground(
    state: AnalysisState,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> tuple[list[BinnedData], dict[str, Any]]:
    import hist
    import numpy as np

    if state.training_result is None or state.dataset is None:
        raise ValueError("Cannot generate smoothed background without training.")

    pred_key, sample_key = random.split(rng_key)

    # 1. Get full latent distribution (REAL SPACE)
    # We use all bins in test_data
    pred_mean, pred_cov = predictInRealSpace(
        posterior=state.training_result.posterior,
        dataset_train=state.dataset,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=pred_key,
    )

    samples = drawPoissonSamples(
        rng_key=sample_key,
        mean=pred_mean,
        cov=pred_cov,
        num_samples=num_samples,
    )

    smoothed_hists = []
    shape = tuple(len(e) - 1 for e in state.test_data.edges)
    for i in range(num_samples):
        axes = [
            hist.axis.Variable(np.asarray(e), name=n)
            for e, n in zip(state.test_data.edges, state.test_data.axis_names)
        ]
        h = hist.Hist(*axes)

        # Reconstruct the full grid using the domain mask
        if state.domain_mask is not None:
            full_values = np.zeros(len(state.domain_mask))
            full_values[np.asarray(state.domain_mask)] = np.asarray(samples[i])
            full_vars = np.zeros(len(state.domain_mask))
            full_vars[np.asarray(state.domain_mask)] = np.asarray(samples[i])
        else:
            full_values = np.asarray(samples[i])
            full_vars = np.asarray(samples[i])

        h.values()[:] = full_values.reshape(shape)
        h.variances()[:] = full_vars.reshape(shape)
        smoothed_hists.append(h)

    # 4. Generate diagnostic plots (Comparison with original)
    from .diagnostics.plots import makeSmoothingPlots

    # Use BinnedData for plotting
    smoothed_binned = BinnedData(
        X=state.test_data.X,
        Y=samples[0],
        V=samples[0],
        edges=state.test_data.edges,
        axis_names=state.test_data.axis_names,
    )

    plots = makeSmoothingPlots(
        smoothed_data=smoothed_binned,
        original_data=state.test_data,
        pred_mean=pred_mean,
        pred_cov=pred_cov,
    )

    return smoothed_hists, plots
