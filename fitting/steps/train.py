from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import attrs
import jax
import jax.numpy as jnp
from jax import random
import gpjax
import flax.nnx as nnx

from ..core.data import AnalysisState
from ..data.preprocessing import preprocess
from ..core.transforms import computeNormalization
from ..inference.optimization import train, setAtPath
from ..diagnostics.parameters import logKernelParameters, logLikelihoodParameters

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig

logger = logging.getLogger(__name__)


def _buildStage1Mean(
    mean_cfg,
    transform,
    test_data,
    ndim: int,
):
    """Build a Stage 1 fixed-lengthscale GP on the full (unblinded) data.

    Constructs an RBF GP with frozen hyperparameters, computes the
    posterior, and delegates to the mean config's ``buildStage1Mean()``
    to create the appropriate frozen mean function (InterpolatedMean
    or LookupTableMean).

    Args:
        mean_cfg: The mean function config (InterpolatedMeanConfig or
            LookupTableMeanConfig) carrying Stage 1 hyperparameters.
        transform: DataTransformation for normalizing the full dataset.
        test_data: Full (unblinded) BinnedData.
        ndim: Input dimensionality.

    Returns:
        A frozen gpjax mean function backed by the Stage 1 posterior.
    """
    import gpjax.kernels as gpk
    import gpjax.likelihoods as gpl

    # Normalize the full dataset
    norm_full = transform.applyToBinnedData(test_data)
    full_dataset = gpjax.Dataset(
        X=norm_full.X,
        y=norm_full.Y.reshape(-1, 1),
    )

    ls_val = mean_cfg.stage1_lengthscale

    logger.info(f"  Stage 1 lengthscale: {ls_val}")
    logger.info(f"  Stage 1 variance: {mean_cfg.stage1_variance}")

    stage1_kernel = gpk.RBF(
        lengthscale=ls_val,
        variance=mean_cfg.stage1_variance,
    )

    if mean_cfg.stage1_homoscedastic:
        stage1_likelihood = gpl.Gaussian(
            num_datapoints=full_dataset.n,
            obs_stddev=jnp.array(1.0),
        )
    else:
        if norm_full.V is not None:
            obs_var = norm_full.V.reshape(-1, 1)
            obs_var = jnp.clip(obs_var, a_min=jnp.min(obs_var[obs_var > 0]))
            stage1_likelihood = gpl.Gaussian(
                num_datapoints=full_dataset.n,
                obs_stddev=jnp.sqrt(obs_var),
            )
            stage1_likelihood.obs_stddev = nnx.Variable(
                stage1_likelihood.obs_stddev[...]
            )
        else:
            stage1_likelihood = gpl.Gaussian(
                num_datapoints=full_dataset.n,
                obs_stddev=jnp.array(1.0),
            )

    parameter_filter = (gpjax.parameters.Parameter, nnx.Param)
    for path, node in nnx.graph.iter_graph(stage1_kernel):
        if isinstance(node, parameter_filter):
            setAtPath(stage1_kernel, path, nnx.Variable(node.value))

    stage1_prior = gpjax.gps.Prior(
        mean_function=gpjax.mean_functions.Zero(),
        kernel=stage1_kernel,
    )
    stage1_posterior = stage1_prior * stage1_likelihood

    logger.info(
        f"  Stage 1 GP built: {full_dataset.n} training points, lengthscale={ls_val}"
    )

    return mean_cfg.buildStage1Mean(stage1_posterior, full_dataset)


def trainModel(state: AnalysisState, rng_key: jax.Array) -> AnalysisState:
    if state.background is None:
        raise ValueError("Background data not found in state. Cannot train model.")
    state = preprocess(state, min_counts=state.config.min_counts)
    transform = computeNormalization(state.train_data, config=state.config.transform)
    norm_train = transform.applyToBinnedData(state.train_data)

    state = attrs.evolve(state, transform=transform)
    mean_cfg = state.config.model.mean_function
    pre_fit_mean = None

    if hasattr(mean_cfg, "needsPreFit") and mean_cfg.needsPreFit():
        logger.info("=== Stage 1: Background GP Mean Estimation ===")
        pre_fit_mean = _buildStage1Mean(
            mean_cfg=mean_cfg,
            transform=transform,
            test_data=state.test_data,
            ndim=state.background.ndim,
        )

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
        mean_function=pre_fit_mean,
        domain_mask=state.domain_mask,
        signal_data=transform.applyToBinnedData(state.signal)
        if state.signal is not None
        else None,
        transform=transform,
    )

    logger.info(f"  Training on {dataset.n} data points")
    logger.info(f"  posterior_type={type(posterior).__name__}")

    training_result = train(
        posterior=posterior,
        likelihood=likelihood,
        dataset=dataset,
        config=state.config.optimization,
        rng_key=train_key,
    )

    state = attrs.evolve(state, training_result=training_result, dataset=dataset)

    logger.info("Trained Hyperparameters:")
    logKernelParameters(training_result.posterior)
    logLikelihoodParameters(training_result.likelihood)
    logger.info(f"Final Loss: {training_result.final_loss}")

    return state
