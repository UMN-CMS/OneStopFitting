from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import attrs
import gpjax
import jax.numpy as jnp
from flax import nnx
import gpjax.kernels as gpk
import gpjax.parameters as gpp

from .kernels import KernelConfig, NNKernelConfig, MCEnsembleKernel
from .likelihoods import FixedGaussianNoiseConfig, LikelihoodConfig
from .means import (
    MeanFunctionConfig,
    DoubleSidedCrystalBallMeanConfig,
    ZeroMeanConfig,
    QCDMCMeanFunction,
)
from .priors import PriorConfig, SoftplusNormalPriorConfig, NormalPriorConfig

logger = logging.getLogger(__name__)


@attrs.define
class GPModelConfig(ABC):
    kernel: KernelConfig = attrs.Factory(NNKernelConfig)
    likelihood: LikelihoodConfig = attrs.Factory(FixedGaussianNoiseConfig)
    mean_function: MeanFunctionConfig = attrs.Factory(ZeroMeanConfig)

    @abstractmethod
    def buildModel(
        self,
        dataset: gpjax.Dataset,
        ndim: int,
        rngs: nnx.Rngs | None = None,
        obs_variance: jnp.ndarray | None = None,
        **kwargs,
    ) -> tuple[Any, Any, Any]: ...


@attrs.define
class ExactGPConfig(GPModelConfig):
    def buildModel(
        self,
        dataset: gpjax.Dataset,
        ndim: int,
        rngs: nnx.Rngs | None = None,
        obs_variance: jnp.ndarray | None = None,
        mean_function: MeanFunctionConfig | None = None,
        **kwargs,
    ) -> tuple[Any, Any, Any]:
        kernel = self.kernel.buildKernel(ndim, rngs=rngs, **kwargs)
        if mean_function is not None:
            mean_fn = mean_function
        else:
            mean_fn = self.mean_function.buildMeanFunction(ndim, kernel, **kwargs)

        likelihood_kwargs = {"num_datapoints": dataset.n}
        if obs_variance is not None:
            likelihood_kwargs["obs_variance"] = obs_variance

        likelihood = self.likelihood.buildLikelihood(**likelihood_kwargs)
        prior = gpjax.gps.Prior(mean_function=mean_fn, kernel=kernel)
        posterior = prior * likelihood

        logger.info(
            f"Built ExactGP: kernel={type(kernel).__name__}, "
            f"likelihood={type(likelihood).__name__}, "
            f"mean_function={type(mean_fn).__name__}, "
            f"n_train={dataset.n}"
        )

        return posterior, likelihood, prior


@attrs.define
class SparseGPConfig(GPModelConfig):
    num_inducing: int = 400

    def buildModel(
        self,
        dataset: gpjax.Dataset,
        ndim: int,
        rngs: nnx.Rngs | None = None,
        obs_variance: jnp.ndarray | None = None,
        mean_function: MeanFunctionConfig | None = None,
        **kwargs,
    ) -> tuple[Any, Any, Any]:
        kernel = self.kernel.buildKernel(ndim, rngs=rngs, **kwargs)
        if mean_function is not None:
            mean_fn = mean_function
        else:
            mean_fn = self.mean_function.buildMeanFunction(ndim, kernel, **kwargs)
        likelihood_kwargs = {"num_datapoints": dataset.n}
        if obs_variance is not None:
            likelihood_kwargs["obs_variance"] = obs_variance

        likelihood = self.likelihood.buildLikelihood(**likelihood_kwargs)

        prior = gpjax.gps.Prior(mean_function=mean_fn, kernel=kernel)
        posterior = prior * likelihood

        z = _selectInducingPoints(dataset, self.num_inducing)

        q = gpjax.variational_families.CollapsedVariationalGaussian(
            posterior=posterior,
            inducing_inputs=z,
        )

        logger.info(
            f"Built SparseGP (collapsed): kernel={type(kernel).__name__}, "
            f"n_inducing={len(z)}, n_train={dataset.n}"
        )

        return q, likelihood, prior


@attrs.define
class VariationalGPConfig(GPModelConfig):
    num_inducing: int = 50

    def buildModel(
        self,
        dataset: gpjax.Dataset,
        ndim: int,
        rngs: nnx.Rngs | None = None,
        obs_variance: jnp.ndarray | None = None,
        mean_function: MeanFunctionConfig | None = None,
        **kwargs,
    ) -> tuple[Any, Any, Any]:
        kernel = self.kernel.buildKernel(ndim, rngs=rngs, **kwargs)
        if mean_function is not None:
            mean_fn = mean_function
        else:
            mean_fn = self.mean_function.buildMeanFunction(ndim, kernel, **kwargs)
        likelihood_kwargs = {"num_datapoints": dataset.n}
        if obs_variance is not None:
            likelihood_kwargs["obs_variance"] = obs_variance

        likelihood = self.likelihood.buildLikelihood(**likelihood_kwargs)

        prior = gpjax.gps.Prior(mean_function=mean_fn, kernel=kernel)
        posterior = prior * likelihood

        z = _selectInducingPoints(dataset, self.num_inducing)

        q = gpjax.variational_families.VariationalGaussian(
            posterior=posterior,
            inducing_inputs=z,
        )

        logger.info(
            f"Built VariationalGP (uncollapsed): kernel={type(kernel).__name__}, "
            f"n_inducing={len(z)}, n_train={dataset.n}"
        )

        return q, likelihood, prior


def _selectInducingPoints(dataset: gpjax.Dataset, num_inducing: int) -> jnp.ndarray:
    n_train = dataset.n
    ndim = dataset.X.shape[1]

    if num_inducing >= n_train:
        return dataset.X

    x_min = jnp.min(dataset.X, axis=0)
    x_max = jnp.max(dataset.X, axis=0)

    if ndim == 1:
        z = jnp.linspace(float(x_min), float(x_max), num_inducing).reshape(-1, 1)
    else:
        points_per_dim = max(2, int(num_inducing ** (1.0 / ndim)))
        grids = [
            jnp.linspace(float(lo), float(hi), points_per_dim)
            for lo, hi in zip(x_min, x_max)
        ]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        z = jnp.stack([m.ravel() for m in mesh], axis=-1)
        if len(z) > num_inducing:
            step = max(1, len(z) // num_inducing)
            z = z[::step][:num_inducing]

    return z


@attrs.define
class QCDPriorGPConfig:
    """
    Full Bayesian GP using QCD MC as a robust physics prior.
    
    - Mean function: QCD MC prediction (with learnable scale + tilt)
    - Kernel: empirical MC ensemble covariance + stationary residual kernel
    - Prior on scale: log-normal centred on theory prediction (controlled by theory_scale_uncertainty)
    
    The posterior represents beliefs about the background after seeing data,
    given that QCD MC is the best physics prior.
    """

    residual_lengthscale: float = 0.3
    lengthscale_prior: PriorConfig = attrs.Factory(
        lambda: SoftplusNormalPriorConfig(loc=0.5, scale=0.2)
    )
    theory_scale_uncertainty: float = 0.2

    def buildModel(
        self,
        mc_X,
        mc_Y_nominal,
        mc_Y_ensemble,
        ndim: int = 2,
    ):
        mean_fn = QCDMCMeanFunction(
            mc_X,
            mc_Y_nominal,
            learn_scale=True,
            learn_tilt=True,
            ndim=ndim,
        )

        mc_kernel = MCEnsembleKernel(mc_X, mc_Y_ensemble, mc_Y_nominal)
        residual_kernel = gpk.Matern32(
            lengthscale=gpp.PositiveReal(
                jnp.array([self.residual_lengthscale] * ndim),
                prior=self.lengthscale_prior.buildPrior(),
            )
        )
        kernel = mc_kernel + residual_kernel

        scale_prior = NormalPriorConfig(loc=0.0, scale=self.theory_scale_uncertainty)
        mean_fn.log_scale = gpp.Real(
            jnp.array(0.0),
            prior=scale_prior.buildPrior(),
        )

        return gpjax.gps.Prior(mean_function=mean_fn, kernel=kernel)
