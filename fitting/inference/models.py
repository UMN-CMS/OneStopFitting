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

from .kernels import (
    KernelConfig,
    NNKernelConfig,
    MCEnsembleKernel,
    RBFConfig,
    Matern32Config,
    MultiFidelityResidualKernelConfig,
)
from .likelihoods import FixedGaussianNoiseConfig, LikelihoodConfig
from .means import (
    MeanFunctionConfig,
    ZeroMeanConfig,
    QCDMCMeanFunction,
    MultiFidelityMeanFunction,
)
from .priors import PriorConfig, SoftplusNormalPriorConfig, NormalPriorConfig
from ..data.loading import FileLoader, extractHistogram, histToBinnedData
from .optimization import setAtPath, runMLE, OptimizationConfig
import jax

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
        kernel = self.kernel.buildKernel(ndim, rngs=rngs, dataset=dataset, **kwargs)
        if mean_function is not None:
            mean_fn = mean_function.buildMeanFunction(ndim, kernel, **kwargs)
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
            mean_fn = mean_function.buildMeanFunction(ndim, kernel, **kwargs)
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
    num_inducing: int = 500

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
            mean_fn = mean_function.buildMeanFunction(ndim, kernel, **kwargs)
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


@attrs.define
class MultiFidelityGPConfig(GPModelConfig):
    """Multi-fidelity GP: linear autoregressive model combining QCD MC and data, see https://arxiv.org/abs/2006.16728.
    Model is fit in a recursive fashion.
    Note requires  D_{n-1} is a superset of D_n

        f_data(x) = rho * f_MC(x) + delta(x)

    Stage 1: Fit a GP to QCD MC data (low-fidelity).
    Stage 2: Fit a GP to observed data using the frozen MC GP posterior
             as the mean function (high-fidelity).


    TODO:
    Also implement linear model coregionalization approach and non-linear autoregression.
    """

    mc_path: str = attrs.Factory(lambda: "")
    mc_kernel: KernelConfig = attrs.Factory(lambda: RBFConfig(ard=True))
    mc_likelihood: LikelihoodConfig = attrs.Factory(FixedGaussianNoiseConfig)
    mc_num_iters: int = 150
    mc_lr: float = 0.01

    residual_kernel: KernelConfig = attrs.Factory(lambda: NNKernelConfig())

    learn_rho: bool = True
    learn_tilt: bool = False
    propagate_mc_variance: bool = True
    rho_prior: PriorConfig | None = None

    def buildModel(
        self,
        dataset: gpjax.Dataset,
        ndim: int,
        rngs: nnx.Rngs | None = None,
        obs_variance: jnp.ndarray | None = None,
        mean_function: MeanFunctionConfig | None = None,
        **kwargs,
    ) -> tuple[Any, Any, Any]:
        if not self.mc_path:
            raise ValueError("MultiFidelityGPConfig requires a valid mc_path")

        domain_mask = kwargs.get("domain_mask", None)
        transform = kwargs.get("transform", None)

        logger.info("=== Multi-Fidelity Stage 1: Preparation & Fit ===")
        mc_dataset, mc_norm_V = self._prepareMcDataset(
            ndim, domain_mask, transform, dataset.y
        )

        mc_posterior = self._fitMcGp(mc_dataset, mc_norm_V, ndim, rngs)

        frozen_mc_posterior = mc_posterior
        parameter_filter = (gpjax.parameters.Parameter, nnx.Param)
        for path, node in nnx.graph.iter_graph(frozen_mc_posterior):
            if isinstance(node, parameter_filter):
                setAtPath(frozen_mc_posterior, path, nnx.Variable(node.value))

        logger.info("=== Multi-Fidelity Stage 2: Building Data GP ===")

        mf_mean = MultiFidelityMeanFunction(
            mc_posterior=frozen_mc_posterior,
            mc_dataset=mc_dataset,
            learn_scale=self.learn_rho,
            learn_tilt=self.learn_tilt,
            ndim=ndim,
        )

        if self.rho_prior is not None and mf_mean.rho is not None:
            mf_mean.rho = gpp.PositiveReal(
                jnp.array(0.1), prior=self.rho_prior.buildPrior()
            )
        kernel_kwargs = dict(**kwargs)
        if self.propagate_mc_variance and mf_mean.rho is not None:
            x_mc = mc_dataset.X
            k_mc = frozen_mc_posterior.prior.kernel
            Kxx = k_mc.gram(x_mc).to_dense()
            obs_noise = jnp.square(frozen_mc_posterior.likelihood.obs_stddev[...])
            jitter = getattr(frozen_mc_posterior, "jitter", 1e-6)
            Sigma = Kxx + jnp.eye(Kxx.shape[0]) * (obs_noise + jitter)
            mc_L = jnp.linalg.cholesky(Sigma)

            kernel_kwargs["mc_kernel"] = k_mc
            kernel_kwargs["mc_L"] = mc_L
            kernel_kwargs["mc_dataset"] = mc_dataset
            kernel_kwargs["rho"] = mf_mean.rho

        residual_config = MultiFidelityResidualKernelConfig(
            residual_kernel=self.residual_kernel,
            propagate_mc_variance=self.propagate_mc_variance,
        )
        data_kernel = residual_config.buildKernel(ndim, rngs=rngs, **kernel_kwargs)

        likelihood_kwargs = {"num_datapoints": dataset.n}
        if obs_variance is not None:
            likelihood_kwargs["obs_variance"] = obs_variance
        data_likelihood = self.likelihood.buildLikelihood(**likelihood_kwargs)

        data_prior = gpjax.gps.Prior(mean_function=mf_mean, kernel=data_kernel)
        data_posterior = data_prior * data_likelihood

        logger.info(
            f"  Built MultiFidelityGP: "
            f"residual_kernel={type(data_kernel).__name__}, "
            f"learn_rho={self.learn_rho}, "
            f"propagate_mc_var={self.propagate_mc_variance}, "
            f"n_train={dataset.n}"
        )

        return data_posterior, data_likelihood, data_prior

    def _prepareMcDataset(
        self,
        ndim: int,
        domain_mask: jnp.ndarray | None,
        transform: Any | None,
        target_y_norm: jnp.ndarray,
    ) -> tuple[gpjax.Dataset, jnp.ndarray | None]:
        loader = FileLoader.forPath(self.mc_path)
        raw_data = loader.load(self.mc_path)
        histogram = extractHistogram(raw_data)
        mc_binned = histToBinnedData(histogram, variation="central")

        if domain_mask is not None:
            mc_binned = mc_binned.masked(domain_mask)

        if transform is not None:
            raw_data_y = transform.invertY(target_y_norm)
            total_data_yield = jnp.sum(raw_data_y)
            total_mc_yield = jnp.sum(mc_binned.Y)

            scale_factor = total_data_yield / total_mc_yield
            mc_binned.Y = mc_binned.Y * scale_factor
            if mc_binned.V is not None:
                mc_binned.V = mc_binned.V * (scale_factor**2)

            logger.info(
                f"  Scaling MC yield: total_data={total_data_yield:.1f}, "
                f"total_mc={total_mc_yield:.1f}, factor={scale_factor:.4f}"
            )

            mc_norm = transform.applyToBinnedData(mc_binned)
        else:
            mc_norm = mc_binned
            logger.warning("No transform provided; skipping MC scaling/normalization")

        mc_dataset = gpjax.Dataset(
            X=mc_norm.X,
            y=mc_norm.Y.reshape(-1, 1),
        )
        mc_norm_V = mc_norm.V.reshape(-1, 1) if mc_norm.V is not None else None

        return mc_dataset, mc_norm_V

    def _fitMcGp(
        self,
        mc_dataset: gpjax.Dataset,
        mc_norm_V: jnp.ndarray | None,
        ndim: int,
        rngs: nnx.Rngs ,
    ) -> Any:
        mc_stage_kernel = self.mc_kernel.buildKernel(ndim, rngs=rngs)

        mc_likelihood_kwargs = {"num_datapoints": mc_dataset.n}
        if mc_norm_V is not None:
            mc_likelihood_kwargs["obs_variance"] = mc_norm_V

        mc_stage_likelihood = self.mc_likelihood.buildLikelihood(**mc_likelihood_kwargs)

        mc_prior = gpjax.gps.Prior(
            mean_function=gpjax.mean_functions.Zero(),
            kernel=mc_stage_kernel,
        )
        mc_posterior = mc_prior * mc_stage_likelihood
        mc_opt_config = OptimizationConfig(num_iters=self.mc_num_iters, lr=self.mc_lr)
        mc_result = runMLE(
            mc_posterior,
            mc_stage_likelihood,
            mc_dataset,
            config=mc_opt_config,
            rng_key=rngs(),
        )

        logger.info(f"  MC GP trained: final_loss={mc_result.final_loss:.4f}")
        return mc_result.posterior
