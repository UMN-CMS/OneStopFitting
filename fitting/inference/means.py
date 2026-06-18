from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import attrs
import gpjax
import jax
import jax.numpy as jnp
import jax.scipy.stats
from flax import nnx
from typing import Any

from ..data.loading import FileLoader, extractHistogram, histToBinnedData
from ..data.windowing import fitGaussianWindow

from .priors import PriorConfig

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class DeepMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    base_mean: gpjax.mean_functions.AbstractMeanFunction
    network: nnx.Module

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        xt = self.network(x)
        return self.base_mean(xt)


@attrs.define(slots=False)
class NeuralMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    network: nnx.Module

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.network(x).reshape(-1, 1)


class DoubleSidedCrystalBallMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int, init_mu: jnp.ndarray):
        super().__init__()

        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))
        self.mu = gpjax.parameters.Real(init_mu)
        self.log_sigma = gpjax.parameters.Real(jnp.full(ndim, -1.0))

        self.log_alpha_L = gpjax.parameters.Real(jnp.full(ndim, jnp.log(1.5)))
        self.log_n_L = gpjax.parameters.Real(jnp.full(ndim, 0.0))
        self.log_alpha_R = gpjax.parameters.Real(jnp.full(ndim, jnp.log(1.5)))
        self.log_n_R = gpjax.parameters.Real(jnp.full(ndim, 0.0))

        self.theta = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)

        delta = x - self.mu.value
        if x.shape[-1] == 2:
            c = jnp.cos(self.theta.value)
            s = jnp.sin(self.theta.value)
            rot_0 = delta[:, 0] * c - delta[:, 1] * s
            rot_1 = delta[:, 0] * s + delta[:, 1] * c
            z = jnp.stack([rot_0, rot_1], axis=-1) / sigma
        else:
            z = delta / sigma

        alpha_L = jnp.exp(self.log_alpha_L.value)
        n_L = jnp.exp(self.log_n_L.value) + 1.0
        alpha_R = jnp.exp(self.log_alpha_R.value)
        n_R = jnp.exp(self.log_n_R.value) + 1.0

        A_L = (n_L / alpha_L) ** n_L * jnp.exp(-0.5 * alpha_L**2)
        B_L = (n_L / alpha_L) - alpha_L
        safe_z_L = jnp.minimum(z, -alpha_L)
        tail_L = A_L * (B_L - safe_z_L) ** (-n_L)

        A_R = (n_R / alpha_R) ** n_R * jnp.exp(-0.5 * alpha_R**2)
        B_R = (n_R / alpha_R) - alpha_R
        safe_z_R = jnp.maximum(z, alpha_R)
        tail_R = A_R * (B_R + safe_z_R) ** (-n_R)

        core = jnp.exp(-0.5 * z**2)

        is_left = z < -alpha_L
        is_right = z > alpha_R

        cb_1d = jnp.where(is_left, tail_L, jnp.where(is_right, tail_R, core))

        cb_nd = jnp.prod(cb_1d, axis=-1)

        bump = self.amplitude.value * cb_nd
        return (bump + self.baseline.value).reshape(-1, 1)


class AsymmetricGaussianBumpMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.ndim = ndim
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_sigma = gpjax.parameters.Real(jnp.zeros(ndim))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

        n_ltri = ndim * (ndim + 1) // 2
        # One Cholesky factor per orthant: 2^ndim orthants
        n_orthants = 2**ndim
        self.L_raw = gpjax.parameters.Real(jnp.zeros((n_orthants, n_ltri)))

    def _cholesky_precision(self, L_raw_row: jnp.ndarray) -> jnp.ndarray:
        L = jnp.zeros((self.ndim, self.ndim))
        rows, cols = jnp.tril_indices(self.ndim)
        L = L.at[rows, cols].set(L_raw_row)
        diag_idx = jnp.arange(self.ndim)
        L = L.at[diag_idx, diag_idx].set(jnp.exp(jnp.diag(L)))
        return L

    def _orthant_index(self, delta: jnp.ndarray) -> jnp.ndarray:
        signs = (delta >= 0).astype(jnp.int32)  # (n, ndim)
        powers = 2 ** jnp.arange(self.ndim)  # (ndim,)
        return jnp.einsum("nd,d->n", signs, powers)  # (n,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)
        delta = (x - self.mu.value) / sigma  # (n, ndim)

        orthant_idx = self._orthant_index(delta)  # (n,)

        # Precompute all Cholesky factors
        all_L = jax.vmap(self._cholesky_precision)(
            self.L_raw.value
        )  # (2^ndim, ndim, ndim)

        L_per_point = all_L[orthant_idx]  # (n, ndim, ndim)
        Lt_delta = jnp.einsum(
            "nij,nj->ni", jnp.transpose(L_per_point, (0, 2, 1)), delta
        )
        exponent = -0.5 * jnp.sum(Lt_delta**2, axis=-1)

        bump = self.amplitude.value * jnp.exp(exponent)
        return (bump + self.baseline.value).reshape(-1, 1)


class GaussianBumpMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.array([0.3, 0.3]))
        self.log_sigma = gpjax.parameters.Real(jnp.array([0.3, 0.3]))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)
        z = (x - self.mu.value) / sigma
        exponent = -0.5 * jnp.sum(z**2, axis=-1)
        bump = self.amplitude.value * jnp.exp(exponent)
        return (bump + self.baseline.value).reshape(-1, 1)


class PolynomialBackgroundMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.w1 = gpjax.parameters.Real(jnp.zeros(ndim))
        self.w2 = gpjax.parameters.Real(jnp.zeros(ndim))

        self.b = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        linear_term = jnp.dot(x, self.w1.value)
        quad_term = jnp.dot(x**2, self.w2.value)
        baseline = linear_term + quad_term + self.b.value

        return baseline.reshape(-1, 1)


class ParametricBackgroundMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.w1 = gpjax.parameters.Real(jnp.zeros(ndim))
        self.w2 = gpjax.parameters.Real(jnp.zeros(ndim))
        self.b = gpjax.parameters.Real(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        linear_term = jnp.dot(x, self.w1.value)
        quad_term = jnp.dot(x**2, self.w2.value)

        exponent = linear_term + quad_term + self.b.value

        return jnp.exp(exponent).reshape(-1, 1)


class SkewedGaussianMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_sigma = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_alpha = gpjax.parameters.Real(jnp.zeros(ndim))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)
        alpha = jnp.exp(self.log_alpha.value)
        delta = (x - self.mu.value) / sigma  # (n, ndim)
        z = -0.5 * jnp.sum(delta**2, axis=-1)
        skew_term = jnp.prod(1 + jax.scipy.stats.norm.cdf(alpha * delta), axis=-1)
        bump = self.amplitude.value * jnp.exp(z) * skew_term
        return (bump + self.baseline.value).reshape(-1, 1)


class StudentTBumpMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_sigma = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_nu = gpjax.parameters.Real(jnp.array(2.0))  # log(nu), init at 2.0
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)
        nu = jnp.exp(self.log_nu.value)
        ndim = x.shape[-1]

        delta = (x - self.mu.value) / sigma
        z = jnp.sum(delta**2, axis=-1)
        exponent = -(nu + ndim) / 2.0
        scale_term = 1 + z / nu
        bump = self.amplitude.value * jnp.power(scale_term, exponent)

        return (bump + self.baseline.value).reshape(-1, 1)


class AsymmetricLaplaceMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.ndim = ndim
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_scale = gpjax.parameters.Real(jnp.zeros(ndim))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))
        n_ltri = ndim * (ndim + 1) // 2
        n_orthants = 2**ndim
        self.L_raw = gpjax.parameters.Real(jnp.zeros((n_orthants, n_ltri)))

    def _cholesky_transform(self, L_raw_row: jnp.ndarray) -> jnp.ndarray:
        L = jnp.zeros((self.ndim, self.ndim))
        rows, cols = jnp.tril_indices(self.ndim)
        L = L.at[rows, cols].set(L_raw_row)
        diag_idx = jnp.arange(self.ndim)
        L = L.at[diag_idx, diag_idx].set(jnp.exp(jnp.diag(L)))  # Positive diagonal
        return L

    def _orthant_index(self, delta: jnp.ndarray) -> jnp.ndarray:
        signs = (delta >= 0).astype(jnp.int32)
        powers = 2 ** jnp.arange(self.ndim)
        return jnp.einsum("nd,d->n", signs, powers)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.exp(self.log_scale.value)
        delta = (x - self.mu.value) / scale

        orthant_idx = self._orthant_index(delta)
        all_L = jax.vmap(self._cholesky_transform)(self.L_raw.value)
        L_per_point = all_L[orthant_idx]

        # L1 norm after linear transformation
        L_delta = jnp.einsum("nij,nj->ni", jnp.transpose(L_per_point, (0, 2, 1)), delta)
        l1_norm = jnp.sum(jnp.abs(L_delta), axis=-1)

        bump = self.amplitude.value * jnp.exp(-l1_norm)
        return (bump + self.baseline.value).reshape(-1, 1)


class RationalQuadraticBumpMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int):
        super().__init__()
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0))
        self.mu = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_sigma = gpjax.parameters.Real(jnp.zeros(ndim))
        self.log_alpha = gpjax.parameters.Real(
            jnp.array(1.0)
        )  # log(alpha), init at 1.0
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.exp(self.log_sigma.value)
        alpha = jnp.exp(self.log_alpha.value)

        delta = (x - self.mu.value) / sigma
        z = jnp.sum(delta**2, axis=-1)
        scale_term = 1 + z / (2 * alpha)
        bump = self.amplitude.value * jnp.power(scale_term, -alpha)

        return (bump + self.baseline.value).reshape(-1, 1)


class LogNormalWarpingMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(
        self,
        base_mean: gpjax.mean_functions.AbstractMeanFunction,
        log_dims: list[int] = None,
    ):
        super().__init__()
        self.base_mean = base_mean
        self.log_dims = log_dims
        self.offset = gpjax.parameters.Real(jnp.array(1.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_warped = x.copy()
        if self.log_dims is not None:
            for dim in self.log_dims:
                x_warped = x_warped.at[:, dim].set(
                    jnp.log(x_warped[:, dim] + self.offset.value)
                )

        return self.base_mean(x_warped)


class MixtureOfGaussiansMean(gpjax.mean_functions.AbstractMeanFunction):
    def __init__(self, ndim: int, n_components: int = 2):
        super().__init__()
        self.ndim = ndim
        self.n_components = n_components
        self.log_amplitudes = gpjax.parameters.Real(jnp.zeros(n_components))
        self.mus = gpjax.parameters.Real(jnp.zeros((n_components, ndim)))
        self.log_sigmas = gpjax.parameters.Real(jnp.zeros((n_components, ndim)))
        self.baseline = gpjax.parameters.Real(jnp.array(0.0))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        amplitudes = jax.nn.softmax(self.log_amplitudes.value)

        total_bump = jnp.zeros(x.shape[0])
        for k in range(self.n_components):
            mu_k = self.mus.value[k]
            sigma_k = jnp.exp(self.log_sigmas.value[k])
            delta = (x - mu_k) / sigma_k
            z = -0.5 * jnp.sum(delta**2, axis=-1)
            component_bump = amplitudes[k] * jnp.exp(z)
            total_bump += component_bump

        bump = total_bump
        return (bump + self.baseline.value).reshape(-1, 1)


@attrs.define
class MeanFunctionConfig(ABC):
    @abstractmethod
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction: ...


@attrs.define
class ZeroMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return gpjax.mean_functions.Zero()


@attrs.define
class ConstantMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return gpjax.mean_functions.Constant()


@attrs.define
class DeepMeanFunctionConfig(MeanFunctionConfig):
    base_mean: MeanFunctionConfig

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        network = kernel.network
        return DeepMeanFunction(
            self.base_mean.buildMeanFunction(ndim, kernel, **kwargs), network
        )


@attrs.define
class NeuralMeanConfig(MeanFunctionConfig):
    input_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [20, 20])
    activation: str = "silu"
    weight_prior: PriorConfig | None = None
    bias_prior: PriorConfig | None = None

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        from .kernels.nn import Network

        rngs = kwargs.get("rngs")
        if rngs is None:
            raise ValueError(
                "NeuralMeanConfig requires rngs for network initialization"
            )

        network = Network(
            rngs=rngs,
            input_dim=self.input_dim,
            output_dim=1,
            shape=self.hidden_shapes,
            activation_name=self.activation,
            weight_prior=self.weight_prior,
            bias_prior=self.bias_prior,
            output_residual=False,
        )
        return NeuralMeanFunction(network=network)


@attrs.define
class ParametricBackgroundMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return ParametricBackgroundMean(ndim)


@attrs.define
class PolynomialBackgroundMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return PolynomialBackgroundMean(ndim)


@attrs.define
class GaussianBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return GaussianBumpMean(ndim)


@attrs.define
class AsymmetricGaussianBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return GaussianBumpMean(ndim)


@attrs.define
class DoubleSidedCrystalBallMeanConfig(MeanFunctionConfig):
    init_mu: jnp.ndarray = jnp.array([0.3, 0.3])

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return DoubleSidedCrystalBallMean(ndim, self.init_mu)


@attrs.define
class SkewedGaussianMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return SkewedGaussianMean(ndim)


@attrs.define
class StudentTBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return StudentTBumpMean(ndim)


@attrs.define
class AsymmetricLaplaceMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return AsymmetricLaplaceMean(ndim)


@attrs.define
class RationalQuadraticBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return RationalQuadraticBumpMean(ndim)


@attrs.define
class LogNormalWarpingMeanConfig(MeanFunctionConfig):
    base_mean: MeanFunctionConfig = attrs.Factory(GaussianBumpMeanConfig)
    log_dims: list[int] = attrs.Factory(lambda: [0])

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        base_mean_func = self.base_mean.buildMeanFunction(ndim, kernel, **kwargs)
        return LogNormalWarpingMean(base_mean_func, self.log_dims)


@attrs.define
class MixtureOfGaussiansMeanConfig(MeanFunctionConfig):
    n_components: int = 2

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return MixtureOfGaussiansMean(ndim, self.n_components)


class InterpolatedMean(gpjax.mean_functions.AbstractMeanFunction):
    _train_dataset: gpjax.Dataset = nnx.data()

    def __init__(self, posterior, train_dataset):
        super().__init__()
        self._posterior = posterior
        self._train_dataset = train_dataset

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        latent = self._posterior.predict(x, train_data=self._train_dataset)
        return jax.lax.stop_gradient(latent.mean).reshape(-1, 1)


class LookupTableMean(gpjax.mean_functions.AbstractMeanFunction):
    _ref_X: jnp.ndarray = nnx.data()
    _ref_Y: jnp.ndarray = nnx.data()

    def __init__(self, reference_X: jnp.ndarray, reference_Y: jnp.ndarray):
        super().__init__()
        self._ref_X = jax.lax.stop_gradient(reference_X)
        self._ref_Y = jax.lax.stop_gradient(reference_Y)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Pairwise squared distances: (n_query, n_ref)
        diffs = x[:, None, :] - self._ref_X[None, :, :]
        dists = jnp.sum(diffs**2, axis=-1)
        nearest_idx = jnp.argmin(dists, axis=-1)
        return self._ref_Y[nearest_idx].reshape(-1, 1)


@attrs.define
class InterpolatedMeanConfig(MeanFunctionConfig):
    stage1_lengthscale: list[float] = attrs.Factory(lambda: [0.4, 0.4])
    stage1_variance: float = 2.0
    stage1_homoscedastic: bool = False

    def needsPreFit(self) -> bool:
        return True

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        raise RuntimeError(
            "InterpolatedMeanConfig requires a pre-fit step. "
            "Use buildStage1Mean() after fitting Stage 1."
        )

    def buildStage1Mean(
        self, posterior, train_dataset
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        """Build the frozen InterpolatedMean from a fitted Stage 1 posterior."""
        return InterpolatedMean(posterior, train_dataset)


@attrs.define
class LookupTableMeanConfig(MeanFunctionConfig):
    stage1_lengthscale: list[float] = attrs.Factory(lambda: [0.4, 0.4])
    stage1_variance: float = 2.0
    stage1_homoscedastic: bool = False

    def needsPreFit(self) -> bool:
        return True

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        raise RuntimeError(
            "LookupTableMeanConfig requires a pre-fit step. "
            "Use buildStage1Mean() after fitting Stage 1."
        )

    def buildStage1Mean(
        self, posterior, train_dataset
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        """Build the frozen LookupTableMean from a fitted Stage 1 posterior."""
        latent_dist = posterior.predict(train_dataset.X, train_data=train_dataset)
        ref_Y = jax.lax.stop_gradient(latent_dist.mean.ravel())
        return LookupTableMean(train_dataset.X, ref_Y)


class QCDMCMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    """
    Uses QCD MC prediction as the GP mean function.

    The GP then models residuals/corrections to the MC prediction.
    This anchors the extrapolation into the blinded window to the MC
    prediction, rather than to zero or an arbitrary parametric form.
    The GP only needs to learn corrections, which are typically small and smooth.
    """

    _mc_X: jnp.ndarray = nnx.data()
    _mc_Y: jnp.ndarray = nnx.data()

    def __init__(
        self,
        mc_X: jnp.ndarray,
        mc_Y: jnp.ndarray,
        learn_scale: bool = True,
        learn_tilt: bool = True,
        ndim: int = 2,
    ):
        super().__init__()
        self._mc_X = jax.lax.stop_gradient(mc_X)
        self._mc_Y = jax.lax.stop_gradient(mc_Y)
        self.log_scale = gpjax.parameters.Real(jnp.array(0.0))

        if learn_tilt:
            self.tilt = gpjax.parameters.Real(jnp.zeros(ndim))
        else:
            self.tilt = None

    def _interpolate_mc(self, x: jnp.ndarray) -> jnp.ndarray:
        diffs = x[:, None, :] - self._mc_X[None, :, :]  # (n, n_ref, ndim)
        dists = jnp.sum(diffs**2, axis=-1)  # (n, n_ref)
        nearest = jnp.argmin(dists, axis=-1)  # (n,)
        return self._mc_Y[nearest]  # (n,)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mc_pred = self._interpolate_mc(x)  # (n,)
        scale = jnp.exp(self.log_scale.value)

        if self.tilt is not None:
            tilt_correction = jnp.exp(x @ self.tilt.value)
            return (scale * tilt_correction * mc_pred).reshape(-1, 1)

        return (scale * mc_pred).reshape(-1, 1)


@attrs.define
class QCDMCMeanConfig(MeanFunctionConfig):
    mc_path: str = attrs.Factory(lambda: "")
    learn_scale: bool = True
    learn_tilt: bool = True

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        if not self.mc_path:
            raise ValueError("QCDMCMeanConfig requires a valid mc_path")

        domain_mask = kwargs.get("domain_mask", None)
        transform = kwargs.get("transform", None)

        loader = FileLoader.forPath(self.mc_path)
        raw_data = loader.load(self.mc_path)
        histogram = extractHistogram(raw_data)

        # Extract central variation and convert to BinnedData
        binned_data = histToBinnedData(histogram, variation="central")

        # Apply domain mask if provided
        domain_mask = kwargs.get("domain_mask", None)
        if domain_mask is not None:
            masked_data = binned_data.masked(domain_mask)
        else:
            masked_data = binned_data

        return QCDMCMeanFunction(
            mc_X=masked_data.X,
            mc_Y=masked_data.Y,
            learn_scale=self.learn_scale,
            learn_tilt=self.learn_tilt,
            ndim=ndim,
        )


class InterpolatedSignalMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    _signal_x: jnp.ndarray = nnx.data()
    _signal_y: jnp.ndarray = nnx.data()

    def __init__(self, signal_x: jnp.ndarray, signal_y: jnp.ndarray, prior: Any = None):
        super().__init__()
        self._signal_x = jax.lax.stop_gradient(signal_x)
        self._signal_y = jax.lax.stop_gradient(signal_y)
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0), prior=prior)

    def interpolateSignal(self, x: jnp.ndarray) -> jnp.ndarray:
        diffs = x[:, None, :] - self._signal_x[None, :, :]
        dists = jnp.sum(diffs**2, axis=-1)
        nearest = jnp.argmin(dists, axis=-1)
        return self._signal_y[nearest]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        signal_shape = self.interpolateSignal(x)
        return (self.amplitude.value * signal_shape).reshape(-1, 1)


class GaussianFitSignalMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    _shape_amplitude: jnp.ndarray = nnx.data()
    _center: jnp.ndarray = nnx.data()
    _sigma: jnp.ndarray = nnx.data()
    _theta: float = nnx.data()
    _normalization_scale: jnp.ndarray = nnx.data()

    def __init__(
        self,
        shape_amplitude: jnp.ndarray,
        center: jnp.ndarray,
        sigma: jnp.ndarray,
        theta: float,
        normalization_scale: jnp.ndarray,
        prior: Any = None,
    ):
        super().__init__()
        self._shape_amplitude = jax.lax.stop_gradient(shape_amplitude)
        self._center = jax.lax.stop_gradient(center)
        self._sigma = jax.lax.stop_gradient(sigma)
        self._theta = jax.lax.stop_gradient(theta)
        self._normalization_scale = jax.lax.stop_gradient(normalization_scale)
        self.amplitude = gpjax.parameters.Real(jnp.array(1.0), prior=prior)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_norm = x / self._normalization_scale
        ndim = x_norm.shape[-1] if x_norm.ndim > 1 else 1

        if ndim == 1:
            g = self._shape_amplitude * jnp.exp(
                -(((x_norm - self._center) / self._sigma) ** 2)
            )
            if g.ndim == 2 and g.shape[-1] == 1:
                g = g.squeeze(-1)
        else:
            x_, y_ = x_norm[..., 0], x_norm[..., 1]
            xo, yo = self._center[0], self._center[1]
            sx, sy = self._sigma[0], self._sigma[1]
            theta = self._theta

            a = jnp.cos(theta) ** 2 / (2 * sx**2) + jnp.sin(theta) ** 2 / (2 * sy**2)
            b = -jnp.sin(2 * theta) / (4 * sx**2) + jnp.sin(2 * theta) / (4 * sy**2)
            c = jnp.sin(theta) ** 2 / (2 * sx**2) + jnp.cos(theta) ** 2 / (2 * sy**2)
            g = self._shape_amplitude * jnp.exp(
                -(
                    a * (x_ - xo) ** 2
                    + 2 * b * (x_ - xo) * (y_ - yo)
                    + c * (y_ - yo) ** 2
                )
            )

        return (self.amplitude.value * g).reshape(-1, 1)


@attrs.define
class SignalTemplateMeanConfig(MeanFunctionConfig):
    use_gaussian_fit: bool = False
    amplitude_prior: PriorConfig | None = None

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        signal_data = kwargs.get("signal_data")
        if signal_data is None:
            raise ValueError(
                "SignalTemplateMeanConfig requires 'signal_data' provided in kwargs from the pipeline"
            )

        # Apply domain mask if provided
        domain_mask = kwargs.get("domain_mask", None)
        if domain_mask is not None:
            masked_data = signal_data.masked(domain_mask)
        else:
            masked_data = signal_data

        prior_dist = (
            self.amplitude_prior.buildPrior()
            if self.amplitude_prior is not None
            else None
        )

        if self.use_gaussian_fit:
            window = fitGaussianWindow(masked_data)
            return GaussianFitSignalMeanFunction(
                shape_amplitude=window.amplitude,
                center=window.center,
                sigma=window.sigma,
                theta=window.theta or 0.0,
                normalization_scale=window.normalization_scale,
                prior=prior_dist,
            )
        else:
            return InterpolatedSignalMeanFunction(
                signal_x=masked_data.X, signal_y=masked_data.Y, prior=prior_dist
            )


class MultiFidelityMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    """Autoregressive multi-fidelity mean: m(x) = ρ · μ_MC(x) [+ tilt].

    Wraps a frozen low-fidelity (MC) GP posterior. The posterior mean
    is computed with stop_gradient so the MC GP parameters are not
    updated when fitting the high-fidelity (data) GP.

    Args:
        mc_posterior: Frozen GP posterior fit to QCD MC data.
        mc_dataset: Training dataset used for the MC posterior.
        learn_scale: Whether ρ is a trainable parameter.
        learn_tilt: Whether to add a learnable linear tilt correction.
        ndim: Input dimensionality.
    """

    _mc_dataset: gpjax.Dataset = nnx.data()

    def __init__(
        self,
        mc_posterior,
        mc_dataset: gpjax.Dataset,
        learn_scale: bool = True,
        learn_tilt: bool = True,
        ndim: int = 2,
    ):
        super().__init__()
        self._mc_posterior = mc_posterior
        self._mc_dataset = mc_dataset

        if learn_scale:
            self.rho = gpjax.parameters.PositiveReal(jnp.array(0.1))
        else:
            self.rho = None

        if learn_tilt:
            self.tilt = gpjax.parameters.Real(jnp.zeros(ndim))
        else:
            self.tilt = None

    def _computeMcMean(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the frozen MC posterior mean at x."""
        latent = self._mc_posterior.predict(x, train_data=self._mc_dataset)
        return jax.lax.stop_gradient(latent.mean).ravel()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mc_mean = self._computeMcMean(x)

        if self.rho is not None:
            rho = self.rho[...]
        else:
            rho = 1.0

        result = rho * mc_mean

        if self.tilt is not None:
            tilt_correction = jnp.exp(x @ self.tilt.value)
            result = result * tilt_correction

        return result.reshape(-1, 1)


@attrs.define
class MultiFidelityMeanConfig(MeanFunctionConfig):
    learn_scale: bool = True
    learn_tilt: bool = False

    def needsPreFit(self) -> bool:
        return False

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel, **kwargs
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        mc_posterior = kwargs.get("mc_posterior")
        mc_dataset = kwargs.get("mc_dataset")
        if mc_posterior is None or mc_dataset is None:
            raise ValueError(
                "MultiFidelityMeanConfig requires 'mc_posterior' and "
                "'mc_dataset' provided via kwargs from MultiFidelityGPConfig."
            )
        return MultiFidelityMeanFunction(
            mc_posterior=mc_posterior,
            mc_dataset=mc_dataset,
            learn_scale=self.learn_scale,
            learn_tilt=self.learn_tilt,
            ndim=ndim,
        )
