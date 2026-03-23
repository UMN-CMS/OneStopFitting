from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import attrs
import gpjax
import jax
import jax.numpy as jnp
import jax.scipy.stats
from flax import nnx

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class DeepMeanFunction(gpjax.mean_functions.AbstractMeanFunction):
    base_mean: gpjax.mean_functions.AbstractMeanFunction
    network: nnx.Module

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        xt = self.network(x)
        return self.base_mean(xt)


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
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction: ...


@attrs.define
class ZeroMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return gpjax.mean_functions.Zero()


@attrs.define
class ConstantMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return gpjax.mean_functions.Constant()


@attrs.define
class DeepMeanFunctionConfig(MeanFunctionConfig):
    base_mean: MeanFunctionConfig

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        network = kernel.network
        return DeepMeanFunction(self.base_mean.buildMeanFunction(ndim, kernel), network)


@attrs.define
class ParametricBackgroundMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return ParametricBackgroundMean(ndim)


@attrs.define
class PolynomialBackgroundMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return PolynomialBackgroundMean(ndim)


@attrs.define
class GaussianBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return GaussianBumpMean(ndim)


@attrs.define
class AsymmetricGaussianBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return GaussianBumpMean(ndim)


@attrs.define
class DoubleSidedCrystalBallMeanConfig(MeanFunctionConfig):
    init_mu: jnp.ndarray = jnp.array([0.3, 0.3])

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return DoubleSidedCrystalBallMean(ndim, self.init_mu)


@attrs.define
class SkewedGaussianMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return SkewedGaussianMean(ndim)


@attrs.define
class StudentTBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return StudentTBumpMean(ndim)


@attrs.define
class AsymmetricLaplaceMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return AsymmetricLaplaceMean(ndim)


@attrs.define
class RationalQuadraticBumpMeanConfig(MeanFunctionConfig):
    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return RationalQuadraticBumpMean(ndim)


@attrs.define
class LogNormalWarpingMeanConfig(MeanFunctionConfig):
    base_mean: MeanFunctionConfig = attrs.Factory(GaussianBumpMeanConfig)
    log_dims: list[int] = attrs.Factory(lambda: [0])

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        base_mean_func = self.base_mean.buildMeanFunction(ndim, kernel)
        return LogNormalWarpingMean(base_mean_func, self.log_dims)


@attrs.define
class MixtureOfGaussiansMeanConfig(MeanFunctionConfig):
    n_components: int = 2

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
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
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
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
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
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
