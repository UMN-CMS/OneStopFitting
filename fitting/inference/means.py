from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import attrs
import gpjax
import jax.numpy as jnp
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
class DoubleSidedCrystalBallMeanConfig(MeanFunctionConfig):
    init_mu: jnp.ndarray = jnp.array([0.3, 0.3])

    def buildMeanFunction(
        self, ndim: int, kernel: gpjax.kernels.AbstractKernel
    ) -> gpjax.mean_functions.AbstractMeanFunction:
        return DoubleSidedCrystalBallMean(ndim, self.init_mu)
