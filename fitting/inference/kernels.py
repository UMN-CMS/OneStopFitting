"""Kernel configuration hierarchy.

Each KernelConfig subclass wraps a gpjax kernel type. The buildKernel
method constructs the actual gpjax kernel instance. cattrs
include_subclasses handles polymorphic serialization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import attrs
from flax import nnx
import jax
import gpjax.kernels as gpk
import gpjax.parameters as gpp
from typing import Any

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.base import AbstractKernel
from .priors import PriorConfig


@attrs.define
class KernelConfig(ABC):
    """Base kernel configuration.

    Subclasses represent specific kernel types.
    """

    @abstractmethod
    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        """Construct the gpjax kernel.

        Args:
            ndim: Input dimensionality.
            rngs: Flax NNX RNG container.

        Returns:
            A gpjax kernel instance.
        """
        ...

    def _get_param(
        self, 
        name: str, 
        value: Any, 
        prior_config: PriorConfig | None = None,
        param_type: type[gpp.Parameter] = gpp.PositiveReal
    ) -> Any:
        """Wrap a parameter value with a prior if present."""
        if prior_config is not None:
            return param_type(value, prior=prior_config.buildPrior())
        return value


@attrs.define
class RBFConfig(KernelConfig):
    """Radial Basis Function (squared exponential) kernel."""

    active_dims: list[int] | None = None
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        
        return gpk.RBF(
            lengthscale=lengthscale, 
            variance=variance, 
            active_dims=self.active_dims
        )


@attrs.define
class Matern12Config(KernelConfig):
    active_dims: list[int] | None = None
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern12(
            lengthscale=lengthscale, 
            variance=variance,
            active_dims=self.active_dims
        )


@attrs.define
class Matern32Config(KernelConfig):
    active_dims: list[int] | None = None
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern32(
            lengthscale=lengthscale, 
            variance=variance,
            active_dims=self.active_dims
        )


@attrs.define
class Matern52Config(KernelConfig):
    active_dims: list[int] | None = None
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern52(
            lengthscale=lengthscale, 
            variance=variance,
            active_dims=self.active_dims
        )


@attrs.define
class RationalQuadraticConfig(KernelConfig):
    """Rational Quadratic kernel."""

    active_dims: list[int] | None = None
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None
    alpha_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        alpha = self._get_param("alpha", 1.0, self.alpha_prior)
        return gpk.RationalQuadratic(
            lengthscale=lengthscale, 
            variance=variance,
            alpha=alpha,
            active_dims=self.active_dims
        )


@attrs.define
class PeriodicConfig(KernelConfig):
    """Periodic kernel."""

    active_dims: list[int] | None = None
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None
    period_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        lengthscale = self._get_param("lengthscale", 1.0, self.lengthscale_prior)
        variance = self._get_param("variance", 1.0, self.variance_prior)
        period = self._get_param("period", 1.0, self.period_prior)
        return gpk.Periodic(
            lengthscale=lengthscale,
            variance=variance,
            period=period,
            active_dims=self.active_dims
        )


@attrs.define
class WhiteConfig(KernelConfig):
    """White noise kernel."""
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        return gpk.White(variance=variance)


@attrs.define
class LinearConfig(KernelConfig):
    """Linear kernel."""

    active_dims: list[int] | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        return gpk.Linear(variance=variance, active_dims=self.active_dims)


@attrs.define
class PolynomialConfig(KernelConfig):
    """Polynomial kernel."""

    degree: int = 2
    active_dims: list[int] | None = None
    variance_prior: PriorConfig | None = None
    shift_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        shift = self._get_param("shift", 1.0, self.shift_prior)
        return gpk.Polynomial(
            degree=self.degree, 
            variance=variance,
            shift=shift,
            active_dims=self.active_dims
        )


@attrs.define
class MultiScaleKernelConfig(KernelConfig):
    ard: bool = True
    # For composite kernels, we might need a way to pass priors to children
    # but for common cases we can just use defaults or manual composition.

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        len_global = [1.0] * ndim if self.ard else 1.0
        k_global = gpk.Matern52(lengthscale=len_global)

        len_local = [0.1] * ndim if self.ard else 0.1
        k_local = gpk.Matern52(lengthscale=len_local)

        return k_global + k_local


@attrs.define
class SumKernelConfig(KernelConfig):
    """Sum of kernels: k1 + k2 + ..."""

    components: list[KernelConfig] = attrs.Factory(list)

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        if not self.components:
            raise ValueError("SumKernelConfig requires at least one component")
        kernels = [c.buildKernel(ndim, rngs=rngs) for c in self.components]
        result = kernels[0]
        for k in kernels[1:]:
            result = result + k
        return result


@attrs.define
class ProductKernelConfig(KernelConfig):
    """Product of kernels: k1 * k2 * ..."""

    components: list[KernelConfig] = attrs.Factory(list)

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        if not self.components:
            raise ValueError("ProductKernelConfig requires at least one component")
        kernels = [c.buildKernel(ndim, rngs=rngs) for c in self.components]
        result = kernels[0]
        for k in kernels[1:]:
            result = result * k
        return result


@attrs.define
class ScaledKernelConfig(KernelConfig):
    """Kernel multiplied by a learned scale."""

    base: KernelConfig = attrs.Factory(Matern32Config)
    scale_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        kernel = self.base.buildKernel(ndim, rngs=rngs)
        scale_val = self._get_param("constant", 1.0, self.scale_prior)
        return gpk.Constant(constant=scale_val) * kernel


class Network(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        output_dim: int,
        shape: list[int],
        activation_name: str = "relu",
    ) -> None:
        self.in_layer = nnx.Linear(input_dim, shape[0], rngs=rngs)
        self.layers = nnx.List(
            [
                nnx.Linear(shape[i], shape[i + 1], rngs=rngs)
                for i in range(len(shape) - 1)
            ]
        )
        self.out_layer = nnx.Linear(shape[-1], output_dim, rngs=rngs)
        self.activation_name = activation_name
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = getattr(jax.nn, self.activation_name)
        x = self.in_layer(x)
        for layer in self.layers:
            x = activation(layer(x))
        x = self.out_layer(x)
        return x


@attrs.define(slots=False)
class DeepKernelFunction(AbstractKernel):
    base_kernel: AbstractKernel
    network: nnx.Module
    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __call__(self, x, y):
        xt = self.network(x)
        yt = self.network(y)
        return self.base_kernel(xt, yt)


@attrs.define
class NNKernelConfig(KernelConfig):
    base_kernel_config: KernelConfig = attrs.Factory(RBFConfig)
    input_dim: int = 2
    output_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [8, 8])
    activation: str = "relu"

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None
    ) -> gpk.AbstractKernel:
        if rngs is None:
            raise ValueError("NNKernelConfig requires rngs for network initialization")
        base_kernel = self.base_kernel_config.buildKernel(ndim, rngs=rngs)
        forward_linear = Network(
            rngs=rngs,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            shape=self.hidden_shapes,
            activation_name=self.activation,
        )
        return DeepKernelFunction(
            base_kernel=base_kernel,
            network=forward_linear,
        )
