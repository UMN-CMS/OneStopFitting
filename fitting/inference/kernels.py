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
import jax.numpy as jnp
import gpjax.kernels as gpk
import gpjax.parameters as gpp
from typing import Any

from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.base import AbstractKernel
from .priors import PriorConfig

from ..data.loading import (
    FileLoader,
    extractHistogram,
    histToBinnedData,
    variationNames,
)


@attrs.define
class KernelConfig(ABC):
    """Base kernel configuration.

    Subclasses represent specific kernel types.
    """

    @abstractmethod
    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
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
        param_type: type[gpp.Parameter] = gpp.PositiveReal,
    ) -> Any:
        """Wrap a parameter value with a prior if present."""
        if prior_config is not None:
            return param_type(value, prior=prior_config.buildPrior())
        return value


@attrs.define
class RBFConfig(KernelConfig):
    """Radial Basis Function (squared exponential) kernel."""

    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)

        return gpk.RBF(lengthscale=lengthscale, variance=variance)


@attrs.define
class Matern12Config(KernelConfig):
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern12(lengthscale=lengthscale, variance=variance)


@attrs.define
class Matern32Config(KernelConfig):
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern32(lengthscale=lengthscale, variance=variance)


@attrs.define
class Matern52Config(KernelConfig):
    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        return gpk.Matern52(lengthscale=lengthscale, variance=variance)


@attrs.define
class RationalQuadraticConfig(KernelConfig):
    """Rational Quadratic kernel."""

    ard: bool = True
    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None
    alpha_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        ls_val = [0.25] * ndim if self.ard else 0.25
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 2.0, self.variance_prior)
        alpha = self._get_param("alpha", 1.0, self.alpha_prior)
        return gpk.RationalQuadratic(
            lengthscale=lengthscale, variance=variance, alpha=alpha
        )


@attrs.define
class PeriodicConfig(KernelConfig):
    """Periodic kernel."""

    lengthscale_prior: PriorConfig | None = None
    variance_prior: PriorConfig | None = None
    period_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        lengthscale = self._get_param("lengthscale", 1.0, self.lengthscale_prior)
        variance = self._get_param("variance", 1.0, self.variance_prior)
        period = self._get_param("period", 1.0, self.period_prior)
        return gpk.Periodic(lengthscale=lengthscale, variance=variance, period=period)


@attrs.define
class WhiteConfig(KernelConfig):
    """White noise kernel."""

    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1e-6, self.variance_prior)
        return gpk.White(variance=variance)


@attrs.define
class LinearConfig(KernelConfig):
    """Linear kernel."""

    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        return gpk.Linear(variance=variance)


@attrs.define
class PolynomialConfig(KernelConfig):
    """Polynomial kernel."""

    degree: int = 2
    variance_prior: PriorConfig | None = None
    shift_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        shift = self._get_param("shift", 1.0, self.shift_prior)
        return gpk.Polynomial(degree=self.degree, variance=variance, shift=shift)


@attrs.define
class SumKernelConfig(KernelConfig):
    """Sum of kernels: k1 + k2 + ..."""

    kernels: list[KernelConfig] = attrs.field(factory=list)

    @kernels.validator
    def _validateKernels(self, attribute, value):
        if not value:
            raise ValueError("SumKernelConfig requires at least one component")

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        kernels = [c.buildKernel(ndim, rngs=rngs, **kwargs) for c in self.kernels]
        result = kernels[0]
        for k in kernels[1:]:
            result = result + k
        return result


@attrs.define
class ProductKernelConfig(KernelConfig):
    """Product of kernels: k1 * k2 * ..."""

    kernels: list[KernelConfig] = attrs.field(factory=list)

    @kernels.validator
    def _validateKernels(self, attribute, value):
        if not value:
            raise ValueError("ProductKernelConfig requires at least one component")

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        kernels = [c.buildKernel(ndim, rngs=rngs, **kwargs) for c in self.kernels]
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
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        kernel = self.base.buildKernel(ndim, rngs=rngs, **kwargs)
        scale_val = self._get_param("constant", 1.0, self.scale_prior)
        return gpk.Constant(constant=scale_val) * kernel


class Network(nnx.Module):
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        output_dim: int,
        shape: list[int],
        activation_name: str = "silu",
    ) -> None:
        self.in_layer = nnx.Linear(input_dim, shape[0], rngs=rngs)
        self.layers = nnx.List(
            [
                nnx.Linear(shape[i], shape[i + 1], rngs=rngs)
                for i in range(len(shape) - 1)
            ]
        )

        self.out_layer = nnx.Linear(
            shape[-1],
            output_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )
        self.activation_name = activation_name
        self.rngs = rngs

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = getattr(jax.nn, self.activation_name)
        init = x
        x = activation(self.in_layer(x))
        for layer in self.layers:
            x = activation(layer(x))
        x = self.out_layer(x)
        return x + init


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

    def cross_covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        xt = self.network(x)
        yt = self.network(y)
        return self.base_kernel.cross_covariance(xt, yt)


@attrs.define
class NNKernelConfig(KernelConfig):
    base_kernel_config: KernelConfig = attrs.Factory(RBFConfig)
    input_dim: int = 2
    output_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [20, 10, 5])
    activation: str = "silu"

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        if rngs is None:
            raise ValueError("NNKernelConfig requires rngs for network initialization")
        base_kernel = self.base_kernel_config.buildKernel(ndim, rngs=rngs, **kwargs)
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


class MCEnsembleKernel(gpk.AbstractKernel):
    """
    K(x, x') = Cov_MC[f(x), f(x')] / (f_nom(x) * f_nom(x'))
    """

    _ensemble_X: jnp.ndarray = nnx.data()
    _ensemble_cov: jnp.ndarray = nnx.data()

    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __init__(self, ensemble_X, ensemble_Y, nominal_Y, nugget=1e-6):
        super().__init__()
        self._ensemble_X = jax.lax.stop_gradient(ensemble_X)
        frac_variations = ensemble_Y / nominal_Y[None, :]  # (n_var, n_ref)
        emp_cov = jnp.cov(frac_variations, rowvar=False)  # (n_ref, n_ref)
        emp_cov += nugget * jnp.eye(len(nominal_Y))
        self._ensemble_cov = jax.lax.stop_gradient(emp_cov)
        self.log_amplitude = gpp.Real(jnp.array(0.0))

    def _interpolate_cov(self, x1, x2):
        diffs1 = x1[:, None, :] - self._ensemble_X[None, :, :]
        i1 = jnp.argmin(jnp.sum(diffs1**2, axis=-1), axis=-1)
        diffs2 = x2[:, None, :] - self._ensemble_X[None, :, :]
        i2 = jnp.argmin(jnp.sum(diffs2**2, axis=-1), axis=-1)
        return self._ensemble_cov[i1[:, None], i2[None, :]]

    def __call__(self, x, y):
        amp = jnp.exp(self.log_amplitude.value)
        return amp * self._interpolate_cov(x.reshape(1, -1), y.reshape(1, -1))[0, 0]

    def cross_covariance(self, x, y):
        amp = jnp.exp(self.log_amplitude.value)
        return amp * self._interpolate_cov(x, y)


@attrs.define
class MCEnsembleKernelConfig(KernelConfig):
    mc_path: str = attrs.Factory(lambda: "")
    nugget: float = 1e-6

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        if not self.mc_path:
            raise ValueError("MCEnsembleKernelConfig requires a valid mc_path")

        loader = FileLoader.forPath(self.mc_path)
        raw_data = loader.load(self.mc_path)
        histogram = extractHistogram(raw_data)

        central_data = histToBinnedData(histogram, variation="central")
        domain_mask = kwargs.get("domain_mask", None)
        if domain_mask is not None:
            masked_central = central_data.masked(domain_mask)
        else:
            masked_central = central_data

        mc_X = masked_central.X
        nominal_Y = masked_central.Y
        ensemble_Y_list = []
        variations = variationNames(histogram)
        for var in variations:
            if var == "central":
                continue
            var_data = histToBinnedData(histogram, variation=var)
            if domain_mask is not None:
                masked_var_Y = var_data.Y[domain_mask]
            else:
                masked_var_Y = var_data.Y
            ensemble_Y_list.append(masked_var_Y)

        if not ensemble_Y_list:
            raise ValueError(
                f"No non-central variations found in {self.mc_path}. "
                "MCEnsembleKernel requires an ensemble of variations."
            )

        ensemble_Y = jnp.stack(ensemble_Y_list, axis=0)  # (n_variations, n_ref)

        return MCEnsembleKernel(
            ensemble_X=mc_X,
            ensemble_Y=ensemble_Y,
            nominal_Y=nominal_Y,
            nugget=self.nugget,
        )
