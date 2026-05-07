from __future__ import annotations

from abc import ABC, abstractmethod
import attrs
from flax import nnx
import jax
import jax.numpy as jnp
import gpjax
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
    @abstractmethod
    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel: ...

    def _get_param(
        self,
        name: str,
        value: Any,
        prior_config: PriorConfig | None = None,
        param_type: type[gpp.Parameter] = gpp.PositiveReal,
    ) -> Any:
        """Wrap a parameter value with a prior if present."""
        if prior_config is not None:
            prior = prior_config.buildPrior()
            val_array = jnp.array(value)
            if val_array.ndim > 0 and val_array.size > 1:
                if prior.batch_shape == () and prior.event_shape == ():
                    prior = prior.expand(val_array.shape).to_event(val_array.ndim)

            return param_type(value, prior=prior)
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
        ls_val = [0.1] * ndim if self.ard else 0.1
        lengthscale = self._get_param("lengthscale", ls_val, self.lengthscale_prior)
        variance = self._get_param("variance", 0.5, self.variance_prior)

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
        variance = self._get_param("variance", 0.5, self.variance_prior)
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

    variance: float = 1e-6
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", self.variance, self.variance_prior)
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
        weight_prior: PriorConfig | None = None,
        bias_prior: PriorConfig | None = None,
        out_kernel_init=nnx.initializers.zeros,
        out_bias_init=nnx.initializers.zeros,
    ) -> None:
        def wrap_layer(layer: nnx.Linear):
            if weight_prior:
                prior = weight_prior.buildPrior()
                val = layer.kernel.value
                if val.ndim > 0 and val.size > 1:
                    if prior.batch_shape == () and prior.event_shape == ():
                        prior = prior.expand(val.shape).to_event(val.ndim)
                layer.kernel = gpp.Real(val, prior=prior)

            if bias_prior:
                prior = bias_prior.buildPrior()
                val = layer.bias.value
                if val.ndim > 0 and val.size > 1:
                    if prior.batch_shape == () and prior.event_shape == ():
                        prior = prior.expand(val.shape).to_event(val.ndim)
                layer.bias = gpp.Real(val, prior=prior)

        self.in_layer = nnx.Linear(input_dim, shape[0], rngs=rngs)
        wrap_layer(self.in_layer)

        self.layers = nnx.List(
            [
                nnx.Linear(shape[i], shape[i + 1], rngs=rngs)
                for i in range(len(shape) - 1)
            ]
        )
        for layer in self.layers:
            wrap_layer(layer)

        self.out_layer = nnx.Linear(
            shape[-1],
            output_dim,
            rngs=rngs,
            kernel_init=out_kernel_init,
            bias_init=out_bias_init,
        )
        wrap_layer(self.out_layer)

        self.activation_name = activation_name
        self.rngs = rngs
        self.scale = gpp.PositiveReal(jnp.ones(input_dim))

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = getattr(jax.nn, self.activation_name)
        z = activation(self.in_layer(x))
        for layer in self.layers:
            z = activation(layer(z))
        delta = self.out_layer(z)
        return x + self.scale.value * delta  # per-dim gated residual


class AxisDecoupledNetwork(nnx.Module):
    def __init__(self, rngs, input_dim, output_dim, shape, activation_name="silu"):
        self.axis_nets = nnx.List(
            [nnx.Linear(1, shape[0] // input_dim, rngs=rngs) for _ in range(input_dim)]
        )
        joint_dim = shape[0]
        self.layers = nnx.List(
            [
                nnx.Linear(joint_dim if i == 0 else shape[i], shape[i], rngs=rngs)
                for i in range(len(shape))
            ]
        )
        self.out_layer = nnx.Linear(
            shape[-1],
            output_dim,
            rngs=rngs,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
        )
        self.scale = gpp.PositiveReal(jnp.ones(input_dim))
        self.activation_name = activation_name

    def __call__(self, x):
        activation = getattr(jax.nn, self.activation_name)
        axis_feats = [
            activation(net(x[..., i : i + 1])) for i, net in enumerate(self.axis_nets)
        ]
        h = jnp.concatenate(axis_feats, axis=-1)
        for layer in self.layers:
            h = activation(layer(h))
        delta = self.out_layer(h)
        return x + self.scale.value * delta


@attrs.define(slots=False)
class DeepWarpingKernel(AbstractKernel):
    base_kernel: AbstractKernel
    network: nnx.Module
    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __call__(self, x, y):
        xt = self.network(x)
        yt = self.network(y)
        ret = self.base_kernel(xt, yt)
        return ret


@attrs.define(slots=False)
class DeepTransformKernel(AbstractKernel):
    base_kernel: AbstractKernel
    network: nnx.Module
    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __call__(self, x, y):
        xt = self.network(x)
        yt = self.network(y)
        ret = self.base_kernel(xt, yt)
        return ret

    def cross_covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        xt = self.network(x)
        yt = self.network(y)
        return self.base_kernel.cross_covariance(xt, yt)


@attrs.define
class NNWarpingKernelConfig(KernelConfig):
    base_kernel_config: KernelConfig = attrs.Factory(RBFConfig)
    input_dim: int = 2
    output_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [20, 20])
    activation: str = "silu"
    weight_prior: PriorConfig | None = None
    bias_prior: PriorConfig | None = None

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
            weight_prior=self.weight_prior,
            bias_prior=self.bias_prior,
        )

        forward_linear = AxisDecoupledNetwork(
            rngs=rngs,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            shape=self.hidden_shapes,
            activation_name=self.activation,
        )
        nnx.display(forward_linear)
        breakpoint()
        return DeepWarpingKernel(
            base_kernel=base_kernel,
            network=forward_linear,
        )


@attrs.define
class NNTransformKernelConfig(KernelConfig):
    base_kernel_config: KernelConfig = attrs.Factory(RBFConfig)
    input_dim: int = 2
    output_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [20, 20])
    activation: str = "silu"
    weight_prior: PriorConfig | None = None
    bias_prior: PriorConfig | None = None

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
            weight_prior=self.weight_prior,
            bias_prior=self.bias_prior,
        )
        return DeepTransformKernel(
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


class MultiFidelityResidualKernel(gpk.AbstractKernel):
    """Kernel for multi-fidelity residual: K(x,x') = ρ² · Σ_MC(x,x') + K_δ(x,x').

    Combines the frozen low-fidelity (MC) GP posterior covariance with a
    learnable residual kernel that captures data-MC discrepancy.

    Args:
        mc_kernel: Prior kernel of the MC GP.
        mc_L: Precomputed Cholesky factor of the MC training covariance (Kxx + σ²I).
        mc_dataset: Training dataset of the MC posterior (X_mc).
        residual_kernel: Base kernel for the δ(x) residual.
        log_rho: Log-scale parameter (shared with mean function).
    """

    _mc_dataset: Any = nnx.data()
    _mc_L: jnp.ndarray = nnx.data()
    residual_kernel: gpk.AbstractKernel
    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __init__(
        self,
        mc_kernel: gpk.AbstractKernel,
        mc_L: jnp.ndarray,
        mc_dataset: gpjax.Dataset,
        residual_kernel: gpk.AbstractKernel,
        rho: gpp.PositiveReal | None = None,
    ):
        super().__init__()
        self.mc_kernel = mc_kernel
        self._mc_L = mc_L
        self._mc_dataset = mc_dataset
        self.residual_kernel = residual_kernel
        self.rho = rho

    def _computeMcCovariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute frozen MC posterior covariance between x and y."""
        # K_post(x, y) = K(x, y) - K(x, X) (L Lᵀ)⁻¹ K(X, y)
        #             = K(x, y) - (L⁻¹ K(X, x))ᵀ (L⁻¹ K(X, y))
        K_xy = self.mc_kernel.cross_covariance(x, y)
        K_Xx = self.mc_kernel.cross_covariance(self._mc_dataset.X, x)
        K_Xy = self.mc_kernel.cross_covariance(self._mc_dataset.X, y)

        # L is lower triangular Cholesky of (Kxx + σ²I)
        Lx = jax.scipy.linalg.solve_triangular(self._mc_L, K_Xx, lower=True)
        Ly = jax.scipy.linalg.solve_triangular(self._mc_L, K_Xy, lower=True)

        mc_post_cov = K_xy - jnp.matmul(Lx.T, Ly)
        return jax.lax.stop_gradient(mc_post_cov)

    def __call__(self, x, y):
        residual_val = self.residual_kernel(x, y)
        if self.rho is not None:
            rho = self.rho[...]
            # Point-wise MC variance
            x_r = x.reshape(1, -1)
            y_r = y.reshape(1, -1)
            mc_cov = self._computeMcCovariance(x_r, y_r)
            return rho**2 * mc_cov[0, 0] + residual_val
        return residual_val

    def cross_covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        residual_cov = self.residual_kernel.cross_covariance(x, y)
        if self.rho is not None:
            rho = jnp.exp(self.rho.value)
            mc_cov = self._computeMcCovariance(x, y)
            return rho**2 * mc_cov + residual_cov
        return residual_cov


@attrs.define
class MultiFidelityResidualKernelConfig(KernelConfig):
    residual_kernel: KernelConfig = attrs.Factory(Matern32Config)
    propagate_mc_variance: bool = True

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        residual = self.residual_kernel.buildKernel(ndim, rngs=rngs, **kwargs)

        if self.propagate_mc_variance:
            mc_kernel = kwargs.get("mc_kernel")
            mc_L = kwargs.get("mc_L")
            mc_dataset = kwargs.get("mc_dataset")
            rho = kwargs.get("rho")
            if mc_kernel is None or mc_L is None or mc_dataset is None:
                raise ValueError(
                    "MultiFidelityResidualKernelConfig with "
                    "propagate_mc_variance=True requires 'mc_kernel', "
                    "'mc_L', and 'mc_dataset' in kwargs."
                )
            return MultiFidelityResidualKernel(
                mc_kernel=mc_kernel,
                mc_L=mc_L,
                mc_dataset=mc_dataset,
                residual_kernel=residual,
                rho=rho,
            )

        return residual


class HeteroscedasticWhiteKernel(AbstractKernel):
    _ref_X: jnp.ndarray = nnx.data()
    _ref_var: jnp.ndarray = nnx.data()
    # offset: gpp.NonNegativeReal | None = None
    # scale: gpp.NonNegativeReal | None = None
    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __init__(
        self,
        ref_X: jnp.ndarray,
        ref_var: jnp.ndarray,
        scale,
        offset,
        scale_prior=None,
        offset_prior=None,
    ):
        self._ref_X = jax.lax.stop_gradient(ref_X)
        self._ref_var = jax.lax.stop_gradient(ref_var)
        if isinstance(scale, nnx.Variable):
            self.scale = scale
        else:
            self.scale = gpp.NonNegativeReal(scale)

        if isinstance(offset, nnx.Variable):
            self.offset = offset
        else:
            self.offset = gpp.NonNegativeReal(offset)

    def __call__(self, x, y):
        dist = jnp.sum((x - y) ** 2)
        is_same = dist < 1e-10

        diffs = jnp.sum((self._ref_X - x[None, :]) ** 2, axis=-1)
        idx = jnp.argmin(diffs)
        var_x = self._ref_var[idx]
        noise = self.scale * var_x + self.offset

        ret = jnp.where(is_same, noise, 0.0)
        return ret


@attrs.define
class HeteroscedasticWhiteConfig(KernelConfig):
    scale_prior: PriorConfig | None = None
    offset_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> AbstractKernel:
        dataset = kwargs.get("dataset")
        if not dataset:
            raise RuntimeError(f"Heteroscedastic config must be provided the dataset")
        scale = self._get_param("scale", 0.01, self.scale_prior)
        offset = self._get_param("offset", 1e-8, self.offset_prior)
        base = HeteroscedasticWhiteKernel(
            dataset.X,
            dataset.y[:, 0],
            scale=scale,
            offset=offset,
            scale_prior=self.scale_prior,
            offset_prior=self.offset_prior,
        )


class SpectralMixtureKernel(AbstractKernel):
    """Spectral Mixture kernel (Wilson & Adams, 2013).

    k(τ) = Σ_q w_q Π_d exp(-2π²τ_d²/ℓ_{q,d}²) cos(2πμ_{q,d}τ_d)

    Parameters are stored with shapes weights (Q,), lengthscales (Q, D),
    frequencies (Q, D).
    """

    compute_engine: AbstractKernelComputation = attrs.field(
        factory=DenseKernelComputation
    )

    def __init__(self, weights, lengthscales, frequencies):
        super().__init__()

        def _wrap(val):
            if isinstance(val, nnx.Variable):
                return val
            return gpp.PositiveReal(jnp.asarray(val, dtype=float))

        self.weights = _wrap(weights)
        self.lengthscales = _wrap(lengthscales)
        self.frequencies = _wrap(frequencies)

    def __call__(self, x, y):
        tau = x - y
        w = self.weights.value
        ls = self.lengthscales.value
        freq = self.frequencies.value

        exp_term = jnp.exp(-2.0 * jnp.pi**2 * tau**2 / ls**2)
        cos_term = jnp.cos(2.0 * jnp.pi * freq * tau)
        per_component = jnp.prod(exp_term * cos_term, axis=-1)
        return jnp.sum(w * per_component)


@attrs.define
class SpectralMixtureConfig(KernelConfig):
    """Spectral Mixture kernel (Wilson & Adams, 2013)."""

    n_components: int = 3
    ard: bool = True
    weight_prior: PriorConfig | None = None
    lengthscale_prior: PriorConfig | None = None
    frequency_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        Q = self.n_components

        w_init = [1.0 / Q] * Q
        weights = self._get_param("weights", w_init, self.weight_prior)

        ls_base = 0.5
        freq_values = [(q + 1.0) / Q for q in range(Q)]

        if self.ard:
            ls_init = [[ls_base] * ndim for _ in range(Q)]
            freq_init = [[f] * ndim for f in freq_values]
        else:
            ls_init = [[ls_base] * ndim for _ in range(Q)]
            freq_init = [[f] * ndim for f in freq_values]

        lengthscales = self._get_param("lengthscales", ls_init, self.lengthscale_prior)
        frequencies = self._get_param("frequencies", freq_init, self.frequency_prior)

        return SpectralMixtureKernel(
            weights=weights,
            lengthscales=lengthscales,
            frequencies=frequencies,
        )
