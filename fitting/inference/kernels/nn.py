from __future__ import annotations

import attrs
from flax import nnx
import jax
import jax.numpy as jnp
import gpjax.kernels as gpk
import gpjax.parameters as gpp
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)
from gpjax.kernels.base import AbstractKernel

from .base import KernelConfig
from .standard import RBFConfig
from ..priors import PriorConfig


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
        output_scale_prior: PriorConfig | None = None,
        output_residual: bool = True,
    ) -> None:
        def wrap_layer(layer: nnx.Linear):
            if weight_prior:
                prior = weight_prior.buildPrior()
                val = layer.kernel[...]
                if val.ndim > 0 and val.size > 1:
                    if prior.batch_shape == () and prior.event_shape == ():
                        prior = prior.expand(val.shape).to_event(val.ndim)
                layer.kernel = gpp.Real(val, prior=prior)

            if bias_prior:
                prior = bias_prior.buildPrior()
                val = layer.bias[...]
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
        self.output_residual = output_residual
        if self.output_residual:
            if output_scale_prior:
                prior = output_scale_prior.buildPrior()
                self.scale = gpp.PositiveReal(0.01, prior=prior)
            else:
                self.scale = gpp.PositiveReal(0.01)

    def __call__(self, x: jax.Array) -> jax.Array:
        activation = getattr(jax.nn, self.activation_name)
        z = activation(self.in_layer(x))
        for layer in self.layers:
            z = activation(layer(z))
        delta = self.out_layer(z)
        if self.output_residual:
            return x + self.scale[...] * delta  # per-dim gated residual
        return delta


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
        return x + self.scale[...] * delta


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

    def cross_covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        xt = self.network(x)
        yt = self.network(y)
        return self.base_kernel.cross_covariance(xt, yt)

    def gram(self, x: jax.Array):
        xt = self.network(x)
        return self.base_kernel.gram(xt)

    def diagonal(self, x: jax.Array):
        xt = self.network(x)
        return self.base_kernel.diagonal(xt)


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

    def gram(self, x: jax.Array):
        xt = self.network(x)
        return self.base_kernel.gram(xt)

    def diagonal(self, x: jax.Array):
        xt = self.network(x)
        return self.base_kernel.diagonal(xt)


@attrs.define
class NNWarpingKernelConfig(KernelConfig):
    base_kernel_config: KernelConfig = attrs.Factory(RBFConfig)
    input_dim: int = 2
    output_dim: int = 2
    hidden_shapes: list[int] = attrs.Factory(lambda: [20, 20])
    activation: str = "silu"
    weight_prior: PriorConfig | None = None
    bias_prior: PriorConfig | None = None
    output_scale_prior: PriorConfig | None = None
    axis_decoupled: bool = False

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        if rngs is None:
            raise ValueError("NNKernelConfig requires rngs for network initialization")
        base_kernel = self.base_kernel_config.buildKernel(ndim, rngs=rngs, **kwargs)

        if self.axis_decoupled:
            forward_linear = AxisDecoupledNetwork(
                rngs=rngs,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                shape=self.hidden_shapes,
                activation_name=self.activation,
            )
        else:
            forward_linear = Network(
                rngs=rngs,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                shape=self.hidden_shapes,
                activation_name=self.activation,
                weight_prior=self.weight_prior,
                bias_prior=self.bias_prior,
                output_scale_prior=self.output_scale_prior,
            )

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
