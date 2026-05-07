from __future__ import annotations

import attrs
from flax import nnx
import jax
import jax.numpy as jnp
import gpjax.parameters as gpp
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)

from .base import KernelConfig
from ..priors import PriorConfig


class HeteroscedasticWhiteKernel(AbstractKernel):
    _ref_X: jnp.ndarray = nnx.data()
    _ref_var: jnp.ndarray = nnx.data()

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
        super().__init__()
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
        noise = self.scale.value * var_x + self.offset.value

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
            raise RuntimeError("Heteroscedastic config must be provided the dataset")
        scale = self._get_param("scale", 0.01, self.scale_prior)
        offset = self._get_param("offset", 1e-8, self.offset_prior)
        return HeteroscedasticWhiteKernel(
            dataset.X,
            dataset.y[:, 0],
            scale=scale,
            offset=offset,
            scale_prior=self.scale_prior,
            offset_prior=self.offset_prior,
        )
