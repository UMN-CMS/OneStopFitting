from __future__ import annotations

import attrs
from flax import nnx
import gpjax.kernels as gpk

from .base import KernelConfig
from .standard import Matern32Config
from ..priors import PriorConfig


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
