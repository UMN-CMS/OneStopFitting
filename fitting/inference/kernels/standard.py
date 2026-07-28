from __future__ import annotations

import attrs
from flax import nnx
import gpjax.kernels as gpk

from .base import KernelConfig
from ..priors import PriorConfig


@attrs.define
class RBFConfig(KernelConfig):
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
    variance: float = 1e-6
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", self.variance, self.variance_prior)
        return gpk.White(variance=variance)


@attrs.define
class LinearConfig(KernelConfig):
    variance_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        return gpk.Linear(variance=variance)


@attrs.define
class PolynomialConfig(KernelConfig):
    degree: int = 2
    variance_prior: PriorConfig | None = None
    shift_prior: PriorConfig | None = None

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> gpk.AbstractKernel:
        variance = self._get_param("variance", 1.0, self.variance_prior)
        shift = self._get_param("shift", 1.0, self.shift_prior)
        return gpk.Polynomial(degree=self.degree, variance=variance, shift=shift)
