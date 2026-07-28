from __future__ import annotations

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

from .base import KernelConfig
from .standard import Matern32Config


class MultiFidelityResidualKernel(gpk.AbstractKernel):
    """
    Combines the frozen low-fidelity (MC) GP posterior covariance with a learnable residual kernel that captures data-MC discrepancy.
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

    def computeMcCov(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute frozen MC posterior covariance between x and y."""
        K_xy = self.mc_kernel.cross_covariance(x, y)
        K_Xx = self.mc_kernel.cross_covariance(self._mc_dataset.X, x)
        K_Xy = self.mc_kernel.cross_covariance(self._mc_dataset.X, y)

        Lx = jax.scipy.linalg.solve_triangular(self._mc_L, K_Xx, lower=True)
        Ly = jax.scipy.linalg.solve_triangular(self._mc_L, K_Xy, lower=True)

        mc_post_cov = K_xy - jnp.matmul(Lx.T, Ly)
        return jax.lax.stop_gradient(mc_post_cov)

    def __call__(self, x, y):
        residual_val = self.residual_kernel(x, y)
        if self.rho is not None:
            rho = self.rho[...]
            x_r = x.reshape(1, -1)
            y_r = y.reshape(1, -1)
            mc_cov = self.computeMcCov(x_r, y_r)
            return rho**2 * mc_cov[0, 0] + residual_val
        return residual_val

    def cross_covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        residual_cov = self.residual_kernel.cross_covariance(x, y)
        if self.rho is not None:
            rho = self.rho[...]
            mc_cov = self.computeMcCov(x, y)
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
