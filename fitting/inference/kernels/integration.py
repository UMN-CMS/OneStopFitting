from __future__ import annotations

import itertools
import logging

import attrs
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import DenseKernelComputation
from gpjax.linalg import Dense, Diagonal, psd
from gpjax.mean_functions import AbstractMeanFunction

from .base import KernelConfig

logger = logging.getLogger(__name__)


def computeQuadratureGrid(
    X: jnp.ndarray,
    edges: tuple[jnp.ndarray, ...],
    n_quad: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Gauss-Legendre quadrature points and weights for every bin.
    """
    ndim = len(edges)
    n_bins = X.shape[0]

    ref_nodes, ref_weights = np.polynomial.legendre.leggauss(n_quad)
    ref_nodes = jnp.array(ref_nodes, dtype=jnp.float64)
    ref_weights = jnp.array(ref_weights, dtype=jnp.float64)

    bin_lo = jnp.empty((n_bins, ndim))
    bin_hi = jnp.empty((n_bins, ndim))

    for d in range(ndim):
        x_d = X[:, d]
        idx = jnp.searchsorted(edges[d], x_d, side="right") - 1
        idx = jnp.clip(idx, 0, len(edges[d]) - 2)
        bin_lo = bin_lo.at[:, d].set(edges[d][idx])
        bin_hi = bin_hi.at[:, d].set(edges[d][idx + 1])

    half_widths = (bin_hi - bin_lo) / 2.0  # (n_bins, ndim)
    centres = (bin_hi + bin_lo) / 2.0  # (n_bins, ndim)

    dim_nodes = half_widths[:, None, :] * ref_nodes[None, :, None] + centres[:, None, :]
    dim_weights = half_widths[:, None, :] * ref_weights[None, :, None]

    multi_idx = jnp.array(
        list(itertools.product(range(n_quad), repeat=ndim))
    )  # (n_total, ndim)
    n_total = multi_idx.shape[0]  # n_quad ** ndim

    points_list = []
    weight_factors = []
    for d in range(ndim):
        points_list.append(dim_nodes[:, multi_idx[:, d], d])  # (n_bins, n_total)
        weight_factors.append(dim_weights[:, multi_idx[:, d], d])

    points = jnp.stack(points_list, axis=-1)  # (n_bins, n_total, ndim)

    weights = jnp.ones((n_bins, n_total))
    for wf in weight_factors:
        weights = weights * wf  # (n_bins, n_total)

    weights = weights / jnp.sum(weights, axis=1, keepdims=True)

    return points, weights


class BinIntegratedKernel(AbstractKernel):
    def __init__(
        self,
        base_kernel: AbstractKernel,
        n_quad: int = 3,
    ):
        super().__init__(compute_engine=DenseKernelComputation())
        self.base_kernel = base_kernel
        self.n_quad = n_quad
        self._quad_registry: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = {}


    def registerQuadrature(
        self,
        n_points: int,
        points: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> None:
        self._quad_registry[n_points] = (points, weights)
        logger.debug(
            f"BinIntegratedKernel: registered quadrature for n={n_points}, "
            f"n_quad_total={points.shape[1]}"
        )

    def _lookupQuadrature(self, n: int) -> tuple[jnp.ndarray, jnp.ndarray] | None:
        found = self._quad_registry.get(n)
        if found is None:
            return None
        points, weights = found
        return jax.lax.stop_gradient(points), jax.lax.stop_gradient(weights)

    def __call__(self, x, y):
        return self.base_kernel(x, y)

    def gram(self, x):
        n = x.shape[0]
        quad = self._lookupQuadrature(n)
        if quad is None:
            logger.warning(
                f"BinIntegratedKernel.gram: no quadrature registered for "
                f"n={n}, falling back to point evaluation."
            )
            return self.base_kernel.gram(x)

        points, weights = quad
        K = self._integratedCrossCovariance(points, weights, points, weights)
        return psd(Dense(K))

    def cross_covariance(self, x, y):
        nx, ny = x.shape[0], y.shape[0]
        quad_x = self._lookupQuadrature(nx)
        quad_y = self._lookupQuadrature(ny)

        if quad_x is None or quad_y is None:
            logger.warning(
                f"BinIntegratedKernel.cross_covariance: missing quadrature "
                f"(nx={nx}, ny={ny}), falling back to point evaluation."
            )
            return self.base_kernel.cross_covariance(x, y)

        px, wx = quad_x
        py, wy = quad_y
        return self._integratedCrossCovariance(px, wx, py, wy)


    def diagonal(self, x):
        """Compute the integrated diagonal of the Gram matrix."""
        n = x.shape[0]
        quad = self._lookupQuadrature(n)
        if quad is None:
            return self.base_kernel.diagonal(x)

        points, weights = quad

        def oneDiag(pts, wts):
            k_mat = jax.vmap(
                lambda xi: jax.vmap(lambda yj: self.base_kernel(xi, yj))(pts)
            )(pts)
            return jnp.sum(wts[:, None] * wts[None, :] * k_mat)

        diag = jax.vmap(oneDiag)(points, weights)
        return psd(Diagonal(diag))

    def _integratedCrossCovariance(
        self,
        points_x: jnp.ndarray,
        weights_x: jnp.ndarray,
        points_y: jnp.ndarray,
        weights_y: jnp.ndarray,
    ) -> jnp.ndarray:

        def onePair(pts_i, wts_i, pts_j, wts_j):
            k_mat = jax.vmap(
                lambda xi: jax.vmap(lambda yj: self.base_kernel(xi, yj))(pts_j)
            )(pts_i)
            return jnp.sum(wts_i[:, None] * wts_j[None, :] * k_mat)

        def overJ(pts_i, wts_i):
            return jax.vmap(lambda pj, wj: onePair(pts_i, wts_i, pj, wj))(
                points_y, weights_y
            )

        return jax.vmap(overJ)(points_x, weights_x)  # (nx, ny)


class BinIntegratedMeanFunction(AbstractMeanFunction):
    def __init__(self, base_mean_fn: AbstractMeanFunction):
        super().__init__()
        self.base_mean_fn = base_mean_fn
        self._quad_registry: dict[int, tuple[jnp.ndarray, jnp.ndarray]] = {}

    def registerQuadrature(
        self,
        n_points: int,
        points: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> None:
        self._quad_registry[n_points] = (points, weights)

    def __call__(self, x):
        n = x.shape[0]
        quad = self._quad_registry.get(n)
        if quad is None:
            return self.base_mean_fn(x)

        points, weights = quad  # (n, nq, D), (n, nq)

        def _one_bin(pts, wts):
            vals = self.base_mean_fn(pts)  # (nq, 1) or (nq,)
            vals = vals.squeeze(-1) if vals.ndim > 1 else vals
            return jnp.sum(wts * vals)

        return jax.vmap(_one_bin)(points, weights).reshape(-1, 1)


def getDefaultKern():
    from .standard import RBFConfig

    return RBFConfig()


@attrs.define
class BinIntegratedKernelConfig(KernelConfig):
    base_kernel: KernelConfig = attrs.Factory(lambda: getDefaultKern())
    n_quad: int = 3

    def buildKernel(
        self, ndim: int, rngs: nnx.Rngs | None = None, **kwargs
    ) -> AbstractKernel:
        base = self.base_kernel.buildKernel(ndim, rngs=rngs, **kwargs)
        wrapped = BinIntegratedKernel(base_kernel=base, n_quad=self.n_quad)

        quad_points = kwargs.get("quad_points")
        quad_weights = kwargs.get("quad_weights")
        if quad_points is not None and quad_weights is not None:
            wrapped.registerQuadrature(quad_points.shape[0], quad_points, quad_weights)
            logger.info(
                f"BinIntegratedKernelConfig: registered train quadrature "
                f"({quad_points.shape[0]} bins, {quad_points.shape[1]} sub-points each)"
            )

        return wrapped
