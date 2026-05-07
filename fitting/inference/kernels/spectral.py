from __future__ import annotations

import attrs
from flax import nnx
import jax.numpy as jnp
import gpjax.kernels as gpk
import gpjax.parameters as gpp
from gpjax.kernels.base import AbstractKernel
from gpjax.kernels.computations import (
    AbstractKernelComputation,
    DenseKernelComputation,
)

from .base import KernelConfig
from ..priors import PriorConfig


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
