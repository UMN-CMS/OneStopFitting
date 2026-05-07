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

from .base import KernelConfig
from ...data.loading import (
    FileLoader,
    extractHistogram,
    histToBinnedData,
    variationNames,
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
