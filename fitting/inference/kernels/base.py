from __future__ import annotations

from abc import ABC, abstractmethod
import attrs
from flax import nnx
import jax.numpy as jnp
import gpjax.kernels as gpk
import gpjax.parameters as gpp
from typing import Any

from ..priors import PriorConfig


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
