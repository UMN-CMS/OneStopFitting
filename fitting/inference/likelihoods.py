"""Likelihood configuration hierarchy.

Each LikelihoodConfig subclass wraps a gpjax likelihood type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import attrs
import gpjax.likelihoods as gpl
import gpjax.parameters as gpp
from .priors import PriorConfig


@attrs.define
class LikelihoodConfig(ABC):
    """Base likelihood configuration."""

    @abstractmethod
    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        """Construct the gpjax likelihood.

        Keyword args typically include:
            num_datapoints: Length of the dataset.
            obs_variance: Array of variances for each bin (optional).
        """
        ...

    def _get_param(
        self,
        name: str,
        value: Any,
        prior_config: PriorConfig | None = None,
        param_type: type[gpp.Parameter] = gpp.PositiveReal,
    ) -> Any:
        """Wrap a parameter value with a prior if present."""
        if prior_config is not None:
            return param_type(value, prior=prior_config.buildPrior())
        return value

@attrs.define
class UniformGaussianNoiseConfig(LikelihoodConfig):
    """Homoscedastic Gaussian likelihood with a single learned scalar variance."""

    obs_stddev_prior: PriorConfig | None = None

    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        obs_stddev = self._get_param("obs_stddev", 1.0, self.obs_stddev_prior)
        return gpl.Gaussian(
            num_datapoints=kwargs["num_datapoints"], obs_stddev=obs_stddev
        )


@attrs.define
class FixedGaussianNoiseConfig(LikelihoodConfig):
    """Gaussian likelihood with fixed per-bin variance."""

    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        if "obs_variance" not in kwargs or kwargs["obs_variance"] is None:
            raise ValueError(
                "FixedGaussianNoiseConfig requires an 'obs_variance' array."
            )
        import jax.numpy as jnp
        from flax import nnx

        obs_var = kwargs["obs_variance"]
        obs_var = jnp.clip(obs_var, a_min=jnp.min(obs_var[obs_var > 0]))
        variances = jnp.atleast_1d(obs_var).reshape(-1, 1)
        likelihood = gpl.Gaussian(
            num_datapoints=kwargs["num_datapoints"],
            obs_stddev=jnp.sqrt(variances),
        )
        likelihood.obs_stddev = nnx.Variable(likelihood.obs_stddev[...])
        return likelihood


@attrs.define
class HeteroscedasticGaussianConfig(LikelihoodConfig):
    """Heteroscedastic Gaussian likelihood where noise is modeled as a GP."""

    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        import gpjax

        noise_mean = gpjax.mean_functions.Constant()
        noise_kernel = gpjax.kernels.RBF()
        noise_prior = gpjax.gps.Prior(mean_function=noise_mean, kernel=noise_kernel)

        return gpl.HeteroscedasticGaussian(
            num_datapoints=kwargs["num_datapoints"],
            noise_prior=noise_prior,
        )


@attrs.define
class PoissonConfig(LikelihoodConfig):
    """Poisson likelihood for count data."""

    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        return gpl.Poisson(num_datapoints=kwargs["num_datapoints"])
