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

    variance_floor_quantile: float = 0.05
    pad_variance_quantile: float | None = None
    scale_variance_value: float | None = None

    def buildLikelihood(self, **kwargs) -> gpl.AbstractLikelihood:
        if "obs_variance" not in kwargs or kwargs["obs_variance"] is None:
            raise ValueError(
                "FixedGaussianNoiseConfig requires an 'obs_variance' array."
            )
        import logging
        import jax.numpy as jnp
        from flax import nnx

        logger = logging.getLogger(__name__)

        obs_var = kwargs["obs_variance"]
        positive_vars = obs_var[obs_var > 0]
        floor = jnp.percentile(positive_vars, self.variance_floor_quantile * 100)
        n_clipped = int(jnp.sum(obs_var < floor))
        logger.info(
            f"Variance floor: {float(floor):.6f} "
            f"(quantile={self.variance_floor_quantile}), "
            f"clipping {n_clipped}/{len(obs_var.ravel())} bins"
        )

        obs_var = jnp.clip(obs_var, a_min=floor)
        variances = jnp.atleast_1d(obs_var).reshape(-1)
        if self.pad_variance_quantile is not None:
            pad_val = jnp.percentile(positive_vars, self.pad_variance_quantile * 100)
            variances += pad_val
            logger.info(
                f"Padding variance with {float(pad_val):.6f} "
                f"(quantile={self.pad_variance_quantile})"
            )

        if self.scale_variance_value is not None:
            variances *= self.scale_variance_value
            logger.info(f"Scaling variance by {float(self.scale_variance_value):.6f}")

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
