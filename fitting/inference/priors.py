"""Prior configuration hierarchy.

Defines serializable configurations for numpyro distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import attrs
import numpyro.distributions as dist


@attrs.define
class PriorConfig(ABC):
    """Base prior configuration."""

    @abstractmethod
    def buildPrior(self) -> dist.Distribution:
        """Construct the numpyro distribution."""
        ...


@attrs.define
class LogNormalPriorConfig(PriorConfig):
    """Log-Normal distribution."""

    loc: float = 0.0
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.LogNormal(loc=self.loc, scale=self.scale)


@attrs.define
class NormalPriorConfig(PriorConfig):
    """Normal distribution."""

    loc: float = 0.0
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Normal(loc=self.loc, scale=self.scale)


@attrs.define
class HalfNormalPriorConfig(PriorConfig):
    """Half-Normal distribution."""

    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.HalfNormal(scale=self.scale)


@attrs.define
class GammaPriorConfig(PriorConfig):
    """Gamma distribution."""

    concentration: float = 1.0
    rate: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Gamma(concentration=self.concentration, rate=self.rate)


@attrs.define
class UniformPriorConfig(PriorConfig):
    """Uniform distribution."""

    low: float = 0.0
    high: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Uniform(low=self.low, high=self.high)
