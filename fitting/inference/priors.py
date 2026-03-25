from __future__ import annotations

from abc import ABC, abstractmethod

import attrs
import numpyro.distributions as dist


@attrs.define
class PriorConfig(ABC):
    @abstractmethod
    def buildPrior(self) -> dist.Distribution: ...


@attrs.define
class LogNormalPriorConfig(PriorConfig):
    loc: float = 0.0
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.LogNormal(loc=self.loc, scale=self.scale)


@attrs.define
class NormalPriorConfig(PriorConfig):
    loc: float = 0.0
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Normal(loc=self.loc, scale=self.scale)


@attrs.define
class HalfNormalPriorConfig(PriorConfig):
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.HalfNormal(scale=self.scale)


@attrs.define
class GammaPriorConfig(PriorConfig):
    concentration: float = 1.0
    rate: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Gamma(concentration=self.concentration, rate=self.rate)


@attrs.define
class UniformPriorConfig(PriorConfig):
    low: float = 0.0
    high: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.Uniform(low=self.low, high=self.high)


@attrs.define
class SoftplusNormalPriorConfig(PriorConfig):
    loc: float = 0.0
    scale: float = 1.0

    def buildPrior(self) -> dist.Distribution:
        return dist.TransformedDistribution(
            dist.Normal(loc=self.loc, scale=self.scale),
            dist.transforms.SoftplusTransform(),
        )
