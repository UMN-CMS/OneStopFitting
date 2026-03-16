"""Data transformations for normalization and scaling.

Provides a hierarchy of TransformConfig classes that build
DataTransformation objects. DataTransformation handles both
values and variances (for weighted histograms), and includes
invertMVN for back-transforming predictions to real space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import attrs
import jax.numpy as jnp
from numpyro.distributions.transforms import Transform, AffineTransform

from .data import BinnedData

logger = logging.getLogger(__name__)


@attrs.define
class DataTransformation:
    """Paired X and Y affine transformations.

    Handles forward/inverse transforms for both values and variances,
    critical for weighted histogram data. The invertMVN method
    back-transforms a multivariate normal from normalized space
    to real bin counts.
    """

    transform_x: Transform
    transform_y: Transform

    def applyX(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.transform_x(X)

    def invertX(self, X: jnp.ndarray) -> jnp.ndarray:
        return self.transform_x.inv(X)

    def applyY(self, Y: jnp.ndarray) -> jnp.ndarray:
        return self.transform_y(Y)

    def invertY(self, Y: jnp.ndarray) -> jnp.ndarray:
        return self.transform_y.inv(Y)

    def applyVariance(self, V: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.atleast_1d(self.transform_y.scale)
        return V * scale**2

    def invertVariance(self, V: jnp.ndarray) -> jnp.ndarray:
        scale = jnp.atleast_1d(self.transform_y.scale)
        return V / scale**2

    def applyEdges(self, edges: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        loc = jnp.atleast_1d(self.transform_x.loc)
        scale = jnp.atleast_1d(self.transform_x.scale)
        return tuple(e * scale[i] + loc[i] for i, e in enumerate(edges))

    def invertEdges(self, edges: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, ...]:
        loc = jnp.atleast_1d(self.transform_x.loc)
        scale = jnp.atleast_1d(self.transform_x.scale)
        return tuple((e - loc[i]) / scale[i] for i, e in enumerate(edges))

    def applyToBinnedData(self, data: BinnedData) -> BinnedData:
        return BinnedData(
            X=self.applyX(data.X),
            Y=self.applyY(data.Y),
            V=self.applyVariance(data.V),
            edges=self.applyEdges(data.edges),
            axis_names=data.axis_names,
        )

    def invertBinnedData(self, data: BinnedData) -> BinnedData:
        return BinnedData(
            X=self.invertX(data.X),
            Y=self.invertY(data.Y),
            V=self.invertVariance(data.V),
            edges=self.invertEdges(data.edges),
            axis_names=data.axis_names,
        )

    def invertMVN(
        self, mean: jnp.ndarray, cov: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        scale = jnp.atleast_1d(self.transform_y.scale)
        loc = jnp.atleast_1d(self.transform_y.loc)
        iscale = 1 / scale
        real_mean = iscale * (mean - loc)
        real_cov = cov * (iscale * iscale)
        return real_mean, real_cov

    def applyToVariations(
        self,
        histogram: Any,
        rebin: int = 1,
    ) -> dict[str, BinnedData]:
        from ..data.loading import histToBinnedData, variationNames

        result = {}
        for name in variationNames(histogram):
            raw = histToBinnedData(histogram, rebin=rebin, variation=name)
            result[name] = self.applyToBinnedData(raw)
        return result


@attrs.define
class TransformConfig(ABC):
    @abstractmethod
    def buildTransform(self, data: BinnedData) -> DataTransformation: ...


@attrs.define
class StandardizationConfig(TransformConfig):
    def buildTransform(self, data: BinnedData) -> DataTransformation:
        X, Y = data.X, data.Y

        x_min = jnp.min(X, axis=0)
        x_max = jnp.max(X, axis=0)
        x_range = x_max - x_min

        y_mean = jnp.mean(Y)
        y_std = jnp.std(Y)

        transform_x = AffineTransform(
            loc=-x_min / x_range,
            scale=1.0 / x_range,
        )

        transform_y = AffineTransform(
            loc=-y_mean / y_std,
            scale=1.0 / y_std,
        )

        return DataTransformation(transform_x=transform_x, transform_y=transform_y)


@attrs.define
class MinMaxConfig(TransformConfig):
    def buildTransform(self, data: BinnedData) -> DataTransformation:
        X, Y = data.X, data.Y

        x_min = jnp.min(X, axis=0)
        x_max = jnp.max(X, axis=0)
        y_min = jnp.min(Y)
        y_max = jnp.max(Y)

        transform_x = AffineTransform(
            loc=-x_min / (x_max - x_min),
            scale=1.0 / (x_max - x_min),
        )
        transform_y = AffineTransform(
            loc=-y_min / (y_max - y_min),
            scale=1.0 / (y_max - y_min),
        )

        return DataTransformation(transform_x=transform_x, transform_y=transform_y)


@attrs.define
class IdentityTransformConfig(TransformConfig):
    def buildTransform(self, data: BinnedData) -> DataTransformation:
        ndim = data.ndim
        transform_x = AffineTransform(loc=jnp.zeros(ndim), scale=jnp.ones(ndim))
        transform_y = AffineTransform(loc=0.0, scale=1.0)
        return DataTransformation(transform_x=transform_x, transform_y=transform_y)


def computeNormalization(
    data: BinnedData, config: TransformConfig | None = None
) -> DataTransformation:
    if config is None:
        config = StandardizationConfig()
    return config.buildTransform(data)
