"""Window definitions for blinding signal regions.

Window is the base class. Subclasses define different blinding
geometries. cattrs include_subclasses handles polymorphic
serialization automatically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import attrs
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from ..core.data import BinnedData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Window hierarchy
# ---------------------------------------------------------------------------


@attrs.define
class Window(ABC):
    """Base window class for blinding regions.

    A Window is a callable: given bin centers X, it returns a boolean
    mask identifying which bins fall inside the blinded region.
    """

    spread: float = 1.0

    @abstractmethod
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return boolean mask: True for bins inside the window."""
        ...


@attrs.define
class GaussianWindow(Window):
    """Gaussian-shaped blinding window.

    Fits a Gaussian to the signal distribution and blinds all bins
    where the Gaussian value exceeds the value at `spread * sigma`
    from the center. Works for any dimensionality.
    """

    amplitude: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))
    center: jnp.ndarray = attrs.field(factory=lambda: jnp.array(0.0))
    sigma: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))
    theta: float | None = None  # rotation angle, 2D only
    normalization_scale: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))

    def _gaussianValue(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Gaussian at points X."""
        X_norm = X / self.normalization_scale
        ndim = X_norm.shape[-1] if X_norm.ndim > 1 else 1

        if ndim == 1:
            return self.amplitude * jnp.exp(
                -(((X_norm - self.center) / self.sigma) ** 2)
            ).ravel()
        else:
            # 2D Gaussian with optional rotation
            x, y = X_norm[..., 0], X_norm[..., 1]
            xo, yo = self.center[0], self.center[1]
            sx, sy = self.sigma[0], self.sigma[1]
            theta = self.theta or 0.0

            a = jnp.cos(theta) ** 2 / (2 * sx**2) + jnp.sin(theta) ** 2 / (
                2 * sy**2
            )
            b = -jnp.sin(2 * theta) / (4 * sx**2) + jnp.sin(2 * theta) / (
                4 * sy**2
            )
            c = jnp.sin(theta) ** 2 / (2 * sx**2) + jnp.cos(theta) ** 2 / (
                2 * sy**2
            )
            g = self.amplitude * jnp.exp(
                -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
            )
            return g

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        vals = self._gaussianValue(X)
        # Threshold: value at spread * sigma from center
        ndim = X.shape[-1] if X.ndim > 1 else 1
        if ndim == 1:
            threshold_point = (
                self.normalization_scale * (self.center + self.spread * self.sigma)
            )
            threshold = self._gaussianValue(threshold_point.reshape(1, 1))
        else:
            theta = self.theta or 0.0
            rot = jnp.array(
                [[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]
            )
            target = self.spread * (rot @ self.sigma) + self.center
            threshold = self._gaussianValue(
                (self.normalization_scale * target).reshape(1, -1)
            )
        return vals > threshold


@attrs.define
class RectangularWindow(Window):
    """Rectangular blinding window defined by lower/upper bounds."""

    lower: jnp.ndarray = attrs.field(factory=lambda: jnp.array(0.0))
    upper: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.atleast_1d(self.lower)
        upper = jnp.atleast_1d(self.upper)
        inside = jnp.all((X >= lower) & (X <= upper), axis=-1)
        return inside


@attrs.define
class EllipseWindow(Window):
    """Elliptical blinding window."""

    center: jnp.ndarray = attrs.field(factory=lambda: jnp.array([0.5, 0.5]))
    axes: jnp.ndarray = attrs.field(factory=lambda: jnp.array([0.1, 0.1]))

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        rel = ((X - self.center) ** 2) / self.axes**2
        return jnp.sum(rel, axis=-1) <= 1.0


# ---------------------------------------------------------------------------
# Window fitting
# ---------------------------------------------------------------------------


def _numpyGaussian1D(X, amplitude, xo, sigma_x):
    return amplitude * np.exp(-(((X - xo) / sigma_x) ** 2))


def _numpyGaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = X[..., 0], X[..., 1]
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (
        2 * sigma_y**2
    )
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (
        4 * sigma_y**2
    )
    c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (
        2 * sigma_y**2
    )
    return amplitude * np.exp(
        -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
    )


def fitGaussianWindow(
    signal_data: BinnedData, spread: float = 1.3
) -> GaussianWindow:
    """Fit a Gaussian to the signal distribution to define a blinding window.

    Args:
        signal_data: Signal histogram data.
        spread: Number of sigma to extend the window.

    Returns:
        A GaussianWindow fitted to the signal shape.

    Raises:
        RuntimeError: If the Gaussian fit fails.
    """
    X_np = np.asarray(signal_data.X)
    Y_np = np.asarray(signal_data.Y)

    # Normalize X for fitting stability
    scale = np.max(X_np, axis=0)
    X_norm = X_np / scale

    ndim = signal_data.ndim

    if ndim == 1:
        X_flat = X_norm.ravel()
        popt, _ = scipy.optimize.curve_fit(_numpyGaussian1D, X_flat, Y_np)
        amplitude, center, sigma = popt
        return GaussianWindow(
            amplitude=jnp.array(amplitude),
            center=jnp.array(center),
            sigma=jnp.array(abs(sigma)),
            spread=spread,
            normalization_scale=jnp.array(scale.ravel()),
        )
    elif ndim == 2:
        peak_idx = np.argmax(Y_np)
        p0 = [1.0, *X_norm[peak_idx], 0.05, 0.05, 0.0]
        popt, _ = scipy.optimize.curve_fit(_numpyGaussian2D, X_norm, Y_np, p0=p0)
        return GaussianWindow(
            amplitude=jnp.array(popt[0]),
            center=jnp.array(popt[1:3]),
            sigma=jnp.array(np.abs(popt[3:5])),
            theta=float(popt[5]),
            spread=spread,
            normalization_scale=jnp.array(scale),
        )
    else:
        raise NotImplementedError(f"Gaussian window fitting for {ndim}D not supported")
