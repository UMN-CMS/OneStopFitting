from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import attrs
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import scipy.ndimage

from ..core.data import BinnedData

logger = logging.getLogger(__name__)


@attrs.define
class Window(ABC):
    @abstractmethod
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        """Return boolean mask: True for bins inside the window."""
        ...


@attrs.define
class AndWindow(Window):
    windows: list[Window]

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.all(jnp.array([w(X) for w in self.windows]), axis=0)


@attrs.define
class OrWindow(Window):
    windows: list[Window]

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(jnp.array([w(X) for w in self.windows]), axis=0)


@attrs.define
class NotWindow(Window):
    window: Window

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return ~self.window(X)


@attrs.define
class CutWindow(Window):
    axis: int
    lower: float = -jnp.inf
    upper: float = jnp.inf

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        return (X[:, self.axis] >= self.lower) & (X[:, self.axis] <= self.upper)





@attrs.define
class GaussianWindow(Window):
    amplitude: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))
    center: jnp.ndarray = attrs.field(factory=lambda: jnp.array(0.0))
    sigma: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))
    theta: float | None = None  # rotation angle, 2D only
    normalization_scale: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))
    spread: float = 1.0

    def _gaussianValue(self, X: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the Gaussian at points X."""
        X_norm = X / self.normalization_scale
        ndim = X_norm.shape[-1] if X_norm.ndim > 1 else 1

        if ndim == 1:
            return (
                self.amplitude
                * jnp.exp(-(((X_norm - self.center) / self.sigma) ** 2)).ravel()
            )
        else:
            # 2D Gaussian with optional rotation
            x, y = X_norm[..., 0], X_norm[..., 1]
            xo, yo = self.center[0], self.center[1]
            sx, sy = self.sigma[0], self.sigma[1]
            theta = self.theta or 0.0

            a = jnp.cos(theta) ** 2 / (2 * sx**2) + jnp.sin(theta) ** 2 / (2 * sy**2)
            b = -jnp.sin(2 * theta) / (4 * sx**2) + jnp.sin(2 * theta) / (4 * sy**2)
            c = jnp.sin(theta) ** 2 / (2 * sx**2) + jnp.cos(theta) ** 2 / (2 * sy**2)
            g = self.amplitude * jnp.exp(
                -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
            )
            return g

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        vals = self._gaussianValue(X)
        # Threshold: value at spread * sigma from center
        ndim = X.shape[-1] if X.ndim > 1 else 1
        if ndim == 1:
            threshold_point = self.normalization_scale * (
                self.center + self.spread * self.sigma
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
    lower: jnp.ndarray = attrs.field(factory=lambda: jnp.array(0.0))
    upper: jnp.ndarray = attrs.field(factory=lambda: jnp.array(1.0))

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        lower = jnp.atleast_1d(self.lower)
        upper = jnp.atleast_1d(self.upper)
        inside = jnp.all((X >= lower) & (X <= upper), axis=-1)
        return inside


@attrs.define
class EllipseWindow(Window):
    center: jnp.ndarray = attrs.field(factory=lambda: jnp.array([0.5, 0.5]))
    axes: jnp.ndarray = attrs.field(factory=lambda: jnp.array([0.1, 0.1]))

    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        rel = ((X - self.center) ** 2) / self.axes**2
        return jnp.sum(rel, axis=-1) <= 1.0




 
@attrs.define
class ConvexHullWindow(Window):
    hull_vertices: np.ndarray = attrs.field()
    ndim: int = attrs.field()
 
    def __call__(self, X: jnp.ndarray) -> jnp.ndarray:
        X_np = np.asarray(X)
 
        if self.ndim == 1:
            lo = self.hull_vertices[:, 0].min()
            hi = self.hull_vertices[:, 0].max()
            return jnp.array((X_np[:, 0] >= lo) & (X_np[:, 0] <= hi))
        elif self.ndim == 2:
            from scipy.spatial import ConvexHull, Delaunay
            hull = ConvexHull(self.hull_vertices)
            tri = Delaunay(self.hull_vertices[hull.vertices])
            inside = tri.find_simplex(X_np) >= 0
            return jnp.array(inside)
        else:
            lo = self.hull_vertices.min(axis=0)
            hi = self.hull_vertices.max(axis=0)
            return jnp.array(np.all((X_np >= lo) & (X_np <= hi), axis=-1))
 










def _numpyGaussian1D(X, amplitude, xo, sigma_x):
    return amplitude * np.exp(-(((X - xo) / sigma_x) ** 2))


def _numpyGaussian2D(X, amplitude, xo, yo, sigma_x, sigma_y, theta):
    x, y = X[..., 0], X[..., 1]
    a = np.cos(theta) ** 2 / (2 * sigma_x**2) + np.sin(theta) ** 2 / (2 * sigma_y**2)
    b = -np.sin(2 * theta) / (4 * sigma_x**2) + np.sin(2 * theta) / (4 * sigma_y**2)
    c = np.sin(theta) ** 2 / (2 * sigma_x**2) + np.cos(theta) ** 2 / (2 * sigma_y**2)
    return amplitude * np.exp(
        -(a * (x - xo) ** 2 + 2 * b * (x - xo) * (y - yo) + c * (y - yo) ** 2)
    )


def fitGaussianWindow(signal_data: BinnedData, spread: float = 1.3) -> GaussianWindow:
    X_np = np.asarray(signal_data.X)
    Y_np = np.asarray(signal_data.Y)

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



def _smoothedGrid2D(
    X_np: np.ndarray, Y_np: np.ndarray, smooth_sigma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0_vals = np.unique(X_np[:, 0])
    x1_vals = np.unique(X_np[:, 1])
 
    x0_idx = np.searchsorted(x0_vals, X_np[:, 0])
    x1_idx = np.searchsorted(x1_vals, X_np[:, 1])
 
    grid = np.zeros((len(x0_vals), len(x1_vals)))
    for i, (i0, i1) in enumerate(zip(x0_idx, x1_idx)):
        grid[i0, i1] = Y_np[i]
 
    Y_smooth = scipy.ndimage.gaussian_filter(grid.astype(float), smooth_sigma)
    return x0_vals, x1_vals, Y_smooth
 



def _dilateHull(vertices: np.ndarray, margin: float) -> np.ndarray:
    centroid = vertices.mean(axis=0)
    directions = vertices - centroid
    radii = np.linalg.norm(directions, axis=1, keepdims=True)
    safe_radii = np.where(radii > 0, radii, 1.0)
    unit_dirs = directions / safe_radii
    mean_radius = radii.mean()
    return vertices + unit_dirs * (margin * mean_radius)
 
 
 
def fitCoreDilatedWindow(
    signal_data: BinnedData,
    core_threshold_fraction: float = 0.08,
    smooth_sigma: float = 1.5,
    dilation_margin: float = 0.25,
) -> ConvexHullWindow:
    X_np = np.asarray(signal_data.X)
    Y_np = np.asarray(signal_data.Y)
    ndim = signal_data.ndim
 
    # Normalise each axis to [0, 1]
    x_min = X_np.min(axis=0)
    x_max = X_np.max(axis=0)
    x_range = np.where(x_max > x_min, x_max - x_min, 1.0)
    X_norm = (X_np - x_min) / x_range
 
    if ndim == 1:
        order = np.argsort(X_norm[:, 0])
        X_sorted = X_norm[order]
        Y_sorted = Y_np[order]
 
        Y_smooth = scipy.ndimage.gaussian_filter1d(Y_sorted.astype(float), smooth_sigma)
        peak = Y_smooth.max()
        if peak <= 0:
            raise ValueError("Signal has non-positive peak after smoothing.")
 
        mask = Y_smooth >= core_threshold_fraction * peak
        selected = X_sorted[mask]
        if len(selected) == 0:
            raise ValueError(
                f"No bins survive core_threshold_fraction={core_threshold_fraction}."
            )
 
        lo = selected[:, 0].min()
        hi = selected[:, 0].max()
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 * (1.0 + dilation_margin)
        hull_norm = np.array([[center - half], [center + half]])
 
        logger.info(
            f"CoreDilatedWindow 1D: core [{lo:.3f}, {hi:.3f}] → "
            f"dilated [{center - half:.3f}, {center + half:.3f}] (normalised), "
            f"{mask.sum()}/{len(mask)} core bins"
        )
 
    elif ndim == 2:
        from scipy.spatial import ConvexHull
 
        x0_vals_raw, x1_vals_raw, _ = _smoothedGrid2D(X_np, Y_np, smooth_sigma)
        x0_vals = (x0_vals_raw - x_min[0]) / x_range[0]
        x1_vals = (x1_vals_raw - x_min[1]) / x_range[1]
        _, _, Y_smooth = _smoothedGrid2D(X_norm, Y_np, smooth_sigma)
 
        peak = Y_smooth.max()
        if peak <= 0:
            raise ValueError("Signal has non-positive peak after smoothing.")
 
        mask_grid = Y_smooth >= core_threshold_fraction * peak
        n_core = mask_grid.sum()
        if n_core == 0:
            raise ValueError(
                f"No bins survive core_threshold_fraction={core_threshold_fraction}."
            )
 
        i0s, i1s = np.where(mask_grid)
        core_pts = np.stack([x0_vals[i0s], x1_vals[i1s]], axis=-1)
 
        if len(core_pts) < 3:
            lo = core_pts.min(axis=0)
            hi = core_pts.max(axis=0)
            center = (lo + hi) / 2.0
            half = (hi - lo) / 2.0 * (1.0 + dilation_margin)
            hull_norm = np.array([
                center + np.array([ half[0],  half[1]]),
                center + np.array([-half[0],  half[1]]),
                center + np.array([-half[0], -half[1]]),
                center + np.array([ half[0], -half[1]]),
            ])
        else:
            hull = ConvexHull(core_pts)
            core_verts = core_pts[hull.vertices]
            hull_norm = _dilateHull(core_verts, dilation_margin)
 
        logger.info(
            f"CoreDilatedWindow 2D: {n_core}/{mask_grid.size} core bins, "
            f"hull {len(hull_norm)} vertices, dilation={dilation_margin:.2f}"
        )
 
    else:
        raise NotImplementedError(f"fitCoreDilatedWindow not implemented for ndim={ndim}")
    hull_vertices = hull_norm * x_range + x_min
    return ConvexHullWindow(hull_vertices=hull_vertices, ndim=ndim)
