from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import attrs
import jax.numpy as jnp
import numpy as np
from uhi.numpy_plottable import NumPyPlottableHistogram
from ..utils import dictToDot, dotFormat

logger = logging.getLogger(__name__)


def _jnpArrayValidator(instance, attribute, value):
    """Validate that the value is a JAX or numpy array."""
    if not isinstance(value, (jnp.ndarray, np.ndarray)):
        raise TypeError(
            f"{attribute.name} must be a JAX or numpy array, got {type(value)}"
        )


def _edgesValidator(instance, attribute, value):
    """Validate edges is a tuple of arrays."""
    if not isinstance(value, tuple):
        raise TypeError(
            f"{attribute.name} must be a tuple of arrays, got {type(value)}"
        )
    for i, e in enumerate(value):
        if not isinstance(e, (jnp.ndarray, np.ndarray)):
            raise TypeError(
                f"{attribute.name}[{i}] must be a JAX or numpy array, got {type(e)}"
            )


@attrs.define
class BinnedData:
    """N-dimensional binned histogram data container.

    This is the fundamental data structure passed through the pipeline.
    It holds flattened bin centers, values, and variances alongside
    the original bin edges for reconstruction.

    Attributes:
        X: Bin centers, shape (N, D) where N is the number of bins
           and D is the dimensionality.
        Y: Bin values (counts or weighted counts), shape (N,).
        V: Bin variances (for weighted histograms, sum of weights²), shape (N,).
        edges: Tuple of bin edge arrays, one per axis.
    """

    X: jnp.ndarray = attrs.field(validator=_jnpArrayValidator)
    Y: jnp.ndarray = attrs.field(validator=_jnpArrayValidator)
    V: jnp.ndarray = attrs.field(validator=_jnpArrayValidator)
    edges: tuple[jnp.ndarray, ...] = attrs.field(validator=_edgesValidator)
    axis_names: tuple[str, ...] = attrs.field(default=())

    @property
    def ndim(self) -> int:
        """Number of histogram dimensions."""
        return len(self.edges)

    @property
    def nbins(self) -> int:
        """Total number of bins (flattened)."""
        return self.X.shape[0]

    def masked(self, mask: jnp.ndarray) -> BinnedData:
        """Return a new BinnedData with only the bins where mask is True."""
        return BinnedData(
            X=self.X[mask],
            Y=self.Y[mask],
            V=self.V[mask],
            edges=self.edges,
            axis_names=self.axis_names,
        )

    def toHist(self) -> NumPyPlottableHistogram:
        """Convert back to a UHI-compatible plottable histogram."""
        return _pointsToPlottable(self.X, self.Y, self.edges, self.V)


def _pointsToPlottable(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    edges: tuple[jnp.ndarray, ...],
    V: jnp.ndarray | None = None,
) -> NumPyPlottableHistogram:
    """Reconstruct a plottable histogram from flattened bin data."""
    np_edges = tuple(np.asarray(e) for e in edges)
    np_X = np.asarray(X)
    np_Y = np.asarray(Y)

    if len(edges) == 1:
        hist_vals = np.histogram(np_X.ravel(), bins=np_edges[0], weights=np_Y)[0]
        filled = np.histogram(
            np_X.ravel(), bins=np_edges[0], weights=np.ones_like(np_Y)
        )[0].astype(bool)
        vals = np.where(filled, hist_vals, np.nan)
        variances = None
        if V is not None:
            np_V = np.asarray(V)
            var_hist = np.histogram(np_X.ravel(), bins=np_edges[0], weights=np_V)[0]
            variances = np.where(filled, var_hist, np.nan)
    else:
        hist_vals = np.histogramdd(np_X, bins=np_edges, weights=np_Y)[0]
        filled = np.histogramdd(np_X, bins=np_edges, weights=np.ones(np_Y.shape[0]))[
            0
        ].astype(bool)
        vals = np.where(filled, hist_vals, np.nan)
        variances = None
        if V is not None:
            np_V = np.asarray(V)
            var_hist = np.histogramdd(np_X, bins=np_edges, weights=np_V)[0]
            variances = np.where(filled, var_hist, np.nan)

    return NumPyPlottableHistogram(vals, *np_edges, variances=variances)


@attrs.define
class TrainingResult:
    posterior: Any
    likelihood: Any
    loss_history: list[float]
    final_loss: float
    metric_histories: dict[str, list[float]] = attrs.Factory(dict)
    samples: dict[str, jnp.ndarray] | None = None


def floatToStr(f):
    if isinstance(f, float):
        return str(f).replace(".", "p")
    return f


@attrs.define
class AnalysisState:
    # --- Config (always present) ---
    config: Any = attrs.field(factory=dict)

    background: BinnedData | None = None
    signal: BinnedData | None = None
    injection_rate: float = 0.0

    background_hist: Any | None = None
    signal_hist: Any | None = None

    train_data: BinnedData | None = None
    test_data: BinnedData | None = None
    domain_mask: jnp.ndarray | None = None
    blind_mask: jnp.ndarray | None = None
    window: Any | None = None

    transform: Any | None = None

    training_result: TrainingResult | None = None
    dataset: Any | None = None

    pred_mean: jnp.ndarray | None = None
    pred_cov: jnp.ndarray | None = None
    ppc_results: dict[str, Any] | None = None
    diagnostic_metrics: dict[str, float] | None = None

    metadata: dict[str, Any] = attrs.field(factory=dict)
    background_metadata: dict[str, Any] = attrs.field(factory=dict)

    def getRealOutPath(self):
        replace_floats = {k: floatToStr(v) for k, v in dictToDot(self.metadata)}

        return Path(dotFormat(self.config.output_dir_format, **replace_floats))
