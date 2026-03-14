"""Core data structures for the fitting2 pipeline.

BinnedData is the fundamental container for N-D binned histogram data.
AnalysisState is the main data interchange object passed through the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import attrs
import jax.numpy as jnp
import numpy as np
from uhi.numpy_plottable import NumPyPlottableHistogram

logger = logging.getLogger(__name__)


def _jnp_array_validator(instance, attribute, value):
    """Validate that the value is a JAX or numpy array."""
    if not isinstance(value, (jnp.ndarray, np.ndarray)):
        raise TypeError(
            f"{attribute.name} must be a JAX or numpy array, got {type(value)}"
        )


def _edges_validator(instance, attribute, value):
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

    X: jnp.ndarray = attrs.field(validator=_jnp_array_validator)
    Y: jnp.ndarray = attrs.field(validator=_jnp_array_validator)
    V: jnp.ndarray = attrs.field(validator=_jnp_array_validator)
    edges: tuple[jnp.ndarray, ...] = attrs.field(validator=_edges_validator)
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
class AnalysisState:
    """Main data interchange object passed through the entire pipeline.

    Fields are populated progressively as pipeline stages run.
    A None field means that stage hasn't executed yet, enabling
    save/resume at any point.

    Attributes:
        background: The input background histogram data.
        signal: Optional signal histogram data (None for signal-free runs).
        signal_name: Optional label for the signal.
        injection_rate: Signal injection strength (0 = no injection).
        train_data: Training data (background with blinded region removed).
        test_data: Full domain data for prediction.
        domain_mask: Boolean mask for bins within the fit domain.
        blind_mask: Boolean mask identifying the blinded window bins.
        window: The Window object used for blinding.
        transform: The normalization transform applied to training data.
        trained_params: Optimized GP model parameters (serializable dict).
        loss_history: Training loss values per iteration.
        pred_mean: Predicted mean in REAL SPACE (un-transformed bin counts).
        pred_cov: Predicted covariance in REAL SPACE.
        config: The PipelineConfig that produced this state.
        metadata: Flexible dict for any additional bookkeeping.
    """

    # --- Config (always present) ---
    config: Any = attrs.field(factory=dict)

    # --- Input data ---
    background: BinnedData | None = None
    signal: BinnedData | None = None
    signal_name: str | None = None
    injection_rate: float = 0.0

    # --- Raw histograms (with all variations for downstream use) ---
    background_hist: Any | None = None  # hist.Hist with variation axis
    signal_hist: Any | None = None  # hist.Hist with variation axis

    # --- After preprocessing ---
    train_data: BinnedData | None = None
    test_data: BinnedData | None = None
    domain_mask: jnp.ndarray | None = None
    blind_mask: jnp.ndarray | None = None
    window: Any | None = None  # Window type (forward ref to avoid circular import)

    # --- After normalization ---
    transform: Any | None = None  # DataTransformation (forward ref)

    # --- After inference ---
    trained_params: dict | None = None
    loss_history: list[float] | None = None
    samples: dict[str, jnp.ndarray] | None = None

    # --- After prediction (REAL SPACE) ---
    pred_mean: jnp.ndarray | None = None
    pred_cov: jnp.ndarray | None = None
    ppc_results: dict[str, Any] | None = None

    # --- Flexible bookkeeping ---
    metadata: dict[str, Any] = attrs.Factory(dict)
