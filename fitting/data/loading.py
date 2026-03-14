"""Histogram loading from various file formats.

Provides a polymorphic FileLoader hierarchy for loading histogram
files, plus free functions for extracting histograms, metadata,
and converting to BinnedData.

Supported formats:
    - .pkl / .pickle — plain pickle
    - .pklz4 / .pkl.lz4 — lz4-compressed pickle

The expected file structure is ``{'metadata': dict, 'item': hist.Hist}``,
where the hist has an optional StrCategory ``'variation'`` axis.
"""

from __future__ import annotations

import logging
import pickle as pkl
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import attrs
import hist
import jax.numpy as jnp
import lz4.frame
import numpy as np

from ..core.data import BinnedData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Polymorphic file loaders
# ---------------------------------------------------------------------------


@attrs.define
class FileLoader(ABC):
    """Base file loader.

    Subclass to add support for new file formats (ROOT, HDF5, etc.).
    Uses cattrs include_subclasses for polymorphic serialization.
    """

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load the raw content of a file.

        Args:
            path: Path to the file.

        Returns:
            The deserialized object (typically a dict or hist.Hist).
        """
        ...

    @staticmethod
    def forPath(path: Path) -> FileLoader:
        """Auto-detect the appropriate loader from file extension.

        Args:
            path: Path to the file.

        Returns:
            A FileLoader instance for the detected format.

        Raises:
            ValueError: If the format is not recognized.
        """
        name = Path(path).name.lower()
        if name.endswith(".lz4") or name.endswith(".pklz4"):
            return Lz4PickleLoader()
        elif name.endswith(".pkl") or name.endswith(".pickle"):
            return PickleLoader()
        else:
            raise ValueError(f"Unknown file format: {path}")


@attrs.define
class PickleLoader(FileLoader):
    """Plain pickle file loader."""

    def load(self, path: Path) -> Any:
        path = Path(path)
        with open(path, "rb") as f:
            return pkl.load(f)


@attrs.define
class Lz4PickleLoader(FileLoader):
    """LZ4-compressed pickle file loader."""

    def load(self, path: Path) -> Any:
        path = Path(path)
        with lz4.frame.open(path, "rb") as f:
            return pkl.load(f)


# ---------------------------------------------------------------------------
# Histogram / metadata extraction
# ---------------------------------------------------------------------------


def extractHistogram(raw_data: Any, key: str | None = None) -> hist.Hist:
    """Extract a hist.Hist from loaded file data.

    Handles:
    - Direct hist.Hist objects
    - ``{'metadata': ..., 'item': hist.Hist}`` structure
    - Dict with ``{'hist': ..., 'params': ...}`` entries
    - Dict keyed by strings or tuples

    Args:
        raw_data: The object loaded from file.
        key: Optional key to select a specific entry from a dict.

    Returns:
        A hist.Hist object.

    Raises:
        KeyError: If key is not found.
        ValueError: If the data format is not recognized.
    """
    if isinstance(raw_data, hist.Hist):
        return raw_data

    if isinstance(raw_data, dict):
        # Handle {'metadata': ..., 'item': hist.Hist} structure
        if "item" in raw_data and isinstance(raw_data["item"], hist.Hist):
            return raw_data["item"]

        entry = _findEntry(raw_data, key)
        if isinstance(entry, hist.Hist):
            return entry
        elif isinstance(entry, dict):
            if "item" in entry and isinstance(entry["item"], hist.Hist):
                return entry["item"]
            elif "hist" in entry:
                return entry["hist"]
        raise ValueError(
            f"Cannot extract histogram from entry of type {type(entry)}"
        )

    raise ValueError(f"Cannot extract histogram from data of type {type(raw_data)}")


def extractMetadata(raw_data: Any, key: str | None = None) -> dict:
    """Extract metadata from loaded file data.

    Recognizes:
    - ``{'metadata': dict, 'item': ...}`` — returns the ``'metadata'`` dict
    - ``{'hist': ..., 'params': dict}`` — returns the ``'params'`` dict

    Args:
        raw_data: The object loaded from file.
        key: Optional key to select a specific entry from a dict.

    Returns:
        Dict of metadata, or empty dict if none found.
    """
    if isinstance(raw_data, dict):
        # Top-level {'metadata': ..., 'item': ...}
        if "metadata" in raw_data:
            return raw_data["metadata"]
        # Try entry-level
        entry = _findEntry(raw_data, key)
        if isinstance(entry, dict):
            if "metadata" in entry:
                return entry["metadata"]
            return entry.get("params", {})
    return {}


def _findEntry(data: dict, key: str | None) -> Any:
    """Find an entry in a dict by key, with flexible matching."""
    if key is not None:
        if key in data:
            return data[key]
        for k, v in data.items():
            if isinstance(k, tuple) and key in k:
                return v
            elif isinstance(k, str) and key in k:
                return v
        raise KeyError(f"Key '{key}' not found in file data")
    else:
        return next(iter(data.values()))


# ---------------------------------------------------------------------------
# Variation axis handling
# ---------------------------------------------------------------------------


def hasVariationAxis(histogram: hist.Hist) -> bool:
    """Check whether the histogram has a StrCategory 'variation' axis."""
    for ax in histogram.axes:
        if isinstance(ax, hist.axis.StrCategory) and ax.name == "variation":
            return True
    return False


def variationNames(histogram: hist.Hist) -> list[str]:
    """Return the list of variation names if a variation axis exists.

    Returns an empty list if no variation axis is present.
    """
    for ax in histogram.axes:
        if isinstance(ax, hist.axis.StrCategory) and ax.name == "variation":
            return list(ax)
    return []


def sliceVariation(histogram: hist.Hist, variation: str = "central") -> hist.Hist:
    """Slice the variation axis, returning a hist without that axis.

    If the histogram has no variation axis, returns it unchanged.

    Args:
        histogram: A hist.Hist potentially containing a StrCategory
            ``'variation'`` axis.
        variation: Name of the variation to select.

    Returns:
        A hist.Hist with the variation axis removed.

    Raises:
        KeyError: If the named variation is not present.
    """
    if not hasVariationAxis(histogram):
        return histogram
    return histogram[{"variation": variation}]


# ---------------------------------------------------------------------------
# Conversion to BinnedData
# ---------------------------------------------------------------------------


def histToBinnedData(
    histogram: hist.Hist,
    rebin: int = 1,
    variation: str = "central",
) -> BinnedData:
    """Convert a hist.Hist to a BinnedData.

    If the histogram has a StrCategory ``'variation'`` axis, it is
    sliced to the requested variation first.

    Args:
        histogram: A hist.Hist with Weight() storage.
        rebin: Rebin factor (merge every N bins along each axis).
        variation: Which variation to extract (default: 'central').

    Returns:
        BinnedData with flattened bin centers, values, variances, and edges.
    """
    # Slice variation axis if present
    h = sliceVariation(histogram, variation)

    if rebin > 1:
        for ax_idx in range(len(h.axes)):
            h = h[:: hist.rebin(rebin)]

    edges = tuple(jnp.array(a.edges) for a in h.axes)
    centers = tuple(jnp.array(jnp.diff(e) / 2 + e[:-1]) for e in edges)
    axis_names = tuple(a.name for a in h.axes)

    bin_values = jnp.array(h.values())
    bin_vars = jnp.array(h.variances())

    # Create meshgrid of bin centers
    if len(edges) == 1:
        flat_centers = centers[0].reshape(-1, 1)
        flat_values = bin_values.ravel()
        flat_vars = bin_vars.ravel()
    else:
        centers_grid = jnp.meshgrid(*centers, indexing="ij")
        centers_stacked = jnp.stack(centers_grid, axis=-1)
        flat_centers = centers_stacked.reshape(-1, len(edges))
        flat_values = bin_values.ravel()
        flat_vars = bin_vars.ravel()

    return BinnedData(
        X=flat_centers,
        Y=flat_values,
        V=flat_vars,
        edges=edges,
        axis_names=axis_names,
    )
