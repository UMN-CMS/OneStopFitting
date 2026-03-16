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

from ..core.data import BinnedData

logger = logging.getLogger(__name__)


@attrs.define
class FileLoader(ABC):
    @abstractmethod
    def load(self, path: Path) -> Any: ...

    @staticmethod
    def forPath(path: Path) -> FileLoader:
        name = Path(path).name.lower()
        if name.endswith(".lz4") or name.endswith(".pklz4"):
            return Lz4PickleLoader()
        elif name.endswith(".pkl") or name.endswith(".pickle"):
            return PickleLoader()
        else:
            raise ValueError(f"Unknown file format: {path}")


@attrs.define
class PickleLoader(FileLoader):
    def load(self, path: Path) -> Any:
        path = Path(path)
        with open(path, "rb") as f:
            return pkl.load(f)


@attrs.define
class Lz4PickleLoader(FileLoader):
    def load(self, path: Path) -> Any:
        path = Path(path)
        with lz4.frame.open(path, "rb") as f:
            return pkl.load(f)


def extractHistogram(raw_data: Any, key: str | None = None) -> hist.Hist:
    if isinstance(raw_data, hist.Hist):
        return raw_data

    if isinstance(raw_data, dict):
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
        raise ValueError(f"Cannot extract histogram from entry of type {type(entry)}")

    raise ValueError(f"Cannot extract histogram from data of type {type(raw_data)}")


def extractMetadata(raw_data: Any, key: str | None = None) -> dict:
    if isinstance(raw_data, dict):
        if "metadata" in raw_data:
            return raw_data["metadata"]
        entry = _findEntry(raw_data, key)
        if isinstance(entry, dict):
            if "metadata" in entry:
                return entry["metadata"]
            return entry.get("params", {})
    return {}


def _findEntry(data: dict, key: str | None) -> Any:
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


def hasVariationAxis(histogram: hist.Hist) -> bool:
    for ax in histogram.axes:
        if isinstance(ax, hist.axis.StrCategory) and ax.name == "variation":
            return True
    return False


def variationNames(histogram: hist.Hist) -> list[str]:
    for ax in histogram.axes:
        if isinstance(ax, hist.axis.StrCategory) and ax.name == "variation":
            return list(ax)
    return []


def sliceVariation(histogram: hist.Hist, variation: str = "central") -> hist.Hist:
    if not hasVariationAxis(histogram):
        return histogram
    return histogram[{"variation": variation}]


def histToBinnedData(
    histogram: hist.Hist,
    rebin: int = 1,
    variation: str = "central",
) -> BinnedData:
    h = sliceVariation(histogram, variation)

    if rebin > 1:
        rebin_slice = tuple(slice(None, None, hist.rebin(rebin)) for _ in h.axes)
        h = h[rebin_slice]

    edges = tuple(jnp.array(a.edges) for a in h.axes)
    centers = tuple(jnp.array(jnp.diff(e) / 2 + e[:-1]) for e in edges)
    axis_names = tuple(a.name for a in h.axes)

    bin_values = jnp.array(h.values())
    bin_vars = jnp.array(h.variances())

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
