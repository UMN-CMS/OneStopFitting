from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
import attrs
from typing import Any, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def iterSummaryFiles(inputs: Iterable[str | Path]) -> Iterator[Path]:
    """Yield `summary.json` paths under given inputs.

    Inputs may be:
    - A directory (searched recursively for summary.json)
    - A file path to summary.json
    - A glob pattern (e.g. '**/summary.json')
    """
    seen: set[Path] = set()
    for inp in inputs:
        inp_str = str(inp)
        expanded: list[Path]

        if any(ch in inp_str for ch in ["*", "?", "["]):
            expanded = [Path(p) for p in glob.glob(inp_str, recursive=True)]
            if not expanded:
                logger.warning(f"No matches for input pattern: {inp_str}")
        else:
            expanded = [Path(inp_str)]

        for path in expanded:
            if path.is_dir():
                for summary_path in path.rglob("summary.json"):
                    resolved = summary_path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        yield resolved
            else:
                if path.name != "summary.json":
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield resolved


def readSummary(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return data


def getByDotpath(dct: dict[str, Any], dotpath: str) -> Any:
    cur: Any = dct
    for part in dotpath.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"Cannot descend into non-dict at '{part}'")
        if part not in cur:
            raise KeyError(part)
        cur = cur[part]
    return cur


@attrs.define(frozen=True)
class AggregatePoint:
    mstop: float
    mchi: float
    value: float
    source: Path


def collectPoints(
    summary_files: Iterable[Path],
    *,
    metric_dotpath: str,
    stop_dotpath: str = "metadata.other_data.stop_mass",
    chi_dotpath: str = "metadata.other_data.chargino_mass",
) -> list[AggregatePoint]:
    points: list[AggregatePoint] = []
    for path in summary_files:
        try:
            summary = readSummary(path)
            mstop = float(getByDotpath(summary, stop_dotpath))
            mchi = float(getByDotpath(summary, chi_dotpath))
            value_raw = getByDotpath(summary, metric_dotpath)
            value = float(value_raw)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue
        points.append(AggregatePoint(mstop=mstop, mchi=mchi, value=value, source=path))
    return points


def _edgesFromCenters(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    centers = np.unique(centers)
    centers = np.sort(centers)

    if centers.size == 1:
        c0 = centers[0]
        return np.array([c0 - 1.0, c0 + 1.0], dtype=float)

    mids = 0.5 * (centers[1:] + centers[:-1])
    first = centers[0] - 0.5 * (centers[1] - centers[0])
    last = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.concatenate([[first], mids, [last]])


def makeAggregateMassPlanePlot(
    points: list[AggregatePoint],
    *,
    metric_name: str,
    title: str | None = None,
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    smooth_sigma: float | None = None,
    smooth_truncate: float = 4.0,
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    xs = np.array([p.mstop for p in points], dtype=float)
    ys = np.array([p.mchi for p in points], dtype=float)
    vs = np.array([p.value for p in points], dtype=float)

    x_unique = np.unique(xs)
    y_unique = np.unique(ys)

    grid_n = int(x_unique.size * y_unique.size)
    uniq_pairs = {(p.mstop, p.mchi) for p in points}
    has_full_grid = (len(points) == grid_n) and (len(uniq_pairs) == len(points))

    fig, ax = plt.subplots(layout="tight")
    vmin = cmin if cmin is not None else None
    vmax = cmax if cmax is not None else None

    if (smooth_sigma is None) and (not has_full_grid):
        sc = ax.scatter(
            xs,
            ys,
            c=vs,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            marker="s",
            s=70,
            linewidths=0.0,
        )
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(metric_name)
    else:
        # Grid the points (even if sparse) to support smoothing and/or heatmap rendering.
        x_edges = _edgesFromCenters(x_unique)
        y_edges = _edgesFromCenters(y_unique)
        grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)

        x_index = {float(x): i for i, x in enumerate(x_unique)}
        y_index = {float(y): i for i, y in enumerate(y_unique)}
        for p in points:
            grid[y_index[float(p.mchi)], x_index[float(p.mstop)]] = float(p.value)

        if smooth_sigma is not None and smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter

            finite = np.isfinite(grid)
            values0 = np.where(finite, grid, 0.0)
            weights = finite.astype(float)

            values_s = gaussian_filter(
                values0, sigma=smooth_sigma, truncate=smooth_truncate
            )
            weights_s = gaussian_filter(
                weights, sigma=smooth_sigma, truncate=smooth_truncate
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                grid = np.where(weights_s > 0, values_s / weights_s, np.nan)

        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            grid,
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(metric_name)

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")
    ax.set_title(title or f"Aggregate: {metric_name}")

    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in metric_name
    )
    return {f"aggregate_{safe}": (fig, ax)}
