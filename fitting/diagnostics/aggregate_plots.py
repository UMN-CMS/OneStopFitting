from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from fitting.utils import dictToDot, dotFormat, commonDict
from collections import defaultdict
import attrs
from typing import Any, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

logger = logging.getLogger(__name__)

plt.rcParams["figure.constrained_layout.use"] = True

PVALUE_BOUNDARIES = [0, 0.05, 0.16, 0.84, 0.95, 1]
PVALUE_COLORS = ["red", "yellow", "green", "yellow", "red"]
PVALUE_CMAP = ListedColormap(PVALUE_COLORS)
PVALUE_NORM = BoundaryNorm(PVALUE_BOUNDARIES, PVALUE_CMAP.N)


def iterSummaryFiles(inputs: Iterable[str | Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for inp in inputs:
        inp_str = str(inp)

        if any(ch in inp_str for ch in ["*", "?", "["]):
            expanded = [Path(p) for p in Path(".").glob(inp_str)]
            if not expanded:
                logger.warning(f"No matches for input pattern: {inp_str}")
        else:
            expanded = [Path(inp_str)]

        for path in expanded:
            # if path.is_dir():
            #     for summary_path in path.rglob("summary.json"):
            #         resolved = summary_path.resolve()
            #         if resolved not in seen:
            #             seen.add(resolved)
            #             yield resolved
            # else:
            # if path.name != "summary.json":
            #     continue
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
    value: Any
    source: Path | None = None
    groups: dict | None = None
    metadata: dict | None = None


@attrs.define(frozen=True)
class MultiPoint:
    mstop: float
    mchi: float
    value: list[Any]
    stats: dict
    source: list[Path] | None = None
    groups: dict | None = None
    metadata: dict | None = None


def collectPoints(
    summary_files: Iterable[Path],
    *,
    metric_dotpath: str | tuple[str,...],
    group_by: list[str] | None = None,
    stop_dotpath: str = "metadata.other_data.stop_mass",
    chi_dotpath: str = "metadata.other_data.chargino_mass",
) -> dict[tuple[tuple[str, Any], ...], list[AggregatePoint]]:
    points = defaultdict(list)
    for path in summary_files:
        try:
            summary = readSummary(path)
            dotted = dict(dictToDot(summary))
            if group_by:
                key = tuple((x, dotted[x]) for x in group_by)
            else:
                key = tuple()
            mstop = float(getByDotpath(summary, stop_dotpath))
            mchi = float(getByDotpath(summary, chi_dotpath))

            if isinstance(metric_dotpath, str):
                value_raw = getByDotpath(summary, metric_dotpath)
                value = float(value_raw)
            else:
                value_raw = tuple(float(getByDotpath(summary, d)) for d in metric_dotpath)
                value = value_raw[0] if len(value_raw) == 1 else value_raw

            points[key].append(
                AggregatePoint(
                    mstop=mstop,
                    mchi=mchi,
                    value=value,
                    source=path,
                    groups=dict(key),
                    metadata=summary["metadata"],
                )
            )
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    return dict(points)


def computeStatistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    arr = np.array(values, dtype=float)

    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q05": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
        "n": len(arr),
    }

    return stats


def makeMulti(points):
    grouped = defaultdict(list)

    for p in points:
        k = (p.mstop, p.mchi)
        grouped[k].append(p)

    ret = []
    for k, group in grouped.items():
        values = ([x.value for x in group],)
        ret.append(
            MultiPoint(
                *k,
                values,
                computeStatistics(values),
                [x.source for x in group],
                group[0].groups,
                commonDict([x.metadata for x in group]),
            )
        )
    return ret


def makeAggregateMassPlanePlot(
    points: list[AggregatePoint],
    *,
    metric_name: str,
    get_value_func=lambda x: x.stats["median"],
    title: str | None = None,
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    smooth_sigma: float | None = None,
    smooth_truncate: float = 4.0,
    params: dict[str, Any] | None = None,
    name_format: str = "aggregate_{metric}",
    draw_contours: tuple[float, ...] | None = None,
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    xs = np.array([p.mstop for p in points], dtype=float)
    ys = np.array([p.mchi for p in points], dtype=float)
    vs = np.array([get_value_func(p) for p in points], dtype=float)
    fig, ax = plt.subplots()
    actual_norm = None
    if "pvalue" in metric_name.lower() and cmap == "viridis":
        cmap = PVALUE_CMAP
        actual_norm = PVALUE_NORM
        if cmin is None:
            cmin = 0.0
        if cmax is None:
            cmax = 1.0
    vmin = cmin if cmin is not None else None
    vmax = cmax if cmax is not None else None
    plot_kwargs: dict[str, Any] = {
        "cmap": cmap,
        "norm": actual_norm,
    }
    if actual_norm is None:
        plot_kwargs["vmin"] = vmin
        plot_kwargs["vmax"] = vmax

    sc = ax.scatter(
        xs,
        ys,
        c=vs,
        marker="s",
        s=150,
        linewidths=0.0,
        edgecolors="black",
        **plot_kwargs,
    )
    cb = fig.colorbar(sc, ax=ax)

    cb.set_label(metric_name)
    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")

    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}

def makeAggregateSmoothPlot(
    points: list[AggregatePoint],
    *,
    metric_name: str,
    get_value_func=lambda x: x.stats["median"],
    title: str | None = None,
    cmap: str = "viridis",
    cmin: float | None = None,
    cmax: float | None = None,
    smooth_sigma: float | None = None,
    smooth_truncate: float = 4.0,
    params: dict[str, Any] | None = None,
    name_format: str = "aggregate_smooth_{metric}",
    draw_contours: tuple[float, ...] | None = (1.0,2.0),
) -> dict[str, tuple]:
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import CloughTocher2DInterpolator


    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    xs = np.array([p.mstop for p in points], dtype=float)
    ys = np.array([p.mchi for p in points], dtype=float)
    vs = np.array([get_value_func(p) for p in points], dtype=float)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()


    fig, ax = plt.subplots()
    actual_norm = None
    if "pvalue" in metric_name.lower() and cmap == "viridis":
        cmap = PVALUE_CMAP
        actual_norm = PVALUE_NORM
        if cmin is None:
            cmin = 0.0
        if cmax is None:
            cmax = 1.0

    vmin = cmin if cmin is not None else None
    vmax = cmax if cmax is not None else None

    xls = np.linspace(x_min, x_max, num=200)
    yls = np.linspace(y_min, y_max, num=200)
    
    X,Y=np.meshgrid(xls,yls)
    interp = CloughTocher2DInterpolator(list(zip(xs,ys)), vs)
    Z = interp(X, Y)
    
    plot_kwargs: dict[str, Any] = {
        "cmap": cmap,
        "norm": actual_norm,
        "shading": "auto",
    }
    if actual_norm is None:
        plot_kwargs["vmin"] = vmin
        plot_kwargs["vmax"] = vmax

    mesh = ax.pcolormesh(X, Y, Z, **plot_kwargs)

    if draw_contours:
        for level in draw_contours:
            ax.contour(X, Y, Z, [level], colors="k", linewidths=2)

    cb = fig.colorbar(mesh, ax=ax)

    cb.set_label(metric_name)
    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")
    #ax.set_title(title or f"Aggregate: {metric_name}")


    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}



def makeAggregateViolinPlot(
    points: list[MultiPoint],
    *,
    metric_name: str,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    name_format: str = "aggregate_violin_{metric_name}",
    vlines: tuple[float,...] | None = (0.0,),
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    fig, ax = plt.subplots(figsize=(12, max(6, len(points) * 0.4)))

    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color="black", linestyle='--', linewidth=1)

    sorted_points = sorted(points, key=lambda p: (p.mstop, p.mchi))

    labels = []
    data = []
    for p in sorted_points:
        labels.append(f"({p.mstop}, {p.mchi})")
        vals = p.value[0]
        if vals and isinstance(vals[0], tuple):
            vals = [v[0] for v in vals]
        data.append(vals)

    ax.violinplot(data, vert=False, showmeans=True)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(metric_name)
    ax.set_ylabel(r"$(m_{\tilde{t}}, m_{\tilde{\chi}^{\pm}})$ [GeV]")
    ax.set_title(title or f"Aggregate Violin: {metric_name}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}


def makeAggregateScatterPlot(
    points: list[MultiPoint],
    *,
    metric_name: str,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    name_format: str = "aggregate_scatter_{metric_name}",
    vlines: tuple[float,...] | None = (0.0,),
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    n_points = len(points)
    sorted_points = sorted(points, key=lambda p: (p.mstop, p.mchi))
    
    fig, ax = plt.subplots(figsize=(12, max(8, n_points * 0.6)))

    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color='red', linestyle='--', linewidth=1, zorder=0)

    rng = np.random.default_rng(42)
    
    labels = []
    for i, p in enumerate(sorted_points):
        y_center = n_points - i
        
        vals = p.value[0]
        mstop_str = f"{p.mstop:.0f}" if p.mstop == int(p.mstop) else f"{p.mstop:.1f}"
        mchi_str = f"{p.mchi:.0f}" if p.mchi == int(p.mchi) else f"{p.mchi:.1f}"
        labels.append(f"({mstop_str}, {mchi_str})")

        if not vals:
            continue
            
        if isinstance(vals[0], (tuple, list, np.ndarray)):
            x = np.array([v[0] for v in vals])
        else:
            x = np.array(vals)

        y_jitter = y_center + rng.uniform(-0.3, 0.3, size=len(x))
        plot_color = 'black'

        if isinstance(vals[0], (tuple, list, np.ndarray)) and len(vals[0]) >= 2:
            xerr = np.array([v[1] for v in vals])
            ax.errorbar(x, y_jitter, xerr=xerr, fmt='o', alpha=0.4, markersize=3, elinewidth=1, color=plot_color)
        else:
            ax.scatter(x, y_jitter, alpha=0.4, s=10, color=plot_color)
        
        bg_color = '#fdfdfd' if i % 2 == 0 else '#f5f5f5'
        ax.axhspan(y_center - 0.5, y_center + 0.5, color=bg_color, zorder=-2)

    ax.set_yticks(np.arange(1, n_points + 1))
    ax.set_yticklabels(reversed(labels))
    ax.tick_params(axis='y', which='both', length=0)
    ax.set_ylabel("")
    
    ax.set_xlabel(metric_name)
    ax.set_ylim(0.5, n_points + 1.2)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}
