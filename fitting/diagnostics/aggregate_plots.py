from __future__ import annotations

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
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield resolved


def readSummary(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
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


def transformRToCoupling(value: Any, summary: dict[str, Any]) -> Any:
    pre_scale = float(getByDotpath(summary, "config.signal_pre_scale"))
    base_coupling = 0.1
    used_coupling = base_coupling * np.sqrt(pre_scale)
    if isinstance(value, (tuple, list)):
        return type(value)(used_coupling * np.sqrt(v) for v in value)
    return used_coupling * np.sqrt(value)


transformRegistry = {
    "r_to_coupling": transformRToCoupling,
}


def _handleOneSummary(
    points,
    summary,
    path,
    metric_dotpath: str | tuple[str, ...],
    group_by: list[str] | None = None,
    stop_dotpath: str = "metadata.other_data.stop_mass",
    chi_dotpath: str = "metadata.other_data.chargino_mass",
    transform_name: str | None = None,
) -> dict[tuple[tuple[str, Any], ...], list[AggregatePoint]]:
    try:
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
            value_raw = tuple(getByDotpath(summary, d) for d in metric_dotpath)
            value = value_raw[0] if len(value_raw) == 1 else value_raw

        if transform_name:
            if transform_name not in transformRegistry:
                raise ValueError(f"Unknown transform '{transform_name}'")
            value = transformRegistry[transform_name](value, summary)

        points[key].append(
            AggregatePoint(
                mstop=mstop,
                mchi=mchi,
                value=value,
                source=path,
                groups=dict(key),
                metadata=summary.get("metadata"),
            )
        )
    except Exception as e:
        logger.warning(f"Skipping {path}: {e}")


def collectPoints(
    summary_files: Iterable[Path], *args, transform_name: str | None = None, **kwargs
) -> dict[tuple[tuple[str, Any], ...], list[AggregatePoint]]:
    points = defaultdict(list)
    for path in summary_files:
        summary = readSummary(path)
        if isinstance(summary, list):
            for s in summary:
                _handleOneSummary(
                    points, s, path, *args, transform_name=transform_name, **kwargs
                )
        else:
            _handleOneSummary(
                points, summary, path, *args, transform_name=transform_name, **kwargs
            )

    return dict(points)


def computeStatistics(values: list[float], is_pvalue: bool = False) -> dict[str, Any]:
    if not values:
        return {}

    arr = np.array(values, dtype=float)

    stats: dict[str, Any] = {
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

    if is_pvalue:
        try:
            from scipy.stats import kstest

            ks_res = kstest(arr, "uniform")
            stats["ks_pvalue_uniform"] = float(ks_res.pvalue)
            stats["ks_stat_uniform"] = float(ks_res.statistic)

            frac_below_half = float(np.mean(arr < 0.5))
            stats["frac_below_half"] = frac_below_half

            if ks_res.pvalue > 0.05:
                verdict = "OK"
            elif frac_below_half > 0.55:
                verdict = "CONSERVATIVE"
            elif frac_below_half < 0.45:
                verdict = "DANGEROUS"
            else:
                verdict = "OK (marginal)"

            stats["pvalue_skew_verdict"] = verdict
        except Exception:
            pass

    return stats


def makeMulti(points, is_pvalue: bool = False):
    grouped = defaultdict(list)

    for p in points:
        k = (p.mstop, p.mchi)
        grouped[k].append(p)

    ret = []
    for k, group in grouped.items():
        values = ([x.value for x in group],)
        statistics = {}
        try:
            statistics = computeStatistics(
                values[0] if isinstance(values[0], list) else values,
                is_pvalue=is_pvalue,
            )
        except Exception:
            pass

        ret.append(
            MultiPoint(
                *k,
                values,
                statistics,
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
    draw_contours: tuple[float, ...] | None = (1.0, 2.0),
) -> dict[str, tuple]:
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
    if (
        any(x in metric_name.lower() for x in ["pvalue", "p_value"])
        and cmap == "viridis"
    ):
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

    X, Y = np.meshgrid(xls, yls)
    interp = CloughTocher2DInterpolator(list(zip(xs, ys)), vs)
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
    vlines: tuple[float, ...] | None = (0.0,),
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    fig, ax = plt.subplots(figsize=(12, max(6, len(points) * 0.4)))

    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color="black", linestyle="--", linewidth=1)

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
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}


PVALUE_BOUNDARIES = [0, 0.05, 0.16, 0.84, 0.95, 1]
PVALUE_COLORS = ["red", "yellow", "green", "yellow", "red"]


def addPValBands(ax, alpha=0.2):
    for i in range(len(PVALUE_BOUNDARIES) - 1):
        low, high = PVALUE_BOUNDARIES[i], PVALUE_BOUNDARIES[i + 1]
        color = PVALUE_COLORS[i]
        ax.axvspan(low, high, color=color, alpha=alpha)


def makeAggregateScatterPlot(
    points: list[MultiPoint],
    *,
    metric_name: str,
    title: str | None = None,
    params: dict[str, Any] | None = None,
    name_format: str = "aggregate_scatter_{metric_name}",
    vlines: tuple[float, ...] | None = (0.0,),
    xlim: tuple[float, float] | None = None,
    pval_bands: bool = False,
) -> dict[str, tuple]:
    if not points:
        raise ValueError("No points to plot.")

    params = params or {}
    n_points = len(points)
    sorted_points = sorted(points, key=lambda p: (p.mstop, p.mchi))

    fig, ax = plt.subplots(figsize=(12, max(8, n_points * 0.6)))

    if vlines is not None:
        for vline in vlines:
            ax.axvline(vline, color="red", linestyle="--", linewidth=1, zorder=0)

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
        plot_color = "black"

        median = np.median(x)
        mean = np.mean(x)
        ax.vlines(
            [median], y_center - 0.5, y_center + 0.5, color="blue", lw=0.5, ls="--"
        )
        ax.vlines(
            [mean], y_center - 0.5, y_center + 0.5, color="green", lw=0.5, ls="--"
        )

        if isinstance(vals[0], (tuple, list, np.ndarray)) and len(vals[0]) >= 2:
            xerr = np.stack([v[1] for v in vals], axis=1)

            ax.errorbar(
                x,
                y_jitter,
                xerr=xerr,
                fmt="o",
                alpha=0.4,
                markersize=3,
                elinewidth=1,
                color=plot_color,
            )
        else:
            ax.scatter(x, y_jitter, alpha=0.4, s=10, color=plot_color)

        bg_color = "#fdfdfd" if i % 2 == 0 else "#f5f5f5"
        ax.axhspan(y_center - 0.5, y_center + 0.5, color=bg_color, zorder=-2)

    if pval_bands:
        addPValBands(ax)

    ax.set_yticks(np.arange(1, n_points + 1))
    ax.set_yticklabels(reversed(labels))
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylabel("")

    ax.set_xlabel(metric_name)
    ax.set_ylim(0.5, n_points + 1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)

    n = dotFormat(name_format, metric_name=metric_name, **params)
    n = n.replace(".", "p")
    return {n: (fig, ax)}
