from __future__ import annotations

from pathlib import Path
import itertools as it
import matplotlib.pyplot as plt
import mplhep
import numpy as np
from uhi.numpy_plottable import NumPyPlottableHistogram

from ..core.data import BinnedData


def addAxesToHist(ax, size=0.1, pad=0.1, position="bottom", extend=False):
    new_ax = mplhep.append_axes(ax, size, pad, position, extend)
    current_axes = getattr(ax, f"{position}_axes", [])
    setattr(ax, f"{position}_axes", current_axes + [new_ax])
    return new_ax


def plotBinnedData(ax, data: BinnedData, **kwargs):
    h = data.toHist()
    if data.ndim == 1:
        drop_keys = {"cmin", "cmax", "cmap"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in drop_keys}
        return mplhep.histplot(h, ax=ax, **clean_kwargs)
    elif data.ndim == 2:
        return mplhep.hist2dplot(h, ax=ax, flow=None, **kwargs)


def plotRaw(ax, edges, X, Y, V=None, **kwargs):
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
        h = NumPyPlottableHistogram(vals, *np_edges, variances=variances)
        drop_keys = {"cmin", "cmax", "cmap"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in drop_keys}
        return mplhep.histplot(h, ax=ax, **clean_kwargs)
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
        h = NumPyPlottableHistogram(vals, *np_edges, variances=variances)
        return mplhep.hist2dplot(h, ax=ax, flow=None, **kwargs)


def savePlots(plots: dict[str, tuple], save_dir, formats=("pdf",)):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, (fig, ax) in plots.items():
        for fmt in formats:
            # ensure format string doesn't start with a dot if provided that way
            ext = f".{fmt.lstrip('.')}"
            out = (save_dir / name).with_suffix(ext)
            fig.savefig(out, bbox_inches="tight")
        plt.close(fig)


DEFAULT_QUANTILE_LINES = (("black", 0.5),)
DEFAULT_QUANTILE_AREAS = (
    ("yellow", 0.05, 0.16),
    ("green", 0.16, 0.84),
    ("yellow", 0.84, 0.95),
)


def plotPPD(
    ax,
    dist,
    obs,
    xlabel: str = "Test Statistic",
    quantile_lines=DEFAULT_QUANTILE_LINES,
    quantile_areas=DEFAULT_QUANTILE_AREAS,
    pvalue: float | None = None,
    dist_title: str = "Posterior Predictive Distribution",
):
    import scipy.stats as stats

    dist = np.asarray(dist, dtype=float)

    density = stats.gaussian_kde(dist)
    # Match the old smoothing
    density.covariance_factor = lambda: 0.25
    density._compute_covariance()

    xs = np.linspace(dist.min(), dist.max(), 200)

    ax.plot(xs, density(xs), linewidth=3, label=dist_title)

    for color, quantile in quantile_lines:
        q = np.quantile(dist, quantile)
        y = density(q)
        ax.vlines(q, 0, y[0], color=color)

    for color, left, right in quantile_areas:
        ql = np.quantile(dist, left)
        qr = np.quantile(dist, right)
        points = xs[(xs > ql) & (xs < qr)]
        y = density(points)
        ax.fill_between(points, y, color=color, alpha=0.5)

    label = "Observed"
    if pvalue is not None:
        label += f" (p={pvalue:.3f})"
    ax.axvline(obs, 0, 1, color="red", linewidth=3, linestyle="--", label=label)

    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    try:
        mplhep.sort_legend(ax=ax)
    except Exception:
        pass


def plotBlinding2D(ax, edges, X, blind_mask, color="magenta", linewidth=2):
    mask = np.asarray(blind_mask)
    np_edges = tuple(np.asarray(e) for e in edges)
    mask_grid, _ = np.histogramdd(
        np.asarray(X), bins=np_edges, weights=mask.astype(float)
    )
    mask_grid = mask_grid.astype(bool)
    ex, ey = np_edges
    padded = np.pad(mask_grid, ((1, 1), (1, 1)), mode="constant", constant_values=False)
    for i, j in it.product(range(len(ex) - 1), range(len(ey))):
        if padded[i + 1, j] != padded[i + 1, j + 1]:
            ax.plot([ex[i], ex[i + 1]], [ey[j], ey[j]], color=color, lw=linewidth)

    for j, i in it.product(range(len(ey) - 1), range(len(ex))):
        if padded[i, j + 1] != padded[i + 1, j + 1]:
            ax.plot([ex[i], ex[i]], [ey[j], ey[j + 1]], color=color, lw=linewidth)
