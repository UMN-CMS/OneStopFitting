from __future__ import annotations

from pathlib import Path
import itertools as it
import matplotlib.pyplot as plt
import logging
import mplhep
import numpy as np
from uhi.numpy_plottable import NumPyPlottableHistogram

from ..core.data import BinnedData

CMS_COLORS = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


logger = logging.getLogger(__name__)

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
    logger.info(f"Saved {len(plots)} to directory {save_dir}")


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


def plotFitDiagnostic(
    data: BinnedData,
    prefit_b: BinnedData,
    b_background: BinnedData | None,
    s_background: BinnedData | None,
    s_signal: BinnedData | None = None,
    signal_rate: float | None = None,
    title: str = "",
    show_signal=True,
    xlabel: str = "Bin Index",
    log: bool = True,
):
    gs_kw = dict(height_ratios=[3, 1, 1,1])
    fig, (ax, rax, pax,sbax) = plt.subplots(
        figsize=(12, 16), nrows=4, sharex=True, gridspec_kw=gs_kw, layout="tight"
    )

    color_data = "black"
    color_prefit_b = CMS_COLORS[0]
    color_b_background = CMS_COLORS[1]
    color_b_sb = CMS_COLORS[2]
    color_s_sb = CMS_COLORS[3]
    color_sb_sb = CMS_COLORS[4]


    plotBinnedData(
        ax,
        data,
        label="Data",
        color="black",
        histtype="errorbar",
        marker="o",
        markersize=3,
    )

    plotBinnedData(
        ax,
        prefit_b,
        label="Prefit Background",
        histtype="step",
        linestyle="-",
        color=color_prefit_b,
    )

    if b_background is not None:
        plotBinnedData(
            ax,
            b_background,
            label="Background / B Only Fit",
            histtype="step",
            linestyle="-",
            linewidth=2,
            color=color_b_background,
        )

    if s_background is not None:
        plotBinnedData(
            ax,
            s_background,
            label="Background / SB Fit",
            histtype="step",
            linestyle="-",
            linewidth=2,
            color=color_b_sb,
        )

    if s_signal is not None and show_signal:
        s_label = (
            "Signal / SB Fit " + f"(r={signal_rate:0.2f})"
            if signal_rate is not None
            else ""
        )
        plotBinnedData(
            ax,
            s_signal,
            histtype="step",
            label=s_label,
            linestyle="-",
            linewidth=2,
            color=color_s_sb,
        )
    if s_background is not None and s_signal is not None:
        plotBinnedData(
            ax,
            s_background + s_signal,
            label="S+B / SB Fit",
            histtype="step",
            linestyle="-",
            linewidth=2,
            color=color_sb_sb,
        )

    ax.set_ylabel("Events")
    if log:
        ax.set_yscale("log")

    # CMS Label
    mplhep.cms.label(ax=ax, label="Preliminary", data=True)
    ax.legend(ncols=2, loc="upper right")
    mplhep.utils.yscale_legend(ax)

    def add(num, den, color):
        ratio = num.Y / den.Y
        ratio_err = np.sqrt(num.V) / den.Y
        centers = num.X.ravel()
        pulls = (num.Y - den.Y) / np.sqrt(num.V)
        pulls = np.where(np.isfinite(pulls), pulls, 0)

        rax.errorbar(
            centers, ratio, yerr=ratio_err, fmt="ko", markersize=3, color=color
        )

        pax.bar(
            centers,
            pulls,
            width=np.diff(num.edges[0]),
            align="center",
            color=color,
            alpha=0.8,
        )


    def addPre(num, den, color):
        ratio = num.Y / den.Y
        centers = num.X.ravel()
        # pulls = (num.Y - den.Y) / np.sqrt(num.V)
        # pulls = np.where(np.isfinite(pulls), pulls, 0)


        sbax.errorbar(
            centers, ratio, fmt="ko", markersize=3, color=color
        )

        # sbax.bar(
        #     centers,
        #     pulls,
        #     width=np.diff(num.edges[0]),
        #     align="center",
        #     color=color,
        #     alpha=0.8,
        # )

    add(data, prefit_b, color_prefit_b)
    add(data, b_background, color_b_background)
    add(data, s_background, color_b_sb)
    addPre(prefit_b, s_background, color_b_sb)
    addPre(prefit_b, s_signal+s_background, color_sb_sb)

    rax.axhline(1, color="black", linestyle="--", alpha=0.5)
    rax.set_ylabel(r"$\frac{Data}{Bkg.}$")
    rax.set_ylim(0.5, 1.5)
    
    pax.axhline(0, color="black", linestyle="-", alpha=0.5)
    pax.set_ylabel(r"$\frac{Data - Bkg.}{\sigma_{Data}}$")
    pax.set_ylim(-3, 3)

    sbax.axhline(1, color="black", linestyle="--", alpha=0.5)
    sbax.set_ylabel(r"$\frac{Bkg.}{Prefit}$")
    sbax.set_ylim(0.8, 1.2)


    sbax.set_xlabel(xlabel)

    ax.set_xticklabels([])
    rax.set_xticklabels([])
    sbax.set_xticklabels([])

    return fig, ax


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
