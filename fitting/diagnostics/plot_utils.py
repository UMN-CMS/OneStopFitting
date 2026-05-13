from __future__ import annotations

from pathlib import Path
import itertools as it
import matplotlib.pyplot as plt
import logging
import mplhep
import numpy as np
from uhi.numpy_plottable import NumPyPlottableHistogram
import matplotlib.patheffects as path_effects

from ..core.data import BinnedData

plt.rcParams["figure.constrained_layout.use"] = True

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
    return plotRaw(ax, data.edges, data.X, data.Y, V=data.V, **kwargs)


def plotRaw(ax, edges, X, Y, V=None, cbar_title=None, **kwargs):
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
        drop_keys = {"cmin", "cmax"}
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in drop_keys}
        if V is not None:
            np_V = np.asarray(V)
            var_hist = np.histogramdd(np_X, bins=np_edges, weights=np_V)[0]
            variances = np.where(filled, var_hist, np.nan)
        h = NumPyPlottableHistogram(vals, *np_edges, variances=variances)
        objs = mplhep.hist2dplot(h, ax=ax, flow=None, **clean_kwargs)
        pc = objs.pcolormesh
        cbar = objs.cbar
        if cbar_title and cbar is not None:
            cbar.set_label(cbar_title)

        clim_relevant = {
            x.replace("c", "v"): kwargs[x] for x in kwargs if x in ("cmin", "cmax")
        }
        if clim_relevant:
            pc.set_clim(**clim_relevant)
        return objs


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
    gs_kw = dict(height_ratios=[3, 1, 1, 1])
    fig, (ax, rax, pax, sbax) = plt.subplots(
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
    # mplhep.cms.label(ax=ax, label="Preliminary", data=True)
    ax.legend(ncols=1, loc="upper right")
    mplhep.utils.yscale_legend(ax, soft_fail=True, N=20)

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

        sbax.errorbar(centers, ratio, fmt="ko", markersize=3, color=color)

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
    addPre(prefit_b, s_signal + s_background, color_sb_sb)

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


def _getSampleCategory(all_meta):
    sample_types = set()
    for meta in all_meta:
        st = meta.get("sample_type")
        if st is not None:
            val = st.value if hasattr(st, "value") else str(st)
            sample_types.add(val)
    has_mc = "MC" in sample_types
    has_data = "Data" in sample_types
    return has_mc, has_data


def isSimulationOnly(all_meta):
    has_mc, has_data = _getSampleCategory(all_meta)
    return has_mc and not has_data


def _buildCMSText(cms_text, all_meta):
    has_mc, has_data = _getSampleCategory(all_meta)
    sim_only = has_mc and not has_data
    sim_only = True
    label = cms_text or ""
    is_private = label.lower().startswith("private work")

    if is_private:
        if has_mc and has_data:
            data_label = "CMS Data/Simulation"
        elif sim_only:
            data_label = "CMS Simulation"
        else:
            data_label = "CMS Data"
        return "", f"Private Work\n({data_label})"
    else:
        if sim_only:
            label = f"Simulation {label}" if label else "Simulation"
        return "CMS", label


def addCMSBits(
    ax,
    all_meta,
    extra_text=None,
    cms_text="Private Work",
    text_color="black",
    cms_text_pos=1,
):
    lumis = set(str(x["era"]["lumi"]) for x in all_meta)
    energies = set(str(x["era"]["energy"]) for x in all_meta)
    era = set(str(x["era"]["name"]) for x in all_meta)
    era_text = f"{'/'.join(era)}"
    lumi_text = f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    info_text = era_text + ", " + lumi_text

    exp, text = _buildCMSText(cms_text, all_meta)

    if extra_text is not None:
        text += "\n" + extra_text

    if exp:
        artists = mplhep.cms.text(
            text=text,
            # exp=exp,
            lumi=info_text,
            ax=ax,
            loc=cms_text_pos,
            color=text_color,
        )
    else:
        loc = cms_text_pos
        color = text_color

        lumi_artist = None
        if info_text is not None:
            lumi_artist = mplhep.add_text(
                info_text,
                loc="over right",
                xpad=0,
                ypad=0,
                ax=ax,
            )

        loc_map = {0: "over left", 1: "upper left", 2: "upper left", 3: "over left"}
        text_loc = loc_map.get(loc, "over left")
        label_artist = mplhep.label.add_text(
            text,
            loc=text_loc,
            ax=ax,
            fontstyle="italic",
            color=color,
        )
        artists = (label_artist, None, lumi_artist, None)

    for artist in artists:
        if artist:
            artist.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="white"),
                    path_effects.Normal(),
                ]
            )
    ax._cms_text_artists = artists
    ax.set_title("")
    return artists


def removeCMSAnnotations(ax):
    if hasattr(ax, "_cms_text_artists"):
        for artist in ax._cms_text_artists:
            if artist is not None:
                artist.remove()
        del ax._cms_text_artists


def labelAxis(ax, which, axes, label=None, label_complete=None):
    mapping = dict(x=0, y=1, z=2)
    idx = mapping[which]

    if idx != len(axes):
        this_unit = getattr(axes[idx], "unit", None)
        if not label:
            label = axes[idx].label
            if this_unit:
                label += f" [{this_unit}]"

        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))
    else:
        label = label or "Events"
        units = [getattr(x, "unit", None) for x in axes]
        units = [x for x in units if x]
        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))


def saveFigVariants(
    fig,
    ax,
    out,
    all_meta,
    metadata=None,
    extra_text=None,
    text_color=None,
    cms_texts=None,
    formats=None,
    **save_kwargs,
):

    # cms_texts = cms_texts or ["Preliminary", "Private Work"]
    cms_texts = cms_texts or ["Private Work"]
    suffix_text = len(cms_texts) > 1

    raw_types = formats or [".pdf"]
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in raw_types]
    suffix_ext = len(extensions) > 1

    base_path = Path(out)
    base_path.parent.mkdir(exist_ok=True, parents=True)

    for variant in cms_texts:
        removeCMSAnnotations(ax)
        addCMSBits(ax, all_meta, cms_text=variant, extra_text=extra_text)

        text_suffix = f"_{variant.lower().replace(' ', '_')}" if suffix_text else ""
        for ext in extensions:
            variant_path = base_path.with_stem(
                f"{base_path.stem}{text_suffix}"
            ).with_suffix(ext)
            
            kwargs = {"bbox_inches": "tight"}
            kwargs.update(save_kwargs)
            fig.savefig(variant_path, **kwargs)
            logger.info(f"Saved figure to {variant_path}")


def getPlotSaver(save_dir, all_meta, formats=("pdf",), **save_kwargs):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    def saver(name, fig, ax=None):
        out = save_dir / name
        if ax is None:
            if fig.axes:
                ax = fig.axes[0]
        if isinstance(ax, (np.ndarray, list, tuple)):
            ax = ax[0]
        saveFigVariants(fig, ax, out, all_meta, formats=formats, **save_kwargs)
        plt.close(fig)

    return saver


def savePlots(plots: dict[str, tuple], save_dir, all_meta, formats=("pdf",), **kwargs):
    saver = getPlotSaver(save_dir, all_meta, formats=formats, **kwargs)
    for name, (fig, ax) in plots.items():
        saver(name, fig, ax)
    logger.info(f"Saved {len(plots)} to directory {save_dir}")
