from __future__ import annotations

import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyBboxPatch

from ..core.data import BinnedData
from .metrics import pullDistribution
from .plot_utils import plotBinnedData, plotBlinding2D
from typing import Callable


logger = logging.getLogger(__name__)


def _uniqueAxisValues(X: np.ndarray, axis: int) -> np.ndarray:
    return np.unique(np.asarray(X)[:, axis])


def _sliceMask(X: np.ndarray, axis: int, value: float) -> np.ndarray:
    return np.isclose(np.asarray(X)[:, axis], value)


def _perpendicularAxis(axis: int) -> int:
    return 1 - axis


def _sliceEdges(edges: tuple, perp_axis: int) -> tuple:
    return (edges[perp_axis],)


def _buildSliceBinnedData(
    data: BinnedData,
    mask: np.ndarray,
    perp_axis: int,
) -> BinnedData:
    X_2d = np.asarray(data.X)
    X_1d = X_2d[mask, perp_axis : perp_axis + 1]
    order = np.argsort(X_1d[:, 0])
    axis_name = (
        (data.axis_names[perp_axis],)
        if data.axis_names and len(data.axis_names) > perp_axis
        else ()
    )
    return BinnedData(
        X=jnp.array(X_1d[order]),
        Y=jnp.array(np.asarray(data.Y)[mask][order]),
        V=jnp.array(np.asarray(data.V)[mask][order]),
        edges=_sliceEdges(data.edges, perp_axis),
        axis_names=axis_name,
    )


def _sliceLabel(axis_name: str | None, value: float) -> str:
    name = axis_name or "axis"
    return f"{name} = {value:0.3g}"


def _drawSliceMinimap(
    ax,
    fig,
    edges_2d: tuple,
    X_2d: np.ndarray,
    blind_mask_2d: np.ndarray,
    slice_axis: int,
    slice_value: float,
    anchor_top_right: tuple[float, float] | None = None,
    axis_names: tuple[str, ...] | None = None,
    background_Y: np.ndarray | None = None,
) -> None:
    np_edges = tuple(np.asarray(e) for e in edges_2d)
    ex, ey = np_edges

    inset_w, inset_h = 0.2, 0.2
    if anchor_top_right is not None:
        x0 = anchor_top_right[0] - inset_w
        y0 = anchor_top_right[1] - inset_h
    else:
        x0, y0 = 0.02, 0.98 - inset_h
    inset = ax.inset_axes([x0, y0, inset_w, inset_h])

    if background_Y is not None:
        grid = np.histogramdd(X_2d, bins=np_edges, weights=np.asarray(background_Y))[0]
        filled = np.histogramdd(X_2d, bins=np_edges, weights=np.ones(len(X_2d)))[
            0
        ].astype(bool)
        grid = np.where(filled, grid, np.nan)
        inset.pcolormesh(
            ex, ey, grid.T, shading="flat", cmap="viridis", rasterized=True
        )
    else:
        inset.add_patch(
            FancyBboxPatch(
                (ex[0], ey[0]),
                ex[-1] - ex[0],
                ey[-1] - ey[0],
                boxstyle="round,pad=0",
                facecolor="0.93",
                edgecolor="0.6",
                lw=0.8,
            )
        )
    plotBlinding2D(inset, edges_2d, X_2d, blind_mask_2d, color="magenta", linewidth=1.5)
    slice_color = "red"
    if slice_axis == 0:
        inset.axvline(slice_value, color=slice_color, lw=1.5, ls="--")
    else:
        inset.axhline(slice_value, color=slice_color, lw=1.5, ls="--")

    inset.set_xlim(ex[0], ex[-1])
    inset.set_ylim(ey[0], ey[-1])
    inset.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
        top=False,
        right=False,
    )
    if axis_names and len(axis_names) >= 2:
        inset.set_xlabel(axis_names[0], fontsize="x-small", labelpad=1)
        inset.set_ylabel(axis_names[1], fontsize="x-small", labelpad=1)


def _makeSliceSummaryPlot(
    slice_data: BinnedData,
    slice_pred_mean: np.ndarray,
    slice_pred_var: np.ndarray,
    slice_blind_mask: np.ndarray | None,
    slice_signal_data: BinnedData | dict[str, BinnedData] | None,
    slice_title: str,
    *,
    minimap_context: dict | None = None,
    extra_draw_funcs=None,
    injected_signal_rate: float | None = None,
) -> tuple:
    X = np.asarray(slice_data.X).ravel()
    obs_V = np.asarray(slice_data.V)
    pred_Y = np.asarray(slice_pred_mean)
    pred_V = np.asarray(slice_pred_var)
    pred_std = np.sqrt(pred_V)
    pulls = np.asarray(pullDistribution(slice_data.Y, jnp.array(pred_Y), slice_data.V))

    grid_spec = {
        "height_ratios": [3, 1],
    }
    fig, (ax, ratio_ax) = plt.subplots(
        2, 1, sharex=True, layout="constrained", gridspec_kw=grid_spec
    )

    plotBinnedData(ax, slice_data, histtype="errorbar", color="black", label="Obs.")

    ax.plot(X, pred_Y, color="orange", label="GPR")
    ax.fill_between(
        X,
        pred_Y + pred_std,
        pred_Y - pred_std,
        color="orange",
        alpha=0.3,
        label=r"$\pm\sigma_{pred}$",
    )

    bin_width = (X[-1] - X[0]) / len(X) if len(X) > 1 else 1.0

    if extra_draw_funcs is not None:
        for draw_func in extra_draw_funcs:
            draw_func(ax, ratio_ax)

    if slice_signal_data is not None:
        sigs = (
            slice_signal_data
            if isinstance(slice_signal_data, dict)
            else {"sig": slice_signal_data}
        )
        import matplotlib.cm as cm

        colors = cm.get_cmap("Reds")(np.linspace(0.4, 1.0, len(sigs)))
        for (lbl, sig), color in zip(sigs.items(), colors):
            label = f"Sig. {lbl}" if lbl != "sig" else "Sig."
            if injected_signal_rate is not None:
                label += f" ({injected_signal_rate:.2f})"

            plotBinnedData(
                ax,
                sig,
                histtype="step",
                color=color,
                label=label,
            )

    if slice_blind_mask is not None and np.any(slice_blind_mask):
        w_min = X[slice_blind_mask].min()
        w_max = X[slice_blind_mask].max()
        for boundary in [w_min - bin_width / 2, w_max + bin_width / 2]:
            ax.axvline(boundary, ls="--", color="gray", alpha=0.5)
            ratio_ax.axvline(boundary, ls="--", color="gray", alpha=0.5)

    ratio_ax.set_ylim(-3, 3)
    ratio_ax.plot(X, pulls, "o", color="black", markersize=2)
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ratio_ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ratio_ax.axhline(1, ls="-.", color="gray", alpha=0.3)
    ratio_ax.axhline(-1, ls="-.", color="gray", alpha=0.3)
    ratio_ax.set_ylabel("Pull")

    with np.errstate(divide="ignore", invalid="ignore"):
        if len(X) > 1:
            ratio_ax.bar(
                x=X,
                bottom=np.nan_to_num(-pred_std / np.sqrt(obs_V), nan=0),
                height=np.nan_to_num(2 * pred_std / np.sqrt(obs_V), nan=0),
                width=X[1] - X[0],
                color="orange",
                alpha=0.3,
                fill=True,
                lw=0,
            )

    if slice_data.axis_names:
        ratio_ax.set_xlabel(slice_data.axis_names[0])

    ax.set_ylabel("Events")
    leg = ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=18,
        columnspacing=0.5,
        labelspacing=0.3,
        borderpad=0.1,
    )

    with np.errstate(over="ignore"):
        fig.canvas.draw()
        leg_bb = leg.get_window_extent().transformed(ax.transAxes.inverted())

        txt = ax.text(
            leg_bb.x1,
            leg_bb.y0 - 0.05,
            slice_title,
            transform=ax.transAxes,
            fontsize="small",
            fontstyle="italic",
            ha="right",
            va="top",
            color="0.3",
        )

    if minimap_context is not None:
        with np.errstate(over="ignore"):
            fig.canvas.draw()
            txt_bb = txt.get_window_extent().transformed(ax.transAxes.inverted())
            anchor = (txt_bb.x1, txt_bb.y0 - 0.02)
            _drawSliceMinimap(
                ax,
                fig,
                edges_2d=minimap_context["edges"],
                X_2d=minimap_context["X"],
                blind_mask_2d=minimap_context["blind_mask"],
                slice_axis=minimap_context["slice_axis"],
                slice_value=minimap_context["slice_value"],
                anchor_top_right=anchor,
                axis_names=minimap_context.get("axis_names"),
                background_Y=minimap_context.get("background_Y"),
            )

    return (fig, ax, ratio_ax)


def fixAxisKey(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace(r"tilde{t}", "stop")
        .replace(r"tilde{\chi}", "chi")
        .replace(r"tilde{\chi^{\pm}}", "chi")
        .replace("/", "")
        .replace("\\", "")
        .replace("$", "")
        .replace("{", "")
        .replace("}", "")
    )


def _makeOneSlice(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    blind_mask: jnp.ndarray,
    slice_axis: int,
    val: float,
    axis_name: str,
    plot_saver: Callable,
    key: str,
    signal_data: BinnedData | dict[str, BinnedData] | None = None,
    show_minimap: bool = True,
    minimap_background: bool = True,
    extra_draw_funcs=None,
    injected_signal_rate: float | None = None,
) -> None:
    X = np.asarray(test_data.X)
    perp = _perpendicularAxis(slice_axis)
    mask = _sliceMask(X, slice_axis, val)
    slice_data = _buildSliceBinnedData(test_data, mask, perp)
    order = np.argsort(X[mask, perp])
    slice_pred_mean = pred_mean[mask][order]
    slice_pred_var = pred_var[mask][order]
    slice_blind = blind_mask[mask][order]

    slice_sigs = None
    if signal_data is not None:
        if isinstance(signal_data, dict):
            slice_sigs = {}
            for lbl, sig in signal_data.items():
                sig_X = np.asarray(sig.X)
                sig_mask = _sliceMask(sig_X, slice_axis, val)
                if np.any(sig_mask):
                    slice_sigs[lbl] = _buildSliceBinnedData(sig, sig_mask, perp)
        else:
            sig_X = np.asarray(signal_data.X)
            sig_mask = _sliceMask(sig_X, slice_axis, val)
            if np.any(sig_mask):
                slice_sigs = _buildSliceBinnedData(signal_data, sig_mask, perp)

    label = _sliceLabel(axis_name, val)

    minimap_ctx = None
    if show_minimap:
        minimap_ctx = {
            "edges": test_data.edges,
            "X": X,
            "blind_mask": blind_mask,
            "slice_axis": slice_axis,
            "slice_value": val,
            "axis_names": test_data.axis_names,
            "background_Y": np.asarray(test_data.Y) if minimap_background else None,
        }

    fig, ax, ratio_ax = _makeSliceSummaryPlot(
        slice_data=slice_data,
        slice_pred_mean=slice_pred_mean,
        slice_pred_var=slice_pred_var,
        slice_blind_mask=slice_blind,
        slice_signal_data=slice_sigs,
        slice_title=f"Slice: {label}",
        minimap_context=minimap_ctx,
        extra_draw_funcs=extra_draw_funcs,
        injected_signal_rate=injected_signal_rate,
    )

    with np.errstate(over="ignore"):
        plot_saver(key, fig, ax)


def makeSlicePlots2D(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    plot_saver: Callable,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | dict[str, BinnedData] | None = None,
    *,
    show_minimap: bool = True,
    minimap_background: bool = True,
) -> None:
    if test_data.ndim != 2:
        logger.warning("makeSlicePlots2D called on non-2D data; returning None")
        return

    if blind_mask is None or not np.any(np.asarray(blind_mask)):
        logger.info("makeSlicePlots2D: no blind mask provided; skipping slice plots")
        return

    X = np.asarray(test_data.X)

    for slice_axis in (0, 1):
        window_X = X[blind_mask]
        window_vals = _uniqueAxisValues(window_X, slice_axis)
        axis_name = (
            test_data.axis_names[slice_axis]
            if test_data.axis_names and len(test_data.axis_names) > slice_axis
            else f"axis{slice_axis}"
        )
        axis_key = fixAxisKey(axis_name)
        for val in window_vals:
            val_key = f"{val:0.3g}".replace(".", "p").replace("-", "m")
            key = f"slice/{fixAxisKey(axis_name)}_{val_key}"
            _makeOneSlice(
                pred_mean=pred_mean,
                pred_var=pred_var,
                test_data=test_data,
                blind_mask=blind_mask,
                slice_axis=slice_axis,
                val=val,
                axis_name=axis_name,
                plot_saver=plot_saver,
                key=key,
                signal_data=signal_data,
                show_minimap=show_minimap,
                minimap_background=minimap_background,
            )


def build2D(blind_mask: jnp.ndarray, test_data: BinnedData, blinded_data: BinnedData):
    X = test_data.X
    Y = np.full_like(test_data.Y, np.nan, dtype=float)
    V = np.full_like(test_data.V, np.nan, dtype=float)

    Y[blind_mask] = blinded_data.Y
    V[blind_mask] = blinded_data.V

    return BinnedData(
        X=X,
        Y=Y,
        V=V,
        axis_names=test_data.axis_names,
        edges=test_data.edges,
    )


def makePostCombineSlice(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    plot_saver: Callable,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | dict[str, BinnedData] | None = None,
    post_fit_signal: BinnedData | None = None,
    post_fit_background: BinnedData | None = None,
    injected_signal: float | None = None,
    extracted_signal: float | None = None,
    *,
    show_minimap: bool = True,
    minimap_background: bool = True,
) -> None:
    if test_data.ndim != 2:
        logger.warning("makeSlicePlots2D called on non-2D data; returning None")
        return

    if blind_mask is None or not np.any(np.asarray(blind_mask)):
        logger.info("makeSlicePlots2D: no blind mask provided; skipping slice plots")
        return

    X = np.asarray(test_data.X)
    post_fit_signal_full = build2D(blind_mask, test_data, post_fit_signal)
    post_fit_background_full = build2D(blind_mask, test_data, post_fit_background)

    for slice_axis in (0, 1):
        window_X = X[blind_mask]
        window_vals = _uniqueAxisValues(window_X, slice_axis)
        axis_name = (
            test_data.axis_names[slice_axis]
            if test_data.axis_names and len(test_data.axis_names) > slice_axis
            else f"axis{slice_axis}"
        )
        axis_key = fixAxisKey(axis_name)
        for val in window_vals:
            val_key = f"{val:0.3g}".replace(".", "p").replace("-", "m")
            key = f"slice/{fixAxisKey(axis_name)}_{val_key}"
            perp = _perpendicularAxis(slice_axis)
            mask = _sliceMask(X, slice_axis, val)
            slice_post_fit_signal = _buildSliceBinnedData(
                post_fit_signal_full, mask, perp
            )
            slice_post_fit_background = _buildSliceBinnedData(
                post_fit_background_full, mask, perp
            )

            def postFitDrawFunc(ax, ratio_ax):
                plotBinnedData(
                    ax,
                    slice_post_fit_background.masked(
                        ~np.isnan(slice_post_fit_background.Y)
                    ),
                    color="blue",
                    label="Post Bkg.",
                    ls="--",
                )
                summed = slice_post_fit_signal + slice_post_fit_background
                plotBinnedData(
                    ax,
                    summed.masked(~np.isnan(summed.Y)),
                    color="purple",
                    label="Post S+B",
                )
                plotBinnedData(
                    ax,
                    slice_post_fit_signal.masked(~np.isnan(slice_post_fit_signal.Y)),
                    color="green",
                    label=f"Post Sig. ({extracted_signal:.2f})",
                    ls="--",
                )

            _makeOneSlice(
                pred_mean=pred_mean,
                pred_var=pred_var,
                test_data=test_data,
                blind_mask=blind_mask,
                slice_axis=slice_axis,
                val=val,
                axis_name=axis_name,
                plot_saver=plot_saver,
                key=key,
                signal_data=signal_data,
                show_minimap=show_minimap,
                minimap_background=minimap_background,
                extra_draw_funcs=[postFitDrawFunc],
                injected_signal_rate=injected_signal,
            )
