from __future__ import annotations

import logging

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import FancyBboxPatch

from ..core.data import BinnedData
from .metrics import pullDistribution
from .plot_utils import plotBinnedData, plotBlinding2D

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
    slice_signal_data: BinnedData | None,
    slice_title: str,
    *,
    minimap_context: dict | None = None,
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

    plotBinnedData(ax, slice_data, histtype="errorbar", color="black", label="Observed")

    if slice_signal_data is not None:
        plotBinnedData(
            ax, slice_signal_data, histtype="step", color="red", label="Injected Signal"
        )

    ax.plot(X, pred_Y, color="orange", label="GP Prediction")
    ax.fill_between(
        X,
        pred_Y + pred_std,
        pred_Y - pred_std,
        color="orange",
        alpha=0.3,
        label=r"$\pm\sigma_{pred}$",
    )

    bin_width = (X[-1] - X[0]) / len(X)

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
    leg = ax.legend(loc="upper right")

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

    return (fig, ax)


def makeSlicePlots2D(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | None = None,
    signal_template: BinnedData | None = None,
    *,
    show_minimap: bool = True,
    minimap_background: bool = True,
) -> dict[str, tuple]:
    if test_data.ndim != 2:
        logger.warning("makeSlicePlots2D called on non-2D data; returning empty dict")
        return {}

    if blind_mask is None or not np.any(np.asarray(blind_mask)):
        logger.info("makeSlicePlots2D: no blind mask provided; skipping slice plots")
        return {}

    ret: dict[str, tuple] = {}
    X_np = np.asarray(test_data.X)
    pred_mean_np = np.asarray(pred_mean)
    pred_var_np = np.asarray(pred_var)
    blind_mask_np = np.asarray(blind_mask)
    window_X = X_np[blind_mask_np]

    for slice_axis in (0, 1):
        perp = _perpendicularAxis(slice_axis)
        window_vals = _uniqueAxisValues(window_X, slice_axis)
        axis_name = (
            test_data.axis_names[slice_axis]
            if test_data.axis_names and len(test_data.axis_names) > slice_axis
            else f"axis{slice_axis}"
        )
        axis_key = (
            axis_name.replace(" ", "_")
            .replace(r"tilde{t}", "stop")
            .replace(r"tilde{\chi}", "chi")
            .replace(r"tilde{\chi^{\pm}}", "chi")
            .replace("/", "")
            .replace("\\", "")
            .replace("$", "")
            .replace("{", "")
            .replace("}", "")
        )

        for val in window_vals:
            mask = _sliceMask(X_np, slice_axis, val)
            slice_data = _buildSliceBinnedData(test_data, mask, perp)
            order = np.argsort(X_np[mask, perp])
            slice_pred_mean = pred_mean_np[mask][order]
            slice_pred_var = pred_var_np[mask][order]
            slice_blind = blind_mask_np[mask][order]

            slice_sig = None
            if signal_data is not None:
                sig_X = np.asarray(signal_data.X)
                sig_mask = _sliceMask(sig_X, slice_axis, val)
                if np.any(sig_mask):
                    slice_sig = _buildSliceBinnedData(signal_data, sig_mask, perp)

            label = _sliceLabel(axis_name, val)
            val_key = f"{val:0.2g}".replace(".", "p").replace("-", "m")
            key = f"slice_{axis_key}_{val_key}"

            minimap_ctx = None
            if show_minimap:
                minimap_ctx = {
                    "edges": test_data.edges,
                    "X": X_np,
                    "blind_mask": blind_mask_np,
                    "slice_axis": slice_axis,
                    "slice_value": val,
                    "axis_names": test_data.axis_names,
                    "background_Y": np.asarray(test_data.Y)
                    if minimap_background
                    else None,
                }

            ret["slice/" + key] = _makeSliceSummaryPlot(
                slice_data=slice_data,
                slice_pred_mean=slice_pred_mean,
                slice_pred_var=slice_pred_var,
                slice_blind_mask=slice_blind,
                slice_signal_data=slice_sig,
                slice_title=f"Slice: {label}",
                minimap_context=minimap_ctx,
            )

    return ret
