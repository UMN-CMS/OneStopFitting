"""1D diagnostic plots."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from typing import Any

from ..core.data import BinnedData
from .metrics import pullDistribution
from .plot_utils import addAxesToHist, plotBinnedData, plotPPD




def makePosteriorPredictivePlots1D(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Create posterior predictive check plots.

    Args:
        ppc_results: Dictionary returned by posteriorPredictiveCheck.
        test_data: Test data.
        blind_mask: Boolean mask for blinded bins.

    Returns:
        Dict of plot name -> (figure, axes).
    """
    ret = {}

    X = np.asarray(test_data.X).ravel()

    # --- 1. Percentile Bands Plot ---
    fig, ax = plt.subplots(layout="tight")

    summary = ppc_results["summary"]
    q05 = np.asarray(summary["q05"])
    q95 = np.asarray(summary["q95"])
    median = np.asarray(summary["median"])

    ax.fill_between(X, q05, q95, color="orange", alpha=0.3, label="90% PPD")
    ax.plot(X, median, color="orange", label="Median PPD")

    plotBinnedData(ax, test_data, histtype="errorbar", color="black", label="Observed")

    if blind_mask is not None and np.any(blind_mask):
        np_mask = np.asarray(blind_mask)
        w_min = X[np_mask].min()
        w_max = X[np_mask].max()
        for boundary in [w_min, w_max]:
            ax.axvline(boundary, ls="--", color="gray", alpha=0.5)

    if test_data.axis_names:
        ax.set_xlabel(test_data.axis_names[0])
    ax.set_ylabel("Counts")
    ax.legend()
    ret["ppc_percentile_bands"] = (fig, ax)

    # --- 2. Test Statistic P-Value Distributions ---
    test_stats = ppc_results["test_stats"]
    for stat_name, regions in test_stats.items():
        for region_name, summary_stats in regions.items():
            obs_val = float(summary_stats["obs"])
            rep_vals = np.asarray(summary_stats["rep"])
            pvalue = float(summary_stats["pvalue"])

            fig, ax = plt.subplots(layout="tight")

            # Use the referenced dense styled PPD plot
            plotPPD(
                ax,
                rep_vals,
                obs_val,
                xlabel=f"Test Statistic: {stat_name} ({region_name})",
                pvalue=pvalue,
            )

            ret[f"ppc_dist_{stat_name}_{region_name}"] = (fig, ax)

    return ret


def makeDiagnosticPlots1D(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | None = None,
) -> dict[str, tuple]:
    """Create 1D diagnostic plots.

    Args:
        pred_mean: Predicted mean in real space, shape (N,).
        pred_var: Predicted variance in real space, shape (N,).
        test_data: Full-domain test data.
        blind_mask: Boolean mask for blinded bins.
        signal_data: Optional signal data to overlay.

    Returns:
        Dict of plot name -> (figure, axes).
    """
    ret = {}
    X = np.asarray(test_data.X).ravel()
    obs_V = np.asarray(test_data.V)
    pred_Y = np.asarray(pred_mean)
    pred_V = np.asarray(pred_var)
    pred_std = np.sqrt(pred_V)
    pulls = np.asarray(pullDistribution(test_data.Y, pred_mean, test_data.V))

    # --- Summary plot: data + GP prediction + pull panel ---
    fig, ax = plt.subplots(layout="tight")
    addAxesToHist(ax, size=1.5)
    plotBinnedData(ax, test_data, histtype="errorbar", color="black", label="Observed")

    if signal_data is not None:
        plotBinnedData(
            ax, signal_data, histtype="step", color="red", label="Injected Signal"
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

    # Window indicators
    if blind_mask is not None and np.any(blind_mask):
        np_mask = np.asarray(blind_mask)
        w_min = X[np_mask].min()
        w_max = X[np_mask].max()
        for boundary in [w_min, w_max]:
            ax.axvline(boundary, ls="--", color="gray", alpha=0.5)
            ax.bottom_axes[0].axvline(boundary, ls="--", color="gray", alpha=0.5)

    # Pull panel
    ratio_ax = ax.bottom_axes[0]
    ratio_ax.set_ylim(-3, 3)
    ratio_ax.plot(X, pulls, "o", color="black", markersize=2)
    ax.tick_params(axis="x", which="both", labelbottom=False)
    ratio_ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ratio_ax.axhline(1, ls="-.", color="gray", alpha=0.3)
    ax.bottom_axes[0].axhline(-1, ls="-.", color="gray", alpha=0.3)
    ax.bottom_axes[0].set_ylabel("Pull")
    if test_data.axis_names:
        ax.bottom_axes[0].set_xlabel(test_data.axis_names[0])

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

    ax.legend()

    ret["summary_plot"] = (fig, ax)

    # --- Pull distribution histogram ---
    ret.update(_plotPullHistograms(pulls, blind_mask))

    return ret


def _plotPullHistograms(
    pulls: np.ndarray,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Plot pull distribution histograms."""
    ret = {}
    bins = np.linspace(-5.0, 5.0, 21)
    gauss_x = np.linspace(-5, 5, 100)
    gauss_y = np.exp(-(gauss_x**2) / 2) / np.sqrt(2 * np.pi)

    # Global pulls
    fig, ax = plt.subplots(layout="tight")
    ax.hist(pulls, bins=bins, density=True, alpha=0.7, label="All bins")
    ax.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
    ax.set_xlabel(r"$(N_{obs} - N_{pred}) / \sigma_{obs}$")
    ax.set_ylabel("Density")
    ax.legend()
    ret["global_pulls_hist"] = (fig, ax)

    # Window pulls
    if blind_mask is not None and np.any(np.asarray(blind_mask)):
        np_mask = np.asarray(blind_mask)
        fig, ax = plt.subplots(layout="tight")
        ax.hist(pulls[np_mask], bins=bins, density=True, alpha=0.7, label="Window bins")
        ax.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
        ax.set_xlabel(r"$(N_{obs} - N_{pred}) / \sigma_{obs}$")
        ax.set_ylabel("Density")
        ax.legend()
        ret["window_pulls_hist"] = (fig, ax)

    return ret
