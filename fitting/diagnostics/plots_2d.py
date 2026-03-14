"""2D diagnostic plots."""

from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..core.data import BinnedData
from .metrics import pullDistribution
from typing import Any
from .plot_utils import plotBinnedData, plotRaw, plotPPD, plotBlinding2D


def makeDiagnosticPlots2D(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    train_data: BinnedData | None = None,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | None = None,
    signal_template: BinnedData | None = None,
) -> dict[str, tuple]:
    """Create 2D diagnostic plots.

    Args:
        pred_mean: Predicted mean in real space, shape (N,).
        pred_var: Predicted variance in real space, shape (N,).
        test_data: Full-domain test data.
        train_data: Training data (blinded region removed).
        blind_mask: Boolean mask for blinded bins.
        signal_data: Optional signal data to overlay.

    Returns:
        Dict of plot name -> (figure, axes).
    """
    ret = {}
    edges = test_data.edges
    X = test_data.X
    obs_Y = test_data.Y
    obs_V = test_data.V
    pulls = np.asarray(pullDistribution(test_data.Y, pred_mean, test_data.V))

    # --- Training data ---
    if train_data is not None:
        fig, ax = plt.subplots(layout="tight")
        plotBinnedData(ax, train_data)
        ax.set_title("Training Data (Blinded)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        ret["training_points"] = (fig, ax)

    # --- GP mean prediction ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, pred_mean)
    ax.set_title("GP Mean Prediction")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["gpr_mean"] = (fig, ax)

    # --- Observed ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, obs_Y)
    ax.set_title("Observed")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["observed_outputs"] = (fig, ax)

    # --- Signal (if provided) ---
    if signal_data is not None:
        fig, ax = plt.subplots(layout="tight")
        plotBinnedData(ax, signal_data)
        ax.set_title("Injected Signal")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)
        ret["injected_signal"] = (fig, ax)

    # --- Signal Template (unscaled) ---
    if signal_template is not None:
        fig, ax = plt.subplots(layout="tight")
        plotBinnedData(ax, signal_template)
        ax.set_title("Signal Template")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)
        ret["signal_template"] = (fig, ax)

    # --- Observed variances ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, obs_V)
    ax.set_title("Observed Variances")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["observed_variances"] = (fig, ax)

    # --- Predicted variances ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, pred_var)
    ax.set_title("Predicted Variances")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["predicted_variances"] = (fig, ax)

    # --- Relative uncertainty ---
    rel_unc = np.asarray(jnp.sqrt(pred_var) / jnp.clip(pred_mean, 1e-10))
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, jnp.array(rel_unc), cmin=0, cmax=0.1)
    ax.set_title("Relative Uncertainty (σ/μ)")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["relative_uncertainty"] = (fig, ax)

    # --- Pull map ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, jnp.array(pulls), cmap="coolwarm", cmin=-3, cmax=3)
    ax.set_title("Pull Map")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["pull_map"] = (fig, ax)

    # --- Pull histograms ---
    bins = np.linspace(-5.0, 5.0, 21)
    gauss_x = np.linspace(-5, 5, 100)
    gauss_y = np.exp(-(gauss_x**2) / 2) / np.sqrt(2 * np.pi)

    fig, ax = plt.subplots(layout="tight")
    ax.hist(pulls, bins=bins, density=True, alpha=0.7, label="All bins")
    if blind_mask is not None and np.any(np.asarray(blind_mask)):
        np_mask = np.asarray(blind_mask)
        ax.hist(
            pulls[np_mask],
            bins=bins,
            density=True,
            alpha=0.7,
            label="Window bins",
        )
    ax.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
    ax.set_xlabel(r"$(N_{obs} - N_{pred}) / \sigma_{obs}$")
    ax.set_ylabel("Density")
    ax.legend()
    ret["pulls_hist"] = (fig, ax)

    return ret


def makePosteriorPredictivePlots2D(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Create 2D posterior predictive check plots.

    Args:
        ppc_results: Dictionary returned by posteriorPredictiveCheck.
        test_data: Full-domain test data.
        blind_mask: Boolean mask for blinded bins.

    Returns:
        Dict of plot name -> (figure, axes).
    """
    ret = {}
    edges = test_data.edges
    X = test_data.X

    # --- 1. PPC Summary Maps (Mean and StdDev) ---
    summary = ppc_results["summary"]

    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, np.asarray(summary["mean"]))
    ax.set_title("PPC Predictive Mean")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["ppc_mean_map"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, np.asarray(summary["std"]))
    ax.set_title("PPC Predictive StdDev")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["ppc_std_map"] = (fig, ax)

    # --- 2. Test Statistic P-Value Distributions ---
    test_stats = ppc_results["test_stats"]
    for stat_name, regions in test_stats.items():
        for region_name, summary_stats in regions.items():
            obs_val = float(summary_stats["obs"])
            rep_vals = np.asarray(summary_stats["rep"])
            pvalue = float(summary_stats["pvalue"])

            fig, ax = plt.subplots(layout="tight")
            plotPPD(
                ax,
                rep_vals,
                obs_val,
                xlabel=f"Test Statistic: {stat_name} ({region_name})",
                pvalue=pvalue,
            )
            ret[f"ppc_dist_{stat_name}_{region_name}"] = (fig, ax)

    return ret
