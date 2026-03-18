from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..core.data import BinnedData
from .metrics import pullDistribution, totalPullDistribution
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
    prior_mean: jnp.ndarray | None = None,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ret = {}
    edges = test_data.edges
    X = test_data.X
    obs_Y = test_data.Y
    obs_V = test_data.V
    pulls = np.asarray(pullDistribution(test_data.Y, pred_mean, test_data.V))
    total_pulls = np.asarray(
        totalPullDistribution(test_data.Y, pred_mean, test_data.V, pred_var)
    )

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

    # --- Prior mean prediction ---
    if prior_mean is not None:
        fig, ax = plt.subplots(layout="tight")
        plotRaw(ax, edges, X, np.asarray(prior_mean))
        ax.set_title("Prior Mean (Parametric)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)
        ret["prior_mean"] = (fig, ax)

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

    # --- Total Pull map ---
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, jnp.array(total_pulls), cmap="coolwarm", cmin=-3, cmax=3)
    ax.set_title("Total Pull Map (Stat + Model)")
    if test_data.axis_names and len(test_data.axis_names) >= 2:
        ax.set_xlabel(test_data.axis_names[0])
        ax.set_ylabel(test_data.axis_names[1])
    plotBlinding2D(ax, edges, X, blind_mask)
    ret["total_pull_map"] = (fig, ax)

    # --- Covariance at blinding center ---
    if pred_cov is not None and blind_mask is not None and np.any(np.asarray(blind_mask)):
        mask = np.asarray(blind_mask)
        blinded_X = X[mask]
        center = np.mean(blinded_X, axis=0)
        dist = np.sum((blinded_X - center) ** 2, axis=1)
        # index in the blinded array
        rel_idx = np.argmin(dist)
        # absolute index in the full array
        abs_idx = np.where(mask)[0][rel_idx]

        cov_row = pred_cov[abs_idx, :]
        fig, ax = plt.subplots(layout="tight")
        plotRaw(ax, edges, X, cov_row)
        ax.set_title("Covariance at Blinding Center")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)
        # Mark the center point
        ax.scatter(X[abs_idx, 0], X[abs_idx, 1], color="red", marker="x", s=100, label="Center")
        ret["covariance_at_blind_center"] = (fig, ax)

    # --- Pull histograms ---
    bins = np.linspace(-5.0, 5.0, 21)
    gauss_x = np.linspace(-5, 5, 100)
    gauss_y = np.exp(-(gauss_x**2) / 2) / np.sqrt(2 * np.pi)

    def _plotSinglePullHist(p_vals, tag_name, title_prefix):
        fig_h, ax_h = plt.subplots(layout="tight")
        ax_h.hist(p_vals, bins=bins, density=True, alpha=0.7, label="All bins")
        if blind_mask is not None and np.any(np.asarray(blind_mask)):
            np_mask = np.asarray(blind_mask)
            ax_h.hist(
                p_vals[np_mask], bins=bins, density=True, alpha=0.7, label="Window bins"
            )
        ax_h.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
        ax_h.set_xlabel(rf"{tag_name} Pull: $(N_{{obs}} - N_{{pred}}) / \sigma$")
        ax_h.set_ylabel("Density")
        ax_h.set_title(f"{title_prefix} Distribution")
        ax_h.legend()
        return fig_h, ax_h

    ret["stat_pulls_hist"] = _plotSinglePullHist(pulls, "Stat", "Statistical Pull")
    ret["total_pulls_hist"] = _plotSinglePullHist(total_pulls, "Total", "Total Pull")

    return ret


def makePosteriorPredictivePlots2D(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ret = {}
    edges = test_data.edges
    X = test_data.X

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


def plotNNTransformation2D(
    kernel: Any,
    test_data: BinnedData,
    transform: Any | None = None,
) -> dict[str, tuple]:
    from ..inference.kernels import DeepKernelFunction

    if not isinstance(kernel, DeepKernelFunction):
        return {}
    edges = test_data.edges
    if edges is not None and len(edges) == 2:
        x_min, x_max = edges[0][0], edges[0][-1]
        y_min, y_max = edges[1][0], edges[1][-1]
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1

    nx, ny = 20, 20
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv.ravel(), yv.ravel()], axis=-1)

    points_norm = points
    if transform is not None:
        points_norm = transform.applyX(points)

    points_t = np.asarray(kernel.network(points_norm))
    xt = points_t[:, 0].reshape(nx, ny)
    yt = points_t[:, 1].reshape(nx, ny)

    fig, ax = plt.subplots(layout="tight")
    for i in range(nx):
        ax.plot(xt[i, :], yt[i, :], color="purple", alpha=0.5, lw=0.5)
    for j in range(ny):
        ax.plot(xt[:, j], yt[:, j], color="purple", alpha=0.5, lw=0.5)

    ax.set_title("NN Kernel Grid Transformation")
    ax.set_xlabel("Transformed X")
    ax.set_ylabel("Transformed Y")

    return {"nn_transform_grid": (fig, ax)}


def makeSmoothingPlots2D(
    smoothed_data: BinnedData,
    original_data: BinnedData,
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Generate plots comparing smoothed background to original in 2D."""
    ret = {}
    edges = original_data.edges
    X = original_data.X
    pred_Y = np.asarray(pred_mean)
    smooth_Y = np.asarray(smoothed_data.Y)

    # 1. Original noisy data
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, original_data.Y)
    ax.set_title("Original Noisy MC")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    ret["smoothing_original_2d"] = (fig, ax)

    # 2. GPR Mean Prediction
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, pred_Y)
    ax.set_title("GPR Latent Mean")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    ret["smoothing_gpr_mean_2d"] = (fig, ax)

    # 3. Smoothed Poisson-sampled data
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, smooth_Y)
    ax.set_title("Smoothed Background (Poisson sample)")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    ret["smoothing_sampled_2d"] = (fig, ax)

    # 4. Consistency: (Smoothed - Mean) / sqrt(Mean)
    resid = (smooth_Y - pred_Y) / np.sqrt(np.maximum(pred_Y, 1e-10))
    fig, ax = plt.subplots(layout="tight")
    plotRaw(ax, edges, X, resid, cmap="coolwarm", cmin=-4, cmax=4)
    ax.set_title("Smoothing Consistency (Resid / sqrt(μ))")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    ret["smoothing_resid_2d"] = (fig, ax)

    return ret
