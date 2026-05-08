from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import itertools as it
from ..core.data import BinnedData
from .metrics import pullDistribution, totalPullDistribution
from typing import Any, Callable
from .plot_utils import plotBinnedData, plotRaw, plotPPD, plotBlinding2D
from contextlib import contextmanager


def makeDiagnosticPlots2D(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    plot_saver: Callable,
    train_data: BinnedData | None = None,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | dict[str, BinnedData] | None = None,
    signal_template: BinnedData | dict[str, BinnedData] | None = None,
    prior_mean: jnp.ndarray | None = None,
    pred_cov: jnp.ndarray | None = None,
    kernel: Any | None = None,
    transform: Any | None = None,
) -> None:
    edges = test_data.edges
    X = test_data.X
    obs_Y = test_data.Y
    obs_V = test_data.V
    pulls = np.asarray(pullDistribution(test_data.Y, pred_mean, test_data.V))
    total_pulls = np.asarray(
        totalPullDistribution(test_data.Y, pred_mean, test_data.V, pred_var)
    )

    @contextmanager
    def makePlot(key):
        fig, ax = plt.subplots()
        try:
            yield ax
        finally:
            plot_saver(key, fig, ax)

    if train_data is not None:
        with makePlot("training_points") as ax:
            plotBinnedData(ax, train_data, cbar_title="Events")
            ax.set_title("Training Data (Blinded)")
            if test_data.axis_names and len(test_data.axis_names) >= 2:
                ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])

    with makePlot("gpr_mean") as ax:
        plotRaw(ax, edges, X, pred_mean, cbar_title="Events")
        ax.set_title("GP Mean Prediction")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Observed ---
    with makePlot("observed_outputs") as ax:
        plotRaw(ax, edges, X, obs_Y, cbar_title="Events")
        ax.set_title("Observed")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Signal (if provided) ---
    if signal_data is not None:
        sigs = (
            signal_data if isinstance(signal_data, dict) else {"injected": signal_data}
        )
        for lbl, sig in sigs.items():
            key = "injected_signal" if lbl == "injected" else f"injected_signal_{lbl}"
            with makePlot(key) as ax:
                plotBinnedData(ax, sig, cbar_title="Events")
                ax.set_title(f"Injected Signal: {lbl}")
                if test_data.axis_names and len(test_data.axis_names) >= 2:
                    ax.set_xlabel(test_data.axis_names[0])
                    ax.set_ylabel(test_data.axis_names[1])
                plotBlinding2D(ax, edges, X, blind_mask)

    # --- Signal Template (unscaled) ---
    if signal_template is not None:
        sigs = (
            signal_template
            if isinstance(signal_template, dict)
            else {"template": signal_template}
        )
        for lbl, sig in sigs.items():
            key = "signal_template" if lbl == "template" else f"signal_template_{lbl}"
            with makePlot(key) as ax:
                plotBinnedData(
                    ax, sig, cbar_title=r"Events ($\lambda^{\prime\prime} = 0.1$)"
                )
                ax.set_title(f"Signal Template: {lbl}")
                if test_data.axis_names and len(test_data.axis_names) >= 2:
                    ax.set_xlabel(test_data.axis_names[0])
                    ax.set_ylabel(test_data.axis_names[1])
                plotBlinding2D(ax, edges, X, blind_mask)

    # --- Observed variances ---
    with makePlot("observed_variances") as ax:
        plotRaw(ax, edges, X, obs_V, cbar_title="Observed Variances")
        ax.set_title("Observed Variances")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Predicted variances ---
    with makePlot("predicted_variances") as ax:
        plotRaw(ax, edges, X, pred_var, cbar_title="Predicted Variances")
        ax.set_title("Predicted Variances")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Relative uncertainty ---
    rel_unc = np.asarray(jnp.sqrt(pred_var) / jnp.clip(pred_mean, 1e-10))
    with makePlot("relative_uncertainty") as ax:
        plotRaw(
            ax,
            edges,
            X,
            jnp.array(rel_unc),
            cmin=0,
            cmax=0.1,
            cbar_title="Relative Uncertainty (σ/μ)",
        )
        ax.set_title("Relative Uncertainty (σ/μ)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Pull map ---
    with makePlot("pull_map") as ax:
        plotRaw(
            ax,
            edges,
            X,
            jnp.array(pulls),
            cmap="coolwarm",
            cmin=-3,
            cmax=3,
            cbar_title="Pull",
        )
        ax.set_title("Pull Map")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
        plotBlinding2D(ax, edges, X, blind_mask)

    # --- Covariance at blinding center ---
    if (
        pred_cov is not None
        and blind_mask is not None
        and np.any(np.asarray(blind_mask))
    ):
        mask = np.asarray(blind_mask)
        blinded_X = X[mask]
        center = np.mean(blinded_X, axis=0)
        dist = np.sum((blinded_X - center) ** 2, axis=1)
        rel_idx = np.argmin(dist)
        abs_idx = mask.nonzero()[0][rel_idx]

        cov_row = pred_cov[abs_idx, :]
        with makePlot("covariance_at_blind_center") as ax:
            plotRaw(ax, edges, X, cov_row)
            ax.set_title("Covariance at Blinding Center")
            if test_data.axis_names and len(test_data.axis_names) >= 2:
                ax.set_xlabel(test_data.axis_names[0])
            ax.set_ylabel(test_data.axis_names[1])
            plotBlinding2D(ax, edges, X, blind_mask)
            # Mark the center point
            ax.scatter(
                X[abs_idx, 0],
                X[abs_idx, 1],
                color="red",
                marker="x",
                s=100,
                label="Center",
            )

        if kernel is not None:
            x_norm = test_data.X
            if transform is not None:
                x_norm = transform.applyX(test_data.X)

            center_pt = x_norm[abs_idx : abs_idx + 1]
            try:
                kernel_values = np.asarray(
                    kernel.cross_covariance(center_pt, x_norm)
                ).ravel()
                with makePlot("kernel_at_blind_center") as ax:
                    plotRaw(
                        ax,
                        edges,
                        X,
                        kernel_values,
                        cbar_title=r"$K(x_{window center},x)$",
                    )
                    ax.set_title("Prior Kernel at Blinding Center")
                    if test_data.axis_names and len(test_data.axis_names) >= 2:
                        ax.set_xlabel(test_data.axis_names[0])
                        ax.set_ylabel(test_data.axis_names[1])
                    plotBlinding2D(ax, edges, X, blind_mask)
                    ax.scatter(
                        X[abs_idx, 0],
                        X[abs_idx, 1],
                        color="red",
                        marker="x",
                        s=100,
                        label="Center",
                    )
            except Exception:
                pass

    # --- Pull histograms ---
    bins = np.linspace(-5.0, 5.0, 21)
    gauss_x = np.linspace(-5, 5, 100)
    gauss_y = np.exp(-(gauss_x**2) / 2) / np.sqrt(2 * np.pi)

    def _plotSinglePullHist(p_vals, tag_name, title_prefix, key):
        fig_h, ax_h = plt.subplots()
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
        plot_saver(key, fig_h, ax_h)

    _plotSinglePullHist(pulls, "Stat", "Statistical Pull", "stat_pulls_hist")
    _plotSinglePullHist(total_pulls, "Total", "Total Pull", "total_pulls_hist")

    if transform is not None:
        norm_data = transform.applyToBinnedData(test_data)
        norm_pred_mean = transform.applyY(pred_mean)
        norm_edges = norm_data.edges
        norm_X = norm_data.X

        fig, ax = plt.subplots()
        plotRaw(ax, norm_edges, norm_X, norm_data.Y, cbar_title="Transformed Events")
        ax.set_title("Observed (Transformed Space)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(f"Transformed {test_data.axis_names[0]}")
            ax.set_ylabel(f"Transformed {test_data.axis_names[1]}")
        plotBlinding2D(ax, norm_edges, norm_X, blind_mask)
        plot_saver("transformed_observed", fig, ax)

        fig, ax = plt.subplots()
        plotRaw(ax, norm_edges, norm_X, norm_pred_mean, cbar_title="Transformed Events")
        ax.set_title("GP Mean Prediction (Transformed Space)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(f"Transformed {test_data.axis_names[0]}")
            ax.set_ylabel(f"Transformed {test_data.axis_names[1]}")
        plotBlinding2D(ax, norm_edges, norm_X, blind_mask)
        plot_saver("transformed_gpr_mean", fig, ax)

        fig, ax = plt.subplots()
        plotRaw(ax, norm_edges, norm_X, norm_data.V, cbar_title="Transformed Variances")
        ax.set_title("Observed Variances (Transformed Space)")
        if test_data.axis_names and len(test_data.axis_names) >= 2:
            ax.set_xlabel(f"Transformed {test_data.axis_names[0]}")
            ax.set_ylabel(f"Transformed {test_data.axis_names[1]}")
        plotBlinding2D(ax, norm_edges, norm_X, blind_mask)
        plot_saver("transformed_variances", fig, ax)


def makePosteriorPredictivePlots2D(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    plot_saver: Callable,
    blind_mask: jnp.ndarray | None = None,
    prefix: str = "ppc",
) -> None:
    reps = ppc_results["test_reps"]

    test_stats = ppc_results["test_stats"]
    for stat_name, regions in test_stats.items():
        for region_name, summary_stats in regions.items():
            obs_val = float(summary_stats["obs"])
            pvalue = float(summary_stats["pvalue"])

            fig, ax = plt.subplots()
            plotPPD(
                ax,
                reps[stat_name][region_name],
                obs_val,
                xlabel=f"Test Statistic: {stat_name} ({region_name})",
                pvalue=pvalue,
            )
            plot_saver(f"{prefix}_dist_{stat_name}_{region_name}", fig, ax)


def plotNNTransformation2D(
    kernel: Any,
    test_data: BinnedData,
    plot_saver: Callable,
    transform: Any | None = None,
    blind_mask: jnp.ndarray | None = None,
) -> None:
    from ..inference.kernels import DeepWarpingKernel, DeepTransformKernel

    if not isinstance(kernel, (DeepWarpingKernel, DeepTransformKernel)):
        return
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

    fig, ax = plt.subplots()
    for i in range(nx):
        ax.plot(xt[i, :], yt[i, :], color="purple", alpha=0.5, lw=0.5)
    for j in range(ny):
        ax.plot(xt[:, j], yt[:, j], color="purple", alpha=0.5, lw=0.5)

    if blind_mask is not None and jnp.any(blind_mask):
        np_edges = tuple(np.asarray(e) for e in edges)
        mask_grid, _ = np.histogramdd(
            np.asarray(test_data.X), bins=np_edges, weights=blind_mask.astype(float)
        )
        mask_grid = mask_grid.astype(bool)
        ex, ey = np_edges
        padded = np.pad(
            mask_grid, ((1, 1), (1, 1)), mode="constant", constant_values=False
        )

        def transform_pts(pts):
            pts_norm = pts
            if transform is not None:
                pts_norm = transform.applyX(pts)
            return np.asarray(kernel.network(pts_norm))

        highlight_color = "magenta"
        highlight_lw = 2
        n_interp = 10

        for i, j in it.product(range(len(ex) - 1), range(len(ey))):
            if padded[i + 1, j] != padded[i + 1, j + 1]:
                seg_x = jnp.linspace(ex[i], ex[i + 1], n_interp)
                seg_y = jnp.full_like(seg_x, ey[j])
                pts = jnp.stack([seg_x, seg_y], axis=-1)
                pts_t = transform_pts(pts)
                ax.plot(
                    pts_t[:, 0],
                    pts_t[:, 1],
                    color=highlight_color,
                    lw=highlight_lw,
                    zorder=10,
                )

        for j, i in it.product(range(len(ey) - 1), range(len(ex))):
            if padded[i, j + 1] != padded[i + 1, j + 1]:
                seg_y = jnp.linspace(ey[j], ey[j + 1], n_interp)
                seg_x = jnp.full_like(seg_y, ex[i])
                pts = jnp.stack([seg_x, seg_y], axis=-1)
                pts_t = transform_pts(pts)
                ax.plot(
                    pts_t[:, 0],
                    pts_t[:, 1],
                    color=highlight_color,
                    lw=highlight_lw,
                    zorder=10,
                )

    ax.set_title("NN Kernel Grid Transformation")
    ax.set_xlabel("Transformed X")
    ax.set_ylabel("Transformed Y")

    plot_saver("nn_transform_grid", fig, ax)


def makeSmoothingPlots2D(
    smoothed_data: BinnedData,
    original_data: BinnedData,
    pred_mean: jnp.ndarray,
    plot_saver: Callable,
    pred_cov: jnp.ndarray | None = None,
) -> None:
    """Generate plots comparing smoothed background to original in 2D."""
    edges = original_data.edges
    X = original_data.X
    pred_Y = np.asarray(pred_mean)
    smooth_Y = np.asarray(smoothed_data.Y)

    # 1. Original noisy data
    fig, ax = plt.subplots()
    plotRaw(ax, edges, X, original_data.Y)
    ax.set_title("Original Noisy MC")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    plot_saver("smoothing_original_2d", fig, ax)

    # 2. GPR Mean Prediction
    fig, ax = plt.subplots()
    plotRaw(ax, edges, X, pred_Y)
    ax.set_title("GPR Latent Mean")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    plot_saver("smoothing_gpr_mean_2d", fig, ax)

    # 3. Smoothed Poisson-sampled data
    fig, ax = plt.subplots()
    plotRaw(ax, edges, X, smooth_Y)
    ax.set_title("Smoothed Background (Poisson sample)")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    plot_saver("smoothing_sampled_2d", fig, ax)

    # 4. Consistency: (Smoothed - Mean) / sqrt(Mean)
    resid = (smooth_Y - pred_Y) / np.sqrt(np.maximum(pred_Y, 1e-10))
    fig, ax = plt.subplots()
    plotRaw(ax, edges, X, resid, cmap="coolwarm", cmin=-4, cmax=4)
    ax.set_title("Smoothing Consistency (Resid / sqrt(μ))")
    if original_data.axis_names and len(original_data.axis_names) >= 2:
        ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(original_data.axis_names[1])
    plot_saver("smoothing_resid_2d", fig, ax)

    # 5. GPR Uncertainties
    if pred_cov is not None:
        pred_var = np.diag(pred_cov)
        pred_std = np.sqrt(pred_var)
        rel_unc = pred_std / np.maximum(pred_Y, 1e-10)

        # Absolute Uncertainty (Sigma)
        fig, ax = plt.subplots()
        plotRaw(ax, edges, X, jnp.array(pred_std))
        ax.set_title(r"GPR Absolute Uncertainty ($\sigma$)")
        if original_data.axis_names and len(original_data.axis_names) >= 2:
            ax.set_xlabel(original_data.axis_names[0])
            ax.set_ylabel(original_data.axis_names[1])
        plot_saver("smoothing_sigma_2d", fig, ax)

        # Relative Uncertainty (Sigma/Mu)
        fig, ax = plt.subplots()
        plotRaw(ax, edges, X, jnp.array(rel_unc), cmin=0, cmax=0.5)
        ax.set_title(r"GPR Relative Uncertainty ($\sigma/\mu$)")
        if original_data.axis_names and len(original_data.axis_names) >= 2:
            ax.set_xlabel(original_data.axis_names[0])
            ax.set_ylabel(original_data.axis_names[1])
        plot_saver("smoothing_rel_unc_2d", fig, ax)
