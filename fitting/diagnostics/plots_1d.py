from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from typing import Any

from ..core.data import BinnedData
from .metrics import pullDistribution, totalPullDistribution
from .plot_utils import addAxesToHist, plotBinnedData, plotPPD


def makePosteriorPredictivePlots1D(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ret = {}

    X = np.asarray(test_data.X).ravel()

    fig, ax = plt.subplots(layout="tight")

    summary = ppc_results["summary"]
    reps = ppc_results["test_reps"]
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

    test_stats = ppc_results["test_stats"]
    for stat_name, regions in test_stats.items():
        for region_name, summary_stats in regions.items():
            obs_val = float(summary_stats["obs"])
            rep_vals = reps
            pvalue = float(summary_stats["pvalue"])
            fig, ax = plt.subplots(layout="tight")
            plotPPD(
                ax,
                reps[stat_name][region_name],
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
    prior_mean: jnp.ndarray | None = None,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ret = {}
    X = np.asarray(test_data.X).ravel()
    obs_V = np.asarray(test_data.V)
    pred_Y = np.asarray(pred_mean)
    pred_V = np.asarray(pred_var)
    pred_std = np.sqrt(pred_V)
    pulls = np.asarray(pullDistribution(test_data.Y, pred_mean, test_data.V))
    total_pulls = np.asarray(
        totalPullDistribution(test_data.Y, pred_mean, test_data.V, pred_var)
    )

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

    if prior_mean is not None:
        ax.plot(
            X,
            np.asarray(prior_mean).ravel(),
            color="blue",
            ls="--",
            label="Prior Mean (Parametric)",
        )

    if blind_mask is not None and np.any(blind_mask):
        np_mask = np.asarray(blind_mask)
        w_min = X[np_mask].min()
        w_max = X[np_mask].max()
        for boundary in [w_min, w_max]:
            ax.axvline(boundary, ls="--", color="gray", alpha=0.5)
            ax.bottom_axes[0].axvline(boundary, ls="--", color="gray", alpha=0.5)

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

    ret.update(_plotPullHistograms(pulls, blind_mask, tag="stat"))
    ret.update(_plotPullHistograms(total_pulls, blind_mask, tag="total"))

    return ret


def _plotPullHistograms(
    pulls: np.ndarray,
    blind_mask: jnp.ndarray | None = None,
    tag: str = "stat",
) -> dict[str, tuple]:
    ret = {}
    bins = np.linspace(-5.0, 5.0, 21)
    gauss_x = np.linspace(-5, 5, 100)
    gauss_y = np.exp(-(gauss_x**2) / 2) / np.sqrt(2 * np.pi)

    # Global pulls
    fig, ax = plt.subplots(layout="tight")
    ax.hist(pulls, bins=bins, density=True, alpha=0.7, label="All bins")
    ax.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
    ax.set_xlabel(
        rf"{tag.capitalize()} Pull: $(N_{{obs}} - N_{{pred}}) / \sigma_{{total}}$"
        if tag == "total"
        else r"Stat Pull: $(N_{obs} - N_{pred}) / \sigma_{obs}$"
    )
    ax.set_ylabel("Density")
    ax.legend()
    ret[f"global_{tag}_pulls_hist"] = (fig, ax)

    if blind_mask is not None and np.any(np.asarray(blind_mask)):
        np_mask = np.asarray(blind_mask)
        fig, ax = plt.subplots(layout="tight")
        ax.hist(pulls[np_mask], bins=bins, density=True, alpha=0.7, label="Window bins")
        ax.plot(gauss_x, gauss_y, "k-", label="Unit Normal")
        ax.set_xlabel(
            rf"{tag.capitalize()} Pull: $(N_{{obs}} - N_{{pred}}) / \sigma_{{total}}$"
            if tag == "total"
            else r"Stat Pull: $(N_{obs} - N_{pred}) / \sigma_{obs}$"
        )
        ax.set_ylabel("Density")
        ax.legend()
        ret[f"window_{tag}_pulls_hist"] = (fig, ax)

    return ret


def plotNNTransformation1D(
    kernel: Any,
    test_data: BinnedData,
    transform: Any | None = None,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    from ..inference.kernels import DeepKernelFunction

    if not isinstance(kernel, DeepKernelFunction):
        return {}

    X = np.asarray(test_data.X).ravel()

    X_norm = test_data.X
    if transform is not None:
        X_norm = transform.applyX(test_data.X)

    Xt = np.asarray(kernel.network(X_norm)).ravel()

    fig, ax = plt.subplots(layout="tight")
    ax.plot(X, Xt, color="purple", lw=2)

    if blind_mask is not None and np.any(np.asarray(blind_mask)):
        np_mask = np.asarray(blind_mask)
        w_min = X[np_mask].min()
        w_max = X[np_mask].max()
        ax.axvspan(w_min, w_max, color="magenta", alpha=0.1, label="Blinded Window")
        for boundary in [w_min, w_max]:
            ax.axvline(boundary, ls="--", color="magenta", alpha=0.5)

    ax.set_title("NN Kernel Transformation")
    if test_data.axis_names:
        ax.set_xlabel(f"Input: {test_data.axis_names[0]}")
    ax.set_ylabel("Transformed Space")
    ax.grid(True, alpha=0.3)
    if blind_mask is not None:
        ax.legend()

    return {"nn_transform_1d": (fig, ax)}


def makeSmoothingPlots1D(
    smoothed_data: BinnedData,
    original_data: BinnedData,
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ret = {}
    X = np.asarray(original_data.X).ravel()
    pred_Y = np.asarray(pred_mean)

    fig, ax = plt.subplots(layout="tight")
    addAxesToHist(ax, size=1.5)

    plotBinnedData(
        ax,
        original_data,
        histtype="errorbar",
        color="black",
        alpha=0.3,
        label="Original MC",
    )

    # GPR Mean
    ax.plot(X, pred_Y, color="orange", label="GPR Latent Mean", lw=2)
    if pred_cov is not None:
        pred_std = np.sqrt(np.diag(pred_cov))
        ax.fill_between(
            X,
            pred_Y + pred_std,
            pred_Y - pred_std,
            color="orange",
            alpha=0.3,
            label=r"$\pm\sigma_{GPR}$",
        )

    # Smoothed Poisson-sampled background
    plotBinnedData(
        ax,
        smoothed_data,
        histtype="step",
        color="red",
        label="Smoothed (Poisson)",
        lw=1.5,
    )

    ax.set_ylabel("Counts")
    ax.legend()

    # Residuals: (Smoothed - GPR Mean) / sqrt(GPR Mean)
    # This checks if the Poisson sampling is consistent with the latent mean
    resid = (np.asarray(smoothed_data.Y) - pred_Y) / np.sqrt(np.maximum(pred_Y, 1e-10))

    ratio_ax = ax.bottom_axes[0]
    ratio_ax.set_ylim(-4, 4)
    ratio_ax.plot(X, resid, "o", color="red", markersize=2)
    ratio_ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ratio_ax.set_ylabel(r"$\frac{N_{smooth} - \mu_{GPR}}{\sqrt{\mu_{GPR}}}$")
    if original_data.axis_names:
        ratio_ax.set_xlabel(original_data.axis_names[0])

    ret["smoothing_summary_1d"] = (fig, ax)

    # 2. GPR Uncertainties
    if pred_cov is not None:
        pred_var = np.diag(pred_cov)
        pred_std = np.sqrt(pred_var)
        rel_unc = pred_std / np.maximum(pred_Y, 1e-10)

        # Absolute Uncertainty (Sigma)
        fig, ax = plt.subplots(layout="tight")
        ax.plot(X, pred_std, color="orange", lw=2)
        ax.set_title(r"GPR Absolute Uncertainty ($\sigma$)")
        if original_data.axis_names:
            ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(r"$\sigma$")
        ret["smoothing_sigma_1d"] = (fig, ax)

        # Relative Uncertainty (Sigma/Mu)
        fig, ax = plt.subplots(layout="tight")
        ax.plot(X, rel_unc, color="orange", lw=2)
        ax.set_ylim(0, 0.5)
        ax.set_title(r"GPR Relative Uncertainty ($\sigma/\mu$)")
        if original_data.axis_names:
            ax.set_xlabel(original_data.axis_names[0])
        ax.set_ylabel(r"$\sigma / \mu$")
        ret["smoothing_rel_unc_1d"] = (fig, ax)

    return ret
