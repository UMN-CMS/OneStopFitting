from __future__ import annotations

import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import jax.numpy as jnp
from ..core.data import AnalysisState
from .plot_utils import savePlots, plotRaw
from ..inference.prediction import computeScaledEigenvectors
import jax

logger = logging.getLogger(__name__)


def plotCombineInputs(
    state: AnalysisState, use_window_mask: bool = True
) -> dict[str, tuple]:

    if use_window_mask:
        blind_mask = state.blind_mask
    else:
        blind_mask = np.ones_like(state.test_data.Y, dtype=bool)

    num_active_bins = jnp.count_nonzero(blind_mask)
    x = np.arange(num_active_bins)

    fig, ax = plt.subplots()
    mplhep.cms.label(ax=ax, data=True, label="Preliminary")

    bg = np.asarray(state.pred_mean[blind_mask])
    ax.step(x, bg, where="mid", label="Background (Smoothed)", color="blue", lw=2)

    if state.signal is not None:
        sig_y = state.signal.Y
        if state.domain_mask is not None and len(sig_y) != num_active_bins:
            sig_y = sig_y[np.asarray(state.domain_mask)][blind_mask]
        ax.step(x, sig_y, where="mid", label="Signal", color="red", lw=2)

    obs = np.asarray(state.test_data.Y[blind_mask])
    ax.scatter(x, obs, color="black", label="Data (Observation)", s=10)

    ax.set_yscale("log")
    ax.set_xlabel("Linear Bin Index")
    ax.set_ylabel("Counts")
    ax.legend()

    plots = {"combine_inputs_1d": (fig, ax)}
    return plots


def verifyEigenvariations(
    state: AnalysisState,
    n_samples: int = 1000,
    use_window_mask: bool = True,
    eigenvar_threshold=0.001,
) -> dict[str, tuple]:
    """Verify that eigenvariations faithfully emulate the true MVN."""

    if use_window_mask:
        blind_mask = state.blind_mask
    else:
        blind_mask = np.ones_like(state.test_data.Y, dtype=bool)

    pred_cov = state.pred_cov[blind_mask, :][:, blind_mask]
    eigenvalues, scaled_vecs = computeScaledEigenvectors(
        pred_cov, threshold_fraction=eigenvar_threshold,
    )
    recon_cov = scaled_vecs @ scaled_vecs.T

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(pred_cov, interpolation="none")
    axes[0].set_title("True Covariance")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(recon_cov, interpolation="none")
    axes[1].set_title("Reconstructed Covariance")
    fig.colorbar(im1, ax=axes[1])

    diff = pred_cov - recon_cov
    im2 = axes[2].imshow(diff, interpolation="none", cmap="RdBu_r")
    axes[2].set_title("Difference (True - Recon)")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("Bin Index")
        ax.set_ylabel("Bin Index")

    plots = {"covariance_reconstruction": (fig, axes)}

    rng = jax.random.PRNGKey(42)

    true_samples = jax.random.multivariate_normal(
        rng, jnp.zeros(pred_cov.shape[0]), pred_cov, shape=(n_samples,)
    )
    true_samples = np.asarray(true_samples)

    z = np.random.normal(size=(n_samples, scaled_vecs.shape[1]))
    recon_samples = z @ scaled_vecs.T

    fig2, ax2 = plt.subplots()
    true_vars = np.var(true_samples, axis=0)
    recon_vars = np.var(recon_samples, axis=0)

    ax2.plot(true_vars, label="True Variance (Sampled)", color="black", lw=2)
    ax2.plot(
        recon_vars, label="Reconstructed Variance (Sampled)", color="red", ls="--", lw=2
    )
    ax2.set_xlabel("Bin Index")
    ax2.set_ylabel("Variance")
    ax2.set_title(f"Variance across bins ($N={n_samples}$ samples)")
    ax2.legend()

    plots["variance_comparison_sampled"] = (fig2, ax2)

    frob_norm_diff = np.linalg.norm(diff)
    frob_norm_true = np.linalg.norm(pred_cov)
    rel_error = frob_norm_diff / frob_norm_true

    logger.info(f"Verified eigenvariations: Relative Frobenius Error = {rel_error:.4g}")
    logger.info(f"Number of eigenvariations kept: {scaled_vecs.shape[1]}")

    return plots


def visualizeEigenvariations(
    state: AnalysisState,
    use_window_mask: bool = True,
) -> dict[str, tuple]:
    """Verify that eigenvariations faithfully emulate the true MVN."""

    if use_window_mask:
        blind_mask = state.blind_mask
    else:
        blind_mask = np.ones_like(state.test_data.Y, dtype=bool)

    pred_cov = state.pred_cov[blind_mask, :][:, blind_mask]

    eigenvalues, scaled_vecs = computeScaledEigenvectors(
        pred_cov, threshold_fraction=state.config.combine.eigenvar_threshold
    )

    base = state.test_data
    plots = {}
    for i in range(scaled_vecs.shape[1]):
        ev = scaled_vecs[:,i]
        fig, ax = plt.subplots()
        plotRaw(ax, base.edges, base.X[blind_mask], ev)
        plots[f"eigenvar_{i}"] = (fig, ax)

    return plots
