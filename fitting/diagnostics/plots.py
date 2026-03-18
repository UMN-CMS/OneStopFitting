from __future__ import annotations

import jax.numpy as jnp

from typing import Any

from ..core.data import BinnedData
from .plots_1d import makeDiagnosticPlots1D, makePosteriorPredictivePlots1D, makeSmoothingPlots1D
from .plots_2d import makeDiagnosticPlots2D, makePosteriorPredictivePlots2D, makeSmoothingPlots2D


def makeSmoothingPlots(
    smoothed_data: BinnedData,
    original_data: BinnedData,
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Generate plots comparing smoothed background to original."""
    ndim = original_data.ndim
    if ndim == 1:
        return makeSmoothingPlots1D(
            smoothed_data=smoothed_data,
            original_data=original_data,
            pred_mean=pred_mean,
            pred_cov=pred_cov,
        )
    else:
        return makeSmoothingPlots2D(
            smoothed_data=smoothed_data,
            original_data=original_data,
            pred_mean=pred_mean,
            pred_cov=pred_cov,
        )


def makePosteriorPredictivePlots(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ndim = test_data.ndim
    if ndim == 1:
        return makePosteriorPredictivePlots1D(
            ppc_results=ppc_results,
            test_data=test_data,
            blind_mask=blind_mask,
        )
    else:
        return makePosteriorPredictivePlots2D(
            ppc_results=ppc_results,
            test_data=test_data,
            blind_mask=blind_mask,
        )


def makeDiagnosticPlots(
    pred_mean: jnp.ndarray,
    pred_var: jnp.ndarray,
    test_data: BinnedData,
    train_data: BinnedData | None = None,
    blind_mask: jnp.ndarray | None = None,
    signal_data: BinnedData | None = None,
    signal_template: BinnedData | None = None,
    prior_mean: jnp.ndarray | None = None,
    kernel: Any | None = None,
    transform: Any | None = None,
    pred_cov: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    ndim = test_data.ndim

    if ndim == 1:
        plots = makeDiagnosticPlots1D(
            pred_mean=pred_mean,
            pred_var=pred_var,
            test_data=test_data,
            blind_mask=blind_mask,
            signal_data=signal_data,
            prior_mean=prior_mean,
            pred_cov=pred_cov,
        )
    elif ndim == 2:
        plots = makeDiagnosticPlots2D(
            pred_mean=pred_mean,
            pred_var=pred_var,
            test_data=test_data,
            train_data=train_data,
            blind_mask=blind_mask,
            signal_data=signal_data,
            signal_template=signal_template,
            prior_mean=prior_mean,
            pred_cov=pred_cov,
        )
    else:
        raise NotImplementedError(
            f"Diagnostic plots for {ndim}D data are not yet implemented. "
            f"Consider contributing a plots_{ndim}d.py module."
        )

    # --- NN Kernel Transformation Plots ---
    if kernel is not None:
        from .plots_1d import plotNNTransformation1D
        from .plots_2d import plotNNTransformation2D
        from ..inference.kernels import DeepKernelFunction
        from flax import nnx

        # Recursively find all DeepKernelFunctions
        nn_kernels = []

        def find_nn_kernels(k):
            if isinstance(k, DeepKernelFunction):
                nn_kernels.append(k)
            # Handle standard gpjax combination kernels
            if hasattr(k, "kernels") and isinstance(k.kernels, (list, nnx.List)):
                for sub_k in k.kernels:
                    find_nn_kernels(sub_k)
            # Handle multiplication/addition which might nest them
            if hasattr(k, "k1"):
                find_nn_kernels(k.k1)
            if hasattr(k, "k2"):
                find_nn_kernels(k.k2)

        find_nn_kernels(kernel)

        for i, nnk in enumerate(nn_kernels):
            suffix = f"_{i}" if len(nn_kernels) > 1 else ""
            if ndim == 1:
                nk_plots = plotNNTransformation1D(
                    nnk, test_data, transform=transform, blind_mask=blind_mask
                )
            else:
                nk_plots = plotNNTransformation2D(
                    nnk, test_data, transform=transform, blind_mask=blind_mask
                )

            for name, fig_ax in nk_plots.items():
                plots[f"{name}{suffix}"] = fig_ax

    return plots
