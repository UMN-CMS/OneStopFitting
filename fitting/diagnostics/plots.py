"""Diagnostic plot dispatcher.

Routes to dimension-specific plotting modules based on data dimensionality.
"""

from __future__ import annotations

import jax.numpy as jnp

from typing import Any

from ..core.data import BinnedData
from .plots_1d import makeDiagnosticPlots1D, makePosteriorPredictivePlots1D
from .plots_2d import makeDiagnosticPlots2D, makePosteriorPredictivePlots2D


def makePosteriorPredictivePlots(
    ppc_results: dict[str, Any],
    test_data: BinnedData,
    blind_mask: jnp.ndarray | None = None,
) -> dict[str, tuple]:
    """Create posterior predictive check plots appropriate for the dimensionality.

    Args:
        ppc_results: Dictionary returned by posteriorPredictiveCheck.
        test_data: Full-domain test data.
        blind_mask: Boolean mask for blinded bins.

    Returns:
        Dict of plot name -> (figure, axes).
    """
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
) -> dict[str, tuple]:
    """Create diagnostic plots appropriate for the data dimensionality.

    Dispatches to 1D or 2D plot implementations based on test_data.ndim.

    Args:
        pred_mean: Predicted mean in real space.
        pred_var: Predicted variance in real space.
        test_data: Full-domain test data.
        train_data: Training data (for 2D visualization).
        blind_mask: Boolean mask for blinded bins.
        signal_data: Optional signal data to overlay.

    Returns:
        Dict of plot name -> (figure, axes).

    Raises:
        NotImplementedError: For >2D data.
    """
    ndim = test_data.ndim

    if ndim == 1:
        return makeDiagnosticPlots1D(
            pred_mean=pred_mean,
            pred_var=pred_var,
            test_data=test_data,
            blind_mask=blind_mask,
            signal_data=signal_data,
        )
    elif ndim == 2:
        return makeDiagnosticPlots2D(
            pred_mean=pred_mean,
            pred_var=pred_var,
            test_data=test_data,
            train_data=train_data,
            blind_mask=blind_mask,
            signal_data=signal_data,
            signal_template=signal_template,
        )
    else:
        raise NotImplementedError(
            f"Diagnostic plots for {ndim}D data are not yet implemented. "
            f"Consider contributing a plots_{ndim}d.py module."
        )
