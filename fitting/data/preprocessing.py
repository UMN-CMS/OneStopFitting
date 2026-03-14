"""Preprocessing: extract regression data, apply masks, split train/test."""

from __future__ import annotations

import logging

import jax.numpy as jnp

from ..core.data import AnalysisState, BinnedData
from .windowing import Window, fitGaussianWindow
import attrs

logger = logging.getLogger(__name__)


def applyDomainMask(
    data: BinnedData,
    min_counts: float = 10.0,
    domain_mask_fn: callable | None = None,
) -> tuple[BinnedData, jnp.ndarray]:
    """Apply domain cuts to remove bins outside the fit region.

    Args:
        data: Full histogram data.
        min_counts: Minimum bin count to include.
        domain_mask_fn: Optional callable(X) -> bool mask for additional cuts.

    Returns:
        Tuple of (masked data, boolean mask).
    """
    mask = data.Y >= min_counts

    if domain_mask_fn is not None:
        custom_mask = domain_mask_fn(data.X)
        mask = mask & custom_mask

    return data.masked(mask), mask


def splitTrainTest(
    data: BinnedData,
    window: Window | None,
) -> tuple[BinnedData, BinnedData, jnp.ndarray]:
    """Split data into training (outside window) and test (full domain).

    The test data is the full domain data. The training data is the
    full domain minus the blinded window.

    Args:
        data: Domain-masked data.
        window: Blinding window. If None, all data is used for training.

    Returns:
        Tuple of (train_data, test_data, blind_mask).
        blind_mask is True for bins inside the window.
    """
    test_data = data

    if window is not None:
        blind_mask = window(data.X)
        train_data = data.masked(~blind_mask)
    else:
        blind_mask = jnp.zeros(data.nbins, dtype=bool)
        train_data = data

    return train_data, test_data, blind_mask


def preprocess(
    state: AnalysisState,
    min_counts: float = 10.0,
    domain_mask_fn: callable | None = None,
) -> AnalysisState:
    """Run the full preprocessing pipeline on an AnalysisState.

    Steps:
    1. If signal is available and no window is set, fit a Gaussian window
    2. Optionally inject signal into background
    3. Apply domain mask (min counts + custom)
    4. Split into train/test using the window

    Args:
        state: AnalysisState with background (and optionally signal) populated.
        min_counts: Minimum bin count threshold.
        domain_mask_fn: Optional additional domain mask function.

    Returns:
        Updated AnalysisState with train_data, test_data, domain_mask,
        blind_mask, and window populated.
    """
    if state.background is None:
        raise ValueError(
            "AnalysisState.background must be populated before preprocessing"
        )

    # Determine the window
    window = state.window
    if window is None and state.signal is not None:
        logger.info("Fitting Gaussian window from signal data")
        spread = (
            state.config.get("window_spread", 1.3)
            if isinstance(state.config, dict)
            else getattr(state.config, "window_spread", 1.3)
        )
        window = fitGaussianWindow(state.signal, spread=spread)

    # Inject signal if requested
    to_estimate = state.background
    if state.signal is not None and state.injection_rate != 0.0:
        logger.info(f"Injecting signal with rate {state.injection_rate}")
        injected_Y = state.background.Y + state.injection_rate * state.signal.Y
        injected_V = state.background.V + state.injection_rate**2 * state.signal.V
        to_estimate = BinnedData(
            X=state.background.X,
            Y=injected_Y,
            V=injected_V,
            edges=state.background.edges,
        )

    # Apply domain mask
    domain_data, domain_mask = applyDomainMask(
        to_estimate, min_counts=min_counts, domain_mask_fn=domain_mask_fn
    )

    # Split train/test
    train_data, test_data, blind_mask = splitTrainTest(domain_data, window)

    logger.info(
        f"Preprocessing complete: {train_data.nbins} train bins, "
        f"{test_data.nbins} test bins, "
        f"{int(jnp.sum(blind_mask))} blinded bins"
    )

    return attrs.evolve(
        state,
        train_data=train_data,
        test_data=test_data,
        domain_mask=domain_mask,
        blind_mask=blind_mask,
        window=window,
    )
