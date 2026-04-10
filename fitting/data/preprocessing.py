from __future__ import annotations

import logging

import jax.numpy as jnp

from ..core.data import AnalysisState, BinnedData
from .windowing import Window, fitGaussianWindow
import attrs

logger = logging.getLogger(__name__)


def applyDomainMask(
    data: BinnedData, min_counts: float | None = None, window: Window | None = None
) -> tuple[BinnedData, jnp.ndarray]:
    min_counts = min_counts if min_counts is not None else 1.0

    mask = data.Y >= min_counts

    logger.info(f"Dropping {jnp.count_nonzero(~mask)} bins with < {min_counts} counts")

    if window is not None:
        custom_mask = window(data.X)
        mask = mask & custom_mask

    return data.masked(mask), mask


def splitTrainTest(
    data: BinnedData,
    window: Window | None,
) -> tuple[BinnedData, BinnedData, jnp.ndarray]:
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
) -> AnalysisState:
    if state.background is None:
        raise ValueError(
            "AnalysisState.background must be populated before preprocessing"
        )

    # Determine the window
    domain_window = state.config.domain_window

    window = state.window
    if window is None and state.signal is not None:
        logger.info("Fitting Gaussian window from signal data")
        spread = (
            state.config.get("window_spread", 1.3)
            if isinstance(state.config, dict)
            else getattr(state.config, "window_spread", 1.3)
        )
        logger.info(f"Fitting Gaussian window from signal data with spread {spread}")
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
    else:
        logger.info("Background only fit, no signal injection")

    # Apply domain mask
    domain_data, domain_mask = applyDomainMask(
        to_estimate, min_counts=min_counts, window=domain_window
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
