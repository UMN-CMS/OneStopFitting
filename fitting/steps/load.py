from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..core.data import AnalysisState
from ..data.loading import (
    FileLoader,
    extractHistogram,
    extractMetadata,
    histToBinnedData,
)

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig


logger = logging.getLogger(__name__)


def loadData(config: PipelineConfig) -> AnalysisState:
    """Load background and optional signal data."""

    logger.info(f"Loading background from {config.background_path}")
    loader = FileLoader.forPath(config.background_path)
    bkg_raw = loader.load(config.background_path)
    bkg_hist = extractHistogram(bkg_raw)
    background = histToBinnedData(bkg_hist, rebin=config.rebin, variation="central")

    # Extract and merge file-level metadata
    file_metadata = extractMetadata(bkg_raw)
    combined_metadata = {
        **config.metadata,
        **file_metadata,
        "injection_rate": config.injection_rate,
        "window_spread": config.window_spread,
        "rebin": config.rebin,
        "min_counts": config.min_counts,
    }

    signal = None
    signal_hist = None
    if config.signal_path is not None:
        logger.info(f"Loading signal from {config.signal_path}")
        sig_loader = FileLoader.forPath(config.signal_path)
        sig_raw = sig_loader.load(config.signal_path)
        sig_hist_full = extractHistogram(sig_raw)
        signal = histToBinnedData(
            sig_hist_full, rebin=config.rebin, variation="central"
        )
        signal_hist = sig_hist_full  # keep all variations for downstream

        # Merge signal metadata (overrides background metadata on conflict)
        sig_metadata = extractMetadata(sig_raw)
        combined_metadata = {**combined_metadata, **sig_metadata}

    # FIXME: Quick and dirty sqrt transform (removable)
    if True:
        import jax.numpy as jnp
        from ..core.data import BinnedData

        def _sqrt_transform(data: BinnedData) -> BinnedData:
            return BinnedData(
                X=data.X,
                Y=jnp.sqrt(jnp.maximum(data.Y, 0)),
                V=0.25 * jnp.ones_like(data.Y),
                edges=data.edges,
                axis_names=data.axis_names,
            )

        background = _sqrt_transform(background)
        if signal is not None:
            signal = _sqrt_transform(signal)
    # END QUICK AND DIRTY

    return AnalysisState(
        config=config,
        background=background,
        signal=signal,
        injection_rate=config.injection_rate,
        background_hist=bkg_hist,
        signal_hist=signal_hist,
        metadata=combined_metadata,
        background_metadata=file_metadata,
    )
