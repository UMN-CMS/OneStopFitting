from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import jax.numpy as jnp

from ..core.data import AnalysisState
from ..utils import getSignal, getCategory, getRecoCategory
from ..data.loading import (
    FileLoader,
    extractHistogram,
    extractMetadata,
    histToBinnedData,
)

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig


logger = logging.getLogger(__name__)


def _deriveSignalLabel(metadata: dict, path) -> str:
    return metadata["dataset_name"]


def _loadSignals(config):
    signals = {}
    signal_hists = {}
    signal_metadata = {}
    first_signal = None
    first_signal_hist = None
    first_sig_metadata = None

    for path in config.signalPaths:
        loader = FileLoader.forPath(path)
        raw = loader.load(path)
        sig_hist = extractHistogram(raw)
        if config.signal_pre_scale != 1.0:
            logger.info(f"Pre-scaling signal by {config.signal_pre_scale}")
            sig_hist = sig_hist * config.signal_pre_scale

        sig_binned = histToBinnedData(sig_hist, rebin=config.rebin, variation="central")
        sig_meta = extractMetadata(raw)
        label = _deriveSignalLabel(sig_meta, path)

        logger.info(f"Loaded signal from {path}. label: {label}")

        signals[label] = sig_binned
        signal_hists[label] = sig_hist
        signal_metadata[label] = sig_meta

        if first_signal is None:
            first_signal = sig_binned
            first_signal_hist = sig_hist
            first_sig_metadata = sig_meta

    return (
        signals,
        signal_hists,
        signal_metadata,
        first_signal,
        first_signal_hist,
        first_sig_metadata,
    )


def loadData(config: PipelineConfig) -> AnalysisState:
    """Load background and optional signal data."""

    logger.info(f"Loading background from {config.background_path}")
    loader = FileLoader.forPath(config.background_path)
    bkg_raw = loader.load(config.background_path)
    bkg_hist = extractHistogram(bkg_raw)
    background = histToBinnedData(bkg_hist, rebin=config.rebin, variation="central")

    file_metadata = extractMetadata(bkg_raw)
    combined_metadata = {
        **config.metadata,
        **file_metadata,
        "injection_rate": config.injection_rate,
        "window_type": type(config.window).__name__ if config.window else "none",
        "rebin": config.rebin,
        "min_counts": config.min_counts,
    }

    (
        signals,
        signal_hists,
        signal_metadata,
        first_signal,
        first_signal_hist,
        first_sig_metadata,
    ) = _loadSignals(config)

    logger.info(f"Loaded {len(signals)} signals")
    for signal in signals:
        logger.info(f"  {signal}: {jnp.sum(signals[signal].Y)} events")

    if first_sig_metadata is not None:
        reco_category = getRecoCategory(first_sig_metadata["name"])
        combined_metadata = {
            **combined_metadata,
            **first_sig_metadata,
            "reco_category": reco_category,
        }

    return AnalysisState(
        config=config,
        background=background,
        signal=first_signal,
        injection_rate=config.injection_rate,
        background_hist=bkg_hist,
        signal_hist=first_signal_hist,
        signals=signals,
        signal_hists=signal_hists,
        signal_metadata=signal_metadata,
        metadata=combined_metadata,
        background_metadata=file_metadata,
    )
