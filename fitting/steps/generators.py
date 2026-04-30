from __future__ import annotations

from typing import Any

from ..inference.prediction import drawPoissonSamples
from ..data.loading import histToBinnedData
import jax
import hist
import numpy as np
from jax import random
import jax.numpy as jnp

from ..core.data import AnalysisState, BinnedData
from ..inference.prediction import predictInRealSpace
from ..diagnostics.plots import makeSmoothingPlots


def generateAsimovSmoothed(
    state: AnalysisState,
    rng_key: jax.Array,
) -> tuple[list[hist.Hist], dict[str, Any]]:
    assert state.test_data is not None
    assert state.transform is not None
    assert state.background is not None

    if state.training_result is None or state.dataset is None:
        raise ValueError("Cannot generate smoothed background without training.")

    pred_key, sample_key = random.split(rng_key)
    test_data = state.test_data

    pred_mean, pred_cov = predictInRealSpace(
        posterior=state.training_result.posterior,
        dataset_train=state.dataset,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=pred_key,
    )
    if len(state.test_data.edges) == 2:
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axis_y = hist.axis.Variable(
            state.test_data.edges[1], name=state.test_data.axis_names[1]
        )
        axes = [axis_x, axis_y]
    else:
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axes = [axis_x]

    h = hist.Hist(*axes, storage=hist.storage.Weight())
    shape = tuple(len(edges) - 1 for edges in state.test_data.edges)
    counts_grid = pred_mean.reshape(shape)
    counts_grid = np.round(counts_grid)
    counts_grid = np.maximum(counts_grid, 0)
    h.view().value = counts_grid
    h.view().variance = counts_grid
    b_data = histToBinnedData(h, rebin=1, variation=None)
    plots = makeSmoothingPlots(
        smoothed_data=b_data,
        original_data=state.background,
        pred_mean=pred_mean,
        pred_cov=pred_cov,
    )

    return h, plots


def generateSmoothedBackground(
    state: AnalysisState,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> tuple[list[hist.Hist], dict[str, Any]]:

    if state.training_result is None or state.dataset is None:
        raise ValueError("Cannot generate smoothed background without training.")

    pred_key, sample_key = random.split(rng_key)

    pred_mean, pred_cov = predictInRealSpace(
        posterior=state.training_result.posterior,
        dataset_train=state.dataset,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=pred_key,
    )

    counts_samples = drawPoissonSamples(
        mean=pred_mean,
        cov=None,  # pred_cov,
        rng_key=sample_key,
        num_samples=num_samples,
    )
    if len(state.test_data.edges) == 2:
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axis_y = hist.axis.Variable(
            state.test_data.edges[1], name=state.test_data.axis_names[1]
        )
        axes = [axis_x, axis_y]
    else:
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axes = [axis_x]

    sample_hists = []
    sample_binned_data = []
    for i in range(num_samples):
        h = hist.Hist(*axes, storage=hist.storage.Weight())
        shape = tuple(len(edges) - 1 for edges in state.test_data.edges)
        counts_grid = counts_samples[i].reshape(shape)
        h.view().value = counts_grid
        h.view().variance = counts_grid

        b_data = histToBinnedData(h, rebin=1, variation=None)
        sample_hists.append(h)
        sample_binned_data.append(b_data)

    plots = makeSmoothingPlots(
        smoothed_data=sample_binned_data[0],
        original_data=state.background,
        pred_mean=pred_mean,
        pred_cov=pred_cov,
    )

    return sample_hists, plots


def generateSmoothedBackgroundBinned(
    state: AnalysisState,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> BinnedData:
    """Convenience wrapper to return exactly one BinnedData."""
    hists, _ = generateSmoothedBackground(state, rng_key, num_samples)
    return histToBinnedData(hists[0], rebin=1, variation=None)
