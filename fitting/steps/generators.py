from __future__ import annotations

from typing import Any

import jax
from jax import random
import jax.numpy as jnp

from ..core.data import AnalysisState, BinnedData
from ..inference.prediction import predictInRealSpace


def generateSmoothedBackground(
    state: AnalysisState,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> tuple[list[BinnedData], dict[str, Any]]:
    import hist
    import numpy as np

    if state.training_result is None or state.dataset is None:
        raise ValueError("Cannot generate smoothed background without training.")

    pred_key, sample_key = random.split(rng_key)

    # 1. Get full latent distribution (REAL SPACE)
    # We use all bins in test_data
    pred_mean, pred_cov = predictInRealSpace(
        posterior=state.training_result.posterior,
        dataset_train=state.dataset,
        test_data=state.test_data,
        transform=state.transform,
        samples=state.training_result.samples,
        rng_key=pred_key,
    )

    # 2. Draw Poisson samples directly from the predicted mean
    from ..inference.prediction import drawPoissonSamples

    counts_samples = drawPoissonSamples(
        mean_rate=pred_mean, rng_key=sample_key, num_samples=num_samples
    )

    # Convert test_data edges to hist axes
    # We assume 2D for now, generalized later
    if len(state.test_data.edges) == 2:
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axis_y = hist.axis.Variable(
            state.test_data.edges[1], name=state.test_data.axis_names[1]
        )
        axes = [axis_x, axis_y]
    else:
        # Fallback to 1D
        axis_x = hist.axis.Variable(
            state.test_data.edges[0], name=state.test_data.axis_names[0]
        )
        axes = [axis_x]

    # 3. Create histograms for each sample
    sample_binned_data = []
    for i in range(num_samples):
        # Create a hist.Hist object
        h = hist.Hist(*axes, storage=hist.storage.Weight())

        # Fill the histogram with the sampled counts
        # We need to reshape the counts array to match the histogram shape
        # In BinnedData, Y is flattened. We must unflatten it.
        shape = tuple(len(edges) - 1 for edges in state.test_data.edges)
        counts_grid = counts_samples[i].reshape(shape)
        # BinnedData flattening uses 'C' order (row-major), which is numpy default
        # No transpose needed if axis order matches

        # Set bin contents directly
        h.view().value = counts_grid
        # Set variance equal to counts (Poisson approximation)
        h.view().variance = counts_grid

        # Convert back to BinnedData
        from ..data.loading import histToBinnedData

        # histToBinnedData expects a dict of variations, pass a dummy one
        dummy_variations = {"central": h}
        b_data = histToBinnedData(dummy_variations, rebin=1, variation="central")
        sample_binned_data.append(b_data)

    # Return some metadata/diagnostics
    metadata = {
        "pred_mean": pred_mean,
        "pred_var": jnp.diag(pred_cov),
        "state_metadata": state.metadata,
    }

    return sample_binned_data, metadata


def generateSmoothedBackgroundBinned(
    state: AnalysisState,
    rng_key: jax.Array,
    num_samples: int = 1,
) -> BinnedData:
    """Convenience wrapper to return exactly one BinnedData."""
    samples, _ = generateSmoothedBackground(state, rng_key, num_samples)
    return samples[0]
