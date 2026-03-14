"""ROOT histogram export for Combine.

Creates ROOT files containing shape histograms using uproot.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ..core.data import BinnedData

logger = logging.getLogger(__name__)


def exportHistograms(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    test_data: BinnedData,
    output_path: Path,
    process_name: str = "background",
    n_eigenvariations: int | None = None,
) -> None:
    """Export prediction histograms to a ROOT file for Combine.

    Creates:
    - {process_name}: Nominal shape (predicted mean)
    - {process_name}_{syst}_Up / _Down: Eigenvariation shapes

    Args:
        pred_mean: Predicted mean in real space.
        pred_cov: Predicted covariance in real space.
        test_data: Full-domain test data (for bin edges).
        output_path: Path for the output ROOT file.
        process_name: Name of the process in the ROOT file.
        n_eigenvariations: Number of eigenvariations to write. None = all.
    """
    from ..inference.prediction import computeScaledEigenvectors

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np_mean = np.asarray(pred_mean)
    np_edges = tuple(np.asarray(e) for e in test_data.edges)

    histograms = {}

    # Nominal
    if test_data.ndim == 1:
        histograms[process_name] = (np_mean, np_edges[0])
    else:
        shape = tuple(len(e) - 1 for e in np_edges)
        histograms[process_name] = (np_mean.reshape(shape), np_edges)

    # Eigenvariations
    eigenvalues, scaled_vecs = computeScaledEigenvectors(pred_cov)
    n_vars = scaled_vecs.shape[1]
    if n_eigenvariations is not None:
        n_vars = min(n_vars, n_eigenvariations)

    for i in range(n_vars):
        variation = np.asarray(scaled_vecs[:, i])
        up = np_mean + variation
        down = np_mean - variation

        syst_name = f"gpr_eigen{i}"
        if test_data.ndim == 1:
            histograms[f"{process_name}_{syst_name}Up"] = (
                np.maximum(up, 0),
                np_edges[0],
            )
            histograms[f"{process_name}_{syst_name}Down"] = (
                np.maximum(down, 0),
                np_edges[0],
            )
        else:
            shape = tuple(len(e) - 1 for e in np_edges)
            histograms[f"{process_name}_{syst_name}Up"] = (
                np.maximum(up.reshape(shape), 0),
                np_edges,
            )
            histograms[f"{process_name}_{syst_name}Down"] = (
                np.maximum(down.reshape(shape), 0),
                np_edges,
            )

    import uproot

    with uproot.recreate(output_path) as f:
        for name, data in histograms.items():
            f[name] = data

    logger.info(
        f"Wrote {len(histograms)} histograms to {output_path} "
        f"({n_vars} eigenvariations)"
    )
