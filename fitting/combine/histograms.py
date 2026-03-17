from __future__ import annotations

import logging
from pathlib import Path

import re
import jax.numpy as jnp
import numpy as np

from ..core.data import AnalysisState, BinnedData

logger = logging.getLogger(__name__)


def exportHistograms(
    pred_mean: jnp.ndarray,
    pred_cov: jnp.ndarray,
    test_data: BinnedData,
    output_path: Path,
    process_name: str = "background",
    n_eigenvariations: int | None = None,
) -> None:
    from ..inference.prediction import computeScaledEigenvectors
    import uproot

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

    with uproot.recreate(output_path) as f:
        for name, data in histograms.items():
            f[name] = data

    logger.info(
        f"Wrote {len(histograms)} histograms to {output_path} "
        f"({n_vars} eigenvariations)"
    )


UP_RES = [
    "Up$",
    "_up_",
    "_up$",
]
DOWN_RES = [
    "Down$",
    "Dn$",
    "_down_",
    "_down$",
]


def normalizeVarName(var_name: str) -> tuple[str, str | None]:
    if var_name == "central":
        return "central", None
    for expr in UP_RES:
        if re.search(expr, var_name):
            return re.sub(expr, "", var_name), "Up"

    for expr in DOWN_RES:
        if re.search(expr, var_name):
            return re.sub(expr, "", var_name), "Down"

    return var_name, None


def exportCombineData(
    state: AnalysisState,
    output_path: Path,
    use_window_mask: bool = True,
) -> None:
    import uproot
    from .eigenvariations import computeEigenvariations
    from ..data.loading import hasVariationAxis, variationNames, histToBinnedData

    if state.pred_mean is None or state.pred_cov is None:
        raise ValueError("Cannot export Combine data without prediction results.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if use_window_mask:
        blind_mask = state.blind_mask
    else:
        blind_mask = jnp.ones_like(state.pred_mean, dtype=bool)

    pred_mean_masked = state.pred_mean[blind_mask]
    pred_cov_masked = state.pred_cov[blind_mask, :][:, blind_mask]

    num_active_bins = jnp.count_nonzero(blind_mask)
    linear_edges = np.arange(num_active_bins + 1)

    def doMask(values: jnp.ndarray) -> jnp.ndarray:
        if state.domain_mask is None:
            return values
        return values[state.domain_mask]

    histograms = {}

    histograms["background"] = (np.asarray(pred_mean_masked), linear_edges)
    variations = computeEigenvariations(pred_mean_masked, pred_cov_masked)

    for var in variations:
        idx = var["index"]
        up = np.asarray(var["up"])
        down = np.asarray(var["down"])
        histograms[f"background_gpr_eigen{idx}Up"] = (up, linear_edges)
        histograms[f"background_gpr_eigen{idx}Down"] = (down, linear_edges)

    if state.signal_hist is not None:
        sig_name = "signal"
        if hasVariationAxis(state.signal_hist):
            for var_name in variationNames(state.signal_hist):
                sig_binned = histToBinnedData(
                    state.signal_hist, rebin=state.config.rebin, variation=var_name
                )
                full_sig = doMask(sig_binned.Y)[blind_mask]
                if var_name == "central":
                    histograms[sig_name] = (np.asarray(full_sig), linear_edges)
                elif var_name.endswith("_disabled"):
                    continue
                else:
                    base, direction = normalizeVarName(var_name)
                    name = (
                        f"{sig_name}_{base}{direction}"
                        if direction
                        else f"{sig_name}_{base}"
                    )
                    histograms[name] = (
                        np.asarray(full_sig),
                        linear_edges,
                    )
        else:
            histograms[sig_name] = (
                np.asarray(state.signal.Y[blind_mask]),
                linear_edges,
            )

    histograms["data_obs"] = (np.asarray(state.test_data.Y[blind_mask]), linear_edges)

    with uproot.recreate(output_path) as f:
        for name, data in histograms.items():
            f[name] = data

    logger.info(f"Exported {len(histograms)} histograms to {output_path}")

    return len(variations)
