from __future__ import annotations

import logging
from pathlib import Path

import re
import jax.numpy as jnp
import numpy as np

from ..core.data import AnalysisState

logger = logging.getLogger(__name__)

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
    eigenvar_threshold=0.01,
    hist_renames: dict[str, dict[str, str]] | None = None,
) -> int:
    import uproot
    import hist
    from .eigenvariations import computeEigenvariations
    from ..data.loading import hasVariationAxis, variationNames, histToBinnedData

    if state.pred_mean is None or state.pred_cov is None:
        raise ValueError("Cannot export Combine data without prediction results.")

    hist_renames = hist_renames or {}
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

    background_hist = hist.Hist(
        hist.axis.Variable(linear_edges, name="gpr_bin"), storage=hist.storage.Weight()
    )

    background_hist[...] = np.stack(
        [np.asarray(pred_mean_masked), np.zeros_like(pred_mean_masked)], axis=-1
    )

    histograms["background"] = background_hist

    year = state.background_metadata["era"]["name"]
    postfix = f"_{year}" if year else ""

    signal_size = 0.0
    if state.signals:
        for lbl, sig_binned in state.signals.items():
            full_sig = doMask(sig_binned.Y)[blind_mask]
            signal_size = max(signal_size, float(jnp.max(full_sig)))

    variations = computeEigenvariations(
        pred_mean_masked, pred_cov_masked, threshold_fraction=eigenvar_threshold, signal_size=signal_size
    )

    for var in variations:
        idx = var["index"]
        up = np.asarray(var["up"])
        down = np.asarray(var["down"])
        histograms[f"background_gpr_eigen{idx}{postfix}Up"] = (up, linear_edges)
        histograms[f"background_gpr_eigen{idx}{postfix}Down"] = (down, linear_edges)

    for lbl, sig_hist_entry in state.signal_hists.items():
        rename_map = hist_renames.get(lbl, {})

        if hasVariationAxis(sig_hist_entry):
            for var_name in variationNames(sig_hist_entry):
                sig_binned = histToBinnedData(
                    sig_hist_entry, rebin=state.config.rebin, variation=var_name
                )
                full_sig = doMask(sig_binned.Y)[blind_mask]
                if var_name == "central":
                    histograms[lbl] = (np.asarray(full_sig), linear_edges)
                elif var_name.endswith("_disabled"):
                    continue
                elif var_name in rename_map:
                    histograms[f"{lbl}_{rename_map[var_name]}"] = (
                        np.asarray(full_sig),
                        linear_edges,
                    )
                else:
                    base, direction = normalizeVarName(var_name)
                    name = f"{lbl}_{base}{direction}" if direction else f"{lbl}_{base}"
                    histograms[name] = (np.asarray(full_sig), linear_edges)
        else:
            sig_binned = state.signals[lbl]
            histograms[lbl] = (
                np.asarray(doMask(sig_binned.Y)[blind_mask]),
                linear_edges,
            )

    histograms["data_obs"] = (np.asarray(state.test_data.Y[blind_mask]), linear_edges)

    with uproot.recreate(output_path) as f:
        for name, data in histograms.items():
            f[name] = data

    logger.info(f"Exported {len(histograms)} histograms to {output_path}")

    return len(variations)
