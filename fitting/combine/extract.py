from __future__ import annotations

import logging
import re
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from ..diagnostics.plot_utils import plotFitDiagnostic
from typing import Callable
import uproot
import lz4.frame
import numpy as np
from ..core.data import BinnedData
from ..data.loading import histToBinnedData
from ..diagnostics.plot_utils import plotPPD
from contextlib import ExitStack

logger = logging.getLogger(__name__)


def extractLimits(tree: uproot.TTree) -> tuple[dict, dict]:
    tree = tree["limit"]
    limit_vals = tree["limit"].array()
    quantiles = tree["quantileExpected"].array()

    def near(x, y, tol=1e-4):
        return abs(x - y) < tol

    result = {}
    for lim, q in zip(limit_vals, quantiles):
        if q == -1.0:
            result["observed"] = float(lim)
        elif near(q, 0.5):
            result["expected"] = float(lim)
        elif near(q, 0.16):
            result["expected_minus_1sigma"] = float(lim)
        elif near(q, 0.84):
            result["expected_plus_1sigma"] = float(lim)
        elif near(q, 0.025):
            result["expected_minus_2sigma"] = float(lim)
        elif near(q, 0.975):
            result["expected_plus_2sigma"] = float(lim)

    return {"limits": result}, {}


def extractSignificance(tree: uproot.TTree) -> tuple[dict, dict]:
    tree = tree["limit"]
    limit_vals = tree["limit"].array()

    if len(limit_vals) > 0:
        return {"significance": float(limit_vals[0])}, {}
    return {}, {}


def extractGof(obs_tree: uproot.TTree, toys_tree: uproot.TTree) -> tuple[dict, dict]:
    obs_tree, toys_tree = obs_tree["limit"], toys_tree["limit"]
    obs_vals, toys_vals = obs_tree["limit"].array(), toys_tree["limit"].array()
    ret = {}
    if len(obs_vals):
        ret["gof_test_statistic"] = float(obs_vals[0])
    if len(toys_vals):
        ret["gof_test_statistic_toys"] = [float(val) for val in toys_vals]
    ret["gof_p_value"] = np.mean(np.array(toys_vals) <= obs_vals[0])

    fig, ax = plt.subplots()
    plotPPD(
        ax,
        np.array(toys_vals),
        obs_vals[0],
        dist_title="GOF Test Statistic Toys",
    )

    return ret, {"gof_test": (fig, ax)}


def extractFitDiagnostics(root_file: uproot.ReadOnlyDirectory) -> tuple[dict, dict]:
    ret = {}
    plots = {}

    fit_trees = ["tree_fit_sb", "tree_fit_b"]
    for fit_tree in fit_trees:
        ret[fit_tree] = {}
        t = root_file[fit_tree]
        ret[fit_tree] = {
            "r": float(t["r"].array()[0]),
            "r_err": float(t["rErr"].array()[0]),
        }

    fit_types = ["shapes_prefit", "shapes_fit_b", "shapes_fit_s"]
    cat = ["data", "total_background", "total_signal"]
    channels = ["ch1"]

    for channel in sorted(channels):
        hists = {}
        for fit_type in fit_types:
            ft_short = fit_type.split("_", 1)[1]
            hists[fit_type] = {}
            for c in cat:
                hists[ft_short, c] = rootToBinnedData(
                    root_file[fit_type][channel][c]
                )

        data_hist = hists["prefit", "data"]
        prefit_background = hists["prefit", "total_background"]
        b_background = hists["fit_b", "total_background"]
        s_background = hists["fit_s", "total_background"]


        fig, ax = plotFitDiagnostic(
            data=data_hist,
            prefit_b=prefit_background,
            b_background=b_background,
            s_background=s_background,
            title=f"{t} / {channel}",
        )
        plots[f"fit_diagnostic_{channel}"] = (fig, ax)

    return ret, plots


def extractMultiDimFit(tree: uproot.TTree) -> tuple[dict, dict]:
    tree = tree["limit"]
    limit_vals = tree["limit"].array()
    limit_err_vals = tree["limitErr"].array()

    if len(limit_vals) > 0:
        return {
            "multidim_fit": {
                "r": float(limit_vals[0]),
                "r_err": float(limit_err_vals[0]) if len(limit_err_vals) > 0 else None,
            }
        }, {}
    return {}, {}


EXTRACTORS: list[
    tuple[tuple[re.Pattern] | re.Pattern, Callable[[uproot.TTree], dict]]
] = [
    (re.compile(r"fitDiagnosticsTest\.root"), extractFitDiagnostics),
    (re.compile(r"\.AsymptoticLimits\."), extractLimits),
    (re.compile(r"\.Significance\."), extractSignificance),
    (
        (
            re.compile(r"(?<!toys)\.GoodnessOfFit\."),
            re.compile(r"toys.*\.GoodnessOfFit\."),
        ),
        extractGof,
    ),
    (re.compile(r"\.MultiDimFit\."), extractMultiDimFit),
]


def rootToBinnedData(obj) -> BinnedData | None:
    try:
        if hasattr(obj, "to_hist"):
            h = obj.to_hist()
            return histToBinnedData(h)
        elif "TGraphAsymmErrors" in str(type(obj)):
            import jax.numpy as jnp

            # TGraphAsymmErrors doesn't have a robust to_hist yet in standard uproot
            x = np.asarray(obj.member("fX"))
            y = np.asarray(obj.member("fY"))

            # Symmetric variance as proxy for BinnedData.V
            ey_high = np.asarray(obj.member("fEYhigh"))
            ey_low = np.asarray(obj.member("fEYlow"))
            var = ((ey_high + ey_low) / 2.0) ** 2

            # Reconstruct edges from fEXlow/high
            ex_low = np.asarray(obj.member("fEXlow"))
            ex_high = np.asarray(obj.member("fEXhigh"))
            if len(x) > 0:
                edges_list = [x[0] - ex_low[0]]
                for i in range(len(x)):
                    edges_list.append(x[i] + ex_high[i])
                edges = (jnp.array(edges_list),)
            else:
                edges = (jnp.array([]),)

            return BinnedData(
                X=jnp.array(x[:, np.newaxis]),
                Y=jnp.array(y),
                V=jnp.array(var),
                edges=edges,
                axis_names=("x",),
            )
    except Exception as e:
        logger.warning(f"Error converting {type(obj)} to BinnedData: {e}")
        return None

    return None


def extractCombineResults(combine_dir: Path) -> dict:
    if not combine_dir.exists() or not combine_dir.is_dir():
        logger.warning(f"Combine directory {combine_dir} does not exist.")
        return {}

    merged_results = {}
    plots = {}

    files = list(combine_dir.glob("*.root"))
    for patterns, extractor in EXTRACTORS:
        patterns = patterns if isinstance(patterns, (list, tuple)) else [patterns]
        matched = []
        for pattern in patterns:
            matched_files = [f for f in files if pattern.search(f.name)]
            if len(matched_files) == 0:
                break

            if len(matched_files) > 1:
                raise ValueError(
                    f"Multiple files found for pattern {pattern}: {matched_files}"
                )
            matched.append(matched_files[0])
        if len(matched) != len(patterns):
            logger.debug(f"No extractor found for {patterns}, skipping.")
            continue

        logger.info(f"Extracting results from {matched} using {extractor.__name__}")
        try:
            with ExitStack() as stack:
                f = [stack.enter_context(uproot.open(f)) for f in matched]
                extracted_data, extracted_plots = extractor(*f)
                merged_results.update(extracted_data)
                plots.update(extracted_plots)
        except Exception as e:
            raise e
            logger.error(f"Failed to extract from {matched}: {e}")

    return merged_results, plots
