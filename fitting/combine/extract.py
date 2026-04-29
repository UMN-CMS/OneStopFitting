from __future__ import annotations

import logging
import re
import matplotlib.pyplot as plt
from pathlib import Path
from ..diagnostics.plot_utils import plotFitDiagnostic
from typing import Callable
import uproot
import numpy as np
from ..core.data import BinnedData
from ..data.loading import histToBinnedData
from ..diagnostics.plot_utils import plotPPD
from contextlib import ExitStack


class OptionalPattern:
    def __init__(self, pattern: re.Pattern):
        self.pattern = pattern
        self.is_optional = True

    def search(self, string: str):
        return self.pattern.search(string)


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
        if fit_tree not in root_file:
            continue
        t = root_file[fit_tree]
        ret[fit_tree] = {
            "r": float(t["r"].array()[0]),
            "r_err": float(t["rErr"].array()[0]),
        }

    fit_types = ["shapes_prefit", "shapes_fit_b", "shapes_fit_s"]
    cat = ["data", "total_background", "total_signal"]

    # Dynamically find channels (directories in shapes_prefit)
    channels = [
        k.split(";")[0] for k in root_file["shapes_prefit"].keys() if "/" not in k
    ]

    ret["histograms"] = {}
    for channel in sorted(channels):
        ret["histograms"][channel] = {}
        hists = {}
        for fit_type in fit_types:
            ft_short = fit_type.split("_", 1)[1]
            for c in cat:
                obj_path = f"{fit_type}/{channel}/{c}"
                if obj_path in root_file:
                    bd = rootToBinnedData(root_file[fit_type][channel][c])
                    hists[ft_short, c] = bd
                    ret["histograms"][channel][f"{ft_short}_{c}"] = bd

        if ("prefit", "data") not in hists:
            continue

        data_hist = hists["prefit", "data"]
        prefit_background = hists.get(("prefit", "total_background"))
        b_background = hists.get(("fit_b", "total_background"))
        s_background = hists.get(("fit_s", "total_background"))
        s_signal = hists.get(("fit_s", "total_signal"))

        continue
        if all(
            x is not None
            for x in [data_hist, prefit_background, b_background, s_background]
        ):
            fig, ax = plotFitDiagnostic(
                data=data_hist,
                prefit_b=prefit_background,
                b_background=b_background,
                s_background=s_background,
                s_signal=s_signal,
                signal_rate=ret.get("tree_fit_sb", {}).get("r", 1.0),
                title=f"Fit Diagnostics / {channel}",
            )
            plots[f"fit_diagnostic_{channel}_show_signal"] = (fig, ax)
            fig, ax = plotFitDiagnostic(
                data=data_hist,
                prefit_b=prefit_background,
                b_background=b_background,
                s_background=s_background,
                s_signal=s_signal,
                signal_rate=ret.get("tree_fit_sb", {}).get("r", 1.0),
                show_signal=False,
                title=f"Fit Diagnostics / {channel}",
            )
            plots[f"fit_diagnostic_{channel}"] = (fig, ax)

    return ret, plots


def extractMultiDimFit(tree: uproot.TTree) -> tuple[dict, dict]:
    tree = tree["limit"]
    rates = tree["r"].array()
    r = rates[0]
    r_err = [float(abs(rates[1] - r)), float(abs(rates[2] - r))]
    return {"multidim_fit": {"r": float(r), "r_err": r_err}}, {}


def extractLikelihoodScan(std_file, froz_file) -> tuple[dict, dict]:
    from scipy.interpolate import make_interp_spline

    ret = {}
    plots = {}

    def get_data(f):
        tree = f["limit"]
        limit_vals = tree["r"].array()
        limit_err_vals = tree["deltaNLL"].array()
        q = tree["quantileExpected"].array()

        mask = q > -1.5
        r_vals = np.array(limit_vals[mask])
        dnll = np.array(limit_err_vals[mask]) * 2.0

        idx = np.argsort(r_vals)
        r_vals = r_vals[idx]
        dnll = dnll[idx]

        r_vals, unique_idx = np.unique(r_vals, return_index=True)
        dnll = dnll[unique_idx]
        return r_vals, dnll

    def makeSpline(r_vals, dnll):
        new_points = np.linspace(r_vals.min(), r_vals.max(), 1000)
        spl = make_interp_spline(r_vals, dnll, k=3)
        return new_points, spl(new_points)

    def maskData(r_vals, dnll, max_val=6):
        mask = dnll <= max_val
        return r_vals[mask], dnll[mask]

    fig, ax = plt.subplots()

    if std_file is not None:
        std_r, std_dnll = maskData(*makeSpline(*get_data(std_file)))

        ax.plot(std_r, std_dnll, "k-", label="Standard", linewidth=2)

        bestfit_idx = np.argmin(std_dnll)
        ret["likelihood_scan_best_r"] = float(std_r[bestfit_idx])

    if froz_file is not None:
        froz_r, froz_dnll = maskData(*makeSpline(*get_data(froz_file)))
        ax.plot(froz_r, froz_dnll, "b-", label="Frozen", linewidth=2)

    ax.axhline(1.0, color="gray", linestyle="--")
    ax.axhline(4.0, color="gray", linestyle="--")
    ax.set_xlabel("r")
    ax.set_ylabel("-2 $\Delta ln L$")
    ax.set_ylim(0, 6.0)

    ax.legend()

    plots["likelihood_scan"] = (fig, ax)
    return ret, plots


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
    (re.compile(r"higgsCombine\.mdimnon\.MultiDimFit\."), extractMultiDimFit),
    (
        (
            re.compile(r"higgsCombine\.mdimgrid\.MultiDimFit\."),
            OptionalPattern(re.compile(r"higgsCombine\.mdimgridfreeze\.MultiDimFit\.")),
        ),
        extractLikelihoodScan,
    ),
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
        skip = False
        for pattern in patterns:
            is_optional = hasattr(pattern, "is_optional")
            actual_pattern = pattern.pattern if is_optional else pattern
            matched_files = [f for f in files if actual_pattern.search(f.name)]
            if len(matched_files) == 0:
                if is_optional:
                    matched.append(None)
                    continue
                skip = True
                break

            if len(matched_files) > 1:
                raise ValueError(
                    f"Multiple files found for pattern {actual_pattern}: {matched_files}"
                )
            matched.append(matched_files[0])

        if skip or len(matched) != len(patterns):
            logger.debug(f"No extractor found for {patterns}, skipping.")
            continue

        logger.info(f"Extracting results from {matched} using {extractor.__name__}")
        try:
            with ExitStack() as stack:
                f = [
                    stack.enter_context(uproot.open(m)) if m is not None else None
                    for m in matched
                ]
                extracted_data, extracted_plots = extractor(*f)
                merged_results.update(extracted_data)
                plots.update(extracted_plots)
        except Exception as e:
            raise e
            logger.error(f"Failed to extract from {matched}: {e}")

    return merged_results, plots
