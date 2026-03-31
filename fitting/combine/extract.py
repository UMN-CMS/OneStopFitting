from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable
import uproot

logger = logging.getLogger(__name__)


def extractLimits(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()
    quantiles = tree["quantileExpected"].array()

    result = {}
    for lim, q in zip(limit_vals, quantiles):
        if q == -1.0:
            result["observed"] = float(lim)
        elif abs(q - 0.5) < 1e-4:
            result["expected"] = float(lim)
        elif abs(q - 0.16) < 1e-4:
            result["expected_minus_1sigma"] = float(lim)
        elif abs(q - 0.84) < 1e-4:
            result["expected_plus_1sigma"] = float(lim)
        elif abs(q - 0.025) < 1e-4:
            result["expected_minus_2sigma"] = float(lim)
        elif abs(q - 0.975) < 1e-4:
            result["expected_plus_2sigma"] = float(lim)

    return {"limits": result}


def extractSignificance(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()

    if len(limit_vals) > 0:
        return {"significance": float(limit_vals[0])}
    return {}


def extractGOFToys(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()

    if len(limit_vals):
        return {"gof_test_statistic_toys": [float(val) for val in limit_vals]}
    return {}


def extractGof(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()

    if len(limit_vals) > 0:
        return {"gof_test_statistic": float(limit_vals[0])}
    return {}


def extractFitDiagnostics(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()
    limit_err_vals = tree["limitErr"].array()

    if len(limit_vals) > 0:
        return {
            "fit_diagnostics": {
                "r": float(limit_vals[0]),
                "r_err": float(limit_err_vals[0]) if len(limit_err_vals) > 0 else None,
            }
        }
    return {}


def extractMultiDimFit(tree: uproot.TTree) -> dict:
    limit_vals = tree["limit"].array()
    limit_err_vals = tree["limitErr"].array()

    if len(limit_vals) > 0:
        return {
            "multidim_fit": {
                "r": float(limit_vals[0]),
                "r_err": float(limit_err_vals[0]) if len(limit_err_vals) > 0 else None,
            }
        }
    return {}


EXTRACTORS: list[tuple[re.Pattern, Callable[[uproot.TTree], dict]]] = [
    (re.compile(r"\.AsymptoticLimits\."), extractLimits),
    (re.compile(r"\.Significance\."), extractSignificance),
    (re.compile(r"toys.*\.GoodnessOfFit\."), extractGOFToys),
    (re.compile(r"\.GoodnessOfFit\."), extractGof),
    (re.compile(r"\.FitDiagnostics\."), extractFitDiagnostics),
    (re.compile(r"\.MultiDimFit\."), extractMultiDimFit),
]


def extractCombineResults(combine_dir: Path) -> dict:
    if not combine_dir.exists() or not combine_dir.is_dir():
        logger.warning(f"Combine directory {combine_dir} does not exist.")
        return {}

    merged_results = {}

    for root_file in combine_dir.glob("higgsCombine*.root"):
        file_name = root_file.name
        matched_extractor = None
        for pattern, extractor in EXTRACTORS:
            if pattern.search(file_name):
                matched_extractor = extractor
                break

        if not matched_extractor:
            logger.debug(f"No extractor found for {file_name}, skipping.")
            continue

        logger.info(
            f"Extracting results from {file_name} using {matched_extractor.__name__}"
        )
        try:
            with uproot.open(root_file) as f:
                if "limit;1" not in f:
                    logger.warning(f"No 'limit' TTree found in {file_name}")
                    continue

                tree = f["limit;1"]
                extracted = matched_extractor(tree)
                merged_results.update(extracted)
        except Exception as e:
            logger.error(f"Failed to extract from {file_name}: {e}")

    return merged_results
