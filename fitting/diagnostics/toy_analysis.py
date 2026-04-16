import logging
from pathlib import Path
from typing import Any, Iterable, Iterator
from collections import defaultdict, ChainMap
from rich import print
import json

import matplotlib.pyplot as plt
import mplhep
import numpy as np

from .aggregate_plots import iterSummaryFiles, getByDotpath, readSummary
from ..utils import dictToDot, dotFormat

logger = logging.getLogger(__name__)


def collectToyData(
    summary_files: Iterable[Path],
    *,
    group_by: list[str],
    metric_dotpath: str,
    stop_dotpath: str = "metadata.other_data.stop_mass",
    chi_dotpath: str = "metadata.other_data.chargino_mass",
) -> dict[tuple[tuple[str, Any], ...], list[dict]]:
    grouped_data = defaultdict(list)

    for path in summary_files:
        try:
            summary = readSummary(path)
            value = float(getByDotpath(summary, metric_dotpath))
            toy_index = getByDotpath(summary, "metadata.toy_index")
            mstop = float(getByDotpath(summary, stop_dotpath))
            mchi = float(getByDotpath(summary, chi_dotpath))
            group_values = {}
            for gp in group_by:
                try:
                    group_values[gp] = getByDotpath(summary, gp)
                except KeyError:
                    logger.debug(f"Group-by key '{gp}' not found in {path}")
                    continue
            group_key = tuple(
                (gp, group_values[gp]) for gp in group_by if gp in group_values
            )
            toy_data = {
                "toy_index": toy_index,
                "value": value,
                "mstop": mstop,
                "mchi": mchi,
                "source": path,
                "group_info": group_values.copy(),
            }

            grouped_data[group_key].append(toy_data)

        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

    return dict(grouped_data)


def computeStatistics(values: list[float]) -> dict[str, float]:
    if not values:
        return {}

    arr = np.array(values, dtype=float)

    stats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "q05": float(np.percentile(arr, 5)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "q95": float(np.percentile(arr, 95)),
        "n": len(arr),
    }

    return stats


def drawBox(ax, values, all_vars, toy_indices):
    bp = ax.boxplot(
        [values],
        labels=[all_vars.get("metadata.dataset_name", "unknown")],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel(all_vars["metric_name"])
    # ax.set_ylim([-5,5])
    ax.grid(True, alpha=0.3, axis="y")


def drawViolin(ax, values, all_vars, toy_indices):
    parts = ax.violinplot([values], showmeans=True, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("lightblue")
        pc.set_alpha(0.7)
    ax.set_ylabel(all_vars["metric_name"])
    # ax.set_ylim([-5,5])
    ax.grid(True, alpha=0.3, axis="y")


def drawHistogram(ax, values, all_vars, toy_indices):
    n, bins, patches = ax.hist(
        values, bins=20, alpha=0.7, edgecolor="black", color="lightblue"
    )
    ax.set_xlabel(all_vars["metric_name"])
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3, axis="both")


def drawScatter(ax, values, all_vars, toy_indices):
    ax.scatter(
        toy_indices,
        values,
        alpha=0.6,
        color="blue",
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.set_xlabel("Toy Index")
    ax.set_ylabel(metric_dotpath)
    ax.grid(True, alpha=0.3, axis="both")

    # Add trend line
    if len(toy_indices) > 1:
        z = np.polyfit(toy_indices, values, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(toy_indices), max(toy_indices), 100)
        ax.plot(
            x_trend,
            p(x_trend),
            "r--",
            alpha=0.8,
            linewidth=2,
            label="Trend",
        )
        ax.legend()


PLOT_TYPES = {
    "histogram": drawHistogram,
    "box": drawBox,
    "violin": drawViolin,
    "scatter": drawScatter,
}


def makeToyVariationPlots(
    grouped_data: dict[tuple[tuple[str, Any], ...], list[dict]],
    metric_dotpath: str,
    metric_short: str,
    plot_types: list[str],
    output_dir: Path,
    formats: list[str],
    name_format: str = "{plot_type}_{metric_short}",
) -> int:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mplhep.style.use("CMS")
    num_plots = 0

    logger.info(f"Generating plot for {len(grouped_data)} groups")

    for group_key, toys in grouped_data.items():
        all_vars = ChainMap(
            dict(group_key),
            {
                "metric_name": metric_dotpath,
                "metric_short": metric_short,
            },
        )

        values = [toy["value"] for toy in toys]
        toy_indices = [toy["toy_index"] for toy in toys]

        if not values:
            logger.warning(f"No values for group {group_key}")
            continue

        for plot_type in plot_types:
            try:
                fig, ax = plt.subplots(layout="tight")
                PLOT_TYPES[plot_type](ax, values, all_vars, toy_indices)

                mplhep.cms.label(ax=ax, label="Preliminary")
                all_vars = all_vars.new_child({"plot_type": plot_type})
                filename = dotFormat(name_format, **all_vars)
                for fmt in formats:
                    ext = f".{fmt.lstrip('.')}"
                    o = output_dir / filename
                    outpath = o.with_suffix(ext)
                    outpath.parent.mkdir(exist_ok=True, parents=True)
                    logger.info(f"Saving figure to {outpath}")
                    fig.savefig(outpath, bbox_inches="tight")
                plt.close(fig)
                num_plots += 1
            except Exception as e:
                logger.error(
                    f"Error creating {plot_type} plot for group {group_key}: {e}"
                )
                plt.close(fig)
                continue
    return num_plots


def saveData(
    grouped_data: dict[tuple[tuple[str, Any], ...], list[dict]],
    compute_stats_func,
    output_path: Path,
) -> None:

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "groups": [],
        "metadata": {
            "n_groups": len(grouped_data),
            "total_toys": sum(len(toys) for toys in grouped_data.values()),
        },
    }

    for group_key, toys in grouped_data.items():
        group_dict = dict(group_key)
        values = [toy["value"] for toy in toys]

        group_info = {
            "group_key": dict(group_key),
            "n_toys": len(toys),
            "values": values,
            "statistics": compute_stats_func(values),
            # "toy_indices": [toy["toy_index"] for toy in toys],
        }

        output_data["groups"].append(group_info)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved data to {output_path}")
