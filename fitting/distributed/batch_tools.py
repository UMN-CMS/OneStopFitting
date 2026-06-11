from __future__ import annotations

import functools as ft
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import attrs
import yaml

from .condor_tools import EraJob
from ..utils import evolveCombos, dotFormat

logger = logging.getLogger("fitting")


@ft.cache
def loadConfig(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def writeConfigFile(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def formatParamGrid(param_grid: dict, format_kwargs: dict) -> dict:
    return {
        k: [dotFormat(v, **format_kwargs) if isinstance(v, str) else v for v in vals]
        for k, vals in param_grid.items()
    }


def expandParameters(
    jobs: list[EraJob],
    param_grid: dict[str, list],
    config_dir: Path,
) -> list[EraJob]:
    """Explode EraJobs across parameter combinations, writing config files."""
    config_dir.mkdir(parents=True, exist_ok=True)

    by_config: dict[str, list[EraJob]] = defaultdict(list)
    for j in jobs:
        by_config[j.config].append(j)

    expanded = []
    idx = 0
    for config_path, group in by_config.items():
        base_config = loadConfig(config_path)
        grid = formatParamGrid(param_grid, group[0].metadata)
        total_combos = 0
        for combo in evolveCombos(base_config, **grid):
            total_combos += 1
            new_path = config_dir / f"batch_{idx:04d}.yaml"
            writeConfigFile(combo, new_path)
            for job in group:
                expanded.append(attrs.evolve(job, config=str(new_path)))
            idx += 1

    logger.info(
        f"Expanded {len(jobs)} jobs x {total_combos} configs = {len(expanded)} total"
    )
    return expanded
