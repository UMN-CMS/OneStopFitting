from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from fitting.utils import evolveCombos

import yaml

logger = logging.getLogger("fitting")


def writeConfigFile(config: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config written to {output_path}")


def generateBatchSubmit(
    signal_pattern: str,
    background_pattern: str,
    years: list[str],
    config_base: Path | None,
    output_dir: Path,
    subdir_format: str,
    venv_path: str | None,
    container: str | None,
    combine_cmds: list[str] | None,
    rates: list[float] | None,
    rebin: list[int] | None,
    min_counts: list[float] | None,
    window_spread: list[float] | None,
) -> None:
    from .condor_tools import (
        getJobs,
        compressNeededFiles,
        makeRunFitScript,
        makeSubmitScript,
        COMBINE_SHORT_COMMANDS,
    )

    base_config = None
    if config_base is not None:
        with open(config_base, "r") as f:
            base_config = yaml.safe_load(f)

    # Build parameter grids
    param_grids = {}
    if rates:
        param_grids["injection_rate"] = rates
    if rebin:
        param_grids["rebin"] = rebin
    if min_counts:
        param_grids["min_counts"] = min_counts
    if window_spread:
        param_grids["window_spread"] = window_spread

    if not param_grids:
        logger.warning("No batch parameters specified. Generating single submit file.")
        from .condor_tools import generateCondorSubmit

        generateCondorSubmit(
            signal_pattern=signal_pattern,
            background_pattern=background_pattern,
            years=years,
            config_pattern=str(config_base) if config_base else None,
            output_dir=output_dir,
            subdir_format=subdir_format,
            venv_path=venv_path,
            container=container,
            combine_cmds=combine_cmds,
        )
        return

    total_combinations = 1
    for values in param_grids.values():
        total_combinations *= len(values)
    logger.info(
        f"Generating {total_combinations} parameter combinations: {param_grids.keys()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    base_jobs = getJobs(
        signal_pattern=signal_pattern,
        background_pattern=background_pattern,
        years=years,
        output_dir=output_dir,
        config_pattern="config_placeholder",
    )
    if not base_jobs:
        logger.error("No base jobs found from signal/background patterns!")
        return

    batch_config_dir = output_dir / "batch_configs"
    batch_config_dir.mkdir(parents=True, exist_ok=True)
    all_jobs = []
    config_index = 0
    for config in evolveCombos(base_config, **param_grids):
        config_name = f"batch_config_{config_index:04d}.yaml"
        config_path = batch_config_dir / config_name
        writeConfigFile(config, config_path)
        for base_job in base_jobs:
            job = base_job.copy()
            job["config"] = str(config_path)
            all_jobs.append(job)
        config_index += 1

    logger.info(
        f"Generated {len(all_jobs)} total jobs from {config_index} parameter combinations"
    )
    if not venv_path:
        import os

        venv_path = os.environ.get("VIRTUAL_ENV")
        if not venv_path:
            logger.warning(
                "VIRTUAL_ENV not found in environment and not provided. Defaulting to '.venv'."
            )
            venv_path = ".venv"

    configs = list(set(job["config"] for job in all_jobs))
    transfer_files = compressNeededFiles(
        venv_path=venv_path,
        condor_temp_loc=Path(".condor_temp/"),
        extra_files=configs,
    )

    venv_activate_path = Path(Path(venv_path).name) / "bin" / "activate"

    expanded_cmds = []
    if combine_cmds:
        for cmd in combine_cmds:
            if cmd in COMBINE_SHORT_COMMANDS:
                expanded_cmds.append(COMBINE_SHORT_COMMANDS[cmd])
            else:
                expanded_cmds.append(cmd)

    run_fit_script = makeRunFitScript(
        venv_activate_path=str(venv_activate_path),
        output_dir=output_dir,
        files_to_unzip=transfer_files,
        container=container,
        combine_cmds=expanded_cmds,
    )

    transfer_files.append(run_fit_script)

    submit_file_path = makeSubmitScript(
        jobs=all_jobs,
        transfer_files=transfer_files,
        output_dir=output_dir,
        executable=run_fit_script,
        container=container,
    )

    logger.info(
        f"Batch generation complete. Single submit file at {submit_file_path} "
        f"with {len(all_jobs)} jobs ({config_index} parameter combinations)"
    )
    logger.info(f"Batch config files saved to {batch_config_dir}")
