from __future__ import annotations

import datetime
import fnmatch
import glob
import itertools as it
import logging
import os
from collections import defaultdict
from pathlib import Path

import yaml
from attrs import define
from jinja2 import Environment, FileSystemLoader

import fitting
from .file_tools import tarDirectory, tarFiles
from ..utils import getSignal, getCategory, getRecoCategory, MatchRules, deepCommonDict

logger = logging.getLogger("fitting")

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
ERA_DELIMITER = "@"
DEFAULT_CONTAINER = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmsml/cmsml:3.11-cuda"
)
DEFAULT_ERA_GROUP_FIELDS = (
    "mstop",
    "mchi",
    "category",
    "pipeline",
    "toy_index",
    "config",
)
DEFAULT_SIGNAL_GROUP_FIELDS = ("mstop", "mchi", "category", "era", "background")


@define
class EraJob:
    era: str
    background: str
    signals: list[str]
    config: str
    output_dir: str
    metadata: dict[str, str]

    @property
    def all_meta(self):
        return self.metadata | {"era": self.era} | {"config": self.config}




@define
class JobGroup:
    """Single Condor Job"""

    era_jobs: list[EraJob]
    combined_output_format: str | None = None

    @property
    def is_combined(self) -> bool:
        return len(self.era_jobs) > 1

    @property
    def common_meta(self):
        return deepCommonDict(*(x.all_meta for x in self.era_jobs))
        
        


@define
class SubmissionPlan:
    """Set of condor jobs plus run script"""

    job_groups: list[JobGroup]
    transfer_files: list[str]
    run_script: Path
    output_dir: Path


def groupBy(
    jobs: list[EraJob],
    fields: tuple[str, ...],
) -> dict[tuple, list[EraJob]]:
    groups: dict[tuple, list[EraJob]] = defaultdict(list)
    for job in jobs:
        job_meta = job.all_meta
        groups[tuple(job_meta.get(f, "") for f in fields)].append(job)
    return groups


def mergeSignals(
    jobs: list[EraJob],
    fields: tuple[str, ...] = DEFAULT_SIGNAL_GROUP_FIELDS,
) -> list[EraJob]:
    """Merge signals sharing the same group key into single EraJobs."""
    merged = []
    for _, group in groupBy(jobs, fields).items():
        base = group[0]
        merged.append(
            EraJob(
                era=base.era,
                background=base.background,
                signals=[s for j in group for s in j.signals],
                config=base.config,
                output_dir=base.output_dir,
                metadata=base.metadata,
            )
        )
    return merged


def groupIntoJobs(
    jobs: list[EraJob],
    combine_eras: list[str] | None = None,
    fields: tuple[str, ...] = DEFAULT_ERA_GROUP_FIELDS,
) -> list[JobGroup]:
    """Convert EraJobs into JobGroups.

    Without combine_eras each EraJob becomes its own single-era JobGroup.
    With combine_eras, jobs are grouped across eras; incomplete groups
    (missing eras) are dropped with a warning.
    """
    logger.info(f"Combining eras: {combine_eras}")
    if not combine_eras:
        return [JobGroup(era_jobs=[j]) for j in jobs]

    era_set = set(combine_eras)
    grouped = groupBy([j for j in jobs if j.era in era_set], fields)
    logger.info(f"Grouping based on fields {fields}")
    logger.info(f"Found {len(grouped)} groups")

    result = []
    for key, group in grouped.items():
        by_era = {j.era: j for j in group}
        if set(by_era.keys()) != era_set:
            logger.warning(
                f"Incomplete group {dict(zip(fields, key))}: "
                f"missing eras {era_set - set(by_era.keys())}"
            )
            continue
        g= JobGroup(era_jobs=[by_era[e] for e in combine_eras])
        result.append(g)


    logger.info(
        f"Grouped {len(jobs)} era-jobs into {len(result)} job groups "
        f"({len(combine_eras)} eras each)"
    )
    return result


def makeEraJobs(
    signal_pattern: tuple[str, ...],
    background_pattern: str,
    years: list[str],
    pipelines: list[str],
    output_dir: str,
    config_pattern: str,
    toy_index: int | None = None,
    match_rules: MatchRules | None = None,
) -> list[EraJob]:
    """You glob then you job"""
    jobs = []
    for bkg_year, pipeline in it.product(years, pipelines):
        logger.info(f"Handling background year {bkg_year} and pipeline {pipeline}")
        syp = match_rules.resolve(bkg_year, "year") if match_rules else [bkg_year]
        logger.info(f"Signal year pattern: {syp}")

        sig_files: set[str] = set()
        for p in signal_pattern:
            sig_files |= set(
                glob.glob(
                    p.replace("{category}", "*").format(year=syp, pipeline=pipeline),
                    recursive=True,
                )
            )

        if not sig_files:
            logger.warning(f"No signals for year={bkg_year} (pattern: {syp})")
            continue

        sig_param_groups = defaultdict(list)

        num_good_signals = 0
        logger.info(f"Using signal patterns: {signal_pattern}")
        for sig_file in sig_files:
            sig_params = getSignal(sig_file)
            category = getCategory(sig_params[1], sig_params[2])
            reco_category = getRecoCategory(Path(sig_file).name)

            if not reco_category:
                continue

            if not any(
                fnmatch.fnmatch(
                    sig_file,
                    x.format(
                        year=syp,
                        pipeline=pipeline,
                        category=category,
                        reco_category=reco_category,
                    ),
                )
                for x in signal_pattern
            ):
                continue

            num_good_signals += 1

            sig_param_groups[(tuple(sig_params), category, reco_category)].append(
                sig_file
            )

        logger.info(
            f"Found {num_good_signals} / {len(sig_files)} signals for {bkg_year}, {pipeline}"
        )
        logger.info(
            f"Grouped {num_good_signals} signals into {len(sig_param_groups)} groups"
        )

        for (
            sig_params,
            category,
            reco_category,
        ), files in sig_param_groups.items():
            bkg_file = background_pattern.format(
                year=bkg_year,
                pipeline=pipeline,
                category=category,
                toy_index=toy_index,
                reco_category=reco_category,
            )
            if not Path(bkg_file).exists():
                raise FileNotFoundError(f"Background not found: {bkg_file}")

            jobs.append(
                EraJob(
                    era=bkg_year,
                    background=bkg_file,
                    signals=files,
                    config=config_pattern.format(
                        pipeline=pipeline,
                        category=category,
                        reco_category=reco_category,
                    ),
                    output_dir=output_dir,
                    metadata={
                        "mstop": str(sig_params[1]),
                        "mchi": str(sig_params[2]),
                        "category": category,
                        "pipeline": pipeline,
                        "reco_category": reco_category,
                        "coupling": pipeline.removeprefix("Signal"),
                        "toy_index": str(toy_index or 0),
                    },
                )
            )
    return jobs


def compressNeededFiles(venv_path, condor_temp_loc, extra_files=None):
    condor_temp_loc.mkdir(exist_ok=True, parents=True)
    extra_files = extra_files or []
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    compressed_env = condor_temp_loc / "environment.tar.gz"
    compressed_extra = condor_temp_loc / f"extras_{ts}.tar.gz"
    compressed_program = condor_temp_loc / f"program_{ts}.tar.gz"
    program_path = Path(fitting.__file__).parent

    if not compressed_env.exists():
        logger.info("Creating compressed virtual environment (one-time).")
        tarDirectory(venv_path, compressed_env)

    logger.info("Creating compressed program")
    compressed_extra.unlink(missing_ok=True)
    if extra_files:
        tarFiles(extra_files, compressed_extra)
    tarDirectory(program_path, compressed_program)

    transfer = [str(compressed_env), str(compressed_program)]
    if extra_files:
        transfer.append(str(compressed_extra))
    return transfer


def _discoverInjectionFiles(config_paths: list[str]) -> list[str]:
    """
    Needed when doing injections different than target signal.
    """
    found = set()
    for cp in config_paths:
        try:
            with open(cp) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and (sp := cfg.get("injection_signal_path")):
                found.add(str(sp))
        except Exception:
            logger.debug(f"Could not inspect config {cp} for injection files")
    return list(found)


def _renderTemplate(
    template_name: str,
    output_path: Path,
    make_executable: bool = True,
    **kwargs,
) -> Path:
    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR))
    template = env.get_template(template_name)
    content = template.render(
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    )
    with open(output_path, "w") as f:
        f.write(content)
    if make_executable:
        os.chmod(output_path, 0o755)
    logger.info(f"Generated {'executable ' if make_executable else ''}{output_path}")
    return output_path


def _renderJobGroup(group: JobGroup) -> dict:
    """Convert a JobGroup into the dict consumed by the submit template."""
    d = ERA_DELIMITER
    all_files: set[str] = set()
    for j in group.era_jobs:
        all_files.update(j.signals)
        all_files.add(j.background)
    return {
        "transfer_files": "$(COMMA)".join(sorted(all_files)),
        "era_names": d.join(j.era for j in group.era_jobs),
        "era_backgrounds": d.join(j.background for j in group.era_jobs),
        "era_signals": d.join("|".join(j.signals) for j in group.era_jobs),
        "era_configs": d.join(j.config for j in group.era_jobs),
        "era_outputs": d.join(j.output_dir for j in group.era_jobs),
        "combined_output_format": group.combined_output_format or "$(BLANK)",
    }


def buildPlan(
    jobs: list[EraJob],
    output_dir: Path,
    combine_eras: list[str] | None = None,
    combined_dir_format: str | None = None,
    group_by_fields: tuple[str, ...] | None = None,
    venv_path: str | None = None,
    container: str | None = None,
) -> SubmissionPlan:
    container = container or DEFAULT_CONTAINER
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / "logs").mkdir(exist_ok=True, parents=True)

    job_groups = groupIntoJobs(
        jobs,
        combine_eras,
        group_by_fields or DEFAULT_ERA_GROUP_FIELDS,
    )
    if not job_groups:
        raise ValueError("No job groups created — check patterns and eras")

    for g in job_groups:
        if g.is_combined:
            g.combined_output_format = combined_dir_format

    if not venv_path:
        venv_path = os.environ.get("VIRTUAL_ENV")
        if not venv_path:
            logger.warning("VIRTUAL_ENV not set, defaulting to '.venv'")
            venv_path = ".venv"

    configs = list({j.config for g in job_groups for j in g.era_jobs})
    injection_files = _discoverInjectionFiles(configs)
    if injection_files:
        logger.info(f"Discovered {len(injection_files)} injection file(s)")
    transfer_files = compressNeededFiles(
        venv_path,
        Path(".condor_temp/"),
        configs + injection_files,
    )

    venv_activate = Path(Path(venv_path).name) / "bin" / "activate"
    run_script = _renderTemplate(
        "run_job.sh.jinja",
        output_dir / "run_job.sh",
        files_to_unzip=transfer_files,
        venv_activate_path=venv_activate,
        container=container,
    )
    transfer_files.append(str(run_script))

    return SubmissionPlan(
        job_groups=job_groups,
        transfer_files=transfer_files,
        run_script=run_script,
        output_dir=output_dir,
    )


def generateFiles(plan: SubmissionPlan) -> Path:
    """Render submit file and local test script from a SubmissionPlan."""
    render_jobs = [_renderJobGroup(g) for g in plan.job_groups]

    submit_path = _renderTemplate(
        "submit.sub.jinja",
        plan.output_dir / "submit.sub",
        make_executable=False,
        executable=plan.run_script,
        jobs=render_jobs,
        transfer_input_files=plan.transfer_files,
        output_dir=plan.output_dir,
    )

    if plan.job_groups:
        first = plan.job_groups[-1]
        job_files = [f for j in first.era_jobs for f in j.signals + [j.background]]
        _renderTemplate(
            "local_test.sh.jinja",
            plan.output_dir / "local_test.sh",
            transfer_files=plan.transfer_files,
            job_files=job_files,
            job=render_jobs[-1],
            executable=plan.run_script,
        )

    logger.info(f"Submit file: {submit_path} ({len(plan.job_groups)} jobs)")
    return submit_path
