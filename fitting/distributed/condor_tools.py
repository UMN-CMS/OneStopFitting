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
import attrs
from attrs import define
from jinja2 import Environment, FileSystemLoader

import fitting
from .file_tools import tarDirectory, tarFiles
from ..utils import (
    getSignal,
    getCategory,
    getRecoCategory,
    MatchRules,
    deepCommonDict,
    formatLines,
)
from .fit_plan import FitPlan, Point, Profile

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
    "fit_mode",
)
DEFAULT_SIGNAL_GROUP_FIELDS = (
    "mstop",
    "mchi",
    "category",
    "era",
    "background",
    "fit_mode",
)
NO_VALUE = "-"


@define
class EraJob:
    era: str
    background: str
    signals: list[str]
    config: str
    output_dir: str
    metadata: dict[str, str]
    expected_ndim: int | None = None

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
            attrs.evolve(base, signals=[s for j in group for s in j.signals])
        )
    return merged


def groupIntoJobs(
    jobs: list[EraJob],
    combine_eras: list[str] | None = None,
    fields: tuple[str, ...] = DEFAULT_ERA_GROUP_FIELDS,
) -> list[JobGroup]:
    """Convert EraJobs into JobGroups.

    Without combine_eras each EraJob becomes its own single-era JobGroup.
    With combine_eras, jobs are grouped across eras; incomplete groups (missing eras) are dropped with a warning.
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
    require_background: bool = True,
    allow_category_mismatch: bool = False,
) -> list[EraJob]:
    """You glob then you job

    With require_background=False, jobs whose background file is missing are
    still returned, so callers that filter jobs can check only the ones they keep.
    Signals whose file category (reco_category) differs from the category given
    by their mass point are dropped unless allow_category_mismatch is set.
    """
    jobs = []
    for bkg_year, pipeline in it.product(years, pipelines):
        logger.info(f"Handling background year {bkg_year} and pipeline {pipeline}")
        syp = match_rules.resolve(bkg_year, "year") if match_rules else bkg_year
        logger.info(f"Signal year pattern: {syp}")

        sig_files: set[str] = set()
        for p in signal_pattern:
            sig_files |= set(
                glob.glob(
                    p.replace("{category}", "*")
                    .replace("{reco_category}", "*")
                    .format(year=syp, pipeline=pipeline),
                    recursive=True,
                )
            )

        if not sig_files:
            logger.warning(f"No signals for year={bkg_year} (pattern: {syp})")
            continue

        sig_param_groups = defaultdict(list)
        # Points with a template only from a category other than their own
        category_dropped: set[tuple] = set()

        num_good_signals = 0
        logger.info(f"Using signal patterns: {signal_pattern}")
        for sig_file in sig_files:
            sig_params = getSignal(sig_file)
            category = getCategory(sig_params[1], sig_params[2])
            reco_category = getRecoCategory(Path(sig_file).name)

            if not reco_category:
                continue
            # {reco_category} in a pattern matches any prefix, so check explicitly
            if not allow_category_mismatch and reco_category != category:
                category_dropped.add(tuple(sig_params))
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
        lost = sorted(category_dropped - {k[0] for k in sig_param_groups})
        if lost:
            logger.warning(
                f"{len(lost)} point(s) for {bkg_year}, {pipeline} have no template of "
                "their own category and get NO job (pass --allow-category-mismatch "
                "to fit them with other categories' templates):\n"
                + "\n".join(f"  coupling={c} mstop={ms} mchi={mc}" for c, ms, mc in lost)
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
            if require_background and not Path(bkg_file).exists():
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
                        "coupling": str(sig_params[0]),
                        "toy_index": str(toy_index or 0),
                    },
                )
            )
    return jobs


def _jobPoint(job: EraJob) -> Point:
    m = job.metadata
    return Point(
        mstop=int(m["mstop"]),
        mchi=int(m["mchi"]),
        coupling=m["coupling"],
        category=m["category"],
        year=job.era,
        toy_index=int(m["toy_index"]),
    )


def formatJobTable(jobs: list[EraJob]) -> list[str]:
    """One line per job: which fit profile and config each point gets."""

    def sortKey(j):
        m = j.metadata
        return (j.era, m["coupling"], int(m["mstop"]), int(m["mchi"]),
                m["reco_category"], m["toy_index"], m.get("fit_mode", NO_VALUE))

    rows = [["era", "coupling", "mstop", "mchi", "reco_cat", "toy", "fit_mode", "ndim", "config"]]
    for j in sorted(jobs, key=sortKey):
        m = j.metadata
        rows.append([
            j.era, m["coupling"], m["mstop"], m["mchi"], m["reco_category"],
            m["toy_index"], m.get("fit_mode", NO_VALUE),
            str(j.expected_ndim or NO_VALUE), j.config,
        ])
    return formatLines(rows)


_POINT_KEY_FIELDS = ("era", "coupling", "mstop", "mchi", "toy")


def _pointKey(job: EraJob) -> tuple:
    """Identifies the physical point a job fits, whatever its pipeline or template."""
    m = job.metadata
    return (job.era, m["coupling"], int(m["mstop"]), int(m["mchi"]), m["toy_index"])


def _formatPointKey(key: tuple) -> str:
    return " ".join(f"{f}={v}" for f, v in zip(_POINT_KEY_FIELDS, key))


def _usesTemplate(profile: Profile, job: EraJob) -> bool:
    """Whether a job's template is the category this profile fits with."""
    m = job.metadata
    return m["reco_category"] == (profile.reco_category or m["category"])


def _profileJobs(
    name: str,
    profile: Profile,
    pipelines: list[str],
    **kwargs,
) -> list[EraJob]:
    """Jobs from a profile's pipelines, every template category included.

    Callers filter with _usesTemplate; the rest show which points the profile
    has inputs for, but only in the wrong category.
    """
    if profile.pipelines is not None:
        pipelines = [
            p for p in pipelines if any(fnmatch.fnmatch(p, x) for x in profile.pipelines)
        ]
    if not pipelines:
        logger.warning(f"Fit profile '{name}' matches none of the requested pipelines")
        return []
    jobs = makeEraJobs(
        signal_pattern=tuple(profile.signal),
        background_pattern=profile.background,
        pipelines=pipelines,
        config_pattern=profile.config,
        require_background=False,
        allow_category_mismatch=True,
        **kwargs,
    )
    return [
        attrs.evolve(
            j,
            metadata=j.metadata | {"fit_mode": name},
            expected_ndim=profile.ndim,
        )
        for j in jobs
    ]


def makePlannedEraJobs(
    plan: FitPlan,
    years: list[str],
    pipelines: list[str],
    output_dir: str,
    toy_index: int | None = None,
    match_rules: MatchRules | None = None,
    all_matching: bool = False,
    allow_missing: bool = False,
) -> list[EraJob]:
    """Build EraJobs for every profile in a fit plan and keep the assigned ones.

    The plan decides per physical point (_POINT_KEY_FIELDS), and each point is
    fit once per profile it is assigned: once in total unless all_matching.
    A profile with several candidate jobs for one point is an error. A point
    assigned to a profile that has no inputs for it, or only templates of the
    wrong category, is an error, unless allow_missing is set, in which case it
    is skipped with a warning (it is never silently refit with another profile).
    """
    by_point: dict[tuple, dict[str, list[EraJob]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # Profiles with inputs for a point in any category, used to tell gaps apart
    seen: dict[tuple, set[str]] = defaultdict(set)
    points: dict[tuple, Point] = {}
    for name, profile in plan.profiles.items():
        logger.info(f"Collecting jobs for fit profile '{name}'")
        for job in _profileJobs(
            name,
            profile,
            pipelines,
            years=years,
            output_dir=output_dir,
            toy_index=toy_index,
            match_rules=match_rules,
        ):
            key = _pointKey(job)
            points.setdefault(key, _jobPoint(job))
            seen[key].add(name)
            if _usesTemplate(profile, job):
                by_point[key][name].append(job)

    selected: list[EraJob] = []
    missing: dict[str, list[str]] = defaultdict(list)
    ambiguous: list[tuple[str, tuple, list[EraJob]]] = []
    n_extra = 0
    for key in sorted(points, key=str):
        point = points[key]
        for name in plan.assign(point, all_matching=all_matching):
            candidates = by_point[key].get(name, [])
            if len(candidates) == 1:
                selected.append(candidates[0])
            elif candidates:
                ambiguous.append((name, key, candidates))
            elif name in seen[key]:
                wanted = plan.profiles[name].reco_category or point.category
                missing[name].append(
                    f"{_formatPointKey(key)} (no {wanted} template, only other categories)"
                )
            elif name == plan.default:
                # The default profile defines which points exist: a point found only
                # in another profile's inputs but routed to the default is not a gap
                n_extra += 1
            else:
                missing[name].append(f"{_formatPointKey(key)} (no inputs)")

    if ambiguous:
        lines = [
            f"  [{name}] {_formatPointKey(key)}: "
            + ", ".join(
                f"{j.metadata['pipeline']}/{j.metadata['reco_category']}" for j in jobs
            )
            for name, key, jobs in ambiguous
        ]
        raise ValueError(
            f"{len(lines)} point(s) have more than one candidate job for a profile:\n"
            + "\n".join(lines)
            + "\nRestrict the profile with pipelines=... or reco_category=... so it "
            "supplies one job per point."
        )

    if n_extra:
        logger.info(
            f"Ignored {n_extra} point(s) found only in non-default profile inputs and "
            f"routed to the default profile '{plan.default}'"
        )

    if missing and not allow_missing:
        lines = [f"  [{name}] {d}" for name, descs in missing.items() for d in descs]
        raise FileNotFoundError(
            f"{len(lines)} point(s) are assigned to a profile that has no usable inputs for them:\n"
            + "\n".join(lines)
            + "\nFix the plan's choose() or the profile patterns, or pass "
            "--allow-missing-profile-inputs to skip these jobs."
        )

    no_bkg = [j for j in selected if not Path(j.background).exists()]
    if no_bkg:
        listing = "\n".join(f"  [{j.metadata['fit_mode']}] {j.background}" for j in no_bkg)
        raise FileNotFoundError(f"Backgrounds not found for {len(no_bkg)} job(s):\n{listing}")

    _logPlanSummary(plan, selected, missing)
    return selected


def _logPlanSummary(plan: FitPlan, selected: list[EraJob], missing: dict[str, list[str]]):
    counts = defaultdict(int)
    for j in selected:
        counts[j.metadata["fit_mode"]] += 1
    for name in plan.profiles:
        logger.info(f"Fit profile '{name}': {counts[name]} job(s)")

    for name, descs in missing.items():
        logger.warning(
            f"{len(descs)} point(s) assigned to profile '{name}' have no usable '{name}' "
            f"inputs and will NOT be run:"
        )
        for d in descs:
            logger.warning("  " + d)

    non_default = sorted(
        {(j.metadata["fit_mode"], j.metadata["coupling"], int(j.metadata["mstop"]), int(j.metadata["mchi"]))
         for j in selected if j.metadata["fit_mode"] != plan.default}
    )
    for mode, coupling, mstop, mchi in non_default:
        logger.info(f"  [{mode}] coupling={coupling} mstop={mstop} mchi={mchi}")


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
        "era_fit_modes": d.join(
            j.metadata.get("fit_mode", NO_VALUE) for j in group.era_jobs
        ),
        "era_ndims": d.join(
            str(j.expected_ndim) if j.expected_ndim else NO_VALUE
            for j in group.era_jobs
        ),
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
