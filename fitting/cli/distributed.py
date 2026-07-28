from __future__ import annotations

from pathlib import Path

import click
import yaml


@click.command("makesubmit")
@click.option("--signal", required=True, multiple=True, help="Signal pattern")
@click.option("--background", required=True, help="Background pattern")
@click.option("--years", multiple=True, required=True, help="Years to process")
@click.option("--pipelines", multiple=True, required=True, help="Pipelines to process")
@click.option("--config-pattern", "-c", type=str, help="Config file pattern")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("condor_output"),
    help="Output directory",
)
@click.option(
    "--subdir-format", type=str, required=True, help="Format for output subdirectory"
)
@click.option("--venv", type=str, help="Path to virtual environment")
@click.option("--container", type=str, help="Container image")
@click.option("--num-toys", type=int, default=None, help="Number of toys")
@click.option("--toy-offset", type=int, default=0, help="Toy index offset")
@click.option(
    "--multi-signal", is_flag=True, default=False, help="Group signals by mass point"
)
@click.option(
    "--combine-eras",
    default=None,
    help="Comma-separated eras to combine in a fat-job (e.g. '2016,2017,2018')",
)
@click.option(
    "--combined-subdir-format",
    default=None,
    type=str,
)
@click.option(
    "--group-by",
    "group_by_str",
    default=None,
    help="Comma-separated grouping fields (default: mstop,mchi,category,pipeline,toy_index)",
)
@click.option(
    "--param",
    "extra_params",
    multiple=True,
    help="Parameter sweep: key=csv_values (e.g. 'injection_rate=0.0,0.5,1.0')",
)
@click.option(
    "--match-rule",
    "match_rules_cli",
    multiple=True,
    default=None,
    help="Background→signal mapping, e.g. 'year:Run3=202*'",
)
@click.option(
    "--match-rule-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="YAML match rules file",
)
def makesubmitCmd(
    signal: tuple[str, ...],
    background: str,
    years: tuple[str, ...],
    pipelines: tuple[str, ...],
    config_pattern: str | None,
    output: Path,
    subdir_format: str,
    venv: str | None,
    container: str | None,
    num_toys: int | None,
    toy_offset: int,
    multi_signal: bool,
    combine_eras: str | None,
    combined_subdir_format: str | None,
    group_by_str: str | None,
    extra_params: tuple[str, ...],
    match_rules_cli: tuple[str, ...] | None,
    match_rule_config: Path | None,
) -> None:
    """Generate HTCondor submit files."""
    from ..utils import MatchRules
    from ..distributed.condor_tools import (
        makeEraJobs,
        mergeSignals,
        buildPlan,
        generateFiles,
    )
    from ..distributed.batch_tools import expandParameters

    if match_rules_cli:
        match_rules = MatchRules.fromCLI(match_rules_cli)
    elif match_rule_config:
        match_rules = MatchRules.fromConfig(match_rule_config)
    else:
        match_rules = None

    if bool(combined_subdir_format) != bool(combine_eras):
        raise click.UsageError("Must specify either both or neither of combined_subdir_format and combine_eras")

    parsed_eras = [e.strip() for e in combine_eras.split(",")] if combine_eras else None
    group_by_fields = (
        tuple(f.strip() for f in group_by_str.split(",")) if group_by_str else None
    )

    toys = range(num_toys) if num_toys else [0]
    jobs = []
    for t in toys:
        jobs.extend(
            makeEraJobs(
                signal_pattern=signal,
                background_pattern=background,
                years=list(years),
                pipelines=list(pipelines),
                output_dir=str(output / subdir_format),
                config_pattern=config_pattern,
                toy_index=t + toy_offset,
                match_rules=match_rules,
            )
        )

    if not jobs:
        print("No jobs found!")
        return

    print(f"Found {len(jobs)} era-jobs")

    if multi_signal:
        print("Merging {len(jobs)} into common signals")
        jobs = mergeSignals(jobs)
        print(f"Merged into {len(jobs)} multi-signal jobs")

    if extra_params:
        parsed_params = {}
        for p in extra_params:
            key, vals = p.split("=", 1)
            parsed_params[key] = yaml.safe_load("[" + vals + "]")
        jobs = expandParameters(jobs, parsed_params, output / "batch_configs")

    combined_dir_format = None
    if combined_subdir_format is not None:
        combined_dir_format = str(output / combined_subdir_format),
        

    plan = buildPlan(
        jobs=jobs,
        output_dir=output,
        combine_eras=parsed_eras,
        combined_dir_format=combined_dir_format,
        group_by_fields=group_by_fields,
        venv_path=venv,
        container=container,
    )
    submit_path = generateFiles(plan)
    print(f"Generated {len(plan.job_groups)} jobs in {submit_path}")


@click.command("generate-combine-script")
@click.option(
    "--datacard",
    required=True,
    type=click.Path(path_type=Path),
    help="Path to datacard.txt",
)
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Pipeline config YAML",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory",
)
def generateCombineScriptCmd(
    datacard: Path,
    config: Path,
    output: Path,
) -> None:
    """Generate a run_combine_commands.sh from config settings."""
    import os
    from jinja2 import Environment, FileSystemLoader
    from ..combine.commands import CombineContext, Text2Workspace, resolveCommands

    with open(config) as f:
        cfg = yaml.safe_load(f)

    combine_cfg = cfg.get("combine", {})
    cmd_names = combine_cfg.get(
        "combine_commands",
        [
            "limits",
            "fit-diagnostics",
            "multidimfit",
            "significance",
            "gof-saturated",
        ],
    )
    combine_container = combine_cfg.get(
        "combine_container",
        "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest",
    )

    context = CombineContext(signal_labels=["signal"], channel_name="combined")
    cmds = Text2Workspace().render(context)
    for cmd in resolveCommands(cmd_names):
        cmds.extend(cmd.render(context))

    output.mkdir(parents=True, exist_ok=True)
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    script = env.get_template("run_combine_commands.sh.jinja").render(
        container=combine_container,
        commands=cmds,
        enumerate=enumerate,
    )

    script_path = output / "run_combine_commands.sh"
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    print(f"Combine script: {script_path} ({len(cmds)} commands)")
