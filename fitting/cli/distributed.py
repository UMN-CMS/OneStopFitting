from __future__ import annotations
from pathlib import Path
import click


@click.command("makecondor")
@click.option(
    "--signal",
    required=True,
    multiple=True,
    help="Signal pattern (e.g. '**/signal_{year}_*.pklz4')",
)
@click.option(
    "--background",
    required=True,
    help="Background pattern (e.g. '**/bkg_{year}.pklz4')",
)
@click.option("--years", multiple=True, required=True, help="Years to process")
@click.option("--pipelines", multiple=True, required=True, help="Pipelines to process")
@click.option(
    "--config-pattern",
    "-c",
    type=str,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("condor_output"),
    help="Output directory for condor files",
)
@click.option(
    "--subdir-format",
    type=str,
    required=True,
    help="Format for subdirectory",
)
@click.option("--venv", type=str, help="Path to virtual environment to pack")
@click.option("--container", type=str, help="Container image to use")
@click.option("--num-toys", type=int, default=None, help="Number of toys to run over")
@click.option("--toy-offset", type=int, default=0, help="Offset for toy index.")
@click.option(
    "--combine-cmd",
    "combine_cmds",
    multiple=True,
    help="Combine commands to run after the fit",
)
@click.option(
    "--multi-signal",
    is_flag=True,
    default=False,
    help="Group signals by mass point into multi-signal jobs.",
)
def makecondorCmd(
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
) -> None:
    """Generate HTCondor submit files for distributed processing."""
    from ..distributed.condor_tools import generateCondorSubmit

    generateCondorSubmit(
        signal_pattern=signal,
        background_pattern=background,
        years=list(years),
        pipelines=list(pipelines),
        config_pattern=config_pattern,
        output_dir=output,
        subdir_format=subdir_format,
        venv_path=venv,
        container=container,
        num_toys=num_toys,
        toy_offset=toy_offset,
        multi_signal=multi_signal,
    )


@click.command("makebatch")
@click.option(
    "--signal",
    required=True,
    multiple=True,
    help="Signal pattern (e.g. '**/signal_{year}_*.pklz4')",
)
@click.option(
    "--background",
    required=True,
    help="Background pattern (e.g. '**/bkg_{year}.pklz4')",
)
@click.option("--years", multiple=True, required=True, help="Years to process")
@click.option("--pipelines", multiple=True, required=True, help="Pipelines to process")
@click.option(
    "--config-base",
    "-c",
    type=str,
    help="Base config file to use as template",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("batch_output"),
    help="Output directory for batch job files",
)
@click.option(
    "--subdir-format",
    type=str,
    default="{era.name}/{dataset_name}/{injection_rate}",
    show_default=True,
    help="Base format for output subdirectory (will be extended with batch parameters)",
)
@click.option("--venv", type=str, help="Path to virtual environment to pack")
@click.option("--container", type=str, help="Container image to use")
@click.option(
    "--combine-cmd",
    "combine_cmds",
    multiple=True,
    help="Combine commands to run after the fit",
)
@click.option(
    "--rates",
    type=str,
    help="Comma-separated injection rates (e.g., '0.0,0.1,0.5')",
)
@click.option(
    "--rebin",
    type=str,
    help="Comma-separated rebin factors (e.g., '1,2,4')",
)
@click.option(
    "--min-counts",
    type=str,
    help="Comma-separated min counts values (e.g., '1.0,5.0')",
)
@click.option(
    "--injection-rates",
    type=str,
    help="Comma-separated injection rates (e.g., '0.0,1.0,10.0')",
)
@click.option("--num-toys", type=int, default=None, help="Number of toys to run over")
@click.option("--toy-offset", type=int, default=0, help="Offset for toy index.")
@click.option(
    "--param",
    "extra_params",
    multiple=True,
    type=str,
    help="Arbitrary dot-path parameter sweep: key=csv_values (e.g., 'model.likelihood.variance_floor_quantile=0.01,0.05,0.1')",
)
@click.option(
    "--multi-signal",
    is_flag=True,
    default=False,
    help="Group signals by mass point into multi-signal jobs.",
)
def makebatchCmd(
    signal: tuple[str, ...],
    background: str,
    years: tuple[str, ...],
    pipelines: tuple[str, ...],
    config_base: Path | None,
    output: Path,
    subdir_format: str,
    venv: str | None,
    container: str | None,
    rates: str | None,
    rebin: str | None,
    injection_rates: str | None,
    min_counts: str | None,
    num_toys: int | None,
    toy_offset: int,
    extra_params: tuple[str, ...],
    multi_signal: bool,
) -> None:
    """Generate HTCondor submit files for a batch of jobs with parameter sweeps."""
    import yaml
    from ..distributed.batch_tools import generateBatchSubmit

    parsed_extra = {}
    for p in extra_params:
        key, vals = p.split("=", 1)
        parsed_extra[key] = yaml.safe_load("[" + vals + "]")

    generateBatchSubmit(
        signal_pattern=signal,
        background_pattern=background,
        years=list(years),
        pipelines=list(pipelines),
        config_base=config_base,
        output_dir=output,
        subdir_format=subdir_format,
        venv_path=venv,
        container=container,
        rates=parseCsvFloat(rates) if rates else None,
        rebin=parseCsvInt(rebin) if rebin else None,
        min_counts=parseCsvFloat(min_counts) if min_counts else None,
        injection_rates=parseCsvFloat(injection_rates) if injection_rates else None,
        num_toys=num_toys,
        toy_offset=toy_offset,
        extra_params=parsed_extra if parsed_extra else None,
        multi_signal=multi_signal,
    )


def parseCsvFloat(s: str) -> list[float]:
    """Parse comma-separated string to list of floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parseCsvInt(s: str) -> list[int]:
    """Parse comma-separated string to list of ints."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]
