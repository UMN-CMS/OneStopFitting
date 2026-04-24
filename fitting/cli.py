from __future__ import annotations

import copy
import logging
from pathlib import Path

import attrs
import click
import jax
import lz4.frame
import enum
import pickle
from .diagnostics.plot_utils import savePlots
from .distributed.batch_tools import generateBatchSubmit
from .distributed.condor_tools import generateCondorSubmit
from fitting.utils import dotFormat

from .core.serialization import converter, load
from .diagnostics.aggregate_plots import (
    collectPoints,
    iterSummaryFiles,
    makeAggregateMassPlanePlot,
    makeAggregateSmoothPlot,
    makeAggregateViolinPlot,
    makeAggregateScatterPlot,
)

from .diagnostics.point_report import (
    PointReportConfig,
    generatePointReports,
)

from .pipeline import (
    CombineConfig,
    PipelineConfig,
    runPipeline,
    PipelineStep,
)
from .data.windowing import (
    WindowConfig,
    GaussianWindowConfig,
    CoreDilatedWindowConfig,
    RectangularWindowConfig,
    CutWindowConfig,
)

WINDOW_TYPE_MAP: dict[str, type[WindowConfig]] = {
    "gaussian": GaussianWindowConfig,
    "core-dilated": CoreDilatedWindowConfig,
    "rectangular": RectangularWindowConfig,
    "cut": CutWindowConfig,
}


def _parseWindowParams(window_type: str | None, window_params: tuple[str, ...]) -> WindowConfig | None:
    """Build a WindowConfig from CLI --window-type and --window-param flags."""
    if window_type is None:
        return None  # signal: use whatever the config/default provides
    if window_type == "none":
        return None  # explicitly disable windowing

    cls = WINDOW_TYPE_MAP.get(window_type)
    if cls is None:
        valid = ", ".join(list(WINDOW_TYPE_MAP.keys()) + ["none"])
        raise click.UsageError(f"Unknown window type '{window_type}'. Valid: {valid}")

    kwargs: dict[str, object] = {}
    for param in window_params:
        if "=" not in param:
            raise click.UsageError(f"Window param must be key=value, got: '{param}'")
        key, raw = param.split("=", 1)
        # Try numeric conversion
        try:
            value: object = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                # Try list of floats (comma-separated)
                if "," in raw:
                    value = [float(x.strip()) for x in raw.split(",")]
                else:
                    value = raw
        kwargs[key] = value

    return cls(**kwargs)

from .steps.generators import generateSmoothedBackground, generateAsimovSmoothed
from .inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    OptimizerType,
    ObjectiveType,
)
import yaml

logger = logging.getLogger("fitting")


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """GPR Background Estimation for HEP."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JSON config file.",
)
@click.option(
    "--background",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Background histogram file.",
)
@click.option(
    "--signal",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Signal histogram file.",
)
@click.option(
    "--injection-rate", "-r", type=float, default=0.0, help="Signal injection rate."
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="output",
    help="Output directory format.",
)
@click.option("--rebin", type=int, default=1, help="Rebin factor.")
@click.option(
    "--min-counts", type=float, default=1.0, help="Min bin count for fit domain."
)
@click.option("--num-iters", type=int, default=None, help="Training iterations.")
@click.option("--lr", type=float, default=0.01, help="Learning rate.")
@click.option("--seed", type=int, default=0xBEEFBEEF, help="RNG seed.")
@click.option(
    "--window-type",
    type=str,
    default=None,
    help="Window strategy: gaussian, core-dilated, rectangular, cut, none.",
)
@click.option(
    "--window-param",
    "window_params",
    multiple=True,
    help="Window parameter as key=value (repeatable). E.g. --window-param spread=1.5",
)
@click.option(
    "--mode",
    type=click.Choice(InferenceMode, case_sensitive=False),
    default=None,
    help="Inference mode.",
)
@click.option(
    "--optimizer",
    type=click.Choice(OptimizerType, case_sensitive=False),
    default=OptimizerType.ADAM,
    help="Optimizer type.",
)
@click.option(
    "--objective",
    type=click.Choice(ObjectiveType, case_sensitive=False),
    default=ObjectiveType.MLL,
    help="Objective function.",
)
@click.option("--use-map-priors", is_flag=True, help="Use MAP priors.")
@click.option("--num-samples", type=int, default=500, help="MCMC samples.")
@click.option("--num-warmup", type=int, default=200, help="MCMC warmup steps.")
@click.option("--num-chains", type=int, default=1, help="MCMC chains.")
@click.option("--stage1-iters", type=int, default=100, help="Stage 1 iterations.")
@click.option("--stage2-iters", type=int, default=100, help="Stage 2 iterations.")
@click.option(
    "--step",
    type=click.Choice(PipelineStep, case_sensitive=False),
    help="Run only a specific step.",
)
@click.option(
    "--start-from",
    type=click.Choice(PipelineStep, case_sensitive=False),
    help="Start from a specific step.",
)
@click.option(
    "--combine-command",
    "combine_commands",
    multiple=True,
    default=[
        "limits",
        "fit-diagnostics",
        "multidimfit",
        "significance",
        "gof-saturated",
    ],
    help="Combine commands to run (e.g., 'fit', 'limits', 'significance', or full custom commands).",
)
@click.option(
    "--combine-container",
    type=str,
    help="Combine container image path.",
)
def run(
    config: Path | None,
    background: Path | None,
    signal: Path | None,
    injection_rate: float,
    output: str,
    rebin: int,
    min_counts: float,
    num_iters: int,
    lr: float,
    seed: int,
    window_type: str | None,
    window_params: tuple[str, ...],
    mode: InferenceMode,
    optimizer: OptimizerType,
    objective: ObjectiveType,
    use_map_priors: bool,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    stage1_iters: int,
    stage2_iters: int,
    step: PipelineStep | None,
    start_from: PipelineStep | None,
    combine_commands: tuple[str, ...],
    combine_container: str | None,
) -> None:
    start_from_step = start_from or PipelineStep.LOAD

    if config is not None:
        with open(config, "r") as f:
            raw = yaml.safe_load(f)

        if background is not None:
            raw["background_path"] = str(background)
        if signal is not None:
            raw["signal_path"] = str(signal)
        if output is not None:
            raw["output_dir_format"] = str(output)
        if injection_rate is not None:
            raw["injection_rate"] = injection_rate
        if num_iters is not None:
            raw["optimization"]["num_iters"] = num_iters
        if lr is not None:
            raw["optimization"]["lr"] = lr

        # CLI window overrides config-level window
        window_config = _parseWindowParams(window_type, window_params)
        if window_config is not None or window_type == "none":
            if window_config is None:
                raw["window"] = None
            else:
                from .core.serialization import converter as _conv
                raw["window"] = _conv.unstructure(window_config)
        if mode is not None:
            raw["optimization"]["mode"] = mode

        if "combine" not in raw:
            raw["combine"] = {}
        if combine_commands:
            raw["combine"]["combine_commands"] = list(combine_commands)
        if combine_container:
            raw["combine"]["combine_container"] = combine_container

        pipeline_config = converter.structure(raw, PipelineConfig)

    elif background is not None:
        pipeline_config = PipelineConfig(
            background_path=background,
            signal_path=signal,
            injection_rate=injection_rate,
            output_dir_format=output,
            rebin=rebin,
            min_counts=min_counts,
            rng_seed=seed,
            window=_parseWindowParams(window_type, window_params),
            optimization=OptimizationConfig(
                mode=mode,
                lr=lr,
                num_iters=num_iters,
            ),
            combine=CombineConfig(
                combine_commands=list(combine_commands) if combine_commands else [],
                combine_container=combine_container
                or "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest",
            ),
        )
    else:
        raise click.UsageError("Must specify either --config or --background.")

    runPipeline(pipeline_config, start_from=start_from_step, single_step=step)


@main.command("print-params")
@click.argument("input", type=str)
def printParams(input):
    from .diagnostics.parameters import (
        logKernelParameters,
        logLikelihoodParameters,
        meanParameters,
    )
    from .core.serialization import load

    data = load(input)
    if data.training_result is None:
        raise RuntimeError(
            f"Can only print parameters on a state that has been trained"
        )

    posterior = data.training_result.posterior
    logKernelParameters(posterior)
    meanParameters(posterior)


@main.command()
@click.option(
    "--state",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to saved AnalysisState.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the smoothed histogram (compressed pickle).",
)
@click.option(
    "--name",
    "-n",
    type=str,
    required=True,
    help="Name of the smoothed histogram. Will be appended with toy index.",
)
@click.option("--seed", type=int, default=42, help="RNG seed.")
@click.option("--include-smooth", default=True, is_flag=True)
@click.option("--num-samples", type=int, default=1, help="Number of samples to draw.")
def smooth(
    state: Path,
    output_dir: Path,
    name: str,
    seed: int,
    include_smooth: bool,
    num_samples: int,
) -> None:

    jax.config.update("jax_enable_x64", True)
    rng_key = jax.random.key(seed)

    logger.info(f"Loading state from {state}")
    analysis_state = load(state)

    logger.info("Generating smoothed background...")
    hists, plots = generateSmoothedBackground(
        analysis_state, rng_key, num_samples=num_samples
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "smoothing_diagnostics"
    savePlots(plots, plot_dir, [analysis_state.metadata])
    logger.info(f"Smoothing diagnostic plots saved to {plot_dir}")

    for idx, hist in enumerate(hists):
        o = output_dir / f"{name}_{idx}.pklz4"
        metadata = copy.deepcopy(analysis_state.background_metadata)
        metadata["toy_index"] = idx
        with lz4.frame.open(o, "wb") as f:
            to_save = {
                "item": hist,
                "metadata": metadata,
            }
            pickle.dump(to_save, f)

    if include_smooth:
        pure_plot_dir = output_dir / "pure_smooth_diagnostics"
        pure_plot_dir.mkdir(exist_ok=True, parents=True)
        h, plots = generateAsimovSmoothed(analysis_state, rng_key)
        savePlots(plots, pure_plot_dir, [analysis_state.metadata])
        o = output_dir / f"pure_smoothed.pklz4"
        metadata = copy.deepcopy(analysis_state.background_metadata)
        with lz4.frame.open(o, "wb") as f:
            to_save = {
                "item": h,
                "metadata": metadata,
            }
            pickle.dump(to_save, f)

    logger.info(f"Smoothed background saved to {output_dir}")


@main.command("aggregate")
@click.argument(
    "inputs",
    nargs=-1,
)
@click.option(
    "-m",
    "--metric",
    "metric_dotpath",
    required=True,
    multiple=True,
    help="Dot-path into summary.json, e.g. 'metrics.blinded_chi2_per_bin'. Can be specified multiple times.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory where plots will be written.",
)
@click.option(
    "-n",
    "--name-format",
    type=str,
    default="{metric_name}.png",
    help="Format for output filenames.",
)
@click.option(
    "-f",
    "--formats",
    multiple=True,
    default=("png",),
    show_default=True,
    help="Image formats to write (repeatable), e.g. --formats png --formats pdf.",
)
@click.option(
    "--plot-types",
    multiple=True,
    default=("mass_plane",),
    show_default=True,
    help="Types of aggregate plots to generate: mass_plane, violin, scatter.",
)
@click.option(
    "-g",
    "--group-by",
    multiple=True,
)
@click.option("--title", type=str, default=None, help="Plot title override.")
@click.option("--cmap", type=str, default="viridis", show_default=True)
@click.option("--cmin", type=float, default=None, help="Color scale min.")
@click.option("--cmax", type=float, default=None, help="Color scale max.")
@click.option(
    "--smooth-sigma",
    type=float,
    default=None,
    help="Optional Gaussian smoothing sigma (in grid-bin units). If set, uses heatmap rendering.",
)
@click.option(
    "--smooth-truncate",
    type=float,
    default=4.0,
    show_default=True,
    help="Gaussian filter truncate (in sigmas).",
)
@click.option("--save-data", default=True, is_flag=True)
@click.option(
    "--stop-dotpath",
    type=str,
    default="metadata.other_data.stop_mass",
    show_default=True,
)
@click.option(
    "--chi-dotpath",
    type=str,
    default="metadata.other_data.chargino_mass",
    show_default=True,
)
def aggregate(
    inputs: tuple[str, ...],
    metric_dotpath: tuple[str, ...],
    output: Path,
    formats: tuple[str, ...],
    name_format: str,
    group_by: tuple[str, ...],
    title: str | None,
    cmap: str,
    cmin: float | None,
    cmax: float | None,
    smooth_sigma: float | None,
    smooth_truncate: float,
    save_data: bool,
    stop_dotpath: str,
    chi_dotpath: str,
    plot_types: tuple[str, ...],
) -> None:
    """Create an aggregate 2D mass-plane plot from many summary.json files."""
    from .diagnostics.plot_utils import savePlots
    import json
    import cattrs
    from .diagnostics.aggregate_plots import makeMulti

    summary_files = list(iterSummaryFiles(inputs))
    if not summary_files:
        raise click.UsageError("No summary.json files found for given input(s).")

    points = collectPoints(
        summary_files,
        metric_dotpath=metric_dotpath,
        group_by=group_by,
        stop_dotpath=stop_dotpath,
        chi_dotpath=chi_dotpath,
    )

    for k in list(points):
        points[k] = makeMulti(points[k])

    all_points = [x for y in points.values() for x in y]
    logger.info(f"Gathered {len(all_points)} points into {len(points)} groups")
    if not points:
        raise click.ClickException(
            f"Found {len(summary_files)} summary.json files, but none contained "
            f"'{metric_dotpath}' plus required mass metadata."
        )
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)

    if save_data:
        for k, d in points.items():
            n = dotFormat(name_format, metric_name=metric_dotpath, **dict(k))
            n = n.replace(".","p")
            p = (output / n).with_suffix(".json")
            logger.info(f"Saving data to {p}")
            with open(p, "w") as f:
                json.dump(cattrs.unstructure(d), f)
    plots = {}
    metric_name_str = metric_dotpath[0] if isinstance(metric_dotpath, tuple) and len(metric_dotpath) > 0 else metric_dotpath
    for k, p in points.items():
        plots_k = {}
        if "mass_plane" in plot_types:
            plots_k.update(makeAggregateMassPlanePlot(
                p,
                metric_name=metric_name_str,
                title=title,
                cmap=cmap,
                cmin=cmin,
                cmax=cmax,
                smooth_sigma=smooth_sigma,
                smooth_truncate=smooth_truncate,
                name_format="{plot_type}_" + name_format if "{plot_type}" not in name_format else name_format,
                params=dict(k, plot_type="mass_plane"),
            ))
        if "mass_plane_smooth" in plot_types:
            plots_k.update(makeAggregateSmoothPlot(
                p,
                metric_name=metric_name_str,
                title=title,
                cmap=cmap,
                cmin=cmin,
                cmax=cmax,
                smooth_sigma=smooth_sigma,
                smooth_truncate=smooth_truncate,
                name_format="{plot_type}_" + name_format if "{plot_type}" not in name_format else name_format,
                params=dict(k, plot_type="mass_plane"),
            ))
        if "violin" in plot_types:
            from .diagnostics.aggregate_plots import makeAggregateViolinPlot
            plots_k.update(makeAggregateViolinPlot(
                p,
                metric_name=metric_name_str,
                title=title,
                name_format="{plot_type}_" + name_format if "{plot_type}" not in name_format else name_format,
                params=dict(k, plot_type="violin"),
            ))
        if "scatter" in plot_types:
            from .diagnostics.aggregate_plots import makeAggregateScatterPlot
            plots_k.update(makeAggregateScatterPlot(
                p,
                metric_name=metric_name_str,
                title=title,
                name_format="{plot_type}_" + name_format if "{plot_type}" not in name_format else name_format,
                params=dict(k, plot_type="scatter"),
            ))

        savePlots(plots_k, output, [x.metadata for x in p], formats=formats)
    logger.info(f"Aggregate plot saved to {output}")


@main.command("toy-analyze")
@click.option(
    "-i",
    "--inputs",
    "inputs",
    multiple=True,
    required=True,
    help="Directory, summary.json path, or glob pattern.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for analysis results.",
)
@click.option(
    "-m",
    "--metric",
    "metric_dotpath",
    required=True,
    help="Dot-path to metric to analyze (e.g., 'metrics.blinded_chi2_per_bin').",
)
@click.option(
    "-g",
    "--group-by",
    multiple=True,
    default=["metadata.dataset_name"],
    help="Dot-paths to group by (hierarchical, in order).",
)
@click.option(
    "-n",
    "--name-format",
    type=str,
    default="{plot_type}_{metric_short}",
    help="Format for output filenames using dotFormat syntax. "
    "Available: {plot_type}, {metric_name}, {metric_short}, and all group variables.",
)
@click.option(
    "--plot-types",
    multiple=True,
    default=["box", "violin", "histogram"],
    help="Types of plots to generate: box, violin, histogram, scatter.",
)
@click.option(
    "--formats",
    multiple=True,
    default=["png"],
    help="Output image formats (e.g., png, pdf).",
)
@click.option(
    "--save-data",
    is_flag=True,
    default=True,
    help="Save grouped data and statistics as JSON.",
)
@click.option(
    "--correlation",
    is_flag=True,
    help="Generate scatter plots to check for trends across toy index.",
)
def toyAnalyze(
    inputs: tuple[str, ...],
    output: Path,
    metric_dotpath: str,
    group_by: tuple[str, ...],
    name_format: str,
    plot_types: tuple[str, ...],
    formats: tuple[str, ...],
    save_data: bool,
    correlation: bool,
) -> None:
    """Analyze toy-to-toy variation in a single metric."""
    from .diagnostics.toy_analysis import (
        collectToyData,
        computeStatistics,
        makeToyVariationPlots,
        saveData,
    )

    summary_files = list(iterSummaryFiles(inputs))
    if not summary_files:
        raise click.UsageError("No summary.json files found.")

    logger.info(f"Found {len(summary_files)} summary files")

    grouped_data = collectToyData(
        summary_files,
        group_by=list(group_by),
        metric_dotpath=metric_dotpath,
    )

    if not grouped_data:
        raise click.ClickException(
            f"No valid data found for metric '{metric_dotpath}' with grouping {group_by}"
        )

    logger.info(f"Grouped data into {len(grouped_data)} groups")

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_short = metric_dotpath.replace(".", "_")
    all_plot_types = list(plot_types)
    if correlation:
        all_plot_types.append("scatter")

    num_plots = makeToyVariationPlots(
        grouped_data,
        metric_dotpath=metric_dotpath,
        metric_short=metric_short,
        plot_types=all_plot_types,
        output_dir=output_dir,
        formats=list(formats),
        name_format=name_format,
    )

    logger.info(f"Generated {num_plots} plots")

    if save_data:
        saveData(grouped_data, computeStatistics, output_dir / "data.json")

    logger.info(f"Analysis complete. Results saved to {output_dir}")


@main.command()
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
@click.option(
    "--combine-cmd",
    "combine_cmds",
    multiple=True,
    help="Combine commands to run after the fit",
)
def makecondor(
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
    combine_cmds: tuple[str, ...],
) -> None:
    """Generate HTCondor submit files for distributed processing."""

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
        combine_cmds=list(combine_cmds),
        num_toys=num_toys,
    )


@main.command()
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
@click.option("--num-toys", type=int, default=None, help="Number of toys to run over")
def makebatch(
    signal: tuple[str, ...],
    background: str,
    years: tuple[str, ...],
    pipelines: tuple[str, ...],
    config_base: Path | None,
    output: Path,
    subdir_format: str,
    venv: str | None,
    container: str | None,
    combine_cmds: tuple[str, ...],
    rates: str | None,
    rebin: str | None,
    min_counts: str | None,
    num_toys: int | None,
) -> None:
    """Generate HTCondor submit files for a batch of jobs with parameter sweeps."""

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
        combine_cmds=list(combine_cmds),
        rates=parse_csv_float(rates) if rates else None,
        rebin=parse_csv_int(rebin) if rebin else None,
        min_counts=parse_csv_float(min_counts) if min_counts else None,
        num_toys=num_toys,
    )


def parse_csv_float(s: str) -> list[float]:
    """Parse comma-separated string to list of floats."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_int(s: str) -> list[int]:
    """Parse comma-separated string to list of ints."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


@main.command()
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="Directory, summary.json path, or glob pattern. May be passed multiple times.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    help="Output: single PDF or directory (for per-point reports).",
)
@click.option(
    "--single-document",
    is_flag=True,
    help="Combine all points into a single PDF document.",
)
@click.option(
    "--latex-engine",
    default="pdflatex",
    show_default=True,
    help="LaTeX engine to use (pdflatex, xelatex, etc.)",
)
@click.option(
    "--keep-build",
    is_flag=True,
    help="Keep LaTeX build directory for debugging.",
)
@click.option(
    "--keep-tex",
    is_flag=True,
    help="Keep intermediate .tex files.",
)
@click.option(
    "--image-format",
    default="png",
    show_default=True,
    help="Image format for plots (png, pdf, etc.)",
)
def report(
    inputs: tuple[str, ...],
    output: Path | None,
    single_document: bool,
    latex_engine: str,
    keep_build: bool,
    keep_tex: bool,
    image_format: str,
) -> None:
    config = PointReportConfig(
        latex_engine=latex_engine,
        keep_build=keep_build,
        keep_tex=keep_tex,
        image_format=image_format,
    )

    output_paths = generatePointReports(
        inputs=inputs,
        output=output,
        single_document=single_document,
        config=config,
    )
    logger.info(f"Generated {len(output_paths)} report(s)")


@main.command()
@click.option(
    "--background",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Background histogram file.",
)
@click.option(
    "--signal",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    help="Signal histogram file (optional).",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Config file containing pipeline parameters.",
)
@click.option(
    "--output-format",
    "-o",
    type=str,
    required=True,
    help="Output format string with placeholders (e.g., 'output/{era.name}/{dataset_name}/{injection_rate}')",
)
def resolveOutput(
    background: Path,
    signal: Path | None,
    config: Path,
    output_format: str,
) -> None:
    from .pipeline import PipelineConfig, loadData
    import sys

    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    pipeline_config = PipelineConfig(
        background_path=background,
        signal_path=signal,
        output_dir_format=output_format,
    )
    pipeline_config = attrs.evolve(pipeline_config, **config_data)
    state = loadData(pipeline_config)
    output_path = state.getRealOutPath()
    logger.info(f"Resolved output to '{str(output_path)}'")
    print(output_path, flush=True)


@main.command()
@click.argument(
    "summaries",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
def harvest(summaries: tuple[Path, ...]) -> None:
    from .combine.extract import extractCombineResults
    import json
    from .diagnostics.plot_utils import savePlots

    if not summaries:
        logger.warning("No summary files provided to harvest.")
        return

    success_count = 0
    for summary_path in summaries:
        combine_dir = summary_path.parent / "combine"
        plot_dir = summary_path.parent / "diagnostics" / "post_combine"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if not combine_dir.exists():
            logger.debug(f"No combine directory found at {combine_dir}")
            continue

        extracted, plots = extractCombineResults(combine_dir)
        if not extracted:
            logger.debug(f"No Combine results extracted for {summary_path}.")
            continue

        with open(summary_path, "r") as f:
            summary_data = json.load(f)

        summary_data["combine"] = extracted

        savePlots(plots, plot_dir, [summary_data["metadata"]])
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        logger.info(
            f"Updated {summary_path} with {list(extracted.keys())} and generated plots."
        )
        success_count += 1

    logger.info(
        f"Harvest complete. Updated {success_count}/{len(summaries)} summary files."
    )


@main.command("window-fit")
@click.argument("input", nargs=-1)
@click.option(
    "--window-type",
    type=str,
    default="core-dilated",
    help="Window strategy: gaussian, core-dilated, rectangular, cut.",
)
@click.option(
    "--window-param",
    "window_params",
    multiple=True,
    help="Window parameter as key=value (repeatable).",
)
@click.option("--rebin", default=1, type=int)
@click.option("-o", "--output-format", type=str)
def windowFit(
    input: tuple[str, ...],
    window_type: str,
    window_params: tuple[str, ...],
    rebin: int,
    output_format: str,
):
    from .utils import getRecoCategory
    from .data.loading import (
        FileLoader,
        extractHistogram,
        extractMetadata,
        histToBinnedData,
    )
    import matplotlib.pyplot as plt
    from .diagnostics.plot_utils import plotBinnedData, plotRaw, plotPPD, plotBlinding2D
    from .utils import dictToDot, dotFormat
    import numpy as np
    import json
    import mplhep
    from rich.progress import track

    mplhep.style.use("CMS")
    failures = 0

    window_config = _parseWindowParams(window_type, window_params)
    if window_config is None:
        raise click.UsageError("window-fit requires a valid --window-type")

    logger.info(
        f"Producing windows for {len(input)} signals using {type(window_config).__name__}"
    )
    # for signal_path in track(input, total=len(input)):
    for signal_path in input:
        sig_loader = FileLoader.forPath(signal_path)
        sig_raw = sig_loader.load(signal_path)
        sig_hist_full = extractHistogram(sig_raw)
        signal = histToBinnedData(sig_hist_full, rebin=rebin, variation="central")
        sig_metadata = extractMetadata(sig_raw)
        reco_cat = getRecoCategory(sig_metadata["name"])

        try:
            window = window_config.buildWindow(signal)
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Failed to fit {sig_metadata['dataset_name']}: {e}")
            failures += 1
            continue
        blind_mask = window(signal.X)

        fig, ax = plt.subplots(layout="tight")
        plotBinnedData(ax, signal)
        ax.set_title("Injected Signal")
        if signal.axis_names and len(signal.axis_names) >= 2:
            ax.set_xlabel(signal.axis_names[0])
            ax.set_ylabel(signal.axis_names[1])
        plotBlinding2D(ax, signal.edges, signal.X, blind_mask)
        blind_mask_size = int(np.count_nonzero(blind_mask))
        all_data = {
            "reco_category": reco_cat,
            "window_type": window_type,
            "blind_mask_size": blind_mask_size,
            **sig_metadata,
        }

        output_path = Path(dotFormat(output_format, **dict(dictToDot(all_data))))
        image_out = output_path.with_suffix(".png")
        json_out = output_path.with_suffix(".json")
        output_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(image_out)
        plt.close(fig)
        with open(json_out, "w") as f:
            json.dump(all_data, f)
    logger.info(f"Ran for {len(input)} signals with {failures} failures")


if __name__ == "__main__":
    main()

