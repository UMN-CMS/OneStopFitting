from __future__ import annotations

import copy
import logging
from pathlib import Path
import click
from .inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    OptimizerType,
    ObjectiveType,
)
from typing import TYPE_CHECKING
from .pipeline import PipelineStep
import attrs

if TYPE_CHECKING:
    from .data.windowing import WindowConfig


class CommaSeparatedFloat(click.ParamType):
    name = "comma_float"

    def convert(self, value, param, ctx):
        try:
            return [float(v.strip()) for v in value.split(",")]
        except AttributeError:
            self.fail(f"{value} is not a valid comma-separated list", param, ctx)


class CommaSeparatedInt(click.ParamType):
    name = "comma_int"

    def convert(self, value, param, ctx):
        try:
            return [int(v.strip()) for v in value.split(",")]
        except AttributeError:
            self.fail(f"{value} is not a valid comma-separated list", param, ctx)


def _parseWindowParams(
    window_type: str | None, window_params: tuple[str, ...]
) -> WindowConfig | None:

    from .data.windowing import (
        GaussianWindowConfig,
        CoreDilatedWindowConfig,
        RectangularWindowConfig,
        CutWindowConfig,
    )

    """Build a WindowConfig from CLI --window-type and --window-param flags."""
    if window_type is None:
        return None  # signal: use whatever the config/default provides
    if window_type == "none":
        return None  # explicitly disable windowing

    WINDOW_TYPE_MAP: dict[str, type[WindowConfig]] = {
        "gaussian": GaussianWindowConfig,
        "core-dilated": CoreDilatedWindowConfig,
        "rectangular": RectangularWindowConfig,
        "cut": CutWindowConfig,
    }

    cls = WINDOW_TYPE_MAP.get(window_type)
    if cls is None:
        valid = ", ".join(list(WINDOW_TYPE_MAP.keys()) + ["none"])
        raise click.UsageError(f"Unknown window type '{window_type}'. Valid: {valid}")

    kwargs: dict[str, object] = {}
    for param in window_params:
        if "=" not in param:
            raise click.UsageError(f"Window param must be key=value, got: '{param}'")
        key, raw = param.split("=", 1)
        try:
            value: object = int(raw)
        except ValueError:
            try:
                value = float(raw)
            except ValueError:
                if "," in raw:
                    value = [float(x.strip()) for x in raw.split(",")]
                else:
                    value = raw
        kwargs[key] = value

    return cls(**kwargs)


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
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Signal histogram file(s). Can be specified multiple times.",
)
@click.option(
    "--injection-rate", "-r", type=float, default=None, help="Signal injection rate."
)
@click.option(
    "--injection-signal",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Signal file to inject (for bias studies).",
)
@click.option(
    "--output",
    "-o",
    type=str,
    default="output",
    help="Output directory format.",
)
@click.option("--rebin", type=int, default=None, help="Rebin factor.")
@click.option(
    "--min-counts", type=float, default=1.0, help="Min bin count for fit domain."
)
@click.option("--num-iters", type=int, default=None, help="Training iterations.")
@click.option("--lr", type=float, default=None, help="Learning rate.")
@click.option("--seed", type=int, default=None, help="RNG seed.")
@click.option(
    "--signal-pre-scale",
    type=float,
    default=None,
    help="Pre-scale signal by this value.",
)
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
    signal: tuple[Path, ...],
    injection_rate: float,
    injection_signal: Path | None,
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
    signal_pre_scale: float,
) -> None:
    from .core.serialization import converter
    from .pipeline import (
        CombineConfig,
        PipelineConfig,
        runPipeline,
        PipelineStep,
    )
    import yaml

    start_from_step = start_from or PipelineStep.LOAD

    # Normalize signal paths
    signal_path_val = None
    if signal:
        signal_path_val = list(signal) if len(signal) > 1 else signal[0]

    if config is not None:
        with open(config, "r") as f:
            raw = yaml.safe_load(f)

        if background is not None:
            raw["background_path"] = str(background)
        if signal_path_val is not None:
            if isinstance(signal_path_val, list):
                raw["signal_path"] = [str(p) for p in signal_path_val]
            else:
                raw["signal_path"] = str(signal_path_val)
        if rebin is not None:
            raw["rebin"] = rebin
        if output is not None:
            raw["output_dir_format"] = str(output)
        if injection_rate is not None:
            raw["injection_rate"] = injection_rate
        if injection_signal is not None:
            raw["injection_signal_path"] = str(injection_signal)
        if seed is not None:
            raw["rng_seed"] = seed
        if signal_pre_scale is not None:
            raw["signal_pre_scale"] = signal_pre_scale
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
                raw["window"] = converter.unstructure(window_config)
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
            signal_path=signal_path_val,
            injection_rate=injection_rate,
            injection_signal_path=injection_signal,
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
            signal_pre_scale=signal_pre_scale,
        )
    else:
        raise click.UsageError("Must specify either --config or --background.")

    runPipeline(pipeline_config, start_from=start_from_step, single_step=step)


@main.command("print-params")
@click.argument("input", type=str)
def printParams(input):
    from .diagnostics.parameters import (
        logKernelParameters,
        meanParameters,
    )
    from .core.serialization import load

    data = load(input)
    if data.training_result is None:
        raise RuntimeError("Can only print parameters on a state that has been trained")

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
    import lz4.frame
    import pickle
    import jax
    from .core.serialization import load
    from .steps.generators import generateSmoothedBackground, generateAsimovSmoothed
    from .diagnostics.plot_utils import savePlots

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
        o = output_dir / "pure_smoothed.pklz4"
        metadata = copy.deepcopy(analysis_state.background_metadata)
        with lz4.frame.open(o, "wb") as f:
            to_save = {
                "item": h,
                "metadata": metadata,
            }
            pickle.dump(to_save, f)

    logger.info(f"Smoothed background saved to {output_dir}")


@main.command()
@click.argument(
    "inputs",
    nargs=-1,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
)
def gather(inputs: tuple[str, ...], output: Path):
    import json
    from .diagnostics.aggregate_plots import (
        iterSummaryFiles,
        readSummary,
    )

    logger.info(f"Gathering summaries from {inputs} into {output}")
    summary_files = iterSummaryFiles(inputs)
    to_save = [readSummary(x) for x in summary_files]
    logger.info(f"Total summaries: {len(to_save)}")
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(output, "w") as f:
        json.dump(to_save, f, indent=2)


def _aggregateCmdOptions(func):
    func = click.option(
        "-m",
        "--metric",
        "metric_dotpath",
        required=True,
        multiple=True,
        help="Dot-path into summary.json. Can be specified multiple times.",
    )(func)
    func = click.option(
        "-t",
        "--transform",
        type=str,
        default=None,
        help="Name of a registered transformation function.",
    )(func)
    func = click.option(
        "--merge/--no-merge",
        default=True,
        is_flag=True,
        help="Merge points via makeMulti.",
    )(func)
    func = click.option(
        "-n",
        "--name-format",
        type=str,
        default=None,
        help="Output filename format override.",
    )(func)
    return func


def _extractAndMerge(ctx, metric_dotpath, transform, merge):
    from .diagnostics.aggregate_plots import extractPoints, makeMulti

    points = extractPoints(
        ctx.obj["summaries"],
        metric_dotpath=metric_dotpath,
        group_by=ctx.obj["group_by"],
        stop_dotpath=ctx.obj["stop_dotpath"],
        chi_dotpath=ctx.obj["chi_dotpath"],
        transform_name=transform,
    )

    metric_name_str = (
        metric_dotpath[0]
        if isinstance(metric_dotpath, tuple) and len(metric_dotpath) > 0
        else str(metric_dotpath)
    )
    is_pvalue = (
        "pvalue" in metric_name_str.lower()
        or "p_value" in metric_name_str.lower()
        or ctx.obj.get("pval_mode", False)
    )

    if merge:
        for k in list(points):
            points[k] = makeMulti(points[k], is_pvalue=is_pvalue)

    all_points = [x for y in points.values() for x in y]
    logger.info(f"Gathered {len(all_points)} points into {len(points)} groups")
    if not points:
        raise click.ClickException(
            f"Summaries did not contain '{metric_dotpath}' plus required mass metadata."
        )

    return points, all_points, metric_name_str, is_pvalue


def _makeFmt(p, group_key):
    from .utils import dictToDot, dotFormat

    fmt_ctx = (
        dict(dictToDot(p[0].metadata)) if p and getattr(p[0], "metadata", None) else {}
    )
    fmt_ctx.update(dict(group_key))

    def fmt(s):
        if not isinstance(s, str):
            return s
        try:
            return dotFormat(s, **fmt_ctx)
        except KeyError:
            return s

    return fmt, fmt_ctx


def _saveGroupPlots(plots, output, points_group, formats, cms_extra):
    from .diagnostics.plot_utils import savePlots

    savePlots(
        plots,
        output,
        [x.metadata for x in points_group],
        formats=formats,
        extra_text=cms_extra,
    )


@main.group("aggregate")
@click.option(
    "-i",
    "--input",
    "inputs",
    multiple=True,
    required=True,
    help="Input summary files or globs.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory where plots will be written.",
)
@click.option(
    "-f",
    "--formats",
    multiple=True,
    default=("pdf",),
    show_default=True,
    help="Image formats to write.",
)
@click.option("-g", "--group-by", multiple=True)
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
@click.option("--pval-mode", default=False, is_flag=True)
@click.option(
    "--cms-extra",
    type=str,
    default=None,
    help="Extra text for CMS annotation.",
)
@click.option("--title", type=str, default=None, help="Plot title override.")
@click.pass_context
def aggregate(
    ctx,
    inputs,
    output,
    formats,
    group_by,
    stop_dotpath,
    chi_dotpath,
    pval_mode,
    cms_extra,
    title,
):
    """Create aggregate plots from summary.json files."""
    if ctx.resilient_parsing:
        return

    from .diagnostics.aggregate_plots import iterSummaryFiles, readSummary

    summary_files = list(iterSummaryFiles(inputs))
    if not summary_files:
        raise click.UsageError("No summary.json files found for given input(s).")

    logger.info(f"Loading {len(summary_files)} summary files...")
    summaries = [(path, readSummary(path)) for path in summary_files]

    ctx.ensure_object(dict)
    ctx.obj.update(
        {
            "summaries": summaries,
            "output": output,
            "formats": formats,
            "group_by": group_by,
            "stop_dotpath": stop_dotpath,
            "chi_dotpath": chi_dotpath,
            "pval_mode": pval_mode,
            "cms_extra": cms_extra,
            "title": title,
        }
    )


@aggregate.command("mass-plane")
@_aggregateCmdOptions
@click.option("--cmap", type=str, default="viridis", show_default=True)
@click.option("--cmin", type=float, default=None)
@click.option("--cmax", type=float, default=None)
@click.option(
    "--value-func",
    type=str,
    default="median",
    show_default=True,
    help="Stat key: median, mean, slope, intercept, etc.",
)
@click.option("--draw-contours", type=CommaSeparatedFloat(), default=None)
@click.option("--colorbar-label", type=str, default=None)
@click.option("--colorbar-scale", type=str, default="linear")
@click.option("--contour-fmt", type=str, default=None)
@click.pass_context
def massPlaneCmd(
    ctx,
    metric_dotpath,
    transform,
    merge,
    name_format,
    cmap,
    cmin,
    cmax,
    value_func,
    draw_contours,
    colorbar_label,
    colorbar_scale,
    contour_fmt,
):
    from .diagnostics.aggregate_plots import makeAggregateMassPlanePlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)
        get_value_func = lambda x: x.stats.get(value_func, x.stats.get("median"))

        plots = makeAggregateMassPlanePlot(
            p,
            metric_name=fmt(metric_name_str),
            get_value_func=get_value_func,
            title=fmt(ctx.obj["title"]),
            cmap=cmap,
            cmin=cmin,
            cmax=cmax,
            name_format=fmt_name,
            params=dict(fmt_ctx, plot_type="mass_plane"),
            draw_contours=tuple(draw_contours) if draw_contours else None,
            colorbar_label=fmt(colorbar_label),
            colorbar_scale=colorbar_scale,
            contour_fmt=fmt(contour_fmt),
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Mass plane plot saved to {output}")


@aggregate.command("smooth")
@_aggregateCmdOptions
@click.option("--cmap", type=str, default="viridis", show_default=True)
@click.option("--cmin", type=float, default=None)
@click.option("--cmax", type=float, default=None)
@click.option(
    "--smooth-sigma",
    type=float,
    default=None,
    help="Gaussian smoothing sigma (in grid-bin units).",
)
@click.option(
    "--smooth-truncate",
    type=float,
    default=4.0,
    show_default=True,
    help="Gaussian filter truncate (in sigmas).",
)
@click.option(
    "--value-func",
    type=str,
    default="median",
    show_default=True,
    help="Stat key: median, mean, slope, intercept, etc.",
)
@click.option("--draw-contours", type=CommaSeparatedFloat(), default=None)
@click.option("--colorbar-label", type=str, default=None)
@click.option("--colorbar-scale", type=str, default="linear")
@click.option("--contour-fmt", type=str, default=None)
@click.pass_context
def smoothCmd(
    ctx,
    metric_dotpath,
    transform,
    merge,
    name_format,
    cmap,
    cmin,
    cmax,
    smooth_sigma,
    smooth_truncate,
    value_func,
    draw_contours,
    colorbar_label,
    colorbar_scale,
    contour_fmt,
):
    from .diagnostics.aggregate_plots import makeAggregateSmoothPlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)
        get_value_func = lambda x: x.stats.get(value_func, x.stats.get("median"))

        plots = makeAggregateSmoothPlot(
            p,
            metric_name=fmt(metric_name_str),
            get_value_func=get_value_func,
            title=fmt(ctx.obj["title"]),
            cmap=cmap,
            cmin=cmin,
            cmax=cmax,
            smooth_sigma=smooth_sigma,
            smooth_truncate=smooth_truncate,
            name_format=fmt_name,
            params=dict(fmt_ctx, plot_type="smooth"),
            draw_contours=tuple(draw_contours) if draw_contours else (1.0, 2.0),
            colorbar_label=fmt(colorbar_label),
            colorbar_scale=colorbar_scale,
            contour_fmt=fmt(contour_fmt),
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Smooth plot saved to {output}")


@aggregate.command("scatter")
@_aggregateCmdOptions
@click.option("--xlim", type=CommaSeparatedFloat(), default=None)
@click.option("--vlines", type=CommaSeparatedFloat(), default=None)
@click.pass_context
def scatterCmd(ctx, metric_dotpath, transform, merge, name_format, xlim, vlines):
    from .diagnostics.aggregate_plots import makeAggregateScatterPlot

    points, _, metric_name_str, is_pvalue = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)

        plots = makeAggregateScatterPlot(
            p,
            metric_name=fmt(metric_name_str),
            title=fmt(ctx.obj["title"]),
            name_format=fmt_name,
            params=dict(fmt_ctx, plot_type="scatter"),
            xlim=xlim,
            vlines=vlines,
            pval_bands=is_pvalue,
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Scatter plot saved to {output}")


@aggregate.command("violin")
@_aggregateCmdOptions
@click.pass_context
def violinCmd(ctx, metric_dotpath, transform, merge, name_format):
    from .diagnostics.aggregate_plots import makeAggregateViolinPlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)

        plots = makeAggregateViolinPlot(
            p,
            metric_name=fmt(metric_name_str),
            title=fmt(ctx.obj["title"]),
            name_format=fmt_name,
            params=dict(fmt_ctx, plot_type="violin"),
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Violin plot saved to {output}")


@aggregate.command("injection-line")
@_aggregateCmdOptions
@click.option(
    "--error-type",
    type=click.Choice(["sem", "std"]),
    default="sem",
    show_default=True,
    help="Error bar type.",
)
@click.option("--ylim", type=CommaSeparatedFloat(), default=None)
@click.pass_context
def injectionLineCmd(
    ctx, metric_dotpath, transform, merge, name_format, error_type, ylim
):
    from .diagnostics.aggregate_plots import makeInjectionLinePlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "injection_line_{dataset_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)

        plots = makeInjectionLinePlot(
            p,
            title=fmt(ctx.obj["title"]),
            name_format=fmt_name,
            error_type=error_type,
            ylim=tuple(ylim) if ylim else None,
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Injection line plots saved to {output}")


@aggregate.command("save-data")
@_aggregateCmdOptions
@click.pass_context
def saveDataCmd(ctx, metric_dotpath, transform, merge, name_format):
    import json
    import cattrs
    from .utils import dotFormat

    points, all_points, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = Path(ctx.obj["output"])
    output.mkdir(exist_ok=True, parents=True)
    fmt_name = name_format or "{metric_name}"

    for k, d in points.items():
        n = dotFormat(fmt_name, metric_name="__".join(metric_dotpath), **dict(k))
        n = n.replace(".", "p")
        p = (output / n).with_suffix(".json")
        logger.info(f"Saving data to {p}")
        p.parent.mkdir(exist_ok=True, parents=True)
        with open(p, "w") as f:
            json.dump(cattrs.unstructure(d), f, indent=2)

    n = dotFormat("ALL_" + "__".join(metric_dotpath))
    n = n.replace(".", "p")
    p = (output / n).with_suffix(".json")
    p.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving data to {p}")
    with open(p, "w") as f:
        json.dump(cattrs.unstructure(all_points), f, indent=2)


@aggregate.command("diagnose")
@click.option(
    "-n",
    "--name-format",
    type=str,
    default="diagnostic_report",
    help="Output PDF filename format (dotFormat with summary metadata).",
)
@click.option(
    "--latex-engine",
    default="pdflatex",
    show_default=True,
    help="LaTeX engine.",
)
@click.option("--keep-build", is_flag=True, help="Keep LaTeX build directory.")
@click.option("--keep-tex", is_flag=True, help="Keep intermediate .tex files.")
@click.pass_context
def diagnoseCmd(ctx, name_format, latex_engine, keep_build, keep_tex):
    from .diagnostics.diagnostic_report import generateDiagnosticReport
    from .utils import dictToDot
    from collections import defaultdict

    flat = []
    for _path, summary in ctx.obj["summaries"]:
        if isinstance(summary, list):
            flat.extend(summary)
        else:
            flat.append(summary)

    group_by = ctx.obj["group_by"]

    if group_by:
        grouped = defaultdict(list)
        for entry in flat:
            dotted = dict(dictToDot(entry))
            key = tuple((x, dotted.get(x)) for x in group_by)
            grouped[key].append(entry)
    else:
        grouped = {(): flat}

    for group_key, entries in grouped.items():
        generateDiagnosticReport(
            gathered=entries,
            output_dir=ctx.obj["output"],
            name_format=name_format,
            name_ctx=dict(group_key),
            latex_engine=latex_engine,
            keep_build=keep_build,
            keep_tex=keep_tex,
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
    toy_offset: int,
    combine_cmds: tuple[str, ...],
    multi_signal: bool,
) -> None:
    """Generate HTCondor submit files for distributed processing."""
    from .distributed.condor_tools import generateCondorSubmit

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
        toy_offset=toy_offset,
        multi_signal=multi_signal,
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
    injection_rates: str | None,
    min_counts: str | None,
    num_toys: int | None,
    toy_offset: int,
    extra_params: tuple[str, ...],
    multi_signal: bool,
) -> None:
    """Generate HTCondor submit files for a batch of jobs with parameter sweeps."""
    import yaml
    from .distributed.batch_tools import generateBatchSubmit

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
        rates=parse_csv_float(rates) if rates else None,
        rebin=parse_csv_int(rebin) if rebin else None,
        min_counts=parse_csv_float(min_counts) if min_counts else None,
        injection_rates=parse_csv_float(injection_rates) if injection_rates else None,
        num_toys=num_toys,
        toy_offset=toy_offset,
        extra_params=parsed_extra if parsed_extra else None,
        multi_signal=multi_signal,
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
    from .diagnostics.point_report import (
        PointReportConfig,
        generatePointReports,
    )

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
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Signal histogram file(s). Can be specified multiple times.",
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
@click.option(
    "--injection-signal",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Signal file to inject (for bias studies).",
)
def resolveOutput(
    background: Path,
    signal: tuple[Path, ...],
    config: Path,
    output_format: str,
    injection_signal: Path | None,
) -> None:
    from .pipeline import PipelineConfig, loadData
    import yaml

    signal_path_val = None
    if signal:
        signal_path_val = list(signal) if len(signal) > 1 else signal[0]

    with open(config, "r") as f:
        config_data = yaml.safe_load(f)
    pipeline_config = PipelineConfig(
        background_path=background,
        signal_path=signal_path_val,
        injection_signal_path=injection_signal,
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
@click.option(
    "--diagnose/--no-diagnose",
    is_flag=True,
    default=True,
    help="Generate 2D slice plots and save histograms to pickle.",
)
def harvest(summaries: tuple[Path, ...], diagnose: bool) -> None:
    from .combine.extract import extractCombineResults
    import json
    from .diagnostics.plot_utils import getPlotSaver
    import jax.numpy as jnp

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

        json_extracted = {k: v for k, v in extracted.items() if k != "histograms"}

        with open(summary_path, "r") as f:
            summary_data = json.load(f)

        summary_data["combine"] = json_extracted
        saver = getPlotSaver(plot_dir, [summary_data["metadata"]])
        for k, v in plots.items():
            saver(k, *v)

        if diagnose and "histograms" in extracted:
            from .core.serialization import load
            from .diagnostics.plots_2d_slices import makePostCombineSlice
            import pickle
            import lz4.frame

            state_path = summary_path.parent / "state.pklz4"
            if state_path.exists():
                logger.info(f"Loading state from {state_path} for diagnostics")
                state = load(state_path)

                diag_dir = summary_path.parent / "diagnostics" / "combine_slices"
                diag_dir.mkdir(parents=True, exist_ok=True)

                hist_out = diag_dir / "combine_histograms.pklz4"
                with lz4.frame.open(hist_out, "wb") as f_out:
                    pickle.dump(extracted["histograms"], f_out)

                pred_var = (
                    jnp.diag(state.pred_cov)
                    if state.pred_cov is not None
                    else jnp.zeros_like(state.pred_mean)
                )
                sig = state.injection_rate * (state.injection_signal or state.signal)
                for channel, ch_hists in extracted["histograms"].items():
                    makePostCombineSlice(
                        pred_mean=state.pred_mean,
                        pred_var=pred_var,
                        test_data=state.test_data,
                        plot_saver=saver,
                        blind_mask=state.blind_mask,
                        signal_data=sig,
                        post_fit_signal=ch_hists["fit_s_total_signal"],
                        post_fit_background=ch_hists["fit_s_total_background"],
                        injected_signal=state.injection_rate,
                        extracted_signal=extracted["tree_fit_sb"]["r"],
                    )
            else:
                logger.warning(
                    f"Analysis state not found at {state_path}, skipping slice plots."
                )
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
    from .diagnostics.plot_utils import plotBinnedData, plotBlinding2D
    from .utils import dictToDot, dotFormat
    import numpy as np
    import json
    import mplhep

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

        fig, ax = plt.subplots(layout="constrained")
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
            "metadata": sig_metadata,
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


@main.command("check-domain")
@click.option(
    "--background",
    "-b",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Background histogram file.",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Pipeline configuration file.",
)
@click.option("--rebin", type=int, default=1, help="Rebin factor.")
@click.option(
    "--min-counts", type=float, default=1.0, help="Min bin count for fit domain."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("domain_check.png"),
    help="Output plot path.",
)
def checkDomain(
    background: Path,
    config: Path | None,
    rebin: int,
    min_counts: float,
    output: Path,
) -> None:
    from .core.serialization import converter
    from .pipeline import PipelineConfig
    from .data.preprocessing import applyDomainMask
    from .steps.load import loadData
    import matplotlib.pyplot as plt
    from .diagnostics.plot_utils import (
        plotBinnedData,
        plotBlinding2D,
        addCMSBits,
        savePlots,
    )
    import yaml
    import numpy as np
    from .data.windowing import WindowConfig

    if config is not None:
        with open(config, "r") as f:
            raw = yaml.safe_load(f)
        raw["background_path"] = str(background)
        raw["rebin"] = rebin
        raw["min_counts"] = min_counts
        pipeline_config = converter.structure(raw, PipelineConfig)
    else:
        pipeline_config = PipelineConfig(
            background_path=background,
            rebin=rebin,
            min_counts=min_counts,
        )

    state = loadData(pipeline_config)
    bkg = state.background

    domain_window = pipeline_config.domain_window
    if isinstance(domain_window, WindowConfig):
        logger.info(f"Building domain window from {type(domain_window).__name__}")
        domain_window = domain_window.buildWindow(bkg)

    _, domain_mask = applyDomainMask(bkg, min_counts=min_counts, window=domain_window)

    fig, ax = plt.subplots(figsize=(10, 8))
    plotBinnedData(ax, bkg, cbar_title="Events")

    if bkg.ndim == 1:
        x_vals = bkg.X.ravel()
        y_vals = bkg.Y.ravel()
        ax.fill_between(
            x_vals,
            0,
            y_vals,
            where=np.asarray(domain_mask),
            alpha=0.3,
            color="red",
            step="mid",
            label=f"Domain Mask (min_counts={min_counts})",
        )
        ax.legend()
    elif bkg.ndim == 2:
        plotBlinding2D(ax, bkg.edges, bkg.X, domain_mask, color="red", linewidth=3)
        ax.set_title(f"Domain Mask (Red Boundary, min_counts={min_counts})")

    if bkg.axis_names:
        ax.set_xlabel(bkg.axis_names[0])
        if bkg.ndim > 1:
            ax.set_ylabel(bkg.axis_names[1])

    addCMSBits(ax, [state.metadata])

    savePlots(
        {output.stem: (fig, ax)},
        output.parent,
        [state.metadata],
        formats=(output.suffix,),
    )
    logger.info(
        f"Domain check plot saved to {output.parent}/{output.stem}{output.suffix}"
    )
