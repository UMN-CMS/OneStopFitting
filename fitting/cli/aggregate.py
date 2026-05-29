from __future__ import annotations
from pathlib import Path
import click
from .base import logger, CommaSeparatedFloat


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
    from ..diagnostics.aggregate_plots import extractPoints, makeMulti

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
    from ..utils import dictToDot, dotFormat

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
    from ..diagnostics.plot_utils import savePlots

    savePlots(
        plots,
        output,
        [x.metadata for x in points_group],
        formats=formats,
        extra_text=cms_extra,
    )


@click.group("aggregate")
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
def aggregateGroup(
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

    from ..diagnostics.aggregate_plots import iterSummaryFiles, readSummary

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


@aggregateGroup.command("mass-plane")
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
    from ..diagnostics.aggregate_plots import makeAggregateMassPlanePlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)
        get_val = lambda x: x.stats.get(value_func, x.stats.get("median"))

        plots = makeAggregateMassPlanePlot(
            p,
            metric_name=fmt(metric_name_str),
            get_value_func=get_val,
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


@aggregateGroup.command("smooth")
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
    from ..diagnostics.aggregate_plots import makeAggregateSmoothPlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "{plot_type}_{metric_name}"

    for k, p in points.items():
        fmt, fmt_ctx = _makeFmt(p, k)
        get_val = lambda x: x.stats.get(value_func, x.stats.get("median"))

        plots = makeAggregateSmoothPlot(
            p,
            metric_name=fmt(metric_name_str),
            get_value_func=get_val,
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


@aggregateGroup.command("scatter")
@_aggregateCmdOptions
@click.option("--xlim", type=CommaSeparatedFloat(), default=None)
@click.option("--vlines", type=CommaSeparatedFloat(), default=None)
@click.pass_context
def scatterCmd(ctx, metric_dotpath, transform, merge, name_format, xlim, vlines):
    from ..diagnostics.aggregate_plots import makeAggregateScatterPlot

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


@aggregateGroup.command("violin")
@_aggregateCmdOptions
@click.pass_context
def violinCmd(ctx, metric_dotpath, transform, merge, name_format):
    from ..diagnostics.aggregate_plots import makeAggregateViolinPlot

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


@aggregateGroup.command("injection-line")
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
    from ..diagnostics.aggregate_plots import makeInjectionLinePlot

    points, _, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = ctx.obj["output"]
    fmt_name = name_format or "injection_line_{dataset_name}"

    for k, p in points.items():
        fmt, _ = _makeFmt(p, k)

        plots = makeInjectionLinePlot(
            p,
            title=fmt(ctx.obj["title"]),
            name_format=fmt_name,
            error_type=error_type,
            ylim=tuple(ylim) if ylim else None,
        )
        _saveGroupPlots(plots, output, p, ctx.obj["formats"], ctx.obj["cms_extra"])

    logger.info(f"Injection line plots saved to {output}")


@aggregateGroup.command("save-data")
@_aggregateCmdOptions
@click.pass_context
def saveDataCmd(ctx, metric_dotpath, transform, merge, name_format):
    import json
    import cattrs
    from ..utils import dotFormat

    points, all_points, metric_name_str, _ = _extractAndMerge(
        ctx, metric_dotpath, transform, merge
    )
    output = Path(ctx.obj["output"])
    output.mkdir(exist_ok=True, parents=True)
    fmt_name = name_format or "{metric_name}"

    for k, d in points.items():
        name = dotFormat(fmt_name, metric_name="__".join(metric_dotpath), **dict(k))
        name = name.replace(".", "p")
        path = (output / name).with_suffix(".json")
        logger.info(f"Saving data to {path}")
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(cattrs.unstructure(d), f, indent=2)

    name = dotFormat("ALL_" + "__".join(metric_dotpath))
    name = name.replace(".", "p")
    path = (output / name).with_suffix(".json")
    path.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving data to {path}")
    with open(path, "w") as f:
        json.dump(cattrs.unstructure(all_points), f, indent=2)


@aggregateGroup.command("diagnose")
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
    from ..diagnostics.diagnostic_report import generateDiagnosticReport
    from ..utils import dictToDot
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
            key = tuple((x, dotted[x]) for x in group_by)
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
