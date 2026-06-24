from __future__ import annotations
import copy
from pathlib import Path
import click
from fitting.cli.base import logger, _parseWindowParams


@click.command("print-params")
@click.argument("input", type=str)
def printParamsCmd(input):
    from ..diagnostics.parameters import (
        logKernelParameters,
        meanParameters,
    )
    from ..core.serialization import load

    data = load(input)
    if data.training_result is None:
        raise RuntimeError("Can only print parameters on a state that has been trained")

    posterior = data.training_result.posterior
    logKernelParameters(posterior)
    meanParameters(posterior)


@click.command("smooth")
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
@click.option(
    "--scale-to",
    type=click.Path(exists=True, path_type=Path),
    required=False,
    default=None,
    help="Path to histogram whose yield will be matched",
)
def smoothCmd(
    state: Path,
    output_dir: Path,
    name: str,
    seed: int,
    include_smooth: bool,
    num_samples: int,
        scale_to: Path,
) -> None:
    import lz4.frame
    import pickle
    import jax
    from ..core.serialization import load
    from ..steps.generators import generateSmoothedBackground, generateAsimovSmoothed
    from ..diagnostics.plot_utils import getPlotSaver
    from ..data.loading import (
        FileLoader,
        extractHistogram,
        extractMetadata,
        histToBinnedData,
        sliceVariation,
    )

    jax.config.update("jax_enable_x64", True)
    rng_key = jax.random.key(seed)

    logger.info(f"Loading state from {state}")
    analysis_state = load(state)

    plot_dir = output_dir / "smoothing_diagnostics"

    plot_saver = getPlotSaver(
        plot_dir, [analysis_state.metadata], formats=analysis_state.config.image_formats
    )
    
    if scale_to:
        bkg_raw = FileLoader.forPath(scale_to).load(scale_to)
        bkg_hist = extractHistogram(bkg_raw)
        bkg_hist = histToBinnedData(
            bkg_hist,
            variation="central",
        )
        scale_to = bkg_hist.Y.sum()

    logger.info("Generating smoothed background...")
    hists = generateSmoothedBackground(
        analysis_state, rng_key, plot_saver=plot_saver, num_samples=num_samples, scale_to=scale_to
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Smoothing diagnostic plots saved to {plot_dir}")

    for idx, hist in enumerate(hists):
        out_path = output_dir / f"{name}_{idx}.pklz4"
        metadata = copy.deepcopy(analysis_state.background_metadata)
        metadata["toy_index"] = idx
        with lz4.frame.open(out_path, "wb") as f:
            to_save = {
                "item": hist,
                "metadata": metadata,
            }
            pickle.dump(to_save, f)

    if include_smooth:
        pure_plot_dir = output_dir / "pure_smooth_diagnostics"
        plot_saver = getPlotSaver(
            pure_plot_dir,
            [analysis_state.metadata],
            formats=analysis_state.config.image_formats,
        )
        pure_plot_dir.mkdir(exist_ok=True, parents=True)
        hist_asimov = generateAsimovSmoothed(
            analysis_state, rng_key, plot_saver=plot_saver
        )
        out_path = output_dir / "pure_smoothed.pklz4"
        metadata = copy.deepcopy(analysis_state.background_metadata)
        with lz4.frame.open(out_path, "wb") as f:
            to_save = {
                "item": hist_asimov,
                "metadata": metadata,
            }
            pickle.dump(to_save, f)

    logger.info(f"Smoothed background saved to {output_dir}")


@click.command("gather")
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
def gatherCmd(inputs: tuple[str, ...], output: Path):
    import json
    from ..diagnostics.aggregate_plots import (
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


@click.command("report")
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
def reportCmd(
    inputs: tuple[str, ...],
    output: Path | None,
    single_document: bool,
    latex_engine: str,
    keep_build: bool,
    keep_tex: bool,
    image_format: str,
) -> None:
    from ..diagnostics.point_report import (
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


@click.command("harvest")
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
def harvestCmd(summaries: tuple[Path, ...], diagnose: bool) -> None:
    from ..combine.extract import extractCombineResults
    import json
    from ..diagnostics.plot_utils import getPlotSaver
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
            from ..core.serialization import load
            from ..diagnostics.plots_2d_slices import makePostCombineSlice
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


@click.command("window-fit")
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
def windowFitCmd(
    input: tuple[str, ...],
    window_type: str,
    window_params: tuple[str, ...],
    rebin: int,
    output_format: str,
):
    from ..utils import getRecoCategory
    from ..data.loading import (
        FileLoader,
        extractHistogram,
        extractMetadata,
        histToBinnedData,
    )
    import matplotlib.pyplot as plt
    from ..diagnostics.plot_utils import plotBinnedData, plotBlinding2D
    from ..utils import dictToDot, dotFormat
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


@click.command("check-domain")
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
def checkDomainCmd(
    background: Path,
    config: Path | None,
    rebin: int,
    min_counts: float,
    output: Path,
) -> None:
    from ..core.serialization import converter
    from ..pipeline import PipelineConfig
    from ..data.preprocessing import applyDomainMask
    from ..steps.load import loadData
    import matplotlib.pyplot as plt
    from ..diagnostics.plot_utils import (
        plotBinnedData,
        plotBlinding2D,
        addCMSBits,
        savePlots,
    )
    import yaml
    import numpy as np
    from ..data.windowing import WindowConfig

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


@click.command("merge-summaries")
@click.argument(
    "inputs",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output summary JSON file.",
)
@click.option(
    "--era-name",
    type=str,
    default="combined",
    help="New era name for the combined summary.",
)
def mergeSummariesCmd(inputs: tuple[Path, ...], output: Path, era_name: str) -> None:
    """Merge GPR summary JSON files from multiple eras."""
    import json

    if not inputs:
        raise click.UsageError("At least one input summary JSON must be provided.")

    cleaned_era_name = era_name.replace("@", "+")
    logger.info(f"Merging {len(inputs)} GPR summaries into {output} with era name {cleaned_era_name}")

    summaries = []
    for path in inputs:
        with open(path, "r") as f:
            summaries.append(json.load(f))

    merged = copy.deepcopy(summaries[0])

    total_lumi = 0.0
    energies = set()
    for s in summaries:
        metadata = s.get("metadata", {})
        era_info = metadata.get("era", {})
        
        lumi = era_info.get("lumi")
        if lumi is not None:
            try:
                total_lumi += float(lumi)
            except ValueError:
                pass
                
        energy = era_info.get("energy")
        if energy is not None:
            energies.add(energy)

    if "metadata" not in merged:
        merged["metadata"] = {}
    if "era" not in merged["metadata"]:
        merged["metadata"]["era"] = {}

    merged["metadata"]["era"]["name"] = cleaned_era_name
    merged["metadata"]["era"]["lumi"] = total_lumi
    
    if len(energies) == 1:
        merged["metadata"]["era"]["energy"] = list(energies)[0]
    elif len(energies) > 1:
        sorted_energies = sorted(list(energies), key=lambda x: str(x))
        merged["metadata"]["era"]["energy"] = "/".join(str(e) for e in sorted_energies)

    # Sum blind_mask_size if it exists in inputs
    blind_mask_sizes = [s.get("blind_mask_size") for s in summaries if "blind_mask_size" in s]
    if blind_mask_sizes:
        merged["blind_mask_size"] = sum(blind_mask_sizes)
    else:
        merged.pop("blind_mask_size", None)

    # Remove single-era specific metrics/training results
    merged.pop("training", None)
    merged.pop("metrics", None)
    merged.pop("ppc", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(merged, f, indent=2)

    logger.info(f"Successfully wrote merged summary to {output}")

