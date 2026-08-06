from __future__ import annotations
from pathlib import Path
import click
from .base import _parseWindowParams, logger
from ..inference.optimization import (
    InferenceMode,
    OptimizerType,
    ObjectiveType,
)
from ..pipeline import PipelineStep



@click.command()
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
        # "impacts",
    ],
    help="Combine commands to run (e.g., 'fit', 'limits', 'significance', or full custom commands).",
)
@click.option(
    "--combine-container",
    type=str,
    help="Combine container image path.",
)
def runCmd(
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
    from ..core.serialization import converter
    from ..pipeline import (
        CombineConfig,
        PipelineConfig,
        runPipeline,
        PipelineStep,
        OptimizationConfig,
    )
    import yaml

    start_from_step = start_from or PipelineStep.LOAD

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
                or "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:CMSSW_14_1_0_pre4-combine_v10.6.1-harvester_v3.1.0",
            ),
            signal_pre_scale=signal_pre_scale,
        )
    else:
        raise click.UsageError("Must specify either --config or --background.")

    runPipeline(pipeline_config, start_from=start_from_step, single_step=step)

@click.command("resolve-output")
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
def resolveOutputCmd(
    background: Path,
    signal: tuple[Path, ...],
    config: Path,
    output_format: str,
    injection_signal: Path | None,
) -> None:
    from ..pipeline import PipelineConfig, loadData
    import yaml
    import attrs

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
