"""CLI entry point for fitting.

Usage:
    fitting run --config config.json
    fitting run --background data/bg.pkl.lz4 [options]
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .core.serialization import converter, load
from .pipeline import (
    PipelineConfig,
    runPipeline,
    PipelineStep,
    generateSmoothedBackground,
)
from .inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    OptimizerType,
    ObjectiveType,
    MCMCConfig,
    TwoStageConfig,
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
@click.option("--signal-name", type=str, default=None, help="Signal name/key.")
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
    "--min-counts", type=float, default=10.0, help="Min bin count for fit domain."
)
@click.option("--num-iters", type=int, default=500, help="Training iterations.")
@click.option("--lr", type=float, default=0.01, help="Learning rate.")
@click.option("--seed", type=int, default=0xBEEFBEEF, help="RNG seed.")
@click.option("--window-spread", type=float, default=2.0, help="Window spread.")
@click.option(
    "--mode",
    type=click.Choice(InferenceMode, case_sensitive=False),
    default=InferenceMode.OPTIMIZATION,
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
def run(
    config: Path | None,
    background: Path | None,
    signal: Path | None,
    signal_name: str | None,
    injection_rate: float,
    output: str,
    rebin: int,
    min_counts: float,
    num_iters: int,
    lr: float,
    seed: int,
    window_spread: float,
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
) -> None:
    """Run the background estimation pipeline."""
    # Resolve steps
    start_from_step = start_from or PipelineStep.LOAD

    if config is not None:
        with open(config, "r") as f:
            raw = yaml.safe_load(f)
        pipeline_config = converter.structure(raw, PipelineConfig)
    elif background is not None:
        pipeline_config = PipelineConfig(
            background_path=background,
            signal_path=signal,
            signal_name=signal_name,
            injection_rate=injection_rate,
            output_dir_format=output,
            rebin=rebin,
            min_counts=min_counts,
            rng_seed=seed,
            window_spread=window_spread,
            optimization=OptimizationConfig(
                mode=mode,
                lr=lr,
                num_iters=num_iters,
            ),
        )
    else:
        raise click.UsageError("Must specify either --config or --background.")

    runPipeline(pipeline_config, start_from=start_from_step, single_step=step)


@main.command()
@click.option(
    "--state",
    "-s",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to saved AnalysisState.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for the smoothed histogram (compressed pickle).",
)
@click.option("--seed", type=int, default=42, help="RNG seed.")
@click.option("--num-samples", type=int, default=1, help="Number of samples to draw.")
def smooth(state: Path, output: Path, seed: int, num_samples: int) -> None:
    import jax
    import lz4.frame
    import pickle
    from .diagnostics.plot_utils import savePlots

    jax.config.update("jax_enable_x64", True)
    rng_key = jax.random.key(seed)

    logger.info(f"Loading state from {state}")
    analysis_state = load(state)

    logger.info("Generating smoothed background...")
    hists, plots = generateSmoothedBackground(
        analysis_state, rng_key, num_samples=num_samples
    )

    plot_dir = output.parent / "smoothing_diagnostics"
    savePlots(plots, plot_dir)
    logger.info(f"Smoothing diagnostic plots saved to {plot_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with lz4.frame.open(output, "wb") as f:
        if num_samples == 1:
            pickle.dump(hists[0], f)
        else:
            pickle.dump(hists, f)

    logger.info(f"Smoothed background saved to {output}")


if __name__ == "__main__":
    main()
