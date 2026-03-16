"""CLI entry point for fitting.

Usage:
    fitting run --config config.json
    fitting run --background data/bg.pkl.lz4 [options]
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

from .core.serialization import converter
from .pipeline import PipelineConfig, runPipeline
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
    "--min-counts", type=float, default=None, help="Min bin count for fit domain."
)
@click.option("--num-iters", type=int, default=500, help="Training iterations.")
@click.option("--lr", type=float, default=0.01, help="Learning rate.")
def run(
    config: Path | None,
    background: Path | None,
    signal: Path | None,
    signal_name: str | None,
    injection_rate: float,
    output: Path,
    rebin: int,
    min_counts: float,
    num_iters: int,
    lr: float,
) -> None:
    """Run the background estimation pipeline."""
    if config is not None:
        with open(config, "r") as f:
            raw = yaml.safe_load(f)
        pipeline_config = converter.structure(raw, PipelineConfig)
    elif background is not None:
        from .inference.optimization import OptimizationConfig

        pipeline_config = PipelineConfig(
            background_path=background,
            signal_path=signal,
            signal_name=signal_name,
            injection_rate=injection_rate,
            output_dir_format=str(output),
            rebin=rebin,
            min_counts=min_counts,
            optimization=OptimizationConfig(
                num_iters=num_iters,
                lr=lr,
            ),
        )
    else:
        raise click.UsageError("Must specify either --config or --background.")

    runPipeline(pipeline_config)


if __name__ == "__main__":
    main()
