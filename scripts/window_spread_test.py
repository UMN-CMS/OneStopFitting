#!/usr/bin/env python3
"""
Window spread optimization script.

Tests multiple window spread values to find optimal configuration.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import attrs
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from fitting.cli import PipelineConfig
from fitting.pipeline import runPipeline
from fitting.inference.kernels import RationalQuadraticConfig
from fitting.inference.means import ZeroMeanConfig
from fitting.inference.models import ExactGPConfig
from fitting.inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    ObjectiveType,
)
from fitting.inference.likelihoods import FixedGaussianNoiseConfig
from fitting.data.windowing import CutWindow, AndWindow
from fitting.core.serialization import load

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class WindowTestResult:
    """Results from a single window spread test."""

    window_spread: float
    blinded_chi2: float
    global_chi2: float
    pvalue_blinded: float
    pvalue_global: float
    final_loss: float
    training_time: float
    n_train: int
    n_test: int
    n_blinded: int


def createWindowConfigs(
    base_config: PipelineConfig, spreads: list[float]
) -> dict[float, PipelineConfig]:
    """Create configurations for different window spreads."""
    configs = {}

    for spread in spreads:
        configs[spread] = attrs.evolve(
            base_config,
            window_spread=spread,
            output_dir_format=f"{base_config.output_dir_format}/window_spread/{spread}",
        )

    return configs


def runWindowTest(spread: float, config: PipelineConfig) -> WindowTestResult:
    """Run a single window spread test."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing window spread: {spread}")
    logger.info(f"{'=' * 60}")

    start_time = time.time()

    try:
        state = runPipeline(config)

        # Extract metrics
        blinded_chi2 = state.diagnostic_metrics.get(
            "blinded_chi2_per_bin", float("nan")
        )
        global_chi2 = state.diagnostic_metrics.get("global_chi2_per_bin", float("nan"))
        pval_blinded = state.ppc_results["test_stats"]["chi2"]["blinded"]["pvalue"]
        pval_global = state.ppc_results["test_stats"]["chi2"]["all"]["pvalue"]
        final_loss = (
            state.training_result.final_loss
            if state.training_result is not None
            else float("nan")
        )

        # Get bin counts from preprocessing
        n_train = state.train_data.nbins if state.train_data else 0
        n_test = state.test_data.nbins if state.test_data else 0
        n_blinded = jnp.sum(state.blind_mask) if state.blind_mask is not None else 0

        training_time = time.time() - start_time

        result = WindowTestResult(
            window_spread=spread,
            blinded_chi2=blinded_chi2,
            global_chi2=global_chi2,
            pvalue_blinded=pval_blinded,
            pvalue_global=pval_global,
            final_loss=final_loss,
            training_time=training_time,
            n_train=int(n_train),
            n_test=int(n_test),
            n_blinded=int(n_blinded),
        )

        logger.info(f"✓ Completed: window_spread={spread}")
        logger.info(f"  Blinded χ²: {blinded_chi2:.3f}, Global χ²: {global_chi2:.3f}")
        logger.info(
            f"  Train bins: {n_train}, Test bins: {n_test}, Blinded bins: {n_blinded}"
        )
        logger.info(f"  Training time: {training_time:.1f}s")

        return result

    except Exception as e:
        logger.error(f"✗ Failed: window_spread={spread}")
        logger.error(f"  Error: {e}")
        import traceback

        logger.error(f"  Traceback: {traceback.format_exc()}")

        # Load state to get bin counts if available
        n_train, n_test, n_blinded = 0, 0, 0
        try:
            import json
            from pathlib import Path

            summary_path = (
                Path(
                    "testout_window_test/2018/signal_2018_312_1500_600_official/window_spread"
                )
                / f"{spread}"
                / "summary.json"
            )
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                # Bin counts not in summary, default to zeros
                pass
        except Exception:
            pass

        return WindowTestResult(
            window_spread=spread,
            blinded_chi2=float("nan"),
            global_chi2=float("nan"),
            pvalue_blinded=0.0,
            pvalue_global=0.0,
            final_loss=float("nan"),
            training_time=time.time() - start_time,
            n_train=n_train,
            n_test=n_test,
            n_blinded=n_blinded,
        )


def printResults(results: list[WindowTestResult]) -> None:
    """Print formatted results table."""
    table = Table(title="Window Spread Optimization Results")
    table.add_column("Spread", style="cyan")
    table.add_column("Blinded χ²", style="red")
    table.add_column("Global χ²", style="green")
    table.add_column("P(blinded)", style="blue")
    table.add_column("Train/Test/Blind", style="yellow")
    table.add_column("Time (s)", style="magenta")

    # Sort by blinded χ²
    sorted_results = sorted(
        results,
        key=lambda x: abs(x.blinded_chi2 - 1.0)
        if not jnp.isnan(x.blinded_chi2)
        else float("inf"),
    )

    for r in sorted_results:
        # Highlight best blinded χ²
        blinded_style = "red" if r.blinded_chi2 > 1.1 else "green bold"

        table.add_row(
            f"{r.window_spread:.2f}",
            f"[{blinded_style}]{r.blinded_chi2:.3f}[/{blinded_style}]",
            f"{r.global_chi2:.3f}",
            f"{r.pvalue_blinded:.3f}",
            f"{r.n_train}/{r.n_test}/{r.n_blinded}",
            f"{r.training_time:.1f}",
        )

    console.print(table)

    # Find best configuration
    valid_results = [r for r in results if not jnp.isnan(r.blinded_chi2)]
    if valid_results:
        best = min(valid_results, key=lambda x: abs(x.blinded_chi2 - 1.0))
        logger.info(f"\n🏆 Best window spread: {best.window_spread:.2f}")
        logger.info(f"   Blinded χ²: {best.blinded_chi2:.3f} (closest to 1.0)")
        logger.info(
            f"   Train/Test/Blind: {best.n_train}/{best.n_test}/{best.n_blinded}"
        )


def saveResults(results: list[WindowTestResult], output_path: Path) -> None:
    """Save results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    logger.info(f"\nSaved results to {output_path}")


def main() -> None:
    """Main entry point."""
    # Create domain window for uncompressed analysis
    domain_window = AndWindow(
        windows=[
            CutWindow(axis=0, lower=700),
            CutWindow(axis=1, lower=0.1, upper=0.8),
        ]
    )

    # Window spreads to test
    spreads = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

    # Create base config with RationalQuadratic kernel (best performer)
    base_config = PipelineConfig(
        background_path=Path(
            "smoothed/uncomp_qcd_inclusive_2018/uncomp_smoothed.pklz4"
        ),
        signal_path=Path(
            "testexport/2018/Signal312/signal_2018_312_1500_600_official/uncomp_mStop_vs_mChiRatio.pklz4"
        ),
        output_dir_format="testout_window_test/{era.name}/{dataset_name}",
        window_spread=1.75,
        injection_rate=0.0,
        min_counts=1.0,
        rebin=1,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=RationalQuadraticConfig(ard=True),
            mean_function=ZeroMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=150,
        ),
    )

    # Create window spread configs
    configs = createWindowConfigs(base_config, spreads)

    logger.info(f"Testing {len(configs)} window spread values")
    logger.info(f"Spreads: {spreads}")
    logger.info(f"Kernel: RationalQuadratic (best performer from kernel comparison)")
    logger.info(f"Domain window: mStop > 700, mChiRatio in [0.1, 0.8]")
    logger.info(f"Iterations per config: 150")

    # Run tests
    results = []
    for spread, config in configs.items():
        result = runWindowTest(spread, config)
        results.append(result)

    # Print and save results
    printResults(results)
    saveResults(results, Path("testout_window_test/results/window_spread.json"))


if __name__ == "__main__":
    main()
