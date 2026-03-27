#!/usr/bin/env python3
"""
Kernel comparison script for GP background estimation.

Tests multiple kernel/training configurations on a single signal point
to identify optimal setup for background estimation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import attrs
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from fitting.cli import PipelineConfig
from fitting.pipeline import runPipeline
from fitting.inference.kernels import (
    Matern32Config,
    RBFConfig,
    SumKernelConfig,
    RationalQuadraticConfig,
)
from fitting.inference.means import (
    ZeroMeanConfig,
    PolynomialBackgroundMeanConfig,
    InterpolatedMeanConfig,
)
from fitting.inference.models import ExactGPConfig
from fitting.inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    ObjectiveType,
)
from fitting.inference.likelihoods import (
    HeteroscedasticGaussianConfig,
    FixedGaussianNoiseConfig,
)
from fitting.core.data import AnalysisState
from fitting.data.windowing import CutWindow, AndWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class TestResult:
    """Results from a single configuration test."""

    config_name: str
    kernel: str
    mean: str
    objective: str
    mode: str
    blinded_chi2: float
    global_chi2: float
    pvalue_blinded: float
    pvalue_global: float
    final_loss: float
    training_time: float


def createConfigs(
    base_config: PipelineConfig, domain_window: AndWindow, num_iters: int = 150
) -> dict[str, PipelineConfig]:
    """
    Create 8 test configurations.

    1. Matern32 (ARD) + Zero + MLL + Single-stage
    2. Matern32 (ARD) + Zero + LOOCV + Single-stage
    3. Sum(RBF + Matern32) + Zero + MLL + Single-stage
    4. RationalQuadratic (ARD) + Zero + MLL + Single-stage
    5. Matern32 (ARD) + Zero + MLL + Two-stage (mean frozen)
    6. Matern32 (ARD) + PolynomialBackground + MLL + Single-stage
    7. Matern32 (ARD) + InterpolatedMean + MLL + Two-stage (homoscedastic)
    8. Matern32 (ARD) + Heteroscedastic + Zero + MLL + Single-stage
    """

    configs = {}

    # Config 1: Baseline Matern32 + MLL
    configs["matern32_mll"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=ZeroMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_mll",
    )

    # Config 2: Matern32 + LOOCV
    configs["matern32_loocv"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=ZeroMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.LOOCV,
            lr=0.01,
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_loocv",
    )

    # Config 3: Sum Kernel (RBF + Matern32)
    configs["sum_rbf_matern32"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=SumKernelConfig(
                kernels=[RBFConfig(ard=True), Matern32Config(ard=True)]
            ),
            mean_function=ZeroMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/sum_rbf_matern32",
    )

    # Config 4: RationalQuadratic (ARD)
    configs["rational_quadratic"] = attrs.evolve(
        base_config,
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
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/rational_quadratic",
    )

    # Config 5: Matern32 + Two-stage (mean frozen)
    configs["matern32_twostage"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=ZeroMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.TWO_STAGE,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
            two_stage=OptimizationConfig().two_stage,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_twostage",
    )

    # Config 6: Matern32 + PolynomialBackground
    configs["matern32_poly"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=PolynomialBackgroundMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_poly",
    )

    # Config 7: Matern32 + HOMOSCEDASTIC_TWO_STAGE (InterpolatedMean)
    configs["matern32_homoscedastic"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=InterpolatedMeanConfig(),
            likelihood=FixedGaussianNoiseConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.HOMOSCEDASTIC_TWO_STAGE,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
            two_stage=OptimizationConfig().two_stage,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_homoscedastic",
    )

    # Config 8: Matern32 + Heteroscedastic likelihood
    configs["matern32_heteroscedastic"] = attrs.evolve(
        base_config,
        domain_window=domain_window,
        model=ExactGPConfig(
            kernel=Matern32Config(ard=True),
            mean_function=ZeroMeanConfig(),
            likelihood=HeteroscedasticGaussianConfig(),
        ),
        optimization=OptimizationConfig(
            mode=InferenceMode.OPTIMIZATION,
            objective=ObjectiveType.MLL,
            lr=0.01,
            num_iters=num_iters,
        ),
        output_dir_format=f"{base_config.output_dir_format}/kernel_test/matern32_heteroscedastic",
    )

    return configs


def runTest(config_name: str, config: PipelineConfig) -> TestResult:
    """Run a single configuration test."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running config: {config_name}")
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
            state.training_result.final_loss if state.training_result else float("nan")
        )

        training_time = time.time() - start_time

        result = TestResult(
            config_name=config_name,
            kernel=str(type(config.model.kernel).__name__),
            mean=str(type(config.model.mean_function).__name__),
            objective=config.optimization.objective.value,
            mode=config.optimization.mode.value,
            blinded_chi2=blinded_chi2,
            global_chi2=global_chi2,
            pvalue_blinded=pval_blinded,
            pvalue_global=pval_global,
            final_loss=final_loss,
            training_time=training_time,
        )

        logger.info(f"✓ Completed: {config_name}")
        logger.info(f"  Blinded χ²: {blinded_chi2:.3f}, Global χ²: {global_chi2:.3f}")
        logger.info(f"  Training time: {training_time:.1f}s")

        return result

    except Exception as e:
        logger.error(f"✗ Failed: {config_name}")
        logger.error(f"  Error: {e}")
        import traceback

        logger.error(f"  Traceback: {traceback.format_exc()}")

        return TestResult(
            config_name=config_name,
            kernel="error",
            mean="error",
            objective="error",
            mode="error",
            blinded_chi2=float("nan"),
            global_chi2=float("nan"),
            pvalue_blinded=0.0,
            pvalue_global=0.0,
            final_loss=float("nan"),
            training_time=time.time() - start_time,
        )


def printResults(results: list[TestResult]) -> None:
    """Print formatted results table."""
    table = Table(title="Kernel Comparison Results")
    table.add_column("Config", style="cyan")
    table.add_column("Blinded χ²", style="red")
    table.add_column("Global χ²", style="green")
    table.add_column("P(blinded)", style="blue")
    table.add_column("P(global)", style="blue")
    table.add_column("Time (s)", style="yellow")

    for r in results:
        # Highlight best blinded χ²
        blinded_style = "red" if r.blinded_chi2 > 1.1 else "green bold"

        table.add_row(
            r.config_name,
            f"[{blinded_style}]{r.blinded_chi2:.3f}[/{blinded_style}]",
            f"{r.global_chi2:.3f}",
            f"{r.pvalue_blinded:.3f}",
            f"{r.pvalue_global:.3f}",
            f"{r.training_time:.1f}",
        )

    console.print(table)

    # Find best configuration
    valid_results = [r for r in results if not jnp.isnan(r.blinded_chi2)]
    if valid_results:
        best = min(valid_results, key=lambda x: abs(x.blinded_chi2 - 1.0))
        logger.info(f"\n🏆 Best configuration: {best.config_name}")
        logger.info(f"   Blinded χ²: {best.blinded_chi2:.3f} (closest to 1.0)")
        logger.info(f"   Kernel: {best.kernel}")
        logger.info(f"   Mode: {best.mode}")


def saveResults(results: list[TestResult], output_path: Path) -> None:
    """Save results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    logger.info(f"\nSaved results to {output_path}")


def main() -> None:
    """Main entry point."""
    # Create appropriate domain window for uncompressed analysis
    domain_window = AndWindow(
        windows=[
            CutWindow(axis=0, lower=700),
            CutWindow(axis=1, lower=0.1, upper=0.8),
        ]
    )

    # Create minimal base config
    base_config = PipelineConfig(
        background_path=Path(
            "smoothed/uncomp_qcd_inclusive_2018/uncomp_smoothed.pklz4"
        ),
        signal_path=Path(
            "testexport/2018/Signal312/signal_2018_312_1500_600_official/uncomp_mStop_vs_mChiRatio.pklz4"
        ),
        output_dir_format="testout_kernel_test/{era.name}/{dataset_name}",
        window_spread=1.75,
        injection_rate=0.0,
        min_counts=1.0,
        rebin=1,
    )

    # Create test configs with 150 iterations
    configs = createConfigs(base_config, domain_window=domain_window, num_iters=150)

    logger.info(f"Created {len(configs)} test configurations")
    logger.info(f"Domain window: mStop > 700, mChiRatio in [0.1, 0.8]")
    logger.info(f"Iterations per config: 150")

    # Run tests
    results = []
    for config_name, config in configs.items():
        result = runTest(config_name, config)
        results.append(result)

    # Print and save results
    printResults(results)
    saveResults(results, Path("testout_kernel_test/results/kernel_comparison.json"))


if __name__ == "__main__":
    main()
