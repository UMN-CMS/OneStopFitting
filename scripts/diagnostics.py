#!/usr/bin/env python3
"""
Comprehensive diagnostic suite for GP background estimation.

Diagnoses:
1. Pull distribution analysis (spatial patterns, regional statistics)
2. Residual analysis (GP vs simple interpolation)
3. Regional χ² mapping (identify problematic subregions)
4. GP hyperparameter inspection (lengthscales, convergence)
5. Background density visualization
6. Boundary effect analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table

from fitting.core.serialization import load
from fitting.data.preprocessing import fitGaussianWindow
from fitting.data.windowing import CutWindow, AndWindow
from fitting.inference.models import ExactGPConfig
from fitting.inference.kernels import Matern32Config
from fitting.inference.means import ZeroMeanConfig
from fitting.inference.optimization import (
    OptimizationConfig,
    InferenceMode,
    ObjectiveType,
)
from fitting.inference.likelihoods import FixedGaussianNoiseConfig
from fitting.pipeline import runPipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


def computeChi2(
    observed: jnp.ndarray, expected: jnp.ndarray, variance: jnp.ndarray
) -> tuple[float, int]:
    """Compute χ² per bin."""
    residuals = observed - expected
    chi2 = jnp.sum(residuals**2 / variance)
    n_bins = len(observed)
    return float(chi2), int(n_bins)


def computeRegionalChi2(data, pred, domain_mask, blind_mask, n_regions=(3, 3)):
    """
    Compute χ² in sliding windows across the mass plane.

    This identifies where the GP performs poorly.
    """
    logger.info("Computing regional χ² map...")

    # Convert to numpy for easier manipulation
    data_np = np.array(data)
    pred_np = np.array(pred)
    domain_np = np.array(domain_mask)
    blind_np = np.array(blind_mask)

    # Get domain bounds
    valid_mask = domain_np > 0
    valid_indices = np.where(valid_mask)[0]

    mStop_min = np.min(data_np[valid_indices, 0])
    mStop_max = np.max(data_np[valid_indices, 0])
    mChiRatio_min = np.min(data_np[valid_indices, 1])
    mChiRatio_max = np.max(data_np[valid_indices, 1])

    # Create regional bins
    n_mStop, n_mChiRatio = n_regions

    chi2_map = np.zeros((n_mStop, n_mChiRatio))
    count_map = np.zeros((n_mStop, n_mChiRatio))

    for i in range(n_mStop):
        for j in range(n_mChiRatio):
            mStop_low = mStop_min + i * (mStop_max - mStop_min) / n_mStop
            mStop_high = mStop_min + (i + 1) * (mStop_max - mStop_min) / n_mStop

            mChiRatio_low = (
                mChiRatio_min + j * (mChiRatio_max - mChiRatio_min) / n_mChiRatio
            )
            mChiRatio_high = (
                mChiRatio_min + (j + 1) * (mChiRatio_max - mChiRatio_min) / n_mChiRatio
            )

            # Find bins in this region
            in_region = (
                (data_np[:, 0] >= mStop_low)
                & (data_np[:, 0] < mStop_high)
                & (data_np[:, 1] >= mChiRatio_low)
                & (data_np[:, 1] < mChiRatio_high)
                & valid_mask
            )

            if np.any(in_region):
                # Compute local χ² for this region
                region_mask = in_region & domain_np
                region_pred = pred_np[region_mask]
                region_data = data_np[region_mask]
                region_var = region_data  # Assuming Poisson variance

                if np.sum(region_data) > 0:
                    local_chi2, n = computeChi2(
                        jnp.array(region_data),
                        jnp.array(region_pred),
                        jnp.array(region_var),
                    )
                    chi2_map[i, j] = local_chi2
                    count_map[i, j] = n

    return chi2_map, count_map, mStop_min, mStop_max, mChiRatio_min, mChiRatio_max


def analyzePullsFromTrain(X, Y, pred, variance, domain_mask, blind_mask):
    """Analyze pull distributions in different regions for training data."""
    logger.info("Analyzing pulls for training data...")

    X_np = np.array(X)
    Y_np = np.array(Y)
    pred_np = np.array(pred)
    var_np = np.array(variance)
    domain_np = np.array(domain_mask)
    blind_np = np.array(blind_mask)

    valid_mask = domain_np > 0

    # Compute pulls
    with np.errstate(divide="ignore", invalid="ignore"):
        pulls = (Y_np - pred_np) / np.sqrt(var_np)
        pulls = np.nan_to_num(pulls, nan=0.0, posinf=0.0, neginf=0.0)

    overall_mean = float(np.mean(pulls[valid_mask]))
    overall_std = float(np.std(pulls[valid_mask]))
    overall_median = float(np.median(pulls[valid_mask]))

    # Regional analysis
    mStop_median = np.median(X_np[valid_mask, 0])
    mChiRatio_median = np.median(X_np[valid_mask, 1])

    regions = {
        "signal": blind_np & valid_mask,
        "low_mStop": ~blind_np & valid_mask & (X_np[:, 0] < mStop_median),
        "high_mStop": ~blind_np & valid_mask & (X_np[:, 0] >= mStop_median),
        "low_mChiRatio": ~blind_np & valid_mask & (X_np[:, 1] < mChiRatio_median),
        "high_mChiRatio": ~blind_np & valid_mask & (X_np[:, 1] >= mChiRatio_median),
        "sideband": ~blind_np & valid_mask,
    }

    regional_stats = {}
    for region_name, region_mask in regions.items():
        region_pulls = pulls[region_mask]
        if np.sum(region_mask) > 0:
            regional_stats[region_name] = {
                "mean": float(np.mean(region_pulls)),
                "std": float(np.std(region_pulls)),
                "median": float(np.median(region_pulls)),
                "n_bins": int(np.sum(region_mask)),
            }

    return overall_mean, overall_std, overall_median, regional_stats


def analyzeHyperparameters(posterior, state):
    """Analyze learned GP hyperparameters."""
    logger.info("Analyzing GP hyperparameters...")

    # Check for convergence by examining loss history
    loss_history = state.training_result.loss_history if state.training_result else []
    converged = (
        len(loss_history) > 0
        and abs(loss_history[-1] - loss_history[-min(10, len(loss_history))]) < 0.1
    )

    # Basic kernel info
    kernel = posterior.prior.kernel
    kernel_name = type(kernel).__name__

    # Try to get lengthscales (depends on kernel type)
    try:
        if hasattr(kernel, "lengthscale"):
            ls = kernel.lengthscale
            if hasattr(ls, "value"):
                ls_val = np.array(ls.value)
                lengthscale_info = {
                    "shape": list(ls_val.shape),
                    "values": [float(x) for x in ls_val.ravel()],
                }
            else:
                lengthscale_info = {"note": "No .value attribute"}
        else:
            lengthscale_info = {"note": "No lengthscale attribute"}
    except Exception as e:
        lengthscale_info = {"error": str(e)}

    try:
        if hasattr(kernel, "variance"):
            var = kernel.variance
            if hasattr(var, "value"):
                var_val = float(np.array(var.value).ravel()[0])
                variance_info = {"value": var_val}
            else:
                variance_info = {"note": "No .value attribute"}
        else:
            variance_info = {"note": "No variance attribute"}
    except Exception as e:
        variance_info = {"error": str(e)}

    return {
        "kernel_type": kernel_name,
        "lengthscale": lengthscale_info,
        "variance": variance_info,
        "converged": converged,
        "final_loss": state.training_result.final_loss
        if state.training_result
        else None,
        "n_iterations": len(loss_history),
    }
    if hasattr(kernel, "variance"):
        var = kernel.variance
        if hasattr(var, "value"):
            params["variance"] = float(var.value)

    # Check for convergence by examining loss history
    loss_history = state.training_result.loss_history if state.training_result else []
    converged = (
        len(loss_history) > 0
        and abs(loss_history[-1] - loss_history[-min(10, len(loss_history))]) < 0.1
    )

    return {
        "kernel_params": params,
        "converged": converged,
        "final_loss": state.training_result.final_loss
        if state.training_result
        else None,
        "n_iterations": len(loss_history),
    }


def analyzeResiduals(data, pred, variance, domain_mask):
    """Compare GP to simple interpolation methods."""
    logger.info("Analyzing residuals compared to simple interpolation...")

    data_np = np.array(data)
    pred_np = np.array(pred)
    var_np = np.array(variance)

    valid_mask = domain_np > 0

    # Simple 2D linear interpolation
    from scipy.interpolate import LinearNDInterpolator

    valid_coords = data_np[valid_mask]
    if len(valid_coords) > 10:
        interp = LinearNDInterpolator(valid_coords, valid_coords[:, 1])
        interp_pred = interp(data_np)
        interp_residuals = data_np - interp_pred
    else:
        interp_pred = np.zeros_like(pred_np)
        interp_residuals = data_np - pred_np

    # GP residuals
    gp_residuals = data_np - pred_np

    # Compare in different regions
    residuals_comparison = {}
    regions = ["all_valid", "signal", "sideband"]

    mStop_median = np.median(data_np[valid_mask, 0])
    mChiRatio_median = np.median(data_np[valid_mask, 1])

    region_masks = {
        "all_valid": valid_mask,
        "signal": valid_mask
        & (np.abs(data_np[:, 0] - mStop_median) < 0.3)
        & (np.abs(data_np[:, 1] - mChiRatio_median) < 0.3),
        "sideband": valid_mask & ~valid_mask,
    }

    for region_name, region_mask in region_masks.items():
        if np.sum(region_mask) > 0:
            gp_res = gp_residuals[region_mask]
            interp_res = interp_residuals[region_mask]

            # Compute RMS
            gp_rms = float(np.sqrt(np.mean(gp_res**2 / var_np[region_mask])))
            interp_rms = float(np.sqrt(np.mean(interp_res**2 / var_np[region_mask])))

            residuals_comparison[region_name] = {
                "gp_rms": gp_rms,
                "interp_rms": interp_rms,
                "improvement": interp_rms / gp_rms
                if gp_rms > 0
                else gp_rms / interp_rms,
                "n_bins": int(np.sum(region_mask)),
            }

    return residuals_comparison


def runDiagnostics(state_path: Path, output_dir: Path) -> dict:
    """Run all diagnostic analyses and return results."""
    logger.info(f"Loading state from {state_path}")
    state = load(state_path)

    logger.info("Analyzing GP results...")

    # Get training data
    train_Y = np.array(state.train_data.Y).ravel()
    train_V = np.array(state.train_data.V).ravel()
    train_X = np.array(state.train_data.X)
    train_domain_mask = np.array(state.domain_mask).ravel()[: len(train_Y)]
    train_blind_mask = np.array(state.blind_mask).ravel()[: len(train_Y)]

    # Get predictions for training positions
    pred_train = np.array(state.pred_mean).ravel()[: len(train_Y)]

    logger.info(f"Training data: {len(train_Y)} bins")
    logger.info(f"Domain bins: {np.sum(train_domain_mask > 0)}")
    logger.info(f"Blind bins: {np.sum(train_blind_mask)}")

    results = {}

    # 1. Pull analysis on training data
    overall_mean, overall_std, overall_median, regional_stats = analyzePullsFromTrain(
        train_X, train_Y, pred_train, train_V, train_domain_mask, train_blind_mask
    )
    results["pulls"] = {
        "overall_mean": overall_mean,
        "overall_std": overall_std,
        "overall_median": overall_median,
        "regional": regional_stats,
    }

    # 2. Hyperparameter analysis
    posterior = state.training_result.posterior
    results["hyperparameters"] = analyzeHyperparameters(posterior, state)

    # 3. Overall metrics
    valid_mask = train_domain_mask > 0
    blinded_mask = train_blind_mask & train_domain_mask

    valid_residuals = train_Y[valid_mask] - pred_train[valid_mask]
    valid_chi2 = np.sum((valid_residuals**2) / train_V[valid_mask])
    n_valid = np.sum(valid_mask)

    blinded_residuals = train_Y[blinded_mask] - pred_train[blinded_mask]
    blinded_chi2 = np.sum((blinded_residuals**2) / train_V[blinded_mask])
    n_blinded = np.sum(blinded_mask)

    results["overall_metrics"] = {
        "valid_chi2_per_bin": float(valid_chi2 / n_valid),
        "blinded_chi2_per_bin": float(blinded_chi2 / n_blinded),
        "n_valid_bins": int(n_valid),
        "n_blinded_bins": int(n_blinded),
    }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "diagnostic_results.json", "w") as f:
        json.dump(
            {
                "pulls": results["pulls"],
                "hyperparameters": results["hyperparameters"],
                "overall_metrics": results["overall_metrics"],
            },
            f,
            indent=2,
        )

    logger.info(f"Saved diagnostic results to {output_dir / 'diagnostic_results.json'}")

    return results


def printDiagnosticSummary(results: dict):
    """Print formatted summary of diagnostic results."""
    console.rule("[bold blue]Diagnostic Summary[/bold blue]")

    # Overall metrics
    console.rule("[bold green]Overall Metrics[/bold green]")
    table = Table(title="Overall Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")

    om = results["overall_metrics"]
    table.add_row("Valid χ²/bin", f"{om['valid_chi2_per_bin']:.3f}")
    table.add_row("Blinded χ²/bin", f"{om['blinded_chi2_per_bin']:.3f}")
    table.add_row("Valid bins", f"{om['n_valid_bins']}")
    table.add_row("Blinded bins", f"{om['n_blinded_bins']}")
    console.print(table)

    # Pull analysis
    console.rule("[bold green]Pull Analysis[/bold green]")
    table = Table(title="Pull Distribution (data - pred) / sqrt(data)")
    table.add_column("Region", style="cyan")
    table.add_column("Mean", style="yellow")
    table.add_column("Std", style="yellow")
    table.add_column("Median", style="yellow")
    table.add_column("Bins", style="yellow")

    pulls = results["pulls"]
    table.add_row(
        "Overall",
        f"{pulls['overall_mean']:.3f}",
        f"{pulls['overall_std']:.3f}",
        f"{pulls['overall_median']:.3f}",
        "-",
    )

    for region_name in [
        "low_mStop",
        "high_mStop",
        "low_mChiRatio",
        "high_mChiRatio",
        "sideband",
    ]:
        if region_name in pulls["regional"]:
            r = pulls["regional"][region_name]
            table.add_row(
                region_name,
                f"{r['mean']:.3f}",
                f"{r['std']:.3f}",
                f"{r['median']:.3f}",
                f"{r['n_bins']}",
            )

    console.print(table)

    # Residual analysis
    console.rule(
        "[bold green]Residual Analysis (GP vs Linear Interpolation)[/bold green]"
    )
    table = Table(title="Residual RMS Comparison")
    table.add_column("Region", style="cyan")
    table.add_column("GP RMS", style="yellow")
    table.add_column("Interp RMS", style="yellow")
    table.add_column("Ratio", style="green")
    table.add_column("Bins", style="magenta")

    res = results["residuals"]
    for region_name in ["all_valid", "signal", "sideband"]:
        if region_name in res:
            r = res[region_name]
            table.add_row(
                region_name,
                f"{r['gp_rms']:.3f}",
                f"{r['interp_rms']:.3f}",
                f"{r['improvement']:.2f}",
                f"{r['n_bins']}",
            )

    console.print(table)

    # Hyperparameters
    console.rule("[bold green]Hyperparameters[/bold green]")
    hp = results["hyperparameters"]
    console.print(f"Converged: {hp['converged']}")
    console.print(f"Final Loss: {hp['final_loss']:.2f}")
    console.print(f"Iterations: {hp['n_iterations']}")

    if "kernel_params" in hp:
        kp = hp["kernel_params"]
        console.print("Kernel Parameters:")
        for key, val in kp.items():
            console.print(f"  {key}: {val}")

    console.print(f"\nFull results saved to: diagnostic_results.json")


def main():
    """Main entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diagnostics.py <state_path> <output_dir>")
        sys.exit(1)

    state_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    logger.info(f"Running diagnostics on {state_path}")
    logger.info(f"Output directory: {output_dir}")

    results = runDiagnostics(state_path, output_dir)
    printDiagnosticSummary(results)


if __name__ == "__main__":
    main()
