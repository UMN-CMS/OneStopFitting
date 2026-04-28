# OneStopFitting

OneStopFitting is a software package for Gaussian Process (GP) background estimation. It processes one and two-dimensional histograms to generate background estimates and uncertainties for use in the Combine statistical framework.

## Project Structure

The codebase is organized into modules under the `fitting/` directory:

- `core/`: Utilities and state serialization.
- `combine/`: Interface for the Combine framework.
- `data/`: Data handling, rebinning, and masking.
- `diagnostics/`: Plotting and report generation.
- `distributed/`: Parameter sweep management and HTCondor support.
- `inference/`: Models, optimization, and MCMC using JAX and GPJax.
- `steps/`: Pipeline execution steps.

## Installation

The project uses `uv` for dependency management and supports Apptainer/Singularity containers.

To install:

```bash
uv venv
uv sync
```

The `setup.sh` script initializes the environment on clusters with CVMFS access. Python 3.11+ is required.

## Command Line Interface (`fitting.cli`)

The entry point is `python -m fitting`.

### 1. `run`
Executes the fitting pipeline.

**Arguments:**
- `--config`, `-c`: Path to the YAML configuration file.
- `--background`, `-b`: Path to the background histogram (`.pklz4`).
- `--signal`, `-s`: Path to the signal histogram (`.pklz4`).
- `--output`, `-o`: Output directory.
- `--injection-rate`: Signal injection rate multiplier.
- `--start-from`: Pipeline phase to resume from (`LOAD`, `FIT`, `COMBINE`).

```bash
python -m fitting run \
    --config resources/smoothing_configs/Signal312/comp.yaml \
    --background subsetexported/2018/Signal312/qcd_inclusive_2018/comp_mStop_vs_mChiRatio.pklz4 \
    --injection-rate 0.1 \
    --output output/2018/qcd_inclusive/comp
```

### 2. `smooth`
Generates sampled backgrounds from the latent distribution.

**Arguments:**
- `--state`, `-s`: Path to the `state.pklz4` file.
- `--output-dir`, `-o`: Output directory for generated frames.
- `--name`, `-n`: Prefix for naming extractions.
- `--num-samples`: Number of posterior draws.
- `--include-smooth`: Output the GPR mean.

```bash
python -m fitting smooth \
    --state output/2018/qcd_inclusive/comp/state.pklz4 \
    --name qcd_smoothed_category \
    --output-dir smoothed_outputs/ \
    --num-samples 10
```

### 3. `aggregate-plot`
Aggregates metrics from `summary.json` files and generates summary plots.

**Arguments:**
- `--metric`, `-m`: Path to the metric in `summary.json`.
- `--output`, `-o`: Visualization directory.
- `--formats`, `-f`: Image formats (e.g., `png`, `pdf`).
- `--smooth-sigma`: Gaussian filter sigma for outputs.
- `--cmap`: Matplotlib colormap.

```bash
python -m fitting aggregate-plot \
    "output/2018/qcd_inclusive/comp/**/summary.json" \
    --metric metrics.blinded_chi2_per_bin \
    --output diagnostic_plots/ \
    --formats png \
    --formats pdf \
    --smooth-sigma 1.5
```

### 4. `report`
Generates PDF reports from `summary.json` files.

**Arguments:**
- `--input`, `-i`: Input paths or glob strings for `summary.json`.
- `--output`, `-o`: Output directory.
- `--single-document`: Combines outputs into one document.
- `--latex-engine`: LaTeX engine (`pdflatex`, `xelatex`).

```bash
python -m fitting report \
    --input "output/2018/qcd_inclusive/comp/**/summary.json" \
    --output report_output/ \
    --single-document \
    --latex-engine pdflatex
```

### 5. `makecondor` and `makebatch`
Generate submission files for cluster execution.

**Arguments (`makecondor`):**
- `--signal`: Glob pattern for signal templates.
- `--background`: Glob pattern for background templates.
- `--years`: Data years.
- `--subdir-format`: Output directory format string.
- `--output`: Output directory for submission files.

```bash
python -m fitting makecondor \
    --signal "subsetexported/2018/Signal312/**/signal_*.pklz4" \
    --background "subsetexported/2018/Signal312/**/qcd_inclusive*.pklz4" \
    --years 2018 \
    --pipelines smoothing \
    --subdir-format "{era.name}/{dataset_name}" \
    --output condor_submit_files
```

**Arguments (`makebatch`):**
- Features the same arguments as `makecondor` with parameter sweep support.
- `--rates`: Injection rates to sweep.
- `--rebin`: Rebin factors to sweep.
- `--window-spread`: Window spread values.
- `--config-base`: Base configuration template.

```bash
python -m fitting makebatch \
    --signal "subsetexported/2018/Signal312/**/signal_*.pklz4" \
    --background "subsetexported/2018/Signal312/**/qcd_inclusive*.pklz4" \
    --years 2018 \
    --pipelines smoothing \
    --config-base resources/smoothing_configs/Signal312/comp.yaml \
    --rates "0.0,0.1,0.5" \
    --rebin "1,2" \
    --output batch_submit_files
```

### 6. `harvest`
Extracts Combine results and updates `summary.json` files.

**Arguments:**
- `<summaries>`: Paths to `summary.json` files.

```bash
python -m fitting harvest output/2018/qcd_inclusive/comp/**/summary.json
```

## Modeling and Inference

Models and inference methods are implemented in `fitting.inference`.

### 1. GP Models
- `ExactGPConfig`: Full-batch Gaussian Process fit.
- `SparseGPConfig`: Variational inducing points for large datasets.
- `VariationalGPConfig`: Stochastic variational inference.
- `MultiFidelityGPConfig`: Two-stage autoregressive framework using a low-fidelity representation and a high-fidelity surrogate.
- `QCDPriorGPConfig`: Bayesian workflow using hyperpriors from simulation.

### 2. Kernels
Kernels include options from GPJax and physics-specific extensions.
- **Standard**: `RBF`, `Matern12` / `Matern32` / `Matern52`, `RationalQuadratic`, `Polynomial`, `Periodic`, `Linear`, `White`.
- **Composites**: `SumKernelConfig`, `ProductKernelConfig`, `ScaledKernelConfig`.
- **Neural Network Kernel**: Dense neural network applied before a base kernel.
- **`MCEnsembleKernel`**: Covariance matrix derived from systematic variations.
- **`MultiFidelityResidualKernel`**: Residuals between data and simulation.

### 3. Mean Functions
Available mean functions:
- `ZeroMeanConfig`, `ConstantMeanConfig`: Standard means.
- `PolynomialBackgroundMeanConfig`, `ParametricBackgroundMeanConfig`: Parametric backgrounds.
- `DoubleSidedCrystalBallMeanConfig`, `GaussianBumpMeanConfig`, etc.: Resonance structures.
- `QCDMCMeanConfig`: Mean field from simulation with trainable adjustments.
- `SignalTemplateMeanConfig`: Signal injection geometry.
- `LookupTableMeanConfig`, `InterpolatedMeanConfig`: Manual specifications.

### 4. Inference and Optimization
- `OPTIMIZATION`: MLE or MAP estimates using Optax minimizers (Adam, AdamW, SGD).
- `TWO_STAGE`: Iterative procedure for kernel and mean function optimization.
- `SAMPLING`: Bayesian inference using NumPyro.
