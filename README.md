# OneStopFitting: Gaussian Process Background Estimation

OneStopFitting is a comprehensive framework for Gaussian Process (GP) background estimation and smoothing, tailored for High Energy Physics (HEP) applications. The framework is designed to process multi-dimensional background and signal histograms, utilize Multi-Fidelity Gaussian Processes to robustly capture underlying distributions, and interface with statistical evaluation tools like CMS Combine.

## Project Structure

The codebase is organized into several key modules under the `fitting/` directory:

- `core/`: Fundamental utilities and polymorphic state serialization.
- `combine/`: Subprocess wrappers and configuration generation for CMS Combine.
- `data/`: Utilities for handling, rebinning, and masking datasets securely.
- `diagnostics/`: Automated plotting, aggregations, and LaTeX/Jinja2-based PDF report generation.
- `distributed/`: Tools for managing large-scale parameter sweeps and generating HTCondor submit files.
- `inference/`: Autoregressive models, optimization routines (e.g., Adam), and MCMC pipelines leveraging JAX and GPJax.
- `steps/`: Distinct executable steps of the fitting pipeline (preprocessing, generation, evaluation).

## Installation

The project relies on `uv` for dependency management and ensures cross-platform reproducibility using Apptainer/Singularity containers.

To install the project locally or inside an appropriate ML container:

```bash
uv venv
uv sync
```

Alternatively, you can utilize the provided `setup.sh` to initialize the containerized environment on clusters with access to cvmfs. Note that the project utilizes Python 3.11+.

## Command Line Interface (`fitting.cli`)

The primary entry point is `python -m fitting`. The toolkit provides robust subcommands for managing the full lifecycle of background estimation.

### 1. `run`
Executes the fitting pipeline. 

**Key Arguments:**
- `--config`, `-c`: Path to the YAML configuration file determining the model geometry and pipeline parameters.
- `--background`, `-b`: Path to the target background histogram (`.pklz4`).
- `--signal`, `-s`: Optional path to the target signal histogram to inject (`.pklz4`).
- `--output`, `-o`: Output directory or format string (e.g., `output/{era.name}/{dataset_name}`).
- `--injection-rate`: Set a specific rate multiplier for signal injections (default: `0.0`).
- `--start-from`: Allows resuming pipeline execution from a distinct phase (e.g., `LOAD`, `FIT`, `COMBINE`).

```bash
python -m fitting run \
    --config resources/smoothing_configs/Signal312/comp.yaml \
    --background subsetexported/2018/Signal312/qcd_inclusive_2018/comp_mStop_vs_mChiRatio.pklz4 \
    --injection-rate 0.1 \
    --output output/2018/qcd_inclusive/comp
```

### 2. `smooth`
Generates sampled smoothed backgrounds by drawing poisson toys from a smooth latent distribution

**Key Arguments:**
- `--state`, `-s`: The loaded `state.pklz4` resolving from a successful pipeline `run`.
- `--output-dir`, `-o`: Location to save generated sampled background frames.
- `--name`, `-n`: Prefix for standardizing naming conventions of individual toy extractions.
- `--num-samples`: Adjusts the number of posterior draws requested.
- `--include-smooth`: Outputs a pure  "Asimov" background, equal to the GPR mean.

```bash
python -m fitting smooth \
    --state output/2018/qcd_inclusive/comp/state.pklz4 \
    --name qcd_smoothed_category \
    --output-dir smoothed_outputs/ \
    --num-samples 10
```

### 3. `aggregate-plot`
Harvests metrics stored across multiple `summary.json` outcomes and reconstructs them into 2D mass-plane plots. 

**Key Arguments:**
- `--metric`, `-m`: Target dot-path referring into the `summary.json` (e.g., `metrics.blinded_chi2_per_bin`).
- `--output`, `-o`: Final aggregate visualization directory.
- `--formats`, `-f`: Explicit image format parameters (repeatable, `png` and `pdf`).
- `--smooth-sigma`: Applies a pixelated Gaussian filtering over final outputs to track global gradient trends.
- `--cmap`: Matplotlib colorspace representation override.

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
Generate single file pdf reports for a given set of summary jsons.

**Key Arguments:**
- `--input`, `-i`: Directory paths or Glob strings searching for `summary.json`. Can be declared repeatedly.
- `--output`, `-o`: Parent directory path mapping final artifacts via the same relative layout context as inputs.
- `--single-document`: Compresses output rendering spanning multiple endpoints into a unified bulk presentation.
- `--latex-engine`: Selection parameter to align with computing instance capacity (`pdflatex`, `xelatex`).

```bash
python -m fitting report \
    --input "output/2018/qcd_inclusive/comp/**/summary.json" \
    --output report_output/ \
    --single-document \
    --latex-engine pdflatex
```

### 5. `makecondor` and `makebatch`

`makecondor` and `makebatch` are used to generate submission files for running the fitting pipeline on a cluster.


**Key Arguments (`makecondor`):**
- `--signal`: Glob pattern resolving varied signal templates (`**/signal_{year}_*.pklz4`).
- `--background`: Glob pattern targeting concurrent backgrounds (`**/bkg_{year}.pklz4`).
- `--years`: Comma or space separated parameters iterating multiple campaign collections.
- `--subdir-format`: Injection specification matching execution folders mapping output strings (e.g., `{era.name}/{dataset_name}`).
- `--output`: Generates comprehensive Condor sub files alongside accompanying executable instances.

```bash
python -m fitting makecondor \
    --signal "subsetexported/2018/Signal312/**/signal_*.pklz4" \
    --background "subsetexported/2018/Signal312/**/qcd_inclusive*.pklz4" \
    --years 2018 \
    --pipelines smoothing \
    --subdir-format "{era.name}/{dataset_name}" \
    --output condor_submit_files
```

**Key Arguments (`makebatch`):**
- Features the same foundational arguments as `makecondor` but incorporates dynamic parameter sweep processing.
- `--rates`: Comma-separated injection rates to sweep (e.g., `0.0,0.1,0.5`).
- `--rebin`: Comma-separated rebin factors to sweep (e.g., `1,2,4`).
- `--window-spread`: Comma-separated window spread tuning values.
- `--config-base`: Base YAML configuration file functioning as an override template.

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
Extracts the results of the CMS Combine statistical evaluation from the `combine` directory and saves them to a `summary.json` file, and possibly image files.

**Key Arguments:**
- `<summaries>`: Standard positional paths resolving raw `summary.json` locations where respective parallel `/combine` directories simultaneously overlap.

```bash
python -m fitting harvest output/2018/qcd_inclusive/comp/**/summary.json
```

## Modeling and Inference Choices

OneStopFitting provides a flexible framework for modeling complex backgrounds in high energy physics.

### 1. GP Models (`fitting.inference.models`)
- **`ExactGPConfig`**: Standard full-batch Gaussian Process fit. Ideal for lower dimensions and data spaces computationally acceptable for Cholesky decompositions.
- **`SparseGPConfig`**: Employs collapsed variational inducing points suitable for large datasets.
- **`VariationalGPConfig`**: GPR regression using stochasic variational inference.
- **`MultiFidelityGPConfig`**: A two-stage autoregressive framework. Primarily fits against a low-fidelity representation (e.g., QCD Monte Carlo) and refines a correlated high-fidelity surrogate over the observed data. 
- **`QCDPriorGPConfig`**: Bayesian workflow utilizing extracted hyperpriors from MC runs.

### 2. Available Kernels (`fitting.inference.kernels`)
The software integrates a large number of stationary and non-stationary kernels available in GPJax, while adding specialized physics extensions.
- **Standard**: `RBF`, `Matern12` / `Matern32` / `Matern52`, `RationalQuadratic`, `Polynomial`, `Periodic`, `Linear`, `White`.
- **Composites**: `SumKernelConfig`, `ProductKernelConfig`, `ScaledKernelConfig`.
- **Neural Network Kernel (`NNKernelConfig` / `DeepKernelFunction`)**: Apply a dense NN before applying a base kernel (like RBF). Allows dynamic spatial warping for non-stationary localized mass distortions.
- **`MCEnsembleKernel`**: Generates a custom empirical covariance matrix directly from multiple systematic MC variations.
- **`MultiFidelityResidualKernel`**: Represents the underlying discrepancy between the data and QCD MC predictions.

### 3. Mean Functions (`fitting.inference.means`)
Fitting accurately in multi-dimensional space requires defining rigid geometric priors.
- **Standard**: `ZeroMeanConfig`, `ConstantMeanConfig`.
- **Parametric Backgrounds**: Custom polynomial mapping (`PolynomialBackgroundMeanConfig`), exponential parametric structures (`ParametricBackgroundMeanConfig`).
- **Bump Geometries**: Supports complex geometric perturbations such as `DoubleSidedCrystalBallMeanConfig`, `GaussianBumpMeanConfig`, `AsymmetricGaussianBumpMeanConfig`, `StudentTBumpMeanConfig`, `SkewedGaussianMeanConfig`, and `AsymmetricLaplaceMeanConfig`. Includes `MixtureOfGaussiansMeanConfig` for resonance structures.
- **Physics Extractions**: 
  - `QCDMCMeanConfig`: Dynamically constructs the mean field using nearest-neighbor interpolations over a valid QCD MC space paired with trainable linear tilt adjustments.
  - `SignalTemplateMeanConfig`: Directly incorporates a user-provided signal injection geometry.
- **Lookups**: `LookupTableMeanConfig` and `InterpolatedMeanConfig` for manual mean specification.

### 4. Inference and Optimization (`fitting.inference.optimization`)
Inference modes are configurable directly via the CLI (`--mode`) and optimizations via `OptimizationConfig`.
- **Optimization Mode (`OPTIMIZATION`)**: Resolves Maximum Likelihood Estimation (MLE) (standard) or MAP estimates using `Optax` based minimizers (Adam, AdamW, SGD). Objective evaluations resolve standard Marginal Log-Likelihood (`MLL`), `LOOCV`, `ELBO`, or `COLLAPSED_ELBO` architectures.
- **Two-Stage Fits (`TWO_STAGE` / `HOMOSCEDASTIC_TWO_STAGE`)**: Iterative procedure prioritizing convergence robustness. Freezes the kernel bounds to forcefully optimize macro-shape geometries (mean functions) via early-stage large learning rates, before resolving minor spatial residuals independently.
- **Markov Chain Monte Carlo (`SAMPLING`)**: Fully bayesian inference using `NumPyro` backends.