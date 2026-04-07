from __future__ import annotations

import logging
from enum import IntEnum
from pathlib import Path
from typing import Any

import attrs
import jax
from jax import random
import mplhep

from .core.data import AnalysisState
from .core.serialization import save
from .core.transforms import (
    StandardizationConfig,
    SqrtStandardizationConfig,
    SqrtConfig,
    NormalizeAxes,
    TransformConfig,
)
from .data.windowing import Window
from .inference.models import ExactGPConfig, GPModelConfig
from .inference.optimization import OptimizationConfig

# Import the modular pipeline steps
from .steps.load import loadData
from .steps.train import trainModel
from .steps.diagnostics import runDiagnostics
from .steps.plot import generatePlots
from .steps.combine import prepareCombine
from .steps.report import runPointReport

logger = logging.getLogger(__name__)

mplhep.style.use("CMS")


@attrs.define
class CombineConfig:
    """Configuration for combine command execution."""

    combine_commands: list[str] = attrs.Factory(
        lambda: [
            "limits",
            "fit-diagnostics",
            "multidimfit",
            "significance",
            "gof-saturated",
        ]
    )
    eigenvar_threshold: float = 0.01
    combine_container: str = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/combine-container:latest"

@attrs.define
class PipelineConfig:
    background_path: Path
    signal_path: Path | None = None
    injection_rate: float = 0.0
    rebin: int = 1
    min_counts: float = 0.0
    rng_seed: int = 0xBEEFBEEF
    domain_window: Window | None = None
    window_spread: float = 2.0
    transform: TransformConfig = attrs.Factory(StandardizationConfig)
    model: GPModelConfig = attrs.Factory(ExactGPConfig)
    optimization: OptimizationConfig = attrs.Factory(OptimizationConfig)
    combine: CombineConfig = attrs.Factory(CombineConfig)
    output_dir_format: str = "output"
    image_formats: list[str] = attrs.Factory(lambda: ["png"])
    metadata: dict[str, Any] = attrs.Factory(dict)


class PipelineStep(IntEnum):
    LOAD = 0
    TRAIN = 1
    DIAGNOSTICS = 2
    PLOT = 3
    COMBINE = 4
    REPORT = 5

    @classmethod
    def fromStr(cls, s: str | None) -> PipelineStep | None:
        if s is None:
            return None
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid step: {s}. Valid steps are: {[step.name.lower() for step in cls]}"
            )


STEP_FUNCS = {
    PipelineStep.LOAD: loadData,
    PipelineStep.TRAIN: trainModel,
    PipelineStep.DIAGNOSTICS: runDiagnostics,
    PipelineStep.PLOT: generatePlots,
    PipelineStep.COMBINE: prepareCombine,
    PipelineStep.REPORT: runPointReport,
}


def runPipeline(
    config: PipelineConfig,
    single_step: PipelineStep | None = None,
    start_from: PipelineStep = PipelineStep.LOAD,
) -> AnalysisState:
    from .core.serialization import load

    jax.config.update("jax_enable_x64", True)
    rng_key = random.key(config.rng_seed)

    if single_step:
        to_run = [single_step]
    else:
        to_run = [s for s in PipelineStep if s >= start_from]

    if PipelineStep.LOAD in to_run:
        state = loadData(config)
        logger.info(f"Output directory: {state.getRealOutPath()}")
        save(state, state.getRealOutPath())
    else:
        dummy_state = loadData(config)
        out_path = dummy_state.getRealOutPath()
        logger.info(f"Resuming from state at {out_path}")
        state = load(out_path)
        state = attrs.evolve(state, config=config)

    for s in to_run:
        if s == PipelineStep.LOAD:
            continue
        func = STEP_FUNCS[s]
        rng_key, key = random.split(rng_key)
        if s in [PipelineStep.TRAIN, PipelineStep.DIAGNOSTICS]:
            state = func(state, key)
            save(state, state.getRealOutPath())
        elif s == PipelineStep.PLOT:
            func(state, key)
        else:
            func(state, key)

    # Echo the actual output path for scripts to capture
    print(f"FITTING_OUTPUT_PATH: {state.getRealOutPath()}")

    return state
