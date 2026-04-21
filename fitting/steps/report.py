from __future__ import annotations

import logging
from pathlib import Path
from ..diagnostics.point_report import generatePointReport, PointReportConfig

import jax

from ..core.data import AnalysisState

logger = logging.getLogger(__name__)


def runPointReport(state: AnalysisState, rng_key: jax.Array) -> AnalysisState:
    config = PointReportConfig(
        latex_engine="pdflatex",
        keep_build=False,
        keep_tex=False,
        image_format="pdf",
    )

    if state.signal is not None:
        generatePointReport(
            point_dir=Path(state.getRealOutPath()).absolute(),
            output_pdf=Path(state.getRealOutPath()).absolute() / "report.pdf",
            config=config,
        )
