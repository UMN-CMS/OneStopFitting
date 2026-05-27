from __future__ import annotations

import logging
import os
from pathlib import Path

import jax
from jinja2 import Environment, FileSystemLoader

from ..core.data import AnalysisState
from ..combine.builders import buildCombineModel
from ..combine.commands import CombineContext, Text2Workspace
from ..diagnostics.plot_utils import getPlotSaver
from ..diagnostics.combine import (
    plotCombineInputs,
    verifyEigenvariations,
    visualizeEigenvariations,
)

logger = logging.getLogger(__name__)


def _makeScript(state, channel_name):
    resolved_cmds = state.config.combine.resolvedCommands()
    if not resolved_cmds:
        return

    context = CombineContext(
        signal_labels=list(state.signals.keys()),
        channel_name=channel_name,
        expected_r=state.config.injection_rate,
    )

    if state.config.injection_rate is not None and state.config.injection_rate > 0:
        from ..combine.commands import GoodnessOfFit, Impacts

        resolved_cmds = [
            c for c in resolved_cmds if not isinstance(c, (GoodnessOfFit, Impacts))
        ]

    all_shell_cmds = Text2Workspace().render(context)
    for cmd in resolved_cmds:
        all_shell_cmds.extend(cmd.render(context))

    combine_dir = state.getRealOutPath() / "combine"
    combine_dir.mkdir(parents=True, exist_ok=True)

    script_path = combine_dir / "run_combine_commands.sh"
    template_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template("run_combine_commands.sh.jinja")
    script_content = template.render(
        container=state.config.combine.combine_container,
        commands=all_shell_cmds,
        enumerate=enumerate,
    )

    with open(script_path, "w") as f:
        f.write(script_content)

    os.chmod(script_path, 0o755)

    logger.info(f"Combine script generated at {script_path}")
    logger.info(f"Contains {len(all_shell_cmds)} command(s)")
    logger.info(f"To run: bash {script_path}")


def _makePlots(state):
    diag_dir = state.getRealOutPath() / "diagnostics" / "combine"
    diag_dir.mkdir(parents=True, exist_ok=True)

    plot_saver = getPlotSaver(diag_dir, [state.metadata])

    plotCombineInputs(state, plot_saver=plot_saver)
    verifyEigenvariations(
        state,
        plot_saver=plot_saver,
        eigenvar_threshold=state.config.combine.eigenvar_threshold,
    )
    visualizeEigenvariations(state, plot_saver=plot_saver)


def prepareCombine(state: AnalysisState, rng_key: jax.Array) -> AnalysisState:
    if not state.config.signal_path:
        logger.info("Skipping Combine preparation: no signal path")
        return state

    model = buildCombineModel(state)
    out_dir = state.getRealOutPath() / "combine"
    model.write(out_dir)

    if state.config.combine.export_systematics_table:
        csv_table = model.renderSystematicsTable("csv")
        (out_dir / "systematics.csv").write_text(csv_table)
        logger.info(f"Wrote systematics summary CSV to {out_dir / 'systematics.csv'}")

        latex_table = model.renderSystematicsTable("latex")
        (out_dir / "systematics.tex").write_text(latex_table)
        logger.info(f"Wrote systematics summary LaTeX to {out_dir / 'systematics.tex'}")

    _makePlots(state)
    _makeScript(state, model.channels[0].name)
    return state
