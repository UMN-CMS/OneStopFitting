from __future__ import annotations

import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from jinja2 import Environment, FileSystemLoader

from ..core.data import AnalysisState
from ..combine.histograms import exportCombineData, normalizeVarName
from ..combine.datacard import Process, Channel, Systematic, DataCard
from ..data.loading import variationNames
from ..diagnostics.combine import plotCombineInputs, verifyEigenvariations
from ..distributed.condor_tools import COMBINE_SHORT_COMMANDS

logger = logging.getLogger(__name__)


def prepareCombine(state: AnalysisState, rng_key: jax.Array) -> None:
    out_dir = state.getRealOutPath() / "combine"
    shapes_file = "shapes.root"
    shapes_path = out_dir / shapes_file
    datacard_path = out_dir / "datacard.txt"

    logger.info(f"Preparing Combine inputs in {out_dir}")

    n_eigen = exportCombineData(state=state, output_path=shapes_path)

    channels = []

    def doMask(x):
        return x[state.blind_mask]

    ch_name = state.metadata.get("channel", "ch1")
    observation = float(jnp.sum(doMask(state.test_data.Y)))
    processes = []
    bg_rate = float(jnp.sum(doMask(state.pred_mean)))
    processes.append(Process(name="background", rate=bg_rate, index=1))
    if state.signal is not None:
        sig_name = "signal"
        sig_rate = float(jnp.sum(doMask(state.signal.Y[state.domain_mask])))
        processes.append(Process(name=sig_name, rate=sig_rate, index=0))

    channels.append(
        Channel(
            name=ch_name,
            observation=observation,
            processes=processes,
            shapes_file=shapes_file,
        )
    )

    systematics = []
    for i in range(n_eigen):
        syst_values = {"background": "1"}
        systematics.append(
            Systematic(
                name=f"gpr_eigen{i}",
                distribution="shape",
                values=syst_values,
            )
        )

    if state.signal_hist is not None:
        sig_name = "signal"
        all_vars = variationNames(state.signal_hist)
        sig_systs = set()
        for v in all_vars:
            if v == "central" or v.endswith("_disabled"):
                continue

            base, direction = normalizeVarName(v)
            sig_systs.add(base)

        for syst_name in sorted(list(sig_systs)):
            systematics.append(
                Systematic(
                    name=syst_name,
                    distribution="shape",
                    values={sig_name: "1"},
                )
            )
    systematics.append(
        Systematic(
            name="lumi",
            distribution="lnN",
            values={p.name: "1.02" for p in processes if p.name != "background"},
        )
    )

    card = DataCard(channels=channels, systematics=systematics)
    card.write(datacard_path)

    # Combine Diagnostics
    diag_dir = state.getRealOutPath() / "diagnostics" / "combine"
    plotCombineInputs(state, diag_dir)
    verifyEigenvariations(state, diag_dir)

    logger.info(f"Combine preparation complete. Datacard: {datacard_path}")

    # Generate bash script for combine commands if specified
    if state.config.combine.combine_commands:
        combine_dir = state.getRealOutPath() / "combine"
        combine_dir.mkdir(parents=True, exist_ok=True)

        # Expand short command names
        expanded_cmds = []
        for cmd in state.config.combine.combine_commands:
            if cmd in COMBINE_SHORT_COMMANDS:
                expanded_cmds.append(COMBINE_SHORT_COMMANDS[cmd])
                logger.info(f"Expanded '{cmd}' to full command")
            else:
                expanded_cmds.append(cmd)
                logger.info(f"Using custom command: {cmd}")

        # Generate bash script using jinja template
        script_path = combine_dir / "run_combine_commands.sh"
        # Template dir is up one level from steps package
        template_dir = Path(__file__).parent.parent / "templates"
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template("run_combine_commands.sh.jinja")
        script_content = template.render(
            container=state.config.combine.combine_container,
            commands=expanded_cmds,
            enumerate=enumerate,
        )

        with open(script_path, "w") as f:
            f.write(script_content)

        os.chmod(script_path, 0o755)

        logger.info(f"Combine script generated at {script_path}")
        logger.info(
            f"Contains {len(expanded_cmds)} command(s): {', '.join(state.config.combine.combine_commands)}"
        )
        logger.info(f"To run: bash {script_path}")
