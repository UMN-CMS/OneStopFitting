from __future__ import annotations

import logging
import os
from pathlib import Path

import uproot
import jax
import jax.numpy as jnp
import numpy as np
from jinja2 import Environment, FileSystemLoader

from ..core.data import AnalysisState
from ..combine.histograms import exportCombineData
from ..combine.datacard import Process, Channel, Systematic, DataCard, RateParam
from ..combine.commands import CombineContext, Text2Workspace
from ..combine.systematics import (
    collectShapeSystematics,
    resolveRateSystematics,
    DEFAULT_NAME_MAP,
)
from ..diagnostics.plot_utils import getPlotSaver
from ..diagnostics.combine import (
    plotCombineInputs,
    verifyEigenvariations,
    visualizeEigenvariations,
)
from ..combine.contamination import computeContaminationTemplate, plotContamination

logger = logging.getLogger(__name__)


def _buildProcesses(state, doMask):
    ch_name = state.metadata.get("channel", "ch1")
    observation = float(jnp.sum(doMask(state.test_data.Y)))
    bg_rate = float(jnp.sum(doMask(state.pred_mean)))

    processes = [Process(name="background", rate=bg_rate, index=1)]
    for i, (lbl, sig) in enumerate(state.signals.items()):
        sig_vals = sig.Y[state.domain_mask] if state.domain_mask is not None else sig.Y
        sig_rate = float(jnp.sum(doMask(sig_vals)))
        processes.append(Process(name=lbl, rate=sig_rate, index=-i))

    return ch_name, observation, processes, bg_rate


def _addBackgroundRateUnc(state, bg_rate, systematics, rate_params, ch_name, postfix):
    bg_rate_unc = state.config.combine.bg_rate_uncertainty
    if bg_rate_unc != "none" and state.pred_cov is not None:
        blind_mask = state.blind_mask
        pred_cov_masked = state.pred_cov[blind_mask, :][:, blind_mask]
        ones = jnp.ones(pred_cov_masked.shape[0])
        rate_unc = float(
            jnp.sqrt(ones @ pred_cov_masked @ ones) / jnp.maximum(bg_rate, 1.0)
        )
        logger.info(
            f"Background rate uncertainty: {rate_unc:.4f} "
            f"(mode={bg_rate_unc}, bg_rate={bg_rate:.2f})"
        )

        if bg_rate_unc == "lnN":
            lnN_val = 1.0 + rate_unc
            systematics.append(
                Systematic(
                    name=f"bg_norm{postfix}",
                    distribution="lnN",
                    values={"background": f"{lnN_val:.4f}"},
                )
            )
        elif bg_rate_unc == "rateParam":
            lo = max(1.0 - rate_unc, 0.01)
            hi = 1.0 + rate_unc
            rate_params.append(
                RateParam(
                    channel=ch_name,
                    process="background",
                    init_value=1.0,
                    bounds=[lo, hi],
                )
            )


def _buildSystematics(state, sig_syst_entries, processes, n_eigen, ch_name, bg_rate):
    systematics = []
    year = state.background_metadata["era"]["name"]
    postfix = f"_{year}" if year else ""

    for i in range(n_eigen):
        systematics.append(
            Systematic(
                name=f"gpr_eigen{i}{postfix}",
                distribution="shape",
                values={"background": "1"},
            )
        )

    for entry in sig_syst_entries:
        systematics.append(
            Systematic(
                name=entry["name"],
                distribution=entry["distribution"],
                values=entry["values"],
            )
        )

    signal_labels = [p.name for p in processes if p.index <= 0]
    rate_entries = resolveRateSystematics(
        state.config.combine.rate_systematics, signal_labels, state.signal_metadata
    )
    for entry in rate_entries:
        systematics.append(
            Systematic(
                name=entry["name"],
                distribution=entry["distribution"],
                values=entry["values"],
            )
        )

    rate_params = []
    _addBackgroundRateUnc(state, bg_rate, systematics, rate_params, ch_name, postfix)

    return systematics, rate_params


def _makeScript(state, channel_name):

    resolved_cmds = state.config.combine.resolvedCommands()
    if not resolved_cmds:
        return

    context = CombineContext(
        signal_labels=list(state.signals.keys()),
        channel_name=channel_name,
        # has_contamination=has_contamination,
        expected_r=state.config.injection_rate,
    )

    if state.config.injection_rate is not None and state.config.injection_rate > 0:
        from ..combine.commands import GoodnessOfFit, Impacts

        resolved_cmds = [
            c for c in resolved_cmds if not isinstance(c, (GoodnessOfFit, Impacts))
        ]

    # text2workspace is always first
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


def makeSystematicsTable(state, sig_syst_entries):
    pass

def prepareCombine(state: AnalysisState, rng_key: jax.Array) -> None:
    if not state.config.signal_path:
        logger.info("Skipping Combine preparation: no signal path")
        return
    out_dir = state.getRealOutPath() / "combine"
    shapes_file = "shapes.root"
    shapes_path = out_dir / shapes_file
    datacard_path = out_dir / "datacard.txt"

    logger.info(f"Preparing Combine inputs in {out_dir}")

    sig_syst_entries, hist_renames = collectShapeSystematics(
        state.signal_hists,
        state.signal_metadata,
        name_map=state.config.combine.name_map,
    )


    n_eigen = exportCombineData(
        state=state,
        output_path=shapes_path,
        eigenvar_threshold=state.config.combine.eigenvar_threshold,
        hist_renames=hist_renames,
    )

    def doMask(x):
        return x[state.blind_mask]

    ch_name, observation, processes, bg_rate = _buildProcesses(state, doMask)
    channels = [
        Channel(
            name=ch_name,
            observation=observation,
            processes=processes,
            shapes_file=shapes_file,
            use_auto_mc_stats=True,
        )
    ]

    systematics, rate_params = _buildSystematics(
        state, sig_syst_entries, processes, n_eigen, ch_name, bg_rate
    )

    card = DataCard(channels=channels, systematics=systematics, rate_params=rate_params)
    card.write(datacard_path)

    _makePlots(state)
    logger.info(f"Combine preparation complete. Datacard: {datacard_path}")

    _makeScript(state, ch_name)
