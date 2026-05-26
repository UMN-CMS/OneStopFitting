from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np

from ..core.data import AnalysisState
from ..data.loading import hasVariationAxis, variationNames, histToBinnedData
from .eigenvariations import computeEigenvariations
from .model import (
    ChannelModel,
    CombineModel,
    ProcessModel,
    RateParam,
)
from .systematics import SystematicNameMap, DEFAULT_NAME_MAP, RateSystematic

logger = logging.getLogger(__name__)


def _applyBlindMask(state: AnalysisState):
    return state.blind_mask


def _applyDomainMask(values: jnp.ndarray, domain_mask):
    if domain_mask is not None:
        return values[domain_mask]
    return values


def _maxSignalSize(state: AnalysisState, blind_mask) -> float:
    signal_size = 0.0
    for sig_binned in state.signals.values():
        full_sig = _applyDomainMask(sig_binned.Y, state.domain_mask)[blind_mask]
        signal_size = max(signal_size, float(jnp.max(full_sig)))
    return signal_size


def buildBackgroundProcess(
    state: AnalysisState,
    blind_mask: jnp.ndarray,
    eigenvar_threshold: float,
) -> ProcessModel:
    pred_mean_masked = np.asarray(state.pred_mean[blind_mask])
    pred_cov_masked = np.asarray(state.pred_cov[blind_mask, :][:, blind_mask])

    year = state.background_metadata["era"]["name"]
    postfix = f"_{year}" if year else ""

    signal_size = _maxSignalSize(state, blind_mask)
    variations = computeEigenvariations(
        jnp.array(pred_mean_masked),
        jnp.array(pred_cov_masked),
        threshold_fraction=eigenvar_threshold,
        signal_size=signal_size,
    )

    proc = ProcessModel(name="background", index=1, nominal=pred_mean_masked)

    for var in variations:
        proc.addShape(
            name=state.config.combine.background_syst_prefix
            + "_"
            + f"gpr_eigen{var['index']}{postfix}",
            up=np.asarray(var["up"]),
            down=np.asarray(var["down"]),
        )

    bg_rate_unc = state.config.combine.bg_rate_uncertainty
    if bg_rate_unc == "lnN" and state.pred_cov is not None:
        ones = jnp.ones(pred_cov_masked.shape[0])
        rate_unc = float(
            jnp.sqrt(ones @ jnp.array(pred_cov_masked) @ ones)
            / jnp.maximum(proc.rate, 1.0)
        )
        proc.addRate(
            name=f"bg_norm{postfix}",
            distribution="lnN",
            value=f"{1.0 + rate_unc:.4f}",
        )
        logger.info(
            f"Background rate uncertainty: {rate_unc:.4f} "
            f"(mode={bg_rate_unc}, bg_rate={proc.rate:.2f})"
        )

    return proc


def buildSignalProcess(
    state: AnalysisState,
    label: str,
    sig_hist,
    blind_mask: jnp.ndarray,
    signal_index: int,
    name_map: SystematicNameMap,
    rate_systematics: list[RateSystematic],
) -> ProcessModel:
    central = histToBinnedData(sig_hist, rebin=state.config.rebin, variation="central")
    nominal = np.asarray(_applyDomainMask(central.Y, state.domain_mask)[blind_mask])
    proc = ProcessModel(name=label, index=-signal_index, nominal=nominal)

    if hasVariationAxis(sig_hist):
        _addShapeVariations(state, proc, sig_hist, blind_mask, name_map)

    meta = state.signal_metadata.get(label, {})
    for rs in rate_systematics:
        if rs.appliesTo(meta):
            proc.addRate(name=rs.name, distribution=rs.distribution, value=rs.value)

    return proc


def _addShapeVariations(
    state: AnalysisState,
    proc: ProcessModel,
    sig_hist,
    blind_mask: jnp.ndarray,
    name_map: SystematicNameMap,
) -> None:
    pending: dict[str, dict[str, np.ndarray]] = {}

    for raw_var in variationNames(sig_hist):
        if raw_var == "central" or raw_var.endswith("_disabled"):
            continue

        cms_name, direction = name_map.resolve(raw_var)
        if direction is None:
            logger.warning(f"Cannot determine direction for '{raw_var}', skipping")
            continue

        sig_binned = histToBinnedData(
            sig_hist, rebin=state.config.rebin, variation=raw_var
        )
        values = np.asarray(
            _applyDomainMask(sig_binned.Y, state.domain_mask)[blind_mask]
        )

        pending.setdefault(cms_name, {})[direction] = values

    for cms_name, directions in sorted(pending.items()):
        up = directions.get("Up")
        down = directions.get("Down")
        if up is None or down is None:
            logger.warning(
                f"Systematic '{cms_name}' missing "
                f"{'Up' if up is None else 'Down'} variation for process "
                f"'{proc.name}', skipping"
            )
            continue
        proc.addShape(name=cms_name, up=up, down=down)


def buildChannel(state: AnalysisState) -> ChannelModel:
    blind_mask = _applyBlindMask(state)
    ch_name = state.metadata.get("channel", "ch1")
    eigenvar_threshold = state.config.combine.eigenvar_threshold
    name_map = state.config.combine.name_map or DEFAULT_NAME_MAP
    rate_systematics = state.config.combine.rate_systematics

    data_obs = np.asarray(state.test_data.Y[blind_mask])

    bg_proc = buildBackgroundProcess(state, blind_mask, eigenvar_threshold)
    processes = [bg_proc]

    for i, (lbl, sig_hist) in enumerate(state.signal_hists.items()):
        sig_proc = buildSignalProcess(
            state=state,
            label=lbl,
            sig_hist=sig_hist,
            blind_mask=blind_mask,
            signal_index=i,
            name_map=name_map,
            rate_systematics=rate_systematics,
        )
        processes.append(sig_proc)

    return ChannelModel(
        name=ch_name,
        data_obs=data_obs,
        processes=processes,
        use_auto_mc_stats=True,
    )


def _buildRateParams(state: AnalysisState, channel: ChannelModel) -> list[RateParam]:
    rate_params = []
    bg_rate_unc = state.config.combine.bg_rate_uncertainty

    if bg_rate_unc == "rateParam" and state.pred_cov is not None:
        blind_mask = _applyBlindMask(state)
        pred_cov_masked = state.pred_cov[blind_mask, :][:, blind_mask]
        bg_proc = next(p for p in channel.processes if p.name == "background")
        ones = jnp.ones(pred_cov_masked.shape[0])
        rate_unc = float(
            jnp.sqrt(ones @ pred_cov_masked @ ones) / jnp.maximum(bg_proc.rate, 1.0)
        )
        lo = max(1.0 - rate_unc, 0.01)
        hi = 1.0 + rate_unc
        rate_params.append(
            RateParam(
                channel=channel.name,
                process="background",
                init_value=1.0,
                bounds=[lo, hi],
            )
        )

    return rate_params


def buildCombineModel(state: AnalysisState) -> CombineModel:
    channel = buildChannel(state)
    rate_params = _buildRateParams(state, channel)

    return CombineModel(
        channels=[channel],
        rate_params=rate_params,
    )
