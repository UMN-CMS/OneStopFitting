from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any
from ..data.loading import variationNames
from .histograms import normalizeVarName

import attrs

logger = logging.getLogger(__name__)

DIRECTION_PATTERNS = [
    (re.compile(r"^[Uu]p(?:_(.*))?$"), "Up"),
    (re.compile(r"^[Dd]own(?:_(.*))?$"), "Down"),
    (re.compile(r"^[Dd]n(?:_(.*))?$"), "Down"),
    (re.compile(r"^[Uu]P(?:_(.*))?$"), "Up"),
]


def parseRawVariation(raw: str) -> tuple[str, str | None, str | None]:
    """Parse `{category}-variation_{direction}[_{detail}]`.

    Returns (category, detail_suffix, direction).
    direction is "Up" or "Down", or None if unparseable.
    """
    parts = raw.split("-variation_", 1)
    if len(parts) != 2:
        return raw, None, None

    category = parts[0]
    rest = parts[1]

    for pattern, direction in DIRECTION_PATTERNS:
        m = pattern.match(rest)
        if m:
            detail = m.group(1)
            return category, detail, direction

    return category, rest, None


@attrs.define
class SystematicNameRule:
    """Maps a raw variation category to a CMS-compliant nuisance name prefix."""

    raw_prefix: str
    cms_prefix: str
    strip_detail_prefix: str | None = None
    syst_class: str = "custom"

    def matches(self, category: str) -> bool:
        return category == self.raw_prefix

    def apply(self, detail: str | None) -> str:
        if detail is None:
            return self.cms_prefix
        d = detail
        if self.strip_detail_prefix and d.startswith(self.strip_detail_prefix):
            d = d[len(self.strip_detail_prefix) :]
        return f"{self.cms_prefix}_{d}"


@attrs.define
class SystematicNameMap:
    """Collection of rules for mapping raw variation names to CMS names."""

    rules: list[SystematicNameRule] = attrs.Factory(list)

    def resolve(self, raw_variation: str) -> tuple[str, str | None]:
        """Returns (cms_nuisance_name, direction_or_None)."""
        if raw_variation == "central" or raw_variation.endswith("_disabled"):
            return raw_variation, None

        category, detail, direction = parseRawVariation(raw_variation)

        for rule in self.rules:
            if rule.matches(category):
                return rule.apply(detail), direction

        logger.warning(f"No rule for variation '{raw_variation}', using raw name")

        return normalizeVarName(raw_variation)


DEFAULT_RULES = [
    SystematicNameRule("bjetshapesf", "CMS_btag", syst_class="btag"),
    SystematicNameRule(
        "jes",
        "CMS_scale_j",
        strip_detail_prefix="jesRegrouped_",
        syst_class="jet_energy_scale",
    ),
    SystematicNameRule(
        "jer",
        "CMS_res_j",
        strip_detail_prefix="JER",
        syst_class="jet_energy_resolution",
    ),
    SystematicNameRule("pusf", "CMS_pileup", syst_class="pileup"),
    SystematicNameRule(
        "l1prefiring", "CMS_L1Prefiring", syst_class="other_experimental"
    ),
    SystematicNameRule("triggereff", "CMS_trigger", syst_class="other_experimental"),
    SystematicNameRule("puid", "CMS_puid", syst_class="jet_efficiency"),
]

DEFAULT_NAME_MAP = SystematicNameMap(rules=list(DEFAULT_RULES))


@attrs.define
class RateSystematic:
    """A configurable lnN/gmN systematic for the datacard."""

    name: str
    distribution: str
    value: str
    era_scope: list[str] | None = None

    def appliesTo(self, metadata: dict) -> bool:
        if self.era_scope is None:
            return True
        era = metadata["era"]["name"]
        return era in self.era_scope


def collectShapeSystematics(
    signal_hists: dict[str, Any],
    signal_metadata: dict[str, dict],
    name_map: SystematicNameMap = DEFAULT_NAME_MAP,
) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Collect and merge shape systematics across all signals.
    Returns:
        syst_entries: list of Systematic-compatible dicts
            [{name, distribution, values}, ...]
        hist_renames: dict mapping (process_label -> {raw_variation -> cms_hist_suffix})
            Used by exportCombineData to name histograms correctly.
    """

    nuisance_map: dict[str, dict[str, str]] = defaultdict(dict)
    hist_renames: dict[str, dict[str, str]] = defaultdict(dict)

    for lbl, sig_hist in signal_hists.items():
        for raw_var in variationNames(sig_hist):
            if raw_var == "central" or raw_var.endswith("_disabled"):
                continue

            cms_name, direction = name_map.resolve(raw_var)
            if direction is None:
                logger.warning(f"Cannot determine direction for '{raw_var}', skipping")
                continue

            nuisance_map[cms_name][lbl] = "1"
            hist_renames[lbl][raw_var] = f"{cms_name}{direction}"

    syst_entries = [
        {"name": name, "distribution": "shape", "values": dict(values)}
        for name, values in sorted(nuisance_map.items())
    ]

    logger.info(
        f"Collected {len(syst_entries)} shape systematics "
        f"across {len(signal_hists)} signals"
    )


    return syst_entries, dict(hist_renames)


def resolveRateSystematics(
    rate_systematics: list[RateSystematic],
    signal_labels: list[str],
    signal_metadata: dict[str, dict],
) -> list[dict]:
    entries = []
    for rs in rate_systematics:
        values = {}
        for lbl in signal_labels:
            meta = signal_metadata.get(lbl, {})
            if rs.appliesTo(meta):
                values[lbl] = rs.value
        if values:
            entries.append(
                {"name": rs.name, "distribution": rs.distribution, "values": values}
            )
    return entries
