from __future__ import annotations

import logging
import re
from typing import Any

import attrs

logger = logging.getLogger(__name__)

DIRECTION_PATTERNS = [
    (re.compile(r"^[Uu]p(?:_(.*))?$"), "Up"),
    (re.compile(r"^[Dd]own(?:_(.*))?$"), "Down"),
    (re.compile(r"^[Dd]n(?:_(.*))?$"), "Down"),
    (re.compile(r"^[Uu]P(?:_(.*))?$"), "Up"),
]

UP_RES = [re.compile(r"Up$"), re.compile(r"_up_"), re.compile(r"_up$")]
DOWN_RES = [
    re.compile(r"Down$"),
    re.compile(r"Dn$"),
    re.compile(r"_down_"),
    re.compile(r"_down$"),
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


def normalizeVarName(var_name: str) -> tuple[str, str | None]:
    """Fallback parser for variation names not matching the structured format."""
    if var_name == "central":
        return "central", None
    for expr in UP_RES:
        if expr.search(var_name):
            return expr.sub("", var_name), "Up"
    for expr in DOWN_RES:
        if expr.search(var_name):
            return expr.sub("", var_name), "Down"
    return var_name, None


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


DEFAULT_RATE_SYSTEMATICS = [
    RateSystematic("lumi_13TeV_2016", "lnN", "1.02", ["2016_preVFP", "2016_postVFP"]),
    RateSystematic("lumi_13TeV_2017", "lnN", "1.0082", ["2017"]),
    RateSystematic("lumi_13TeV_2018", "lnN", "1.009", ["2018"]),
    RateSystematic("lumi_13p6TeV_2022", "lnN", "1.014", ["2022_preEE", "2022_postEE"]),
    RateSystematic(
        "lumi_13p6TeV_2023", "lnN", "1.013", ["2023_preBPix", "2023_preBPix"]
    ),
]
