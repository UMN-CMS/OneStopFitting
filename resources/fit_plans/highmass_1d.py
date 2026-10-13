"""
Fit plan: 1D (mStop only) fits for high stop / high chargino mass points,
standard 2D (mStop vs mChiRatio) fits everywhere else.

Use with:
    python -m fitting makesubmit --fit-plan resources/fit_plans/highmass_1d.py --dry-run ...

Profile patterns take the same placeholders as makesubmit's --signal/--background/
--config-pattern: {year} {pipeline} {category} {reco_category} {toy_index}.

NOTE: the paths and the 1D region below are placeholders, adjust before use.
"""

from fitting.distributed.fit_plan import FitPlan, Point, Profile

PROFILE_2D = Profile(
    background="smoothed_complete/{year}/{pipeline}/qcd_inclusive_2018/{reco_category}/{reco_category}_{toy_index}.pklz4",
    signal="export_complete_with_large/{year}/{pipeline}/signal*/{reco_category}_mStop_vs_mChiRatio.pklz4",
    config="resources/new_opt/config_{reco_category}_{pipeline}.yaml",
    ndim=2,
)

PROFILE_1D = Profile(
    background="smoothed_1d_verycomp_sel/{year}/{pipeline}VeryComp/{reco_category}/{reco_category}_{toy_index}.pklz4",
    signal="2026_09_10__dbs_verycomp_sel_w_1d_dump/{pipeline}VeryComp/{year}/signal*/comp_mStop.pklz4",  # only comp has 1D inputs
    config="resources/new_opt_1d/config_{reco_category}_{pipeline}VeryComp.yaml",
    ndim=1,
)

# (mstop, mchi) points forced to 1D regardless of the region below
EXTRA_1D_POINTS: set[tuple[int, int]] = set()


def isHighMass(p: Point) -> bool:
    return p.mstop >= 1500 and p.mchi >= 0.9 * p.mstop


def choose(p: Point) -> list[str]:
    """Profiles to use for this job, best first. Empty means the default."""
    # Only the comp reco category has 1D inputs; other reco categories stay 2D.
    if p.reco_category != "comp":
        return []
    if isHighMass(p) or (p.mstop, p.mchi) in EXTRA_1D_POINTS:
        return ["1d"]
    return []


PLAN = FitPlan(
    profiles={"2d": PROFILE_2D, "1d": PROFILE_1D},
    default="2d",
    choose=choose,
)
