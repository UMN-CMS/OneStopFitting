"""
Fit plan: 1D (mStop only) fits for high stop / high chargino mass points,
standard 2D (mStop vs mChiRatio) fits everywhere else.

Use with:
    python -m fitting makesubmit --fit-plan resources/fit_plans/highmass_1d.py --dry-run ...

Profile patterns take the same placeholders as makesubmit's --signal/--background/
--config-pattern: {year} {pipeline} {category} {reco_category} {toy_index}.
"""

from fitting.distributed.fit_plan import FitPlan, Point, Profile

PROFILE_2D = Profile(
    background="smoothed_complete_backgrounds/{year}/{pipeline}/{reco_category}/{reco_category}_{toy_index}.pklz4",
    signal="export_signal/{year}/{pipeline}/signal*/{reco_category}_mStop_vs_mChiRatio.pklz4",
    config="resources/new_opt/config_{reco_category}_{pipeline}.yaml",
    ndim=2,
    pipelines=["Signal312", "Signal313"],
)

PROFILE_1D = Profile(
    background="smoothed_1d_verycomp_sel/{year}/{pipeline}/{reco_category}/{reco_category}_{toy_index}.pklz4",
    signal="filtered_signals_1d_verycomp_pt_cuts/{year}/{pipeline}/signal*/comp_mStop.pklz4",  # only comp has 1D inputs
    config="resources/new_opt_1d/config_{reco_category}_{pipeline}.yaml",
    ndim=1,
    pipelines=["*VeryComp"],
    reco_category="comp",  # comp templates are used for verycomp points
)
EXTRA_1D_POINTS: set[tuple[int, int]] = set()
def isCompressed(p: Point) -> bool:
    return p.mstop >= 1500 and (p.mstop - p.mchi < 100)


def isForced1D(p: Point) -> bool:
    return isCompressed(p) or (p.mstop, p.mchi) in EXTRA_1D_POINTS


def choose(p: Point) -> list[str]:
    if isForced1D(p):
        return ["1d"]
    return []


PLAN = FitPlan(profiles={"2d": PROFILE_2D, "1d": PROFILE_1D},
    default="2d",
    choose=choose,
    )
