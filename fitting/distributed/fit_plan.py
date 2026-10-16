"""
Per-point routing of fits to different input/config "profiles".

A fit plan is a Python file that defines a module-level ``PLAN``:

    from fitting.distributed.fit_plan import FitPlan, Profile, Point

    def choose(p: Point) -> list[str]:
        if p.mstop >= 1500 and p.mchi >= 0.9 * p.mstop:
            return ["1d"]
        return []

    PLAN = FitPlan(
        profiles={
            "2d": Profile(..., pipelines=["Signal312", "Signal313"]),
            "1d": Profile(..., pipelines=["*VeryComp"], reco_category="comp"),
        },
        default="2d",
        choose=choose,
    )

Profile patterns accept the same placeholders as makesubmit's --signal,
--background and --config-pattern. ``choose`` is called once per physical point
(era x coupling x mstop x mchi x toy) and returns profile names, best first. By
default the point gets exactly one fit: the first name, or the default profile
if none is returned. With all_matching, every returned name plus the default is
used. ``exclude`` drops a point entirely, in both modes.

Each profile must supply exactly one job per point it is assigned. A profile
restricts its inputs with ``pipelines`` (glob patterns over --pipelines) and
``reco_category`` (the template category to use; by default the one the mass
point belongs to). A point with several candidate jobs for one profile is an
error, never more than one fit.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Callable

import attrs


# Names end up in condor queue columns, '@'-joined era lists and output paths
_PROFILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.]*")


def _asList(v: str | list[str]) -> list[str]:
    return [v] if isinstance(v, str) else list(v)


def _asOptionalList(v: str | list[str] | None) -> list[str] | None:
    return None if v is None else _asList(v)


@attrs.frozen
class Point:
    """A physical point to fit: what choose() and exclude() decide on."""

    mstop: int
    mchi: int
    coupling: str
    # Category the mass point belongs to (utils.getCategory)
    category: str
    year: str
    toy_index: int


@attrs.frozen
class Profile:
    background: str
    signal: list[str] = attrs.field(converter=_asList)
    config: str
    ndim: int | None = None
    # Glob patterns over --pipelines this profile takes inputs from; None means all
    pipelines: list[str] | None = attrs.field(default=None, converter=_asOptionalList)
    # Template category to fit with; None means the point's own category
    reco_category: str | None = None


def _chooseNothing(point: Point) -> list[str]:
    return []

def _excludeNothing(point: Point) -> bool:
    return False


@attrs.frozen
class FitPlan:
    profiles: dict[str, Profile]
    default: str
    choose: Callable[[Point], list[str] | str | None] = _chooseNothing
    # Set by loadFitPlan; only used to point error messages at the plan file
    exclude: Callable[[Point], bool] = _excludeNothing
    source: str | None = attrs.field(default=None, eq=False)

    def __attrs_post_init__(self):
        bad = [n for n in self.profiles if not _PROFILE_NAME.fullmatch(n)]
        if bad:
            raise ValueError(
                f"Invalid profile name(s) {bad}: start with a letter or digit, then use only letters, digits, '_' and '.'"
            )
        if self.default not in self.profiles:
            raise ValueError(
                f"Default profile '{self.default}' not in profiles {list(self.profiles)}"
            )
        if not callable(self.choose):
            raise TypeError(f"FitPlan.choose must be callable, got {type(self.choose)}")

    def assign(self, point: Point, all_matching: bool = False) -> list[str]:
        """Return the profile name(s) a point should be fit with."""
        try:
            chosen = self.choose(point)
        except Exception as e:
            # Keep the user's traceback, just say where it came from
            e.add_note(f"Raised by choose() in fit plan {self.source or '<unknown>'} for {point}")
            raise
        if chosen is None:
            chosen = []
        elif isinstance(chosen, str):
            chosen = [chosen]
        if not isinstance(chosen, (list, tuple)) or not all(
            isinstance(c, str) for c in chosen
        ):
            raise TypeError(
                f"choose() must return a list of profile names, got {chosen!r} for {point}"
            )
        unknown = [c for c in chosen if c not in self.profiles]
        if unknown:
            raise ValueError(
                f"choose() returned unknown profile(s) {unknown} for {point}; "
                f"known profiles: {list(self.profiles)}"
            )
        if self.exclude(point):
            return []
        if not all_matching:
            return [chosen[0] if chosen else self.default]
        return list(dict.fromkeys([*chosen, self.default]))


def loadFitPlan(path: Path | str) -> FitPlan:
    """Import a fit plan file and return its module-level PLAN."""
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"_fit_plan_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import fit plan from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    plan = getattr(module, "PLAN", None)
    if not isinstance(plan, FitPlan):
        raise TypeError(
            f"{path} must define a module-level PLAN = FitPlan(...), got {type(plan)}"
        )
    return attrs.evolve(plan, source=str(path))
