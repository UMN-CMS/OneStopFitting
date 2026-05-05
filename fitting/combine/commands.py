from __future__ import annotations

from abc import ABC, abstractmethod

import attrs


@attrs.define
class CombineContext:
    signal_labels: list[str]
    channel_name: str
    r_min: float = -20
    r_max: float = 20

    @property
    def isMultiSignal(self) -> bool:
        return len(self.signal_labels) > 1

    def physicsModelArgs(self) -> str:
        if not self.isMultiSignal:
            return ""
        model = "HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"
        po_maps = []
        for i, lbl in enumerate(self.signal_labels):
            if i == 0:
                po_maps.append(f"'map=.*/{lbl}:r[1,{self.r_min},{self.r_max}]'")
            else:
                po_maps.append(f"'map=.*/{lbl}:r'")
        po_flags = " ".join(f"--PO {po}" for po in ["verbose"] + po_maps)
        return f"-P {model} {po_flags}"


@attrs.define
class CombineCommand(ABC):
    @abstractmethod
    def render(self, ctx: CombineContext) -> list[str]: ...


@attrs.define
class Text2Workspace(CombineCommand):
    def render(self, ctx):
        extra = ctx.physicsModelArgs()
        return [f"text2workspace.py datacard.txt -m 125 {extra}".strip()]


@attrs.define
class AsymptoticLimits(CombineCommand):
    def render(self, ctx):
        return [
            f"combine -M AsymptoticLimits -d datacard.root"
            f" --rMin={ctx.r_min} --rMax={ctx.r_max}"
        ]


@attrs.define
class FitDiagnostics(CombineCommand):
    def render(self, ctx):
        return [
            f"combine -M FitDiagnostics -d datacard.root"
            f" --saveShapes --saveNormalizations"
            f" --rMin={ctx.r_min} --rMax={ctx.r_max}"
        ]


@attrs.define
class MultiDimFit(CombineCommand):
    points: int = 100

    def render(self, ctx):
        base = (
            f"combine -M MultiDimFit -d datacard.root"
            f" --rMin={ctx.r_min} --rMax={ctx.r_max}"
        )
        return [
            f"{base} -n .mdimnon --algo singles",
            f"{base} -n .mdimgrid --algo grid --points {self.points}",
            f"{base} -n .mdimgridfreeze --algo grid --points {self.points}"
            f" --freezeParameters allConstrainedNuisances",
        ]


@attrs.define
class Significance(CombineCommand):
    def render(self, ctx):
        return [
            f"combine -M Significance -d datacard.root"
            f" --rMin={ctx.r_min} --rMax={ctx.r_max}"
        ]


@attrs.define
class GoodnessOfFit(CombineCommand):
    algorithm: str = "saturated"
    num_toys: int = 200

    def render(self, ctx):
        base = f"combine -M GoodnessOfFit --algorithm {self.algorithm} -d datacard.root"
        return [
            f"{base} --name gof_{self.algorithm}",
            f"{base} --name gof_{self.algorithm}_toys --toys {self.num_toys}",
        ]


@attrs.define
class Impacts(CombineCommand):
    def render(self, ctx):
        return [
            "combineTool.py -M Impacts -d datacard.root -m 125 --doInitialFit --robustFit 1",
            "combineTool.py -M Impacts -d datacard.root -m 125 --robustFit 1 --doFits",
            "combineTool.py -M Impacts -d datacard.root -m 125 -o impacts.json",
            "plotImpacts.py -i impacts.json -o impacts",
        ]


@attrs.define
class RawCommand(CombineCommand):
    command: str = ""

    def render(self, ctx):
        return [self.command]


COMMAND_REGISTRY: dict[str, type[CombineCommand]] = {
    "limits": AsymptoticLimits,
    "fit-diagnostics": FitDiagnostics,
    "multidimfit": MultiDimFit,
    "significance": Significance,
    "gof-saturated": GoodnessOfFit,
    "gof-ks": GoodnessOfFit,
    "gof-ad": GoodnessOfFit,
    "impacts": Impacts,
}

_COMMAND_DEFAULTS: dict[str, dict] = {
    "gof-saturated": {"algorithm": "saturated"},
    "gof-ks": {"algorithm": "KS"},
    "gof-ad": {"algorithm": "AD"},
}


def resolveCommand(name: str) -> CombineCommand:
    if name in COMMAND_REGISTRY:
        cls = COMMAND_REGISTRY[name]
        kwargs = _COMMAND_DEFAULTS.get(name, {})
        return cls(**kwargs)
    return RawCommand(command=name)


def resolveCommands(names: list[str]) -> list[CombineCommand]:
    return [resolveCommand(n) for n in names]
