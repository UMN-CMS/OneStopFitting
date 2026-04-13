from fitting.diagnostics.aggregate_plots import (
    makeAggregateMassPlanePlot,
    AggregatePoint,
)
import mplhep
from pathlib import Path
from fitting.diagnostics.plot_utils import savePlots
import numpy as np
from collections import defaultdict
from rich import print

import json
import click
import cattrs


def toDict(l):
    return {(x.mstop, x.mchi): x.value for x in l}


@click.command()
@click.argument("path")
@click.option("-o", "--output")
def main(path, output):
    mplhep.style.use("CMS")
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "r") as f:
        data = json.load(f)
    data = cattrs.structure(data, list[AggregatePoint])
    groups = defaultdict(list)
    for d in data:
        groups[d.groups["reco_category"]].append(d)
    uncomp = toDict(groups["uncomp"])
    comp = toDict(groups["comp"])

    sunc = set(uncomp)
    scomp = set(comp)
    common = sunc & scomp
    final = [AggregatePoint(*k, uncomp[k] / comp[k]) for k in common]

    for x in sunc ^ scomp:
        if x in sunc:
            final.append(AggregatePoint(*x, 0))
        else:
            final.append(AggregatePoint(*x, 2))

    p = makeAggregateMassPlanePlot(
        final,
        metric_name="Size Uncomp/Comp",
        cmin=0,
        cmax=2,
        cmap="coolwarm",
        name_format="",
    )
    fig, ax = list(p.values())[0]
    ax.set_title("")
    mplhep.cms.label(llabel="Preliminary", year="2018")
    x = np.linspace(900, 2000, 100)
    y_old = 0.6 * x + 150
    y_new = 0.75 * x 
    ax.plot(x, y_old, color="orange", lw=3, label="0.6*x + 150")
    ax.plot(x, y_new, color="green", lw=3, label="0.75*x")
    ax.legend()
    fig.savefig(output)


if __name__ == "__main__":
    main()
