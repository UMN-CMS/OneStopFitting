from __future__ import annotations
from .base import main
from .run import runCmd, resolveOutputCmd
from .diagnostics import (
    printParamsCmd,
    smoothCmd,
    gatherCmd,
    reportCmd,
    harvestCmd,
    windowFitCmd,
    checkDomainCmd,
    mergeSummariesCmd,
)
from .aggregate import aggregateGroup
from .distributed import makesubmitCmd, generateCombineScriptCmd

main.add_command(runCmd, name="run")
main.add_command(printParamsCmd, name="print-params")
main.add_command(smoothCmd, name="smooth")
main.add_command(gatherCmd, name="gather")
main.add_command(aggregateGroup, name="aggregate")
main.add_command(makesubmitCmd, name="makesubmit")
main.add_command(generateCombineScriptCmd, name="generate-combine-script")
main.add_command(reportCmd, name="report")
main.add_command(resolveOutputCmd, name="resolve-output")
main.add_command(harvestCmd, name="harvest")
main.add_command(windowFitCmd, name="window-fit")
main.add_command(checkDomainCmd, name="check-domain")
main.add_command(mergeSummariesCmd, name="merge-summaries")

__all__ = ["main"]
