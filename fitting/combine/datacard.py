from __future__ import annotations

import logging
from pathlib import Path

import attrs
from fitting.utils import formatLines

logger = logging.getLogger(__name__)


@attrs.define
class Process:
    name: str
    rate: float
    index: int


@attrs.define
class Systematic:
    name: str
    distribution: str
    values: dict[str, str]


@attrs.define
class Channel:
    name: str
    observation: float
    processes: list[Process]
    shapes_file: str | None = None
    use_auto_mc_stats: bool = False


@attrs.define
class RateParam:
    channel: str
    process: str
    init_value: float
    bounds: list[float]


@attrs.define
class DataCard:
    channels: list[Channel]
    systematics: list[Systematic] = attrs.Factory(list)
    rate_params: list[RateParam] = attrs.Factory(list)

    def render(self) -> str:
        lines = []

        lines.append("imax * # number of channels")
        lines.append("jmax * # number of backgrounds")
        lines.append("kmax * # number of nuisance parameters")
        lines.append("-" * 60)

        shape_rows = []
        for ch in self.channels:
            if ch.shapes_file:
                shape_rows.append(
                    [
                        "shapes",
                        "*",
                        ch.name,
                        ch.shapes_file,
                        "$PROCESS",
                        "$PROCESS_$SYSTEMATIC",
                    ]
                )
        if shape_rows:
            lines.extend(formatLines(shape_rows))
            lines.append("-" * 60)

        obs_rows = [["bin", ""], ["observation", ""]]
        for ch in self.channels:
            obs_rows[0].append(ch.name)
            obs_rows[1].append(str(ch.observation))
        lines.extend(formatLines(obs_rows))
        lines.append("-" * 60)

        proc_rows = [["bin", ""], ["process", ""], ["process", ""], ["rate", ""]]
        for ch in self.channels:
            for proc in ch.processes:
                proc_rows[0].append(ch.name)
                proc_rows[1].append(proc.name)
                proc_rows[2].append(str(proc.index))
                proc_rows[3].append(f"{proc.rate:.6g}")

        syst_rows = []
        for syst in self.systematics:
            row = [syst.name, syst.distribution]
            for ch in self.channels:
                for proc in ch.processes:
                    val = syst.values.get(proc.name, "-")
                    row.append(str(val))
            syst_rows.append(row)

        formatted_rows = formatLines(proc_rows + syst_rows)
        lines.extend(formatted_rows[: len(proc_rows)])
        lines.append("-" * 60)

        if syst_rows:
            lines.extend(formatted_rows[len(proc_rows) :])
            lines.append("-" * 60)

        for rp in self.rate_params:
            bounds_str = f"[{rp.bounds[0]:.4f},{rp.bounds[1]:.4f}]"
            lines.append(
                f"{rp.channel} rateParam {rp.process} {rp.init_value:.4f} {bounds_str}"
            )

        for ch in self.channels:
            if ch.use_auto_mc_stats:
                lines.append(f"{ch.name} autoMCStats 0 1")

        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.render())
        logger.info(f"Wrote datacard to {path}")
