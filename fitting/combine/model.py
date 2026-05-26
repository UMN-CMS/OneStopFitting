from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union

import attrs
import numpy as np
import uproot

from ..utils import formatLines

logger = logging.getLogger(__name__)


@attrs.define
class ShapeEffect:
    up: np.ndarray
    down: np.ndarray

    @property
    def distribution(self) -> str:
        return "shape"


@attrs.define
class RateEffect:
    distribution: str
    value: str

    @classmethod
    def lnN(cls, value: float) -> RateEffect:
        return cls(distribution="lnN", value=f"{value:.4f}")


SystematicVariation = Union[ShapeEffect, RateEffect]


@attrs.define
class SystematicEffect:
    name: str
    effect: SystematicVariation


@attrs.define
class ProcessModel:
    name: str
    index: int
    nominal: np.ndarray
    systematics: list[SystematicEffect] = attrs.Factory(list)

    @property
    def rate(self) -> float:
        return float(np.sum(self.nominal))

    def addShape(self, name: str, up: np.ndarray, down: np.ndarray) -> None:
        self.systematics.append(
            SystematicEffect(name=name, effect=ShapeEffect(up=up, down=down))
        )

    def addRate(self, name: str, distribution: str, value: str) -> None:
        self.systematics.append(
            SystematicEffect(
                name=name, effect=RateEffect(distribution=distribution, value=value)
            )
        )


@attrs.define
class RateParam:
    channel: str
    process: str
    init_value: float
    bounds: list[float]


@attrs.define
class ChannelModel:
    name: str
    data_obs: np.ndarray
    processes: list[ProcessModel]
    use_auto_mc_stats: bool = False

    @property
    def nbins(self) -> int:
        return len(self.data_obs)


def computeShapeMetrics(nominal: np.ndarray, up: np.ndarray, down: np.ndarray) -> str:
    mask = nominal > 0
    if not np.any(mask):
        return "0.0% [0.0%, 0.0%]"

    rel_up = np.zeros_like(nominal)
    rel_down = np.zeros_like(nominal)

    rel_up[mask] = (up[mask] - nominal[mask]) / nominal[mask]
    rel_down[mask] = (down[mask] - nominal[mask]) / nominal[mask]

    all_changes = np.concatenate([rel_up[mask], rel_down[mask]])
    if len(all_changes) == 0:
        return "0.0% [0.0%, 0.0%]"

    median_dev = np.median(np.abs(all_changes))
    min_change = np.min(all_changes)
    max_change = np.max(all_changes)

    return (
        f"{median_dev * 100:.1f}% [{min_change * 100:+.1f}%, {max_change * 100:+.1f}%]"
    )


@attrs.define
class CombineModel:
    channels: list[ChannelModel]
    rate_params: list[RateParam] = attrs.Factory(list)
    shapes_file: str = "shapes.root"

    def _collectSystematics(self) -> OrderedDict[str, str]:
        """Collect unique systematic names and their distribution types."""
        syst_order: OrderedDict[str, str] = OrderedDict()
        for ch in self.channels:
            for proc in ch.processes:
                for se in proc.systematics:
                    if se.name not in syst_order:
                        syst_order[se.name] = se.effect.distribution
        return syst_order

    def _systematicValue(self, proc: ProcessModel, syst_name: str) -> str:
        for se in proc.systematics:
            if se.name == syst_name:
                if isinstance(se.effect, ShapeEffect):
                    return "1"
                return se.effect.value
        return "-"

    def writeShapes(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for ch in self.channels:
            edges = np.arange(ch.nbins + 1, dtype=float)
            histograms["data_obs"] = (np.asarray(ch.data_obs), edges)
            for proc in ch.processes:
                histograms[proc.name] = (np.asarray(proc.nominal), edges)

                for se in proc.systematics:
                    if isinstance(se.effect, ShapeEffect):
                        histograms[f"{proc.name}_{se.name}Up"] = (
                            np.asarray(se.effect.up),
                            edges,
                        )
                        histograms[f"{proc.name}_{se.name}Down"] = (
                            np.asarray(se.effect.down),
                            edges,
                        )

        with uproot.recreate(path) as f:
            for name, data in histograms.items():
                f[name] = data

        logger.info(f"Exported {len(histograms)} histograms to {path}")

    def renderDatacard(self) -> str:
        lines = []

        lines.append("imax * # number of channels")
        lines.append("jmax * # number of backgrounds")
        lines.append("kmax * # number of nuisance parameters")
        lines.append("-" * 60)

        shape_rows = []
        for ch in self.channels:
            shape_rows.append(
                [
                    "shapes",
                    "*",
                    ch.name,
                    self.shapes_file,
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
            obs_rows[1].append(str(float(np.sum(ch.data_obs))))
        lines.extend(formatLines(obs_rows))
        lines.append("-" * 60)

        proc_rows = [["bin", ""], ["process", ""], ["process", ""], ["rate", ""]]
        for ch in self.channels:
            for proc in ch.processes:
                proc_rows[0].append(ch.name)
                proc_rows[1].append(proc.name)
                proc_rows[2].append(str(proc.index))
                proc_rows[3].append(f"{proc.rate:.6g}")

        syst_map = self._collectSystematics()
        syst_rows = []
        for syst_name, distribution in syst_map.items():
            row = [syst_name, distribution]
            for ch in self.channels:
                for proc in ch.processes:
                    row.append(self._systematicValue(proc, syst_name))
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

    def write(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.writeShapes(out_dir / self.shapes_file)
        (out_dir / "datacard.txt").write_text(self.renderDatacard())
        logger.info(f"Wrote datacard to {out_dir / 'datacard.txt'}")

    def getSystematicsSummary(self) -> tuple[list[str], list[list[str]]]:
        headers = ["Systematic", "Type"]
        col_keys = []
        for ch in self.channels:
            for proc in ch.processes:
                headers.append(f"{ch.name}_{proc.name}")
                col_keys.append((ch, proc))

        syst_map = self._collectSystematics()
        rows = []
        for syst_name, distribution in syst_map.items():
            row = [syst_name, distribution]
            for ch, proc in col_keys:
                effect = None
                for se in proc.systematics:
                    if se.name == syst_name:
                        effect = se.effect
                        break
                if effect is None:
                    row.append("-")
                elif isinstance(effect, RateEffect):
                    row.append(effect.value)
                elif isinstance(effect, ShapeEffect):
                    row.append(
                        computeShapeMetrics(proc.nominal, effect.up, effect.down)
                    )
                else:
                    row.append("-")
            rows.append(row)

        return headers, rows

    def renderSystematicsTable(self, format: str = "csv") -> str:
        format_type = format.lower()
        if format_type not in ["csv", "latex"]:
            raise ValueError(f"Unknown format: {format}")

        headers, rows = self.getSystematicsSummary()

        if format_type == "csv":
            lines = []
            lines.append(",".join(headers))
            for row in rows:
                lines.append(",".join(row))
            return "\n".join(lines) + "\n"

        elif format_type == "latex":
            escaped_headers = [
                h.replace("_", r"\_").replace("%", r"\%") for h in headers
            ]
            aligns = "l" * 2 + "c" * (len(headers) - 2)
            lines = []
            lines.append(r"\begin{tabular}{" + aligns + r"}")
            lines.append(r"\hline")
            lines.append(" & ".join(escaped_headers) + r" \\")
            lines.append(r"\hline")
            for row in rows:
                escaped_row = [
                    cell.replace("_", r"\_").replace("%", r"\%") for cell in row
                ]
                lines.append(" & ".join(escaped_row) + r" \\")
            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            return "\n".join(lines) + "\n"
