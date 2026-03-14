"""Combine datacard generation.

Defines attrs classes for building Higgs Combine datacards
and writing them to text files.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import attrs

logger = logging.getLogger(__name__)


@attrs.define
class Process:
    """A physics process in the datacard.

    Attributes:
        name: Process name (e.g., "signal", "background").
        rate: Expected event count.
        index: Process index (0 = signal, >0 = background).
    """

    name: str
    rate: float
    index: int


@attrs.define
class Systematic:
    """A systematic uncertainty.

    Attributes:
        name: Systematic name.
        distribution: Distribution type (e.g., "lnN", "shape").
        values: Dict of process_name -> value. Use "-" for no effect.
    """

    name: str
    distribution: str
    values: dict[str, str]


@attrs.define
class Channel:
    """A single channel (bin) in the datacard.

    Attributes:
        name: Channel name.
        observation: Observed event count.
        processes: List of processes in this channel.
        shapes_file: Path to ROOT file with shape histograms.
    """

    name: str
    observation: float
    processes: list[Process]
    shapes_file: str | None = None


@attrs.define
class DataCard:
    """Complete Combine datacard.

    Attributes:
        channels: List of channels.
        systematics: List of systematic uncertainties.
    """

    channels: list[Channel]
    systematics: list[Systematic] = attrs.Factory(list)

    def render(self) -> str:
        """Render the datacard as a string."""
        buf = StringIO()

        # Header
        buf.write("imax * number of channels\n")
        buf.write("jmax * number of backgrounds\n")
        buf.write("kmax * number of nuisance parameters\n")
        buf.write("-" * 60 + "\n")

        # Shapes
        for ch in self.channels:
            if ch.shapes_file:
                buf.write(f"shapes * {ch.name} {ch.shapes_file} $PROCESS $PROCESS_$SYSTEMATIC\n")
        buf.write("-" * 60 + "\n")

        # Observation
        ch_names = [ch.name for ch in self.channels]
        obs_vals = [str(ch.observation) for ch in self.channels]
        buf.write("bin         " + "  ".join(ch_names) + "\n")
        buf.write("observation " + "  ".join(obs_vals) + "\n")
        buf.write("-" * 60 + "\n")

        # Rates
        all_bins = []
        all_procs = []
        all_indices = []
        all_rates = []
        for ch in self.channels:
            for proc in ch.processes:
                all_bins.append(ch.name)
                all_procs.append(proc.name)
                all_indices.append(str(proc.index))
                all_rates.append(f"{proc.rate:.6g}")

        buf.write("bin         " + "  ".join(all_bins) + "\n")
        buf.write("process     " + "  ".join(all_procs) + "\n")
        buf.write("process     " + "  ".join(all_indices) + "\n")
        buf.write("rate        " + "  ".join(all_rates) + "\n")
        buf.write("-" * 60 + "\n")

        # Systematics
        for syst in self.systematics:
            values = []
            for ch in self.channels:
                for proc in ch.processes:
                    val = syst.values.get(proc.name, "-")
                    values.append(val)
            buf.write(f"{syst.name}  {syst.distribution}  " + "  ".join(values) + "\n")

        return buf.getvalue()

    def write(self, path: Path) -> None:
        """Write the datacard to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.render())
        logger.info(f"Wrote datacard to {path}")
