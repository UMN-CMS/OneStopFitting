from __future__ import annotations

import glob
from rich import print
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PointReportConfig:
    latex_engine: str = "pdflatex"
    keep_build: bool = False
    keep_tex: bool = False
    image_format: str = "png"


def discoverPointDirs(inputs: Iterable[str | Path]) -> list[Path]:
    point_dirs: set[Path] = set()

    for inp in inputs:
        inp_str = str(inp)
        inp_path = Path(inp_str)

        patterns: list[str]
        if inp_path.exists() and inp_path.is_dir():
            patterns = [str(inp_path / "**" / "summary.json")]
        else:
            patterns = [inp_str]

        for pattern in patterns:
            for match in glob.glob(pattern, recursive=True):
                match_path = Path(match)
                if match_path.is_dir():
                    for summary_path in glob.glob(
                        str(match_path / "**" / "summary.json"), recursive=True
                    ):
                        point_dirs.add(Path(summary_path).parent.resolve())
                else:
                    point_dirs.add(match_path.parent.resolve())

    return sorted(point_dirs)


def _readJson(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object.")
    return data


def _requireFile(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required asset: {path}")


def _inferNdim(
    summary: dict[str, Any], diagnostics_dir: Path, image_format: str
) -> int:
    meta = summary.get("metadata", {})
    name = str(meta.get("name", ""))
    if "vs" in name:
        return 2
    if (diagnostics_dir / f"pull_map.{image_format}").exists():
        return 2
    return 1


def _selectPlotNames(
    *, ndim: int, diagnostics_dir: Path, image_format: str
) -> dict[str, str]:
    if ndim == 1:
        return {
            "summary": "summary_plot",
            "ppc": "ppc_percentile_bands",
            "pulls": "global_total_pulls_hist",
        }

    if ndim == 2:
        plot_names = {
            "signal": "signal_template",
            "observed": "observed_outputs",
            "gpr": "gpr_mean",
            "pulls": "pull_map",
            "ppc_map": "ppc_mean_map",
            "ppc_blinded": "ppc_dist_chi2_blinded",
            "ppc_unblinded": "ppc_dist_chi2_unblinded",
        }

        signal_path = diagnostics_dir / f"{plot_names['signal']}.{image_format}"
        if not signal_path.exists():
            injected_path = diagnostics_dir / f"injected_signal.{image_format}"
            if injected_path.exists():
                plot_names["signal"] = "injected_signal"

        return plot_names

    raise NotImplementedError(f"Unsupported ndim={ndim}")


def gatherPointContext(*, point_dir: Path, image_format: str) -> dict[str, Any]:
    point_dir = Path(point_dir)
    summary_path = point_dir / "summary.json"
    diagnostics_dir = point_dir / "diagnostics"

    _requireFile(summary_path)
    _requireFile(diagnostics_dir)

    summary = _readJson(summary_path)
    ndim = _inferNdim(summary, diagnostics_dir, image_format)
    plot_names = _selectPlotNames(
        ndim=ndim, diagnostics_dir=diagnostics_dir, image_format=image_format
    )

    image_paths: dict[str, str] = {}
    for role, plot_name in plot_names.items():
        img_path = (diagnostics_dir / f"{plot_name}.{image_format}").resolve()
        _requireFile(img_path)
        image_paths[role] = str(img_path)

    return {
        "point_dir": str(point_dir.resolve()),
        "ndim": ndim,
        "metadata": summary.get("metadata", {}),
        "metrics": summary.get("metrics", {}),
        "training": summary.get("training", {}),
        "ppc": summary.get("ppc", {}),
        "image_paths": image_paths,
        "image_format": image_format,
    }


def renderLatex(
    *, template_dir: Path, template_name: str, context: dict[str, Any]
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        block_start_string="[-",
        block_end_string="-]",
        variable_start_string="[[",
        variable_end_string="]]",
        comment_start_string="%[",
        comment_end_string="%]",
        line_comment_prefix="%%[",
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_name)
    return template.render(**context)


def buildPdfFromLatex(
    *,
    latex_source: str,
    output_pdf: Path,
    latex_engine: str,
    keep_build: bool,
    keep_tex: bool,
) -> None:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_dir = output_pdf.parent / "_latex_build"
    build_dir.mkdir(parents=True, exist_ok=True)

    tex_path = build_dir / "point_report.tex"
    tex_path.write_text(latex_source)

    cmd = [
        latex_engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(build_dir),
        str(tex_path),
    ]

    try:
        for _ in range(2):
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "TEXMFOUTPUT": str(build_dir)},
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stdout)

        built_pdf = build_dir / "point_report.pdf"
        if not built_pdf.exists():
            raise FileNotFoundError("LaTeX build did not produce point_report.pdf")
        shutil.copy2(built_pdf, output_pdf)
    finally:
        if not keep_tex:
            try:
                tex_path.unlink(missing_ok=True)
            except Exception:
                pass


def generatePointReport(
    *,
    point_dir: Path,
    output_pdf: Path | None,
    config: PointReportConfig,
) -> Path:
    if output_pdf is None:
        output_pdf = Path(point_dir) / "report" / "point_report.pdf"

    template_dir = Path(__file__).parent / "templates"
    context = gatherPointContext(
        point_dir=Path(point_dir), image_format=config.image_format
    )
    latex_source = renderLatex(
        template_dir=template_dir, template_name="point_report.tex.j2", context=context
    )
    buildPdfFromLatex(
        latex_source=latex_source,
        output_pdf=Path(output_pdf),
        latex_engine=config.latex_engine,
        keep_build=config.keep_build,
        keep_tex=config.keep_tex,
    )
    logger.info(f"Wrote report: {output_pdf}")
    return Path(output_pdf)


def generatePointReports(
    *,
    inputs: Iterable[str | Path],
    output: Path | None,
    single_document: bool,
    config: PointReportConfig,
) -> list[Path]:
    point_dirs = discoverPointDirs(inputs)
    if not point_dirs:
        raise ValueError("No point directories discovered from inputs.")

    template_dir = Path(__file__).parent / "templates"

    if single_document:
        output_pdf = Path("point_reports.pdf") if output is None else Path(output)
        if output_pdf.suffix.lower() != ".pdf":
            output_pdf = output_pdf / "point_reports.pdf"

        points = [
            gatherPointContext(point_dir=point_dir, image_format=config.image_format)
            for point_dir in point_dirs
        ]
        latex_source = renderLatex(
            template_dir=template_dir,
            template_name="points_report.tex.j2",
            context={"points": points},
        )
        buildPdfFromLatex(
            latex_source=latex_source,
            output_pdf=output_pdf,
            latex_engine=config.latex_engine,
            keep_build=config.keep_build,
            keep_tex=config.keep_tex,
        )
        logger.info(f"Wrote combined report: {output_pdf}")
        return [output_pdf]

    output_paths = []
    for point_dir in point_dirs:
        if output is None:
            out_pdf = None
        else:
            out_dir = Path(output)
            if out_dir.suffix.lower() == ".pdf":
                raise ValueError("For per-point reports, --output must be a directory.")
            out_pdf = out_dir / point_dir.name / "point_report.pdf"
        output_paths.append(
            generatePointReport(point_dir=point_dir, output_pdf=out_pdf, config=config)
        )

    return output_paths
