from __future__ import annotations

import glob
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import attrs
from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)

PLOT_REGISTRY: dict[int, dict[str, str]] = {
    1: {
        "summary": "summary_plot",
        "ppc": "ppc_percentile_bands",
        "pulls": "global_total_pulls_hist",
    },
    2: {
        "signal": "signal_template*",
        "observed": "observed_outputs",
        "gpr": "gpr_mean",
        "predvar": "predicted_variances",
        "pulls": "pull_map",
        "covar_center": "kernel_at_blind_center",
        "rel_unc": "relative_uncertainty",
        "total_pulls": "total_pulls_hist",
        "ppc_blinded": "ppc_dist_chi2_blinded",
        "ppc_unblinded": "ppc_dist_chi2_unblinded",
    },
}

SPOT_CHECK_THRESHOLDS = {
    "limit_vs_unc_ratio": 5.0,
    "ppc_pvalue_low": 0.01,
    "ppc_pvalue_high": 0.99,
    "chi2_per_bin_high": 3.0,
}


@attrs.define(frozen=True)
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


def _resolveImagePaths(
    diagnostics_dir: Path, ndim: int, image_format: str, version: str = ""
) -> dict[str, str]:
    plot_names = PLOT_REGISTRY.get(ndim)
    if plot_names is None:
        raise NotImplementedError(f"Unsupported ndim={ndim}")

    plot_names = {k: v + version for k, v in plot_names.items()}
    image_paths: dict[str, str] = {}

    for role, plot_name in plot_names.items():
        if "*" in plot_name:
            matches = list(
                diagnostics_dir.glob(
                    str(Path(plot_name).with_suffix("." + image_format))
                )
            )
            if not matches:
                raise FileNotFoundError(
                    f"No file matching {plot_name}.{image_format} in {diagnostics_dir}"
                )
            img_path = matches[0]
        else:
            img_path = (diagnostics_dir / f"{plot_name}.{image_format}").resolve()
        _requireFile(img_path)
        image_paths[role] = str(img_path)

    return image_paths


def _gatherPostCombineImages(
    diagnostics_dir: Path, image_format: str, image_paths: dict[str, str]
) -> None:
    post_combine_dir = diagnostics_dir / "post_combine"
    if not (post_combine_dir.exists() and post_combine_dir.is_dir()):
        return

    valid_suffixes = {"png", "pdf", image_format}
    for f in post_combine_dir.glob("*"):
        if f.suffix[1:] in valid_suffixes:
            image_paths[f.stem] = str(f.resolve())

    combine_keys = [k for k in image_paths if k.startswith("fit_diagnostic_")]
    show_sig_keys = [k for k in combine_keys if "show_signal" in k]
    no_sig_keys = [k for k in combine_keys if "show_signal" not in k]

    if no_sig_keys:
        image_paths["combine_fit_diagnostic"] = image_paths[no_sig_keys[0]]
    if show_sig_keys:
        image_paths["combine_fit_diagnostic_show_signal"] = image_paths[
            show_sig_keys[0]
        ]
    if "gof_test" in image_paths:
        image_paths["combine_gof_test"] = image_paths["gof_test"]
    if "likelihood_scan" in image_paths:
        image_paths["combine_likelihood_scan"] = image_paths["likelihood_scan"]
    if "nuisance_pulls" in image_paths:
        image_paths["combine_nuisance_pulls"] = image_paths["nuisance_pulls"]


def _computeExpectedSignificances(combine: dict[str, Any]) -> dict[str, Any]:
    """Derive expected significances from limit bands where available."""
    limits = combine.get("limits", {})
    result: dict[str, Any] = {}

    expected = limits.get("expected")
    if expected is not None and expected > 0:
        result["expected_exclusion"] = expected

    observed = limits.get("observed")
    if observed is not None and observed > 0:
        result["observed_exclusion"] = observed

    significance = combine.get("significance")
    if significance is not None:
        result["significance"] = significance

    # Derive expected significance from observed limit if available
    # For asymptotic CLs, significance ~ 1/limit for signal strength exclusion
    if expected is not None and expected > 0:
        result["expected_sensitivity"] = 1.0 / expected

    tree_sb = combine.get("tree_fit_sb", {})
    if tree_sb:
        result["fitted_signal_strength"] = tree_sb.get("r")
        result["fitted_signal_strength_err"] = tree_sb.get("r_err")

    tree_b = combine.get("tree_fit_b", {})
    if tree_b:
        result["bonly_signal_strength"] = tree_b.get("r")

    mdf = combine.get("multidim_fit", {})
    if mdf:
        result["multidim_r"] = mdf.get("r")
        result["multidim_r_err"] = mdf.get("r_err")

    ls = combine.get("likelihood_scan_best_r")
    if ls is not None:
        result["likelihood_scan_best_r"] = ls

    return result


def _runSpotChecks(
    combine: dict[str, Any], metrics: dict[str, Any], ppc: dict[str, Any]
) -> list[dict[str, Any]]:
    """Run consistency checks and return a list of flags with severity."""
    flags: list[dict[str, Any]] = []

    limits = combine.get("limits", {})
    expected = limits.get("expected", 0)
    observed = limits.get("observed", 0)

    # Check: observed limit far from expected band
    if expected > 0 and observed > 0:
        exp_lo = limits.get("expected_minus_2sigma", expected)
        exp_hi = limits.get("expected_plus_2sigma", expected)
        if observed < exp_lo * 0.5:
            flags.append({
                "check": "observed_limit_below_band",
                "severity": "warning",
                "detail": f"Observed limit ({observed:.3g}) far below expected "
                          f"-2σ band ({exp_lo:.3g})",
            })
        elif observed > exp_hi * 2.0:
            flags.append({
                "check": "observed_limit_above_band",
                "severity": "warning",
                "detail": f"Observed limit ({observed:.3g}) far above expected "
                          f"+2σ band ({exp_hi:.3g})",
            })

    # Check: expected limit extremely large (unconstrained)
    if expected > SPOT_CHECK_THRESHOLDS["limit_vs_unc_ratio"]:
        flags.append({
            "check": "weak_expected_limit",
            "severity": "info",
            "detail": f"Expected limit ({expected:.3g}) is large, "
                      f"indicating weak sensitivity",
        })

    # Check: PPC p-values
    chi2_stats = ppc.get("test_stats", {}).get("chi2", {})
    for region_name in ["all", "blinded", "unblinded"]:
        region_data = chi2_stats.get(region_name, {})
        pval = region_data.get("pvalue")
        if pval is None:
            continue
        if pval < SPOT_CHECK_THRESHOLDS["ppc_pvalue_low"]:
            flags.append({
                "check": f"ppc_pvalue_low_{region_name}",
                "severity": "warning",
                "detail": f"PPC p-value ({region_name}) = {pval:.4f} < "
                          f"{SPOT_CHECK_THRESHOLDS['ppc_pvalue_low']}",
            })
        elif pval > SPOT_CHECK_THRESHOLDS["ppc_pvalue_high"]:
            flags.append({
                "check": f"ppc_pvalue_high_{region_name}",
                "severity": "info",
                "detail": f"PPC p-value ({region_name}) = {pval:.4f} is unusually high",
            })

    # Check: χ²/bin
    for key in ["global_chi2_per_bin", "blinded_chi2_per_bin"]:
        val = metrics.get(key)
        if val is not None and val > SPOT_CHECK_THRESHOLDS["chi2_per_bin_high"]:
            flags.append({
                "check": f"high_{key}",
                "severity": "warning",
                "detail": f"{key} = {val:.3g} exceeds threshold "
                          f"{SPOT_CHECK_THRESHOLDS['chi2_per_bin_high']}",
            })

    # Check: GOF p-value
    gof_p = combine.get("gof_p_value")
    if gof_p is not None:
        if gof_p < SPOT_CHECK_THRESHOLDS["ppc_pvalue_low"]:
            flags.append({
                "check": "gof_pvalue_low",
                "severity": "warning",
                "detail": f"GOF p-value = {gof_p:.4f} indicates poor fit",
            })

    # Check: fitted signal strength consistency with injection
    tree_sb = combine.get("tree_fit_sb", {})
    r_fit = tree_sb.get("r")
    r_err = tree_sb.get("r_err")
    if r_fit is not None and r_err is not None and r_err > 0:
        # For blinded analysis with injection_rate=0, r should be consistent with 0
        if abs(r_fit) > 3 * r_err:
            flags.append({
                "check": "large_signal_strength",
                "severity": "info",
                "detail": f"Fitted r = {r_fit:.3g} ± {r_err:.3g} "
                          f"({abs(r_fit/r_err):.1f}σ from 0)",
            })

    return flags


def gatherPointContext(
    *, point_dir: Path, image_format: str, version=""
) -> dict[str, Any]:
    point_dir = Path(point_dir)
    summary_path = point_dir / "summary.json"
    diagnostics_dir = point_dir / "diagnostics"

    _requireFile(summary_path)
    _requireFile(diagnostics_dir)

    summary = _readJson(summary_path)
    ndim = _inferNdim(summary, diagnostics_dir, image_format)

    image_paths = _resolveImagePaths(diagnostics_dir, ndim, image_format, version)
    _gatherPostCombineImages(diagnostics_dir, image_format, image_paths)

    combine_slices = []
    slice_dir = diagnostics_dir / "post_combine" / "slice"
    if slice_dir.exists() and slice_dir.is_dir():
        for f in sorted(slice_dir.glob(f"*.{image_format}")):
            combine_slices.append(str(f.resolve()))

    combine = summary.get("combine", {})
    metrics = summary.get("metrics", {})
    ppc = summary.get("ppc", {})

    combine_extra = _computeExpectedSignificances(combine) if combine else {}
    spot_checks = _runSpotChecks(combine, metrics, ppc)

    return {
        "point_dir": str(point_dir.relative_to(Path.cwd())),
        "ndim": ndim,
        "metadata": summary.get("metadata", {}),
        "config": summary.get("config", {}),
        "metrics": metrics,
        "training": summary.get("training", {}),
        "ppc": ppc,
        "combine": combine,
        "combine_extra": combine_extra,
        "combine_slices": combine_slices,
        "spot_checks": spot_checks,
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

    def _fmtSigma(value):
        """Format with ± display."""
        if value is None:
            return "---"
        return f"{value:.4g}"

    def _fmtSafe(value, fmt="%.4g"):
        """Safe format that handles None."""
        if value is None:
            return "---"
        try:
            return fmt % value
        except (TypeError, ValueError):
            return str(value)

    env.filters["fmtSigma"] = _fmtSigma
    env.filters["fmtSafe"] = _fmtSafe

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
    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    build_dir = (output_pdf.parent / "_latex_build").resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    tex_path = build_dir / "point_report.tex"
    tex_path.write_text(latex_source)

    if shutil.which(latex_engine) is None:
        script_path = build_dir / "build_report.sh"
        script_content = f"""#!/bin/bash
set -e
export TEXMFOUTPUT="{build_dir}"
for i in {{1..2}}; do
    {latex_engine} -interaction=nonstopmode -halt-on-error -output-directory "{build_dir}" "{tex_path}"
done
cp "{build_dir}/point_report.pdf" "{output_pdf}"
echo "Built PDF: {output_pdf}"
"""
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        logger.warning(
            f"LaTeX engine '{latex_engine}' not found. Skipping PDF build. "
            f"A bash script to build the PDF manually has been created at {script_path}"
        )
        return

    cmd = [
        latex_engine,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(build_dir),
        str(tex_path),
    ]

    logger.info(f"Building PDF from LaTeX: {output_pdf}")
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

        points = []
        for point_dir in point_dirs:
            try:
                points.append(
                    gatherPointContext(
                        point_dir=point_dir, image_format=config.image_format
                    )
                )
            except Exception as e:
                logger.error(f"Failed to gather context for point {point_dir}: {e}")
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
            out_pdf = Path(point_dir) / "point_report.pdf"
        else:
            out_dir = Path(output)
            if out_dir.suffix.lower() == ".pdf":
                raise ValueError("For per-point reports, --output must be a directory.")
            out_pdf = out_dir / point_dir.name / "point_report.pdf"
        output_paths.append(
            generatePointReport(point_dir=point_dir, output_pdf=out_pdf, config=config)
        )

    return output_paths
