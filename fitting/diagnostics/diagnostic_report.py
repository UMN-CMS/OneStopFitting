from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import attrs
import matplotlib.pyplot as plt
import mplhep
import numpy as np

from ..utils import getCategory
from .aggregate_plots import computeStatistics
from .plot_utils import CMS_COLORS, addCMSBits
from .point_report import buildPdfFromLatex, renderLatex

logger = logging.getLogger(__name__)

plt.style.use(mplhep.style.CMS)


@attrs.define(frozen=True)
class PointDiagnosis:
    mstop: float
    mchi: float
    coupling: str
    category: str
    n_toys: int
    median_r: float
    median_r_err: float
    mean_r: float
    std_r: float
    coverage: float
    gof_pvalues: list[float]
    gof_ks_pvalue: float | None
    gof_median_pvalue: float | None
    gof_verdict: str
    r_values: list[float] = attrs.field(repr=False)
    r_err_values: list[float] = attrs.field(repr=False)


@attrs.define(frozen=True)
class CategorySummary:
    category: str
    n_points: int
    n_toys_total: int
    avg_coverage: float
    median_coverage: float
    avg_median_r: float
    avg_gof_ks_pvalue: float | None
    n_ok: int
    n_conservative: int
    n_dangerous: int


@attrs.define(frozen=True)
class RunConfig:
    pvq: float | None
    wvi: float | None
    signal_pre_scale: float | None
    min_counts: float | None
    lr: float | None
    injection_rate: float | None
    rebin: int | None
    num_iters: int | None
    optimizer: str | None
    variance_floor_quantile: float | None
    kernel_type: str | None
    hidden_shapes: list[int] | None
    activation: str | None
    ard: bool | None
    blinding_strategy: str | None
    window_type: str | None
    core_threshold_fraction: float | None
    dilation_margin: float | None
    smooth_sigma: float | None
    eigenvar_threshold: float | None


@attrs.define
class DiagnosticReport:
    coupling: str
    points: list[PointDiagnosis]
    category_summaries: list[CategorySummary]
    run_config: RunConfig
    overall_coverage: float
    overall_n_toys: int
    n_ok: int
    n_conservative: int
    n_dangerous: int
    all_meta: list[dict] = attrs.field(default=attrs.Factory(list))


def _extractToyDataByCoupling(gathered: list[dict]) -> dict[str, dict[tuple[float, float], list[dict]]]:
    by_coupling: dict[str, dict[tuple[float, float], list[dict]]] = defaultdict(lambda: defaultdict(list))
    for entry in gathered:
        meta = entry.get("metadata", {}).get("other_data", {})
        mstop = meta.get("stop_mass")
        mchi = meta.get("chargino_mass")
        coupling = str(meta.get("coupling", "unknown"))
        if mstop is None or mchi is None:
            continue
        by_coupling[coupling][(float(mstop), float(mchi))].append(entry)
    return {k: dict(v) for k, v in by_coupling.items()}


def _computePointDiagnosis(coupling: str, mstop: float, mchi: float, toys: list[dict]) -> PointDiagnosis:
    r_vals = []
    r_err_vals = []
    gof_pvals = []

    for toy in toys:
        combine = toy.get("combine", {})
        tree_fit = combine.get("tree_fit_sb", {})
        r = tree_fit.get("r")
        r_err = tree_fit.get("r_err")
        gof = combine.get("gof_p_value")

        if r is not None and r_err is not None:
            r_vals.append(float(r))
            r_err_vals.append(float(r_err))
        if gof is not None:
            gof_pvals.append(float(gof))

    r_arr = np.array(r_vals) if r_vals else np.array([0.0])
    r_err_arr = np.array(r_err_vals) if r_err_vals else np.array([1.0])

    median_r = float(np.median(r_arr))
    median_r_err = float(np.median(r_err_arr))
    mean_r = float(np.mean(r_arr))
    std_r = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0
    coverage = float(np.mean(np.abs(r_arr) < r_err_arr))

    gof_stats = computeStatistics(gof_pvals, is_pvalue=True) if gof_pvals else {}
    gof_ks = gof_stats.get("ks_pvalue_uniform")
    gof_verdict = gof_stats.get("pvalue_skew_verdict", "UNKNOWN")
    gof_median = gof_stats.get("median")

    return PointDiagnosis(
        mstop=mstop,
        mchi=mchi,
        coupling=coupling,
        category=getCategory(mstop, mchi),
        n_toys=len(r_vals),
        median_r=median_r,
        median_r_err=median_r_err,
        mean_r=mean_r,
        std_r=std_r,
        coverage=coverage,
        gof_pvalues=gof_pvals,
        gof_ks_pvalue=gof_ks,
        gof_median_pvalue=gof_median,
        gof_verdict=gof_verdict,
        r_values=r_vals,
        r_err_values=r_err_vals,
    )


def _computeCategorySummary(cat_name: str, points: list[PointDiagnosis]) -> CategorySummary:
    coverages = [p.coverage for p in points]
    median_rs = [p.median_r for p in points]
    ks_vals = [p.gof_ks_pvalue for p in points if p.gof_ks_pvalue is not None]

    return CategorySummary(
        category=cat_name,
        n_points=len(points),
        n_toys_total=sum(p.n_toys for p in points),
        avg_coverage=float(np.mean(coverages)),
        median_coverage=float(np.median(coverages)),
        avg_median_r=float(np.mean(median_rs)),
        avg_gof_ks_pvalue=float(np.mean(ks_vals)) if ks_vals else None,
        n_ok=sum(1 for p in points if p.gof_verdict == "OK"),
        n_conservative=sum(1 for p in points if p.gof_verdict == "CONSERVATIVE"),
        n_dangerous=sum(1 for p in points if p.gof_verdict == "DANGEROUS"),
    )


def _extractRunConfig(gathered: list[dict]) -> RunConfig:
    if not gathered:
        return RunConfig(None, None, None, None, None, None, None, None, None,
                         None, None, None, None, None, None, None, None, None, None, None)
    cfg = gathered[0].get("config", {})
    model = cfg.get("model", {})
    likelihood = model.get("likelihood", {})
    kernel = model.get("kernel", {})
    base_kernel = kernel.get("base_kernel_config", {})
    optim = cfg.get("optimization", {})
    window = cfg.get("window", {})
    blind = cfg.get("blinding_strategy", {})
    combine = cfg.get("combine", {})
    return RunConfig(
        pvq=likelihood.get("pad_variance_quantile"),
        wvi=cfg.get("window_variance_inflation"),
        signal_pre_scale=cfg.get("signal_pre_scale"),
        min_counts=cfg.get("min_counts"),
        lr=optim.get("lr"),
        injection_rate=cfg.get("injection_rate"),
        rebin=cfg.get("rebin"),
        num_iters=optim.get("num_iters"),
        optimizer=optim.get("optimizer"),
        variance_floor_quantile=likelihood.get("variance_floor_quantile"),
        kernel_type=kernel.get("_type"),
        hidden_shapes=kernel.get("hidden_shapes"),
        activation=kernel.get("activation"),
        ard=base_kernel.get("ard"),
        blinding_strategy=blind.get("_type"),
        window_type=window.get("_type"),
        core_threshold_fraction=window.get("core_threshold_fraction"),
        dilation_margin=window.get("dilation_margin"),
        smooth_sigma=window.get("smooth_sigma"),
        eigenvar_threshold=combine.get("eigenvar_threshold"),
    )


def computeDiagnostics(gathered: list[dict], coupling: str) -> DiagnosticReport:
    all_by_coupling = _extractToyDataByCoupling(gathered)
    grouped = all_by_coupling.get(coupling, {})

    all_meta = [e.get("metadata", {}) for e in gathered if str(e.get("metadata", {}).get("other_data", {}).get("coupling", "")) == coupling][:1]

    points = []
    for (mstop, mchi), toys in sorted(grouped.items()):
        points.append(_computePointDiagnosis(coupling, mstop, mchi, toys))

    cat_groups: dict[str, list[PointDiagnosis]] = defaultdict(list)
    for p in points:
        cat_groups[p.category].append(p)

    cat_summaries = [
        _computeCategorySummary(name, pts)
        for name, pts in sorted(cat_groups.items())
    ]

    all_coverages = [p.coverage for p in points]
    n_ok = sum(1 for p in points if p.gof_verdict == "OK")
    n_conservative = sum(1 for p in points if p.gof_verdict == "CONSERVATIVE")
    n_dangerous = sum(1 for p in points if p.gof_verdict == "DANGEROUS")

    return DiagnosticReport(
        coupling=coupling,
        points=points,
        category_summaries=cat_summaries,
        run_config=_extractRunConfig(gathered),
        overall_coverage=float(np.mean(all_coverages)) if all_coverages else 0.0,
        overall_n_toys=sum(p.n_toys for p in points),
        n_ok=n_ok,
        n_conservative=n_conservative,
        n_dangerous=n_dangerous,
        all_meta=all_meta,
    )


def _coverageColor(coverage: float) -> str:
    if 0.63 <= coverage <= 0.73:
        return "green"
    if 0.55 <= coverage <= 0.80:
        return "yellow"
    return "red"


def _verdictColor(verdict: str) -> str:
    if verdict == "OK":
        return CMS_COLORS[3]
    if verdict == "CONSERVATIVE":
        return CMS_COLORS[1]
    if verdict == "DANGEROUS":
        return CMS_COLORS[2]
    return "gray"


def _addCMSLabel(ax, report, extra_text=""):
    addCMSBits(
        ax,
        report.all_meta,
        extra_text=extra_text,
        cms_text="Private Work",
    )


CAT_COLORS = {
    "uncomp": CMS_COLORS[0],
    "comp": CMS_COLORS[1],
    "verycomp": CMS_COLORS[2],
}


def plotCoverageMap(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    for point in report.points:
        color = CAT_COLORS.get(point.category, "gray")
        ax.scatter(
            point.mstop, point.mchi,
            c=[_coverageColor(point.coverage)],
            marker="s", s=120, linewidths=0.8, edgecolors=color, zorder=3,
        )

    for cat_name, color in CAT_COLORS.items():
        ax.scatter([], [], c="gray", marker="s", s=80, edgecolors=color, label=cat_name)

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="green", label="63--73%"),
        Patch(facecolor="yellow", label="55--80%"),
        Patch(facecolor="red", label="Outside"),
    ]
    ax.legend(handles=legend_elements, title="Coverage", loc="lower left", fontsize="small",
              framealpha=0.9)
    ax.add_artist(ax.legend(title="Category", loc="upper left", fontsize="small", framealpha=0.9))

    _addCMSLabel(ax, report, f"Coverage: $|r| < r_{{err}}$\nOverall: {report.overall_coverage:.1%}, N={report.overall_n_toys}")
    return fig, ax


def plotMedianRMap(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    xs = [p.mstop for p in report.points]
    ys = [p.mchi for p in report.points]
    vs = [p.median_r for p in report.points]

    sc = ax.scatter(xs, ys, c=vs, cmap="RdBu_r", marker="s", s=120,
                    vmin=-2, vmax=2, linewidths=0.5, edgecolors="black")
    fig.colorbar(sc, ax=ax, label="Median $r$")
    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")
    _addCMSLabel(ax, report, "Median $r$ (bkg-only)")
    return fig, ax


def plotVerdictMap(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    xs = [p.mstop for p in report.points]
    ys = [p.mchi for p in report.points]
    ks_vals = [p.gof_ks_pvalue if p.gof_ks_pvalue is not None else np.nan for p in report.points]
    edge_colors = [CAT_COLORS.get(p.category, "gray") for p in report.points]

    sc = ax.scatter(
        xs, ys,
        c=ks_vals, cmap="RdYlGn", marker="s", s=120,
        vmin=0, vmax=1, linewidths=0.8, edgecolors=edge_colors, zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, label="GOF KS $p$-value")
    cb.ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8)

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")

    from matplotlib.patches import Patch
    cat_handles = [Patch(facecolor="none", edgecolor=c, label=n) for n, c in CAT_COLORS.items()]
    ax.legend(handles=cat_handles, title="Category", loc="lower right", fontsize="small",
              framealpha=0.9)
    _addCMSLabel(ax, report, f"GOF KS $p$-value\n{report.n_ok} OK, {report.n_conservative} CONS, {report.n_dangerous} DANG")
    return fig, ax


def plotCoverageBarByCategory(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    cats = [s.category for s in report.category_summaries]
    coverages = [s.avg_coverage for s in report.category_summaries]
    n_points = [s.n_points for s in report.category_summaries]
    labels = [f"{c}\n(n={n})" for c, n in zip(cats, n_points)]

    bar_colors = [CMS_COLORS[i] for i in range(len(cats))]
    bars = ax.bar(labels, coverages, color=bar_colors)
    ax.axhline(0.68, color="black", linestyle="--", linewidth=1.5, label="Target (68%)")
    ax.axhspan(0.63, 0.73, color=CMS_COLORS[3], alpha=0.2)
    ax.set_ylabel("Average Coverage ($|r| < r_{err}$)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize="small")

    for bar, cov in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{cov:.1%}", ha="center", va="bottom", fontsize=10)
    _addCMSLabel(ax, report, "Coverage by Category")
    return fig, ax


def plotGofVerdictSummary(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    cats = [s.category for s in report.category_summaries]
    ok = [s.n_ok for s in report.category_summaries]
    cons = [s.n_conservative for s in report.category_summaries]
    dang = [s.n_dangerous for s in report.category_summaries]

    x = np.arange(len(cats))
    width = 0.25

    ax.bar(x - width, ok, width, label="OK", color=CMS_COLORS[3])
    ax.bar(x, cons, width, label="CONSERVATIVE", color=CMS_COLORS[1])
    ax.bar(x + width, dang, width, label="DANGEROUS", color=CMS_COLORS[2])

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Number of Points")
    ax.legend(fontsize="small")
    _addCMSLabel(ax, report, "GOF $p$-value Verdict")
    return fig, ax


def plotRvsRErr(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    for point in report.points:
        color = CAT_COLORS.get(point.category, "gray")
        r_arr = np.array(point.r_values)
        r_err_arr = np.array(point.r_err_values)
        normalized = r_arr / r_err_arr
        ax.scatter(
            np.full(len(normalized), point.mstop),
            normalized,
            c=color, alpha=0.15, s=8, zorder=2,
        )

    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(-1, color="black", linestyle="--", linewidth=0.8)
    ax.axhspan(-1, 1, color=CMS_COLORS[3], alpha=0.1)

    for cat_name, color in CAT_COLORS.items():
        ax.scatter([], [], c=color, s=20, label=cat_name)

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$r / r_{err}$")
    ax.legend(fontsize="small")
    _addCMSLabel(ax, report, "Normalized Signal Strength (bkg-only)")
    return fig, ax


def generateDiagnosticPlots(report: DiagnosticReport, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "coverage_map": plotCoverageMap,
        "median_r_map": plotMedianRMap,
        "verdict_map": plotVerdictMap,
        "coverage_by_category": plotCoverageBarByCategory,
        "gof_verdict_summary": plotGofVerdictSummary,
        "r_vs_r_err": plotRvsRErr,
    }

    saved: dict[str, Path] = {}
    for name, func in plots.items():
        fig, ax = func(report)
        path = output_dir / f"{name}.pdf"
        fig.savefig(path)
        plt.close(fig)
        saved[name] = path
        logger.info(f"Saved {path}")

    return saved


def loadGathered(path: str | Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def discoverCouplings(gathered: list[dict]) -> list[str]:
    couplings: set[str] = set()
    for entry in gathered:
        c = entry.get("metadata", {}).get("other_data", {}).get("coupling")
        if c is not None:
            couplings.add(str(c))
    return sorted(couplings)


def generateDiagnosticReport(
    *,
    gathered: list[dict],
    output_pdf: Path,
    coupling: str | None = None,
    latex_engine: str = "pdflatex",
    keep_build: bool = False,
    keep_tex: bool = False,
) -> list[Path]:
    output_pdf = Path(output_pdf).resolve()

    if coupling is not None:
        couplings = [coupling]
    else:
        couplings = discoverCouplings(gathered)

    results: list[Path] = []
    for c in couplings:
        report = computeDiagnostics(gathered, c)

        if len(couplings) > 1:
            c_dir = output_pdf.parent / f"coupling_{c}"
            c_pdf = c_dir / output_pdf.name
        else:
            c_dir = output_pdf.parent
            c_pdf = output_pdf

        plots_dir = c_dir / "diagnostic_plots"
        plot_paths = generateDiagnosticPlots(report, plots_dir)

        template_dir = Path(__file__).parent / "templates"
        context = {
            "report": report,
            "plot_paths": {k: str(v.resolve()) for k, v in plot_paths.items()},
            "run_config": report.run_config,
            "points": report.points,
            "category_summaries": report.category_summaries,
        }

        latex_source = renderLatex(
            template_dir=template_dir,
            template_name="diagnostic_report.tex.j2",
            context=context,
        )
        buildPdfFromLatex(
            latex_source=latex_source,
            output_pdf=c_pdf,
            latex_engine=latex_engine,
            keep_build=keep_build,
            keep_tex=keep_tex,
        )
        logger.info(f"Generated diagnostic report for coupling {c}: {c_pdf}")
        results.append(c_pdf)

    return results
