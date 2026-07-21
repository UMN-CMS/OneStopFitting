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
from .aggregate_plots import computeBasicStatistics, computePValueStats
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
    significance: float
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
    pvq: float | None = None
    wvi: float | None = None
    signal_pre_scale: float | None = None
    min_counts: float | None = None
    lr: float | None = None
    injection_rate: float | None = None
    rebin: int | None = None
    num_iters: int | None = None
    optimizer: str | None = None
    variance_floor_quantile: float | None = None
    kernel_type: str | None = None
    hidden_shapes: list[int] | None = None
    activation: str | None = None
    ard: bool | None = None
    blinding_strategy: str | None = None
    window_type: str | None = None
    core_threshold_fraction: float | None = None
    dilation_margin: float | None = None
    smooth_sigma: float | None = None
    eigenvar_threshold: float | None = None


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


def _extractByMassPoint(gathered: list[dict]) -> dict[tuple[float, float], list[dict]]:
    by_mass: dict[tuple[float, float], list[dict]] = defaultdict(list)
    for entry in gathered:
        meta = entry.get("metadata", {}).get("sig", {}).get("other_data", {})
        mstop = meta.get("stop_mass")
        mchi = meta.get("chargino_mass")
        if mstop is None or mchi is None:
            continue
        by_mass[(float(mstop), float(mchi))].append(entry)
    return dict(by_mass)


def _inferCoupling(gathered: list[dict]) -> str:
    for entry in gathered:
        c = entry.get("metadata", {}).get("sig", {}).get("other_data", {}).get("coupling")
        if c is not None:
            return str(c)
    return "unknown"


def _computePointDiagnosis(
    mstop: float, mchi: float, toys: list[dict], use_ppc_pval=False
) -> PointDiagnosis:
    r_vals = []
    r_err_vals = []
    gof_pvals = []
    significances = []

    for toy in toys:
        combine = toy.get("combine", {})
        tree_fit = combine.get("tree_fit_sb", {})
        r, r_err = tree_fit.get("r"), tree_fit.get("r_err")

        if use_ppc_pval:
            gof = toy["ppc"]["test_stats"]["chi2"]["blinded"]["pvalue"]
        else:
            gof = combine.get("gof_p_value")

        if r is not None and r_err is not None:
            r_vals.append(float(r))
            r_err_vals.append(float(r_err))
        if gof is not None:
            gof_pvals.append(float(gof))
        significance = combine.get("significance")
        if significance is not None:
            significances.append(float(significance))
    gof_pvals = np.array(gof_pvals)

    r_arr = np.array(r_vals) if r_vals else np.array([0.0])
    r_err_arr = np.array(r_err_vals) if r_err_vals else np.array([1.0])

    median_r = float(np.median(r_arr))
    median_r_err = float(np.median(r_err_arr))
    mean_r = float(np.mean(r_arr))
    std_r = float(np.std(r_arr, ddof=1)) if len(r_arr) > 1 else 0.0
    coverage = float(np.mean(np.abs(r_arr) < r_err_arr))

    gof_stats = computeBasicStatistics(gof_pvals)
    gof_stats.update(computePValueStats(gof_pvals))
    gof_ks = gof_stats.get("ks_pvalue_uniform")
    gof_verdict = gof_stats.get("pvalue_skew_verdict", "UNKNOWN")
    gof_median = gof_stats.get("median")
    significance = np.median(significances)

    coupling = (
        str(
            toys[0].get("metadata", {}).get("sig", {}).get("other_data", {}).get("coupling", "unknown")
        )
        if toys
        else "unknown"
    )

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
        significance=significance,
        gof_median_pvalue=gof_median,
        gof_verdict=gof_verdict,
        r_values=r_vals,
        r_err_values=r_err_vals,
    )


def _computeCategorySummary(
    cat_name: str, points: list[PointDiagnosis]
) -> CategorySummary:
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
        n_ok=sum(
            1 for p in points if p.gof_verdict not in ["CONSERVATIVE", "DANGEROUS"]
        ),
        n_conservative=sum(1 for p in points if p.gof_verdict == "CONSERVATIVE"),
        n_dangerous=sum(1 for p in points if p.gof_verdict == "DANGEROUS"),
    )


def _extractRunConfig(gathered: list[dict]) -> RunConfig:
    if not gathered:
        return RunConfig()
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


def computeDiagnostics(gathered: list[dict]) -> DiagnosticReport:
    grouped = _extractByMassPoint(gathered)
    coupling = _inferCoupling(gathered)

    all_meta = [e.get("metadata", {}) for e in gathered[:1]]
    points = []
    for (mstop, mchi), toys in sorted(grouped.items()):
        points.append(_computePointDiagnosis(mstop, mchi, toys))

    cat_groups: dict[str, list[PointDiagnosis]] = defaultdict(list)
    for p in points:
        cat_groups[p.category].append(p)

    cat_summaries = [
        _computeCategorySummary(name, pts) for name, pts in sorted(cat_groups.items())
    ]

    all_coverages = [p.coverage for p in points]
    n_ok = sum(1 for p in points if p.gof_verdict not in ["CONSERVATIVE", "DANGEROUS"])
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


def _coverageColor(
    coverage: float, expected_coverage=0.68, ok_band=0.05, medium_band=0.1
) -> str:
    if expected_coverage - ok_band <= coverage <= expected_coverage + ok_band:
        return "green"
    if expected_coverage - medium_band <= coverage <= expected_coverage + medium_band:
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


def plotCoverageMap(
    report: DiagnosticReport, expected_coverage=0.68, ok_band=0.05, medium_band=0.1
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    for point in report.points:
        color = CAT_COLORS.get(point.category, "gray")
        ax.scatter(
            point.mstop,
            point.mchi,
            c=[_coverageColor(point.coverage)],
            marker="s",
            s=120,
            linewidths=0.8,
            edgecolors=color,
            zorder=3,
        )

    for cat_name, color in CAT_COLORS.items():
        ax.scatter([], [], c="gray", marker="s", s=80, edgecolors=color, label=cat_name)

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="green",
            label="{100*(expected_coverage - ok_band):d}--{100*(expected_coverage + ok_band):d}%",
        ),
        Patch(
            facecolor="yellow",
            label="{100*(expected_coverage - medium_band):d}--{100*(expected_coverage + medium_band):d}%",
        ),
        Patch(facecolor="red", label="Outside"),
    ]
    ax.legend(
        handles=legend_elements,
        title="Coverage",
        loc="lower left",
        fontsize="small",
        framealpha=0.9,
    )
    ax.add_artist(
        ax.legend(title="Category", loc="upper left", fontsize="small", framealpha=0.9)
    )

    _addCMSLabel(
        ax,
        report,
        f"Coverage: $|r| < r_{{err}}$\nOverall: {report.overall_coverage:.1%}, N={report.overall_n_toys}",
    )
    return fig, ax


def plotMedianRMap(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    xs = [p.mstop for p in report.points]
    ys = [p.mchi for p in report.points]
    vs = [p.median_r for p in report.points]

    sc = ax.scatter(
        xs,
        ys,
        c=vs,
        cmap="RdBu_r",
        marker="s",
        s=120,
        vmin=-2,
        vmax=2,
        linewidths=0.5,
        edgecolors="black",
    )
    fig.colorbar(sc, ax=ax, label="Median $r$")
    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")
    _addCMSLabel(ax, report, "Median $r$ (bkg-only)")
    return fig, ax


def plotVerdictMap(report: DiagnosticReport) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()

    xs = [p.mstop for p in report.points]
    ys = [p.mchi for p in report.points]
    ks_vals = [
        p.gof_ks_pvalue if p.gof_ks_pvalue is not None else np.nan
        for p in report.points
    ]
    edge_colors = [CAT_COLORS.get(p.category, "gray") for p in report.points]

    sc = ax.scatter(
        xs,
        ys,
        c=ks_vals,
        cmap="RdYlGn",
        marker="s",
        s=120,
        vmin=0,
        vmax=1,
        linewidths=0.8,
        edgecolors=edge_colors,
        zorder=3,
    )
    cb = fig.colorbar(sc, ax=ax, label="GOF KS $p$-value")
    cb.ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8)

    for p in report.points:
        if p.gof_verdict == "CONSERVATIVE":
            ax.annotate(
                "CONS",
                (p.mstop, p.mchi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="#FFDC00",
                    alpha=0.85,
                    ec="black",
                    lw=0.5,
                ),
                zorder=4,
            )
        elif p.gof_verdict == "DANGEROUS":
            ax.annotate(
                "DANG",
                (p.mstop, p.mchi),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="#FF4136",
                    alpha=0.85,
                    ec="black",
                    lw=0.5,
                ),
                zorder=4,
            )

    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\tilde{\chi}^{\pm}}$ [GeV]")

    from matplotlib.patches import Patch

    cat_handles = [
        Patch(facecolor="none", edgecolor=c, label=n) for n, c in CAT_COLORS.items()
    ]
    ax.legend(
        handles=cat_handles,
        title="Category",
        loc="lower right",
        fontsize="small",
        framealpha=0.9,
    )
    _addCMSLabel(
        ax,
        report,
        f"GOF KS $p$-value\n{report.n_ok} OK, {report.n_conservative} CONS, {report.n_dangerous} DANG",
    )
    return fig, ax


def plotCoverageBarByCategory(
    report: DiagnosticReport, expected_coverage=0.68, ok_band=0.05, medium_band=0.1
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots()
    cats = [s.category for s in report.category_summaries]
    coverages = [s.avg_coverage for s in report.category_summaries]
    n_points = [s.n_points for s in report.category_summaries]
    labels = [f"{c}\n(n={n})" for c, n in zip(cats, n_points)]

    bar_colors = [CMS_COLORS[i] for i in range(len(cats))]
    bars = ax.bar(labels, coverages, color=bar_colors)
    ax.axhline(
        expected_coverage,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Target (68%)",
    )
    ax.axhspan(
        expected_coverage - ok_band,
        expected_coverage + ok_band,
        color=CMS_COLORS[3],
        alpha=0.2,
    )
    ax.set_ylabel("Average Coverage ($|r| < r_{err}$)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize="small")

    for bar, cov in zip(bars, coverages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cov:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
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
            c=color,
            alpha=0.15,
            s=8,
            zorder=2,
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


def generateDiagnosticPlots(
    report: DiagnosticReport, output_dir: Path
) -> dict[str, Path]:
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
        c = entry.get("metadata", {}).get("sig", {}).get("other_data", {}).get("coupling")
        if c is not None:
            couplings.add(str(c))
    return sorted(couplings)


def generateDiagnosticReport(
    *,
    gathered: list[dict],
    output_dir: Path,
    name_format: str = "diagnostic_report",
    name_ctx: dict | None = None,
    latex_engine: str = "pdflatex",
    keep_build: bool = False,
    keep_tex: bool = False,
) -> Path:
    from ..utils import dictToDot, dotFormat

    if not gathered:
        return
    output_dir = Path(output_dir).resolve()
    fmt_ctx = dict(dictToDot(gathered[0])) | (name_ctx or {})
    output_name = dotFormat(name_format, **fmt_ctx).replace(".", "p")
    output_pdf = output_dir / f"{output_name}.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    report = computeDiagnostics(gathered)

    plots_dir = output_pdf.parent / "diagnostic_plots"
    plot_paths = generateDiagnosticPlots(report, plots_dir)

    def verdictPriority(p: PointDiagnosis) -> tuple[int, float, float]:
        if p.gof_verdict == "DANGEROUS":
            priority = 0
        elif p.gof_verdict == "CONSERVATIVE":
            priority = 1
        else:
            priority = 2
        return (priority, p.mstop, p.mchi)

    sorted_points = sorted(report.points, key=verdictPriority)

    template_dir = Path(__file__).parent / "templates"
    context = {
        "report": report,
        "plot_paths": {k: str(v.resolve()) for k, v in plot_paths.items()},
        "run_config": report.run_config,
        "points": sorted_points,
        "category_summaries": report.category_summaries,
    }

    latex_source = renderLatex(
        template_dir=template_dir,
        template_name="diagnostic_report.tex.j2",
        context=context,
    )
    buildPdfFromLatex(
        latex_source=latex_source,
        output_pdf=output_pdf,
        latex_engine=latex_engine,
        keep_build=keep_build,
        keep_tex=keep_tex,
    )
    logger.info("=" * 60)
    logger.info(f"DIAGNOSTIC VERDICT SUMMARY for Coupling: {report.coupling}")
    logger.info(f"Total points: {len(report.points)}")
    logger.info(f"  OK: {report.n_ok}")
    logger.info(f"  CONSERVATIVE: {report.n_conservative}")
    logger.info(f"  DANGEROUS: {report.n_dangerous}")

    if report.n_dangerous > 0:
        dang_list = [
            f"({int(p.mstop)}, {int(p.mchi)})"
            for p in report.points
            if p.gof_verdict == "DANGEROUS"
        ]
        logger.info(f"  --> DANGEROUS points: {', '.join(dang_list)}")
    if report.n_conservative > 0:
        cons_list = [
            f"({int(p.mstop)}, {int(p.mchi)})"
            for p in report.points
            if p.gof_verdict == "CONSERVATIVE"
        ]
        logger.info(f"  --> CONSERVATIVE points: {', '.join(cons_list)}")
    logger.info("=" * 60)

    logger.info(f"Generated diagnostic report: {output_pdf}")
    return output_pdf
