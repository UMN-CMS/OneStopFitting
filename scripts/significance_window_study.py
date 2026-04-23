import click
import json
import logging
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
import jax.numpy as jnp

from fitting.data.loading import FileLoader, extractHistogram, histToBinnedData
from fitting.utils import getSignal, getCategory
from fitting.data.windowing import fitCoreDilatedWindow
from fitting.core.data import BinnedData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("significance_study")

def getRebin(category: str, resources_dir: Path) -> int:
    config_path = resources_dir / f"config_{category}.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config.get("rebin", 1)
    return 1

def computeSignificance(s: np.ndarray, b: np.ndarray) -> np.ndarray:
    b_safe = np.maximum(b, 1e-9)
    s_safe = np.maximum(s, 0.0)
    ratio = s_safe / b_safe
    val = 2 * ((s_safe + b_safe) * np.log1p(ratio) - s_safe)
    sig = np.sqrt(np.maximum(val, 0.0))
    sig = np.where(b <= 1e-6, 0.0, sig)
    return sig

def discoverSignals(pipelines: tuple[str, ...], signal_base: Path, year: str) -> dict:
    signals_by_group = defaultdict(list)
    for pipeline in pipelines:
        pipeline_dir = signal_base / year / pipeline
        if not pipeline_dir.exists():
            continue
        for sig_dir in pipeline_dir.glob("signal_*"):
            if not sig_dir.is_dir():
                continue
            try:
                _, mstop, mchi = getSignal(sig_dir.name)
            except Exception:
                continue
            
            category = getCategory(mstop, mchi)
            target_file = sig_dir / f"{category}_mStop_vs_mChiRatio.pklz4"
            if not target_file.exists():
                continue
            signals_by_group[(pipeline, category)].append((mstop, mchi, target_file))
    return signals_by_group

def loadBackground(pipeline: str, category: str, background_base: Path, year: str, resources_dir: Path):
    bkg_path = background_base / year / pipeline / "qcd_inclusive_2018" / category / "pure_smoothed.pklz4"
    if not bkg_path.exists():
        logger.warning(f"Background not found: {bkg_path}")
        return None, None
        
    logger.info(f"Loading background: {bkg_path.name}")
    rebin_factor = getRebin(category, resources_dir)
    
    bkg_raw = FileLoader.forPath(bkg_path).load(bkg_path)
    bkg_hist = extractHistogram(bkg_raw)
    bkg_data = histToBinnedData(bkg_hist, rebin=rebin_factor, variation="central")
    return bkg_data, rebin_factor

def scanParametersForGroup(
    pipeline: str, category: str, signal_infos: list, bkg_data: BinnedData, rebin_factor: int, 
    core_thresholds: list, dilation_margins: list, smooth_sigma: float, signal_smooth_sigma: float
):
    group_results = defaultdict(list)
    diagnostic_rows = []
    
    for mstop, mchi, sig_path in signal_infos:
        logger.info(f"  -> Processing signal mstop={mstop}, mchi={mchi}")
        sig_raw = FileLoader.forPath(sig_path).load(sig_path)
        sig_hist = extractHistogram(sig_raw)
        sig_data = histToBinnedData(sig_hist, rebin=rebin_factor, variation="central")
        
        s_arr = np.asarray(sig_data.Y)
        b_arr = np.asarray(bkg_data.Y)
        
        if signal_smooth_sigma > 0.0:
            from scipy.ndimage import gaussian_filter
            shape = tuple(len(e) - 1 for e in sig_data.edges)
            s_arr = gaussian_filter(s_arr.reshape(shape), sigma=signal_smooth_sigma).ravel()
            
        sig_arr = computeSignificance(s_arr, b_arr)
        
        total_significance = np.sqrt(np.sum(sig_arr**2))
        if total_significance <= 0:
            continue
            
        sig_binned = BinnedData(
            X=sig_data.X, Y=jnp.array(sig_arr), V=sig_data.V,
            edges=sig_data.edges, axis_names=sig_data.axis_names
        )
        
        logging.getLogger("fitting.data.windowing").setLevel(logging.WARNING)

        for ct in core_thresholds:
            for dm in dilation_margins:
                try:
                    window = fitCoreDilatedWindow(
                        sig_binned, core_threshold_fraction=ct,
                        smooth_sigma=smooth_sigma, dilation_margin=dm
                    )
                    mask = window(sig_data.X)
                    mask_np = np.asarray(mask)
                    cap_sig = np.sqrt(np.sum(sig_arr[mask_np]**2))
                    cap_frac = cap_sig / total_significance
                    window_size = mask_np.sum()
                except ValueError:
                    cap_frac = 0.0
                    window_size = 0
                    
                group_results[(ct, dm)].append({
                    "mstop": mstop, "mchi": mchi,
                    "capture_fraction": float(cap_frac),
                    "window_size": int(window_size),
                    "total_significance": float(total_significance)
                })
                
                diagnostic_rows.append(f"{pipeline},{category},{mstop},{mchi},{ct},{dm},{cap_frac},{window_size},{total_significance}")
                
    return group_results, diagnostic_rows

def selectBestParameters(pipeline: str, category: str, group_results: dict, target_capture: float):
    best_params = None
    best_window_size = float('inf')
    
    for (ct, dm), results in group_results.items():
        if not results:
            continue
        cap_fracs = [r["capture_fraction"] for r in results]
        p5_cap = np.percentile(cap_fracs, 5)
        median_cap = np.median(cap_fracs)
        min_cap = np.min(cap_fracs)
        mean_size = np.mean([r["window_size"] for r in results])
        
        if p5_cap >= target_capture:
            if mean_size < best_window_size:
                best_window_size = mean_size
                best_params = {
                    "core_threshold_fraction": ct,
                    "dilation_margin": dm,
                    "median_capture": float(median_cap),
                    "min_capture": float(min_cap),
                    "p5_capture": float(p5_cap),
                    "mean_window_size_bins": float(mean_size)
                }
                
    if best_params:
        logger.info(f"Selected primary params: ct={best_params['core_threshold_fraction']}, dm={best_params['dilation_margin']} (p5_capture={best_params['p5_capture']:.3f}, size={best_params['mean_window_size_bins']:.1f})")
                
    if best_params is None:
        logger.warning(f"Still no parameters met target for {pipeline} {category}. Picking best median.")
        best_median = -1.0
        for (ct, dm), results in group_results.items():
            if not results:
                continue
            cap_fracs = [r["capture_fraction"] for r in results]
            median_cap = np.median(cap_fracs)
            min_cap = np.min(cap_fracs)
            mean_size = np.mean([r["window_size"] for r in results])
            
            if median_cap > best_median:
                best_median = median_cap
                best_window_size = mean_size
                best_params = {
                    "core_threshold_fraction": ct,
                    "dilation_margin": dm,
                    "median_capture": float(median_cap),
                    "min_capture": float(min_cap),
                    "mean_window_size_bins": float(mean_size),
                    "note": "Best median fallback"
                }
        if best_params:
            logger.info(f"Selected fallback params: ct={best_params['core_threshold_fraction']}, dm={best_params['dilation_margin']} (median_capture={best_params['median_capture']:.3f}, size={best_params['mean_window_size_bins']:.1f})")
            
    return best_params

def plotDiagnosticHeatmaps(pipeline, category, year, group_results, best_params, output_dir):
    import matplotlib.pyplot as plt
    import mplhep
    from fitting.diagnostics.plot_utils import addCMSBits
    mplhep.style.use("CMS")
    
    cts = sorted(list(set(k[0] for k in group_results.keys())))
    dms = sorted(list(set(k[1] for k in group_results.keys())))
    
    median_grid = np.zeros((len(cts), len(dms)))
    min_grid = np.zeros((len(cts), len(dms)))
    
    for i, ct in enumerate(cts):
        for j, dm in enumerate(dms):
            results = group_results.get((ct, dm), [])
            if results:
                cap_fracs = [r["capture_fraction"] for r in results]
                median_grid[i, j] = np.median(cap_fracs)
                min_grid[i, j] = np.min(cap_fracs)
            else:
                median_grid[i, j] = np.nan
                min_grid[i, j] = np.nan
                
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    
    all_meta = [{"era": {"lumi": 137, "energy": 13, "name": year}}]
    
    im1 = axes[0].imshow(median_grid, origin='lower', aspect='auto', cmap='viridis', vmin=0.8, vmax=1.0)
    axes[0].set_xticks(np.arange(len(dms)))
    axes[0].set_yticks(np.arange(len(cts)))
    axes[0].set_xticklabels([f"{dm:.2f}" for dm in dms])
    axes[0].set_yticklabels([f"{ct:.2f}" for ct in cts])
    axes[0].set_xlabel("Dilation Margin")
    axes[0].set_ylabel("Core Threshold Fraction")
    addCMSBits(axes[0], all_meta, extra_text=f"{pipeline} {category}\nMedian Capture Fraction", cms_text_pos=1)
    fig.colorbar(im1, ax=axes[0])
    
    if best_params:
        best_ct = best_params["core_threshold_fraction"]
        best_dm = best_params["dilation_margin"]
        if best_ct in cts and best_dm in dms:
            i_best = cts.index(best_ct)
            j_best = dms.index(best_dm)
            axes[0].plot(j_best, i_best, 'r*', markersize=15)
            axes[1].plot(j_best, i_best, 'r*', markersize=15)
            
    im2 = axes[1].imshow(min_grid, origin='lower', aspect='auto', cmap='viridis', vmin=0.8, vmax=1.0)
    axes[1].set_xticks(np.arange(len(dms)))
    axes[1].set_yticks(np.arange(len(cts)))
    axes[1].set_xticklabels([f"{dm:.2f}" for dm in dms])
    axes[1].set_yticklabels([f"{ct:.2f}" for ct in cts])
    axes[1].set_xlabel("Dilation Margin")
    axes[1].set_ylabel("Core Threshold Fraction")
    addCMSBits(axes[1], all_meta, extra_text=f"{pipeline} {category}\nMin Capture Fraction", cms_text_pos=1)
    fig.colorbar(im2, ax=axes[1])
    
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / f"{pipeline}_{category}_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

def plotScatter(pipeline, category, year, best_results, output_dir):
    import matplotlib.pyplot as plt
    import mplhep
    from fitting.diagnostics.plot_utils import addCMSBits
    mplhep.style.use("CMS")
    
    mchi = [r["mchi"] for r in best_results]
    mstop = [r["mstop"] for r in best_results]
    cap = [r["capture_fraction"] for r in best_results]
    
    fig, ax = plt.subplots(layout="constrained")
    
    all_meta = [{"era": {"lumi": 137, "energy": 13, "name": year}}]
    
    sc = ax.scatter(mstop, mchi, c=cap, cmap='viridis', s=100, vmin=0.8, vmax=1.0)
    ax.set_xlabel(r"$m_{\tilde{t}}$ [GeV]")
    ax.set_ylabel(r"$m_{\chi}$ [GeV]")
    addCMSBits(ax, all_meta, extra_text=f"{pipeline} {category}\nBest Params Capture", cms_text_pos=1)
    fig.colorbar(sc, ax=ax, label="Capture Fraction")
    
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    fig.savefig(plot_dir / f"{pipeline}_{category}_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

def getParamsToPlot(best_params, core_thresholds, dilation_margins):
    best_ct = best_params["core_threshold_fraction"]
    best_dm = best_params["dilation_margin"]
    params_to_plot = [(best_ct, best_dm)]
    
    tighter_ct = min(core_thresholds[-1], best_ct + 0.05)
    tighter_dm = max(0.0, best_dm - 0.1)
    if (tighter_ct, tighter_dm) not in params_to_plot:
        params_to_plot.append((tighter_ct, tighter_dm))
        
    even_tighter_ct = min(core_thresholds[-1], best_ct + 0.10)
    even_tighter_dm = max(0.0, best_dm - 0.2)
    if (even_tighter_ct, even_tighter_dm) not in params_to_plot:
        params_to_plot.append((even_tighter_ct, even_tighter_dm))
        
    looser_ct = max(core_thresholds[0], best_ct - 0.05)
    looser_dm = min(dilation_margins[-1], best_dm + 0.1)
    if (looser_ct, looser_dm) not in params_to_plot:
        params_to_plot.append((looser_ct, looser_dm))
        
    even_looser_ct = max(core_thresholds[0], best_ct - 0.10)
    even_looser_dm = min(dilation_margins[-1], best_dm + 0.2)
    if (even_looser_ct, even_looser_dm) not in params_to_plot:
        params_to_plot.append((even_looser_ct, even_looser_dm))
        
    return params_to_plot

def plotSignalPointDetails(pipeline, category, mstop, mchi, sig_path, bkg_data, rebin_factor, smooth_sigma, signal_smooth_sigma, best_params, all_params_to_plot, output_dir, year):
    import matplotlib.pyplot as plt
    import numpy as np
    import jax.numpy as jnp
    from fitting.data.loading import FileLoader, extractHistogram, histToBinnedData
    from fitting.core.data import BinnedData
    from fitting.data.windowing import fitCoreDilatedWindow
    from fitting.diagnostics.plot_utils import plotBinnedData, plotBlinding2D, addCMSBits
    import mplhep
    mplhep.style.use("CMS")
    
    sig_raw = FileLoader.forPath(sig_path).load(sig_path)
    sig_hist = extractHistogram(sig_raw)
    sig_data = histToBinnedData(sig_hist, rebin=rebin_factor, variation="central")
    
    s_raw_arr = np.asarray(sig_data.Y)
    b_arr = np.asarray(bkg_data.Y)
    
    if signal_smooth_sigma > 0.0:
        from scipy.ndimage import gaussian_filter
        shape = tuple(len(e) - 1 for e in sig_data.edges)
        s_arr = gaussian_filter(s_raw_arr.reshape(shape), sigma=signal_smooth_sigma).ravel()
    else:
        s_arr = s_raw_arr
        
    sig_data_smoothed = BinnedData(
        X=sig_data.X,
        Y=jnp.array(s_arr),
        V=sig_data.V,
        edges=sig_data.edges,
        axis_names=sig_data.axis_names
    )
    
    sig_arr = computeSignificance(s_arr, b_arr)
    total_sig = np.sqrt(np.sum(sig_arr**2))
    
    sig_binned = BinnedData(
        X=sig_data.X,
        Y=jnp.array(sig_arr),
        V=sig_data.V,
        edges=sig_data.edges,
        axis_names=sig_data.axis_names
    )
    
    fig = plt.figure(figsize=(24, 14), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_bg = fig.add_subplot(gs[0, 0])
    ax_raw_sig = fig.add_subplot(gs[0, 1])
    ax_smooth_sig = fig.add_subplot(gs[0, 2])
    ax_sigmap = fig.add_subplot(gs[1, 0])
    ax_windows = fig.add_subplot(gs[1, 1])
    ax_empty = fig.add_subplot(gs[1, 2]); ax_empty.axis('off')
    
    all_meta = [{"era": {"lumi": 137, "energy": 13, "name": year}}]
    
    plotBinnedData(ax_bg, bkg_data, cmap="viridis")
    addCMSBits(ax_bg, all_meta, extra_text=f"{pipeline} {category}\nBackground", cms_text_pos=1)
    
    plotBinnedData(ax_raw_sig, sig_data, cmap="viridis")
    addCMSBits(ax_raw_sig, all_meta, extra_text=f"{pipeline} {category} ({mstop}, {mchi})\nRaw Signal", cms_text_pos=1)
    
    plotBinnedData(ax_smooth_sig, sig_data_smoothed, cmap="viridis")
    addCMSBits(ax_smooth_sig, all_meta, extra_text=f"Smoothed Signal ($\\sigma$={signal_smooth_sigma})", cms_text_pos=1)
    
    plotBinnedData(ax_sigmap, sig_binned, cmap="plasma")
    addCMSBits(ax_sigmap, all_meta, extra_text=f"Significance Map\nTotal Sig: {total_sig:.2f}", cms_text_pos=1)
    
    plotBinnedData(ax_windows, sig_binned, cmap="plasma")
    addCMSBits(ax_windows, all_meta, extra_text="Windows & Capture Fractions", cms_text_pos=1)
    
    colors = ['magenta', 'cyan', 'lime', 'red', 'orange', 'yellow']
    for idx, (ct, dm) in enumerate(all_params_to_plot):
        try:
            window = fitCoreDilatedWindow(
                sig_binned,
                core_threshold_fraction=ct,
                smooth_sigma=smooth_sigma,
                dilation_margin=dm
            )
            mask = window(sig_data.X)
            cap_sig = np.sqrt(np.sum(sig_arr[np.asarray(mask)]**2))
            cap_frac = cap_sig / total_sig
            
            c = colors[idx % len(colors)]
            label = f"ct={ct:.2f}, dm={dm:0.2f} (cap={cap_frac:.2f})"
            if ct == best_params["core_threshold_fraction"] and dm == best_params["dilation_margin"]:
                label += " [BEST]"
                lw = 3
            else:
                lw = 2
                
            plotBlinding2D(ax_windows, sig_data.edges, sig_data.X, mask, color=c, linewidth=lw)
            ax_windows.plot([], [], color=c, linewidth=lw, label=label)
        except ValueError:
            pass
            
    handles, labels = ax_windows.get_legend_handles_labels()
    if handles:
        ax_empty.legend(handles, labels, loc='center', ncol=1, fontsize=24)
    
    for ax in [ax_bg, ax_raw_sig, ax_smooth_sig, ax_sigmap, ax_windows]:
        ax.set_xlabel(sig_data.axis_names[0] if sig_data.axis_names else "X")
        ax.set_ylabel(sig_data.axis_names[1] if len(sig_data.axis_names) > 1 else "Y")
        
    plot_dir = output_dir / "plots" / "details"
    plot_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(plot_dir / f"{pipeline}_{category}_{mstop}_{mchi}_details.pdf", bbox_inches="tight")
    plt.close(fig)

def generatePlots(pipeline, category, year, signal_infos, group_results, best_params, bkg_data, rebin_factor, smooth_sigma, signal_smooth_sigma, core_thresholds, dilation_margins, output_dir):
    plotDiagnosticHeatmaps(pipeline, category, year, group_results, best_params, output_dir)
    
    best_results = group_results.get((best_params["core_threshold_fraction"], best_params["dilation_margin"]), [])
    if not best_results:
        return
        
    plotScatter(pipeline, category, year, best_results, output_dir)
    
    sorted_results = sorted(best_results, key=lambda x: x["capture_fraction"])
    n_res = len(sorted_results)
    if n_res == 0:
        return
        
    reps = [
        sorted_results[0], 
        sorted_results[n_res//4],
        sorted_results[n_res//2], 
        sorted_results[3*n_res//4],
        sorted_results[-1]
    ]
    
    params_to_plot = getParamsToPlot(best_params, core_thresholds, dilation_margins)
    
    for rep in reps:
        sig_path = None
        for ms, mc, sp in signal_infos:
            if ms == rep["mstop"] and mc == rep["mchi"]:
                sig_path = sp
                break
        
        if sig_path:
            plotSignalPointDetails(
                pipeline, category, rep["mstop"], rep["mchi"], sig_path, 
                bkg_data, rebin_factor, smooth_sigma, signal_smooth_sigma, best_params, params_to_plot, output_dir, year
            )

@click.command()
@click.option("--signal-base", type=click.Path(exists=True, path_type=Path), default="export_complete_with_large")
@click.option("--background-base", type=click.Path(exists=True, path_type=Path), default="smoothed_complete")
@click.option("--year", type=str, default="2018")
@click.option("--pipelines", multiple=True, default=["Signal312", "Signal313"])
@click.option("--target-capture", type=float, default=0.70)
@click.option("--output", type=click.Path(path_type=Path), default="results/window_study")
@click.option("--resources-dir", type=click.Path(exists=True, path_type=Path), default="resources")
@click.option("--smooth-sigma", type=float, default=1.5, help="Smoothing applied to significance map before thresholding.")
@click.option("--signal-smooth-sigma", type=float, default=0.0, help="Optional pre-smoothing of the signal histogram before significance calculation.")
def main(
    signal_base: Path, 
    background_base: Path, 
    year: str, 
    pipelines: tuple[str, ...], 
    target_capture: float, 
    output: Path, 
    resources_dir: Path, 
    smooth_sigma: float,
    signal_smooth_sigma: float
):
    output.mkdir(parents=True, exist_ok=True)
    
    core_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
    dilation_margins = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    
    signals_by_group = discoverSignals(pipelines, signal_base, year)
            
    final_results = {
        "target_capture_fraction": target_capture,
        "smooth_sigma": smooth_sigma,
        "parameters": {}
    }
    
    diagnostic_rows = []
    
    for (pipeline, category), signal_infos in signals_by_group.items():
        logger.info(f"Processing {pipeline} {category} with {len(signal_infos)} signals")
        
        bkg_data, rebin_factor = loadBackground(pipeline, category, background_base, year, resources_dir)
        if bkg_data is None:
            continue
            
        group_results, group_rows = scanParametersForGroup(
            pipeline, category, signal_infos, bkg_data, rebin_factor, 
            core_thresholds, dilation_margins, smooth_sigma, signal_smooth_sigma
        )
        diagnostic_rows.extend(group_rows)
        
        best_params = selectBestParameters(pipeline, category, group_results, target_capture)
        
        if best_params:
            if pipeline not in final_results["parameters"]:
                final_results["parameters"][pipeline] = {}
            final_results["parameters"][pipeline][category] = best_params
            
            generatePlots(
                pipeline, category, year, signal_infos, group_results, best_params, 
                bkg_data, rebin_factor, smooth_sigma, signal_smooth_sigma, 
                core_thresholds, dilation_margins, output
            )
            
    with open(output / "window_params.json", "w") as f:
        json.dump(final_results, f, indent=2)
        
    with open(output / "detailed_results.csv", "w") as f:
        f.write("pipeline,category,mstop,mchi,core_threshold,dilation_margin,capture_fraction,window_size,total_significance\n")
        f.write("\n".join(diagnostic_rows))
        
    logger.info(f"Done. Results saved to {output}")

if __name__ == "__main__":
    main()
