from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

from ..core.data import AnalysisState, BinnedData
from ..diagnostics.plot_utils import (
    addAxesToHist,
    plotBinnedData,
    plotRaw,
    plotBlinding2D,
)

logger = logging.getLogger(__name__)


def smoothSignal(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_np = np.asarray(X, dtype=np.float64)
    Y_np = np.asarray(Y, dtype=np.float64).ravel()
    if X_np.ndim == 1:
        X_np = X_np[:, None]
    interp = RBFInterpolator(X_np, Y_np, smoothing=0.01)
    return np.asarray(interp(X_np)).ravel()


def computeContaminationTemplate(
    state: AnalysisState,
) -> tuple[np.ndarray, np.ndarray]:
    import gpjax
    from gpjax.variational_families import (
        VariationalGaussian,
        CollapsedVariationalGaussian,
    )

    if not state.signals:
        raise ValueError("Cannot compute contamination without signal templates.")
    if state.training_result is None:
        raise ValueError("Cannot compute contamination without trained model.")

    posterior = state.training_result.posterior
    if isinstance(posterior, (VariationalGaussian, CollapsedVariationalGaussian)):
        raise NotImplementedError(
            "Contamination estimation only supports exact GPs (ConjugatePosterior)."
        )

    kernel = posterior.prior.kernel
    transform = state.transform
    blind_mask = state.blind_mask

    domain_signals = []
    for sig in state.signals.values():
        if state.domain_mask is not None:
            domain_signals.append(np.asarray(sig.Y[state.domain_mask]))
        else:
            domain_signals.append(np.asarray(sig.Y))
    total_signal = np.sum(domain_signals, axis=0)

    smoothed = smoothSignal(state.test_data.X, total_signal)

    signal_train = smoothed[~blind_mask]
    signal_train_norm = np.asarray(transform.applyY(jnp.array(signal_train)))

    X_train_norm = transform.applyX(state.train_data.X)
    X_inside_norm = transform.applyX(state.test_data.X[blind_mask])
    n_train = X_train_norm.shape[0]

    obs_stddev_raw = posterior.likelihood.obs_stddev[...]
    new_prior = gpjax.gps.Prior(
        mean_function=gpjax.mean_functions.Zero(), kernel=kernel
    )
    new_likelihood = gpjax.likelihoods.Gaussian(
        num_datapoints=n_train,
        obs_stddev=obs_stddev_raw,
    )
    contamination_posterior = new_prior * new_likelihood

    signal_dataset = gpjax.Dataset(
        X=X_train_norm,
        y=signal_train_norm.reshape(-1, 1),
    )
    latent = contamination_posterior.predict(X_inside_norm, train_data=signal_dataset)
    contamination_norm = latent.mean.ravel()
    contamination_real = np.asarray(transform.invertY(contamination_norm))

    total_contam = float(np.sum(contamination_real))
    logger.info(
        f"Contamination template: {len(contamination_real)} bins, "
        f"total = {total_contam:.4g}"
    )

    return contamination_real, smoothed


def _buildFullDomainArray(blind_mask, full_size, window_values):
    arr = np.full(full_size, np.nan, dtype=float)
    arr[blind_mask] = window_values
    return arr


def _toBinnedData(test_data, values):
    return BinnedData(
        X=test_data.X,
        Y=jnp.array(values),
        V=jnp.full_like(test_data.V, np.nan),
        edges=test_data.edges,
        axis_names=test_data.axis_names,
    )


def plotContamination(
    state: AnalysisState,
    contamination: np.ndarray,
    smoothed_signal: np.ndarray,
    plot_saver: Callable,
) -> None:
    ndim = state.test_data.ndim
    if ndim == 1:
        _plot1D(state, contamination, smoothed_signal, plot_saver)
    elif ndim == 2:
        _plot2D(state, contamination, smoothed_signal, plot_saver)
        _plotSlices2D(state, contamination, smoothed_signal, plot_saver)

    else:
        logger.warning(f"Contamination plots not implemented for {ndim}D data")


def _plot1D(state, contamination, smoothed, plot_saver):
    blind_mask = np.asarray(state.blind_mask)
    test_data = state.test_data
    X = np.asarray(test_data.X).ravel()
    bg = np.asarray(state.pred_mean)

    contam_full = _buildFullDomainArray(blind_mask, len(X), contamination)
    smoothed_bd = _toBinnedData(test_data, smoothed)
    contam_bd = _toBinnedData(test_data, contam_full)
    bg_minus_contam = np.where(blind_mask, np.maximum(bg - contam_full, 1e-10), bg)

    raw_signal_full = np.zeros(len(X))
    for sig in state.signals.values():
        if state.domain_mask is not None:
            raw_signal_full += np.asarray(sig.Y[state.domain_mask])
        else:
            raw_signal_full += np.asarray(sig.Y)
    raw_signal_bd = _toBinnedData(test_data, raw_signal_full)

    fig, ax = plt.subplots(layout="tight")
    addAxesToHist(ax, size=1.5)

    plotBinnedData(ax, test_data, histtype="errorbar", color="black", label="Observed")

    ax.plot(X, bg, color="orange", label="GP Prediction", lw=2)
    pred_std = (
        np.sqrt(np.asarray(state.pred_cov.diagonal()))
        if state.pred_cov is not None
        else None
    )
    if pred_std is not None:
        ax.fill_between(
            X,
            bg - pred_std,
            bg + pred_std,
            color="orange",
            alpha=0.3,
            label=r"$\pm\sigma_{pred}$",
        )

    plotBinnedData(
        ax, raw_signal_bd, histtype="step", color="red", label="Raw Signal", lw=1
    )
    plotBinnedData(
        ax,
        smoothed_bd,
        histtype="step",
        color="darkred",
        ls="--",
        label="Smoothed Signal",
        lw=1.5,
    )
    plotBinnedData(
        ax, contam_bd, histtype="step", color="purple", label="Contamination", lw=2
    )
    ax.plot(
        X, bg_minus_contam, color="orange", ls="--", label="Bkg $-$ Contam.", lw=1.5
    )

    if np.any(blind_mask):
        w_min = X[blind_mask].min()
        w_max = X[blind_mask].max()
        bin_width = (X[-1] - X[0]) / len(X) if len(X) > 1 else 1.0
        for boundary in [w_min - bin_width / 2, w_max + bin_width / 2]:
            ax.axvline(boundary, ls="--", color="gray", alpha=0.5)
            ax.bottom_axes[0].axvline(boundary, ls="--", color="gray", alpha=0.5)

    ratio_ax = ax.bottom_axes[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((bg > 0) & blind_mask, contam_full / bg, np.nan)
    ratio_ax.plot(X, ratio, "o", color="purple", markersize=3)
    ratio_ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ratio_ax.set_ylabel("Contamination / Bkg")
    ratio_ax.set_ylim(-0.1, None)
    if test_data.axis_names:
        ratio_ax.set_xlabel(test_data.axis_names[0])

    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=10, ncol=2)

    plot_saver("contamination_1d", fig, ax)


def _plot2D(state, contamination, smoothed, plot_saver):
    blind_mask = state.blind_mask
    test_data = state.test_data
    edges = test_data.edges
    X = test_data.X
    n_full = test_data.nbins
    bg = np.asarray(state.pred_mean)

    contam_full = _buildFullDomainArray(blind_mask, n_full, contamination)
    bg_minus_contam = np.where(blind_mask, bg - contam_full, bg)

    raw_signal_full = np.zeros(n_full)
    for sig in state.signals.values():
        if state.domain_mask is not None:
            raw_signal_full += np.asarray(sig.Y[state.domain_mask])
        else:
            raw_signal_full += np.asarray(sig.Y)

    @contextmanager
    def makePlot(key):
        fig, ax = plt.subplots()
        try:
            yield ax
        finally:
            if test_data.axis_names and len(test_data.axis_names) >= 2:
                ax.set_xlabel(test_data.axis_names[0])
                ax.set_ylabel(test_data.axis_names[1])
            plot_saver(key, fig, ax)

    with makePlot("contamination_smoothed_signal") as ax:
        plotRaw(ax, edges, X, smoothed, cbar_title="Events", cmin=0)
        ax.set_title("Smoothed Signal (Sum)")
        plotBlinding2D(ax, edges, X, blind_mask)

    with makePlot("contamination_raw_signal") as ax:
        plotRaw(ax, edges, X, raw_signal_full, cbar_title="Events", cmin=0)
        ax.set_title("Raw Signal (Sum)")
        plotBlinding2D(ax, edges, X, blind_mask)

    with makePlot("contamination_template") as ax:
        plotRaw(ax, edges, X, contam_full, cbar_title="Events")
        ax.set_title("Contamination Template")
        plotBlinding2D(ax, edges, X, blind_mask)

    with makePlot("contamination_bkg_corrected") as ax:
        plotRaw(ax, edges, X, bg_minus_contam, cbar_title="Events", cmin=0)
        ax.set_title("Background $-$ Contamination")
        plotBlinding2D(ax, edges, X, blind_mask)

    with makePlot("contamination_ratio") as ax:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where((bg > 0) & blind_mask, contam_full / bg, np.nan)
        plotRaw(
            ax,
            edges,
            X,
            ratio,
            cmap="viridis",
            cmin=0,
            cbar_title="Contamination / Bkg",
        )
        ax.set_title("Contamination / Background")
        plotBlinding2D(ax, edges, X, blind_mask)


def _plotSlices2D(state, contamination, smoothed, plot_saver):
    from ..diagnostics.plots_2d_slices import (
        _uniqueAxisValues,
        _sliceMask,
        _perpendicularAxis,
        _makeOneSlice,
    )

    blind_mask = np.asarray(state.blind_mask)
    test_data = state.test_data
    X = np.asarray(test_data.X)
    n_full = test_data.nbins
    bg = np.asarray(state.pred_mean)
    pred_var = np.diag(np.asarray(state.pred_cov))

    contam_full = _buildFullDomainArray(blind_mask, n_full, contamination)

    raw_signal_full = np.zeros(n_full)
    for sig in state.signals.values():
        if state.domain_mask is not None:
            raw_signal_full += np.asarray(sig.Y[state.domain_mask])
        else:
            raw_signal_full += np.asarray(sig.Y)

    smoothed_bd = _toBinnedData(test_data, smoothed)
    raw_signal_bd = _toBinnedData(test_data, raw_signal_full)
    contam_bd = _toBinnedData(test_data, np.nan_to_num(contam_full, nan=0.0))
    signal_data = {
        "Contamination": contam_bd,
        "Smoothed": smoothed_bd,
        "Raw": raw_signal_bd,
    }

    print(test_data)
    for slice_axis in (0, 1):
        window_X = X[blind_mask]
        window_vals = _uniqueAxisValues(window_X, slice_axis)
        axis_name = (
            test_data.axis_names[slice_axis]
            if test_data.axis_names and len(test_data.axis_names) > slice_axis
            else f"axis{slice_axis}"
        )

        for val in window_vals:
            perp = _perpendicularAxis(slice_axis)
            mask = _sliceMask(X, slice_axis, val)

            order = np.argsort(X[mask, perp])
            slice_x = X[mask, perp][order]
            slice_contam = contam_full[mask][order]
            slice_bg = bg[mask][order]

            def makeContaminationDrawFunc(sx, sc, sb):
                def drawFunc(ax, ratio_ax):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio_vals = np.where(sb > 0, sc / sb, np.nan)
                    ax.plot(
                        sx,
                        np.maximum(sb - sc, 1e-10),
                        color="orange",
                        ls="--",
                        label="Bkg $-$ Contam.",
                        lw=1.5,
                    )
                    ratio_ax.plot(
                        sx,
                        ratio_vals,
                        "s",
                        color="purple",
                        markersize=3,
                        label="Contam./Bkg",
                    )
                    ratio_ax.legend(fontsize=8, loc="upper right")

                return drawFunc

            draw_func = makeContaminationDrawFunc(slice_x, slice_contam, slice_bg)

            val_key = f"{val:0.3g}".replace(".", "p").replace("-", "m")
            key = f"contamination_slice/axis{slice_axis}_{val_key}"

            _makeOneSlice(
                pred_mean=bg,
                pred_var=pred_var,
                test_data=test_data,
                blind_mask=blind_mask,
                slice_axis=slice_axis,
                val=val,
                axis_name=axis_name,
                plot_saver=plot_saver,
                key=key,
                signal_data=signal_data,
                show_minimap=True,
                minimap_background=True,
                extra_draw_funcs=[draw_func],
            )
