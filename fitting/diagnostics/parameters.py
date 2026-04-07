from __future__ import annotations

import logging
from typing import Any

import jax.numpy as jnp
from flax import nnx
import gpjax

logger = logging.getLogger(__name__)

USEFUL_PARAMETER_NAMES = {
    "lengthscale",
    "variance",
    "period",
    "alpha",
    "shift",
    "constant",
    "rho",
    "obs_stddev",
    "obs_variance",
}


def logParameters(model: nnx.Module, header: str = "Parameters"):
    logger.info(f"--- {header} ---")

    parameter_filter = (gpjax.parameters.Parameter, nnx.Param, nnx.Variable)

    found = False
    for path, node in nnx.graph.iter_graph(model):
        if isinstance(node, parameter_filter):
            if any(
                any(
                    x in p if isinstance(p, str) else False
                    for x in USEFUL_PARAMETER_NAMES
                )
                for p in path
            ):
                if "network" in path:
                    continue

                found = True
                name = ".".join((str(x) for x in path))
                val = node.value
                if isinstance(val, (jnp.ndarray, list)) and jnp.size(val) > 1:
                    val_arr = jnp.atleast_1d(val)
                    if val_arr.size <= 10:
                        val_str = (
                            "["
                            + ", ".join(f"{float(x):.4g}" for x in val_arr.flatten())
                            + "]"
                        )
                    else:
                        val_str = (
                            f"[min={jnp.min(val_arr):.4g}, "
                            f"max={jnp.max(val_arr):.4g}, "
                            f"mean={jnp.mean(val_arr):.4g}] "
                            f"(size {val_arr.size})"
                        )
                else:
                    try:
                        val_str = f"{float(val):.4g}"
                    except (TypeError, ValueError):
                        val_str = str(val)

                logger.info(f"  {name}: {val_str}")

    if not found:
        logger.info("  No useful parameters found.")
    logger.info("-" * (len(header) + 8))


def logKernelParameters(posterior: Any):
    if hasattr(posterior, "prior") and hasattr(posterior.prior, "kernel"):
        kernel = posterior.prior.kernel
    elif hasattr(posterior, "posterior") and hasattr(posterior.posterior, "prior"):
        kernel = posterior.posterior.prior.kernel
    elif hasattr(posterior, "kernel"):
        kernel = posterior.kernel
    else:
        logger.debug(f"Could not find kernel in {type(posterior)}")
        return

    logParameters(kernel, "Kernel Hyperparameters")


def logLikelihoodParameters(likelihood: Any):
    logParameters(likelihood, "Likelihood Parameters")


def meanParameters(mean: Any):
    logParameters(mean, "Mean Parameters")
