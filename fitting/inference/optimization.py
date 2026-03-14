from __future__ import annotations

import enum
import logging
from typing import Any, Callable

import attrs
import gpjax
import jax
import jax.numpy as jnp
import optax
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from gpjax.numpyro_extras import register_parameters

logger = logging.getLogger(__name__)


class InferenceMode(enum.Enum):
    """Inference method."""

    OPTIMIZATION = "optimization"  # MLE or MAP
    SAMPLING = "sampling"  # MCMC


class ObjectiveType(enum.Enum):
    MLL = "mll"
    LOOCV = "loocv"
    COLLAPSED_ELBO = "collapsed_elbo"
    ELBO = "elbo"


class OptimizerType(enum.Enum):
    ADAM = "adam"
    SGD = "sgd"


@attrs.define
class MCMCConfig:
    num_samples: int = 500
    num_warmup: int = 200
    num_chains: int = 1
    thinning: int = 1


@attrs.define
class OptimizationConfig:
    mode: InferenceMode = InferenceMode.OPTIMIZATION
    lr: float = 0.01
    num_iters: int = 1000
    optimizer: OptimizerType = OptimizerType.ADAM
    objective: ObjectiveType = ObjectiveType.MLL
    mcmc: MCMCConfig = attrs.Factory(MCMCConfig)
    log_interval: int = 50
    lr_schedule_gamma: float | None = None
    lr_schedule_step: int | None = None


def _buildOptimizer(config: OptimizationConfig) -> optax.GradientTransformation:
    base_opt = {
        OptimizerType.ADAM: optax.adam,
        OptimizerType.SGD: optax.sgd,
    }[config.optimizer](learning_rate=config.lr)

    if config.lr_schedule_gamma is not None:
        step = config.lr_schedule_step or config.num_iters
        schedule = optax.exponential_decay(
            init_value=config.lr,
            transition_steps=step,
            decay_rate=config.lr_schedule_gamma,
        )
        base_opt = optax.chain(optax.scale_by_schedule(schedule), base_opt)

    return base_opt


def _buildObjective(config: OptimizationConfig) -> Any:
    """Return the gpjax objective function corresponding to the config."""
    objectives = {
        ObjectiveType.MLL: gpjax.objectives.conjugate_mll,
        ObjectiveType.LOOCV: gpjax.objectives.conjugate_loocv,
        ObjectiveType.COLLAPSED_ELBO: gpjax.objectives.collapsed_elbo,
        ObjectiveType.ELBO: gpjax.objectives.elbo,
    }
    obj = objectives.get(config.objective)
    if obj is None:
        raise ValueError(f"Unknown objective: {config.objective}")
    return obj


@attrs.define
class TrainingResult:
    posterior: Any
    likelihood: Any
    loss_history: list[float]
    final_loss: float
    metric_histories: dict[str, list[float]] = attrs.Factory(dict)
    samples: dict[str, jnp.ndarray] | None = None


def train(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    metric_fns: dict[str, Callable] | None = None,
) -> TrainingResult:
    if config.mode == InferenceMode.SAMPLING:
        return runMCMC(posterior, likelihood, dataset, config.mcmc)
    else:
        return runMLE(posterior, likelihood, dataset, config, metric_fns)


def runMLE(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    metric_fns: dict[str, Callable] | None = None,
) -> TrainingResult:
    optimizer = _buildOptimizer(config)
    objective = _buildObjective(config)

    logger.info(
        f"Starting optimization: {config.num_iters} iterations, "
        f"lr={config.lr}, optimizer={config.optimizer.value}, "
        f"objective={config.objective.value}"
    )

    opt_posterior, history = gpjax.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimizer,
        num_iters=config.num_iters,
        safe=True,
        key=jax.random.PRNGKey(0xDEADBEEF),
    )

    loss_values = [float(h) for h in history]

    metric_histories: dict[str, list[float]] = {}
    if metric_fns:
        for name in metric_fns:
            metric_histories[name] = []

    if config.log_interval > 0:
        for i, val in enumerate(loss_values):
            if i % config.log_interval == 0 or i == len(loss_values) - 1:
                metric_strs = [f"loss={val:.4f}"]
                logger.info(f"  Iter {i}: {', '.join(metric_strs)}")

    final_loss = loss_values[-1] if loss_values else float("nan")
    logger.info(f"Optimization complete. Final loss: {final_loss:.4f}")

    return TrainingResult(
        posterior=opt_posterior,
        likelihood=likelihood,
        loss_history=loss_values,
        final_loss=final_loss,
        metric_histories=metric_histories,
    )


def mcmcModelFunc(
    posterior: Any, X: jnp.ndarray, y: jnp.ndarray, X_new: jnp.ndarray | None = None
) -> None:
    p_posterior = register_parameters(posterior)
    dataset = gpjax.Dataset(X=X, y=y)
    mll = gpjax.objectives.conjugate_mll(p_posterior, dataset)
    numpyro.factor("log_lik", mll)
    if X_new is not None:
        latent_dist = p_posterior.predict(X_new, train_data=dataset)
        numpyro.sample("f_new", latent_dist)


def runMCMC(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: MCMCConfig,
) -> TrainingResult:
    logger.info(
        f"Starting MCMC sampling: {config.num_samples} samples, "
        f"{config.num_warmup} warmup, {config.num_chains} chains"
    )

    nuts_kernel = NUTS(mcmcModelFunc)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=config.num_samples,
        num_warmup=config.num_warmup,
        num_chains=config.num_chains,
        thinning=config.thinning,
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(0xCAFE)
    mcmc.run(rng_key, posterior, dataset.X, dataset.y)

    samples = mcmc.get_samples()
    logger.info(f"MCMC sampling complete. Drawn {config.num_samples} samples.")

    return TrainingResult(
        posterior=posterior,
        likelihood=likelihood,
        loss_history=[],
        final_loss=0.0,
        samples=samples,
    )
