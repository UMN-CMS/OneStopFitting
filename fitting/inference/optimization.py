from __future__ import annotations

import enum
import logging
from typing import Any, Callable

from flax import nnx
import attrs
import gpjax
import jax
import jax.numpy as jnp
import optax
import numpyro
from ..core.data import TrainingResult
from numpyro.infer import MCMC, NUTS
from gpjax.numpyro_extras import register_parameters

logger = logging.getLogger(__name__)


class InferenceMode(enum.Enum):
    OPTIMIZATION = "optimization"
    SAMPLING = "sampling"
    TWO_STAGE = "two_stage"
    HOMOSCEDASTIC_TWO_STAGE = "homoscedastic_two_stage"


class ObjectiveType(enum.Enum):
    MLL = "mll"
    LOOCV = "loocv"
    COLLAPSED_ELBO = "collapsed_elbo"
    ELBO = "elbo"


class OptimizerType(enum.Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


@attrs.define
class MCMCConfig:
    num_samples: int = 500
    num_warmup: int = 200
    num_chains: int = 1
    thinning: int = 1


@attrs.define
class TwoStageConfig:
    stage1_iters: int = 100
    stage2_iters: int = 100


@attrs.define
class OptimizationConfig:
    mode: InferenceMode = InferenceMode.OPTIMIZATION
    lr: float = 0.1
    num_iters: int = 200
    optimizer: OptimizerType = OptimizerType.ADAMW
    objective: ObjectiveType = ObjectiveType.MLL
    mcmc: MCMCConfig = attrs.Factory(MCMCConfig)
    two_stage: TwoStageConfig = attrs.Factory(TwoStageConfig)
    use_map_priors: bool = False
    log_interval: int = 50
    weight_decay: float = 1e-4

    lr_schedule_gamma: float | None = None
    lr_schedule_step: int | None = None


def _buildOptimizer(config: OptimizationConfig) -> optax.GradientTransformation:
    base_opt = {
        OptimizerType.ADAM: optax.adam,
        OptimizerType.ADAMW: optax.adamw,
        OptimizerType.SGD: optax.sgd,
    }[config.optimizer](learning_rate=config.lr, weight_decay=config.weight_decay)

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


def train(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    rng_key: jax.Array,
    metric_fns: dict[str, Callable] | None = None,
) -> TrainingResult:
    if config.mode == InferenceMode.SAMPLING:
        return runMCMC(
            posterior,
            likelihood,
            dataset,
            config.mcmc,
            rng_key=rng_key,
        )
    elif config.mode == InferenceMode.TWO_STAGE:
        return runTwoStageFit(
            posterior,
            likelihood,
            dataset,
            config,
            rng_key=rng_key,
        )
    elif config.mode == InferenceMode.HOMOSCEDASTIC_TWO_STAGE:
        return runHomoscedasticTwoStageFit(
            posterior,
            likelihood,
            dataset,
            config,
            rng_key=rng_key,
        )
    else:
        return runMLE(
            posterior,
            likelihood,
            dataset,
            rng_key=rng_key,
            config=config,
            metric_fns=metric_fns,
        )


def _build_log_prior_fn(model: nnx.Module) -> Callable[[nnx.Module], jax.Array]:
    priors_map = {}
    for path, node in nnx.graph.iter_graph(model):
        if (
            isinstance(node, gpjax.parameters.Parameter)
            and hasattr(node, "numpyro_properties")
            and node.numpyro_properties.get("prior") is not None
        ):
            priors_map[path] = node.numpyro_properties["prior"]

    def log_prior_fn(current_model: nnx.Module) -> jax.Array:
        total_log_prob = jnp.array(0.0)
        for path, node in nnx.graph.iter_graph(current_model):
            if path in priors_map:
                prior_dist = priors_map[path]
                total_log_prob += jnp.sum(prior_dist.log_prob(node.value))
        return total_log_prob

    return log_prior_fn


def runMLE(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    rng_key: jax.Array,
    metric_fns: dict[str, Callable] | None = None,
) -> TrainingResult:
    optimizer = _buildOptimizer(config)
    base_objective = _buildObjective(config)
    use_priors = config.use_map_priors

    logger.info(
        f"Starting optimization: {config.num_iters} iterations, "
        f"lr={config.lr}, optimizer={config.optimizer.value}, "
        f"objective={config.objective.value}, use_priors={use_priors}"
    )

    if use_priors:
        log_prior_fn = _build_log_prior_fn(posterior)

        def objective(model: nnx.Module, data: gpjax.Dataset) -> jnp.ndarray:
            nlml = -base_objective(model, data)

            log_prior = log_prior_fn(model)
            jax.debug.print("Log prior: {}", log_prior)
            return nlml - log_prior

    else:

        def objective(p, d):
            return -base_objective(p, d)

    trainable = (gpjax.parameters.Parameter, nnx.Param)

    opt_posterior, history = gpjax.fit(
        model=posterior,
        objective=objective,
        train_data=dataset,
        optim=optimizer,
        num_iters=config.num_iters,
        trainable=trainable,
        safe=True,
        key=rng_key,
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


def modelFunc(
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
    rng_key: jax.Array,
) -> TrainingResult:
    logger.info(
        f"Starting MCMC sampling: {config.num_samples} samples, "
        f"{config.num_warmup} warmup, {config.num_chains} chains"
    )

    nuts_kernel = NUTS(modelFunc)
    mcmc = MCMC(
        nuts_kernel,
        num_samples=config.num_samples,
        num_warmup=config.num_warmup,
        num_chains=config.num_chains,
        thinning=config.thinning,
        progress_bar=True,
    )
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


def setAtPath(obj: Any, path: tuple[str, ...], value: Any) -> None:
    """Helper to set an attribute or list element at a given path."""
    for part in path[:-1]:
        if isinstance(obj, (list, nnx.List)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    last = path[-1]
    if isinstance(obj, (list, nnx.List)):
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def runTwoStageFit(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    rng_key: jax.Array,
) -> TrainingResult:
    logger.info("Starting Two-Stage Fit Strategy")
    logger.info("Stage 1: Fitting Mean Function (Freezing Kernel)...")

    kernel = posterior.prior.kernel
    original_kernel_params = {}

    parameter_filter = (gpjax.parameters.Parameter, nnx.Param)
    for path, node in nnx.graph.iter_graph(kernel):
        if isinstance(node, parameter_filter):
            original_kernel_params[path] = node
            setAtPath(kernel, path, nnx.Variable(node.value))

    stage1_config = attrs.evolve(
        config,
        num_iters=config.two_stage.stage1_iters,
        log_interval=config.log_interval,
    )
    stage1_result = runMLE(
        posterior,
        likelihood,
        dataset,
        config=stage1_config,
        rng_key=rng_key,
    )

    logger.info("Stage 2: Fitting Kernel (Freezing Mean Function)...")

    mean_fn = stage1_result.posterior.prior.mean_function
    original_mean_params = {}
    parameter_filter = (gpjax.parameters.Parameter, nnx.Param)
    for path, node in nnx.graph.iter_graph(mean_fn):
        if isinstance(node, parameter_filter):
            original_mean_params[path] = node
            setAtPath(mean_fn, path, nnx.Variable(node.value))

    learned_kernel = stage1_result.posterior.prior.kernel
    for path, original_node in original_kernel_params.items():
        curr: Any = learned_kernel
        for part in path:
            if isinstance(curr, (list, nnx.List)):
                curr = curr[int(part)]
            else:
                curr = getattr(curr, part)

        current_val = curr.value
        original_node.value = current_val
        setAtPath(learned_kernel, path, original_node)

    stage2_config = attrs.evolve(
        config,
        num_iters=config.two_stage.stage2_iters,
        log_interval=config.log_interval,
    )
    final_result = runMLE(
        stage1_result.posterior,
        likelihood,
        dataset,
        config=stage2_config,
        rng_key=rng_key,
    )

    final_result.loss_history = stage1_result.loss_history + final_result.loss_history

    final_mean_fn = final_result.posterior.prior.mean_function
    for path, original_node in original_mean_params.items():
        curr: Any = final_mean_fn
        for part in path:
            if isinstance(curr, (list, nnx.List)):
                curr = curr[int(part)]
            else:
                curr = getattr(curr, part)
        original_node.value = curr.value
        setAtPath(final_mean_fn, path, original_node)

    return final_result


def runHomoscedasticTwoStageFit(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    rng_key: jax.Array,
) -> TrainingResult:
    logger.info("Starting Homoscedastic Two-Stage Fit Strategy")
    logger.info("Stage 1: Fitting macroscopic shape with Homoscedastic Likelihood...")

    kernel = posterior.prior.kernel
    original_kernel_params = {}
    parameter_filter = (gpjax.parameters.Parameter, nnx.Param)

    for path, node in nnx.graph.iter_graph(kernel):
        if isinstance(node, parameter_filter):
            original_kernel_params[path] = node
            val = node.value
            target_val = jnp.full_like(val, 1e-4)
            setAtPath(kernel, path, nnx.Variable(target_val))

    original_likelihood = likelihood
    flat_likelihood = gpjax.likelihoods.Gaussian(
        num_datapoints=dataset.n, obs_stddev=jnp.array(1.0)
    )

    stage1_posterior = posterior.prior * flat_likelihood

    stage1_config = attrs.evolve(
        config,
        num_iters=config.two_stage.stage1_iters,
        log_interval=config.log_interval,
    )
    stage1_result = runMLE(
        stage1_posterior,
        flat_likelihood,
        dataset,
        config=stage1_config,
        rng_key=rng_key,
    )

    logger.info("Stage 2: Fitting residuals with real Heteroscedastic Likelihood...")

    learned_mean = stage1_result.posterior.prior.mean_function
    original_mean_params = {}
    for path, node in nnx.graph.iter_graph(learned_mean):
        if isinstance(node, parameter_filter):
            original_mean_params[path] = node
            setAtPath(learned_mean, path, nnx.Variable(node.value))

    learned_kernel = stage1_result.posterior.prior.kernel
    for path, original_node in original_kernel_params.items():
        setAtPath(learned_kernel, path, original_node)

    new_prior = gpjax.gps.Prior(mean_function=learned_mean, kernel=learned_kernel)
    final_posterior = new_prior * original_likelihood

    stage2_config = attrs.evolve(
        config,
        num_iters=config.two_stage.stage2_iters,
        log_interval=config.log_interval,
    )
    final_result = runMLE(
        final_posterior,
        original_likelihood,
        dataset,
        config=stage2_config,
        rng_key=rng_key,
    )

    final_result.loss_history = stage1_result.loss_history + final_result.loss_history

    final_mean_fn = final_result.posterior.prior.mean_function
    for path, original_node in original_mean_params.items():
        curr: Any = final_mean_fn
        for part in path:
            if isinstance(curr, (list, nnx.List)):
                curr = curr[int(part)]
            else:
                curr = getattr(curr, part)
        original_node.value = curr.value
        setAtPath(final_mean_fn, path, original_node)

    return final_result
