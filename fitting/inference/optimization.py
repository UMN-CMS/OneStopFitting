from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable
from ..inference.prediction import predictInRealSpace, fixCovarianceMatrix
from ..diagnostics.posterior import posteriorPredictiveCheck
from flax import nnx
import attrs
import gpjax
import jax
import jax.numpy as jnp
import copy
import optax
import numpyro
from ..core.data import TrainingResult
from numpyro.infer import MCMC, NUTS
from gpjax.numpyro_extras import register_parameters

from .kernels import KernelConfig
from .priors import NormalPriorConfig
from ..data.loading import FileLoader, extractHistogram, histToBinnedData

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
    stage1_lr: float = 0.1
    stage2_lr: float = 0.005


@attrs.define
class RestartStrategy(ABC):
    @abstractmethod
    def prepareRun(
        self,
        posterior: Any,
        likelihood: Any,
        run_index: int,
        rng_key: jax.Array,
        model_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> tuple[Any, Any]: ...


@attrs.define
class ReseedConfig(RestartStrategy):
    def prepareRun(
        self,
        posterior: Any,
        likelihood: Any,
        run_index: int,
        rng_key: jax.Array,
        model_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> tuple[Any, Any]:
        return posterior, likelihood


@attrs.define
class ResampleConfig(RestartStrategy):
    def prepareRun(
        self,
        posterior: Any,
        likelihood: Any,
        run_index: int,
        rng_key: jax.Array,
        model_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> tuple[Any, Any]:
        if run_index == 0:
            return posterior, likelihood
        if model_factory is None:
            raise ValueError("ResampleConfig requires a model_factory for restarts > 0")
        new_posterior, new_likelihood = model_factory()
        logger.info(f"  Restart {run_index}: rebuilt model with fresh RNG")
        return new_posterior, new_likelihood


@attrs.define
class PerturbationConfig(RestartStrategy):
    scale: float = 0.1

    def prepareRun(
        self,
        posterior: Any,
        likelihood: Any,
        run_index: int,
        rng_key: jax.Array,
        model_factory: Callable[[], tuple[Any, Any]] | None = None,
    ) -> tuple[Any, Any]:
        if run_index == 0:
            return posterior, likelihood
        perturb_key, _ = jax.random.split(rng_key)
        perturbed = _perturbParameters(posterior, perturb_key, self.scale)
        logger.info(f"  Restart {run_index}: perturbed params with scale={self.scale}")
        return perturbed, likelihood


@attrs.define
class RestartCriterion(ABC):
    @abstractmethod
    def shouldContinue(self, run_result: TrainingResult) -> bool: ...


@attrs.define
class AlwaysContinueConfig(RestartCriterion):
    def shouldContinue(self, run_result: TrainingResult) -> bool:
        return True


@attrs.define
class LossThresholdConfig(RestartCriterion):
    max_loss: float = float("inf")

    def shouldContinue(self, run_result: TrainingResult) -> bool:
        return run_result.final_loss < self.max_loss


@attrs.define
class SelectionStrategy(ABC):
    @abstractmethod
    def score(
        self, result: TrainingResult, context: dict[str, Any] | None = None
    ) -> float: ...


@attrs.define
class BestLossConfig(SelectionStrategy):
    def score(
        self, result: TrainingResult, context: dict[str, Any] | None = None
    ) -> float:
        return result.final_loss


@attrs.define
class BestConvergenceConfig(SelectionStrategy):
    k: int = 20

    def score(
        self, result: TrainingResult, context: dict[str, Any] | None = None
    ) -> float:
        h = result.loss_history
        if len(h) < 2:
            return float("inf")
        tail = h[-min(self.k, len(h)) :]
        return abs(tail[-1] - tail[0]) / max(abs(tail[0]), 1e-10)


@attrs.define
class PredictedChi2Config(SelectionStrategy):
    def score(
        self, result: TrainingResult, context: dict[str, Any] | None = None
    ) -> float:
        if context is None or "dataset" not in context:
            return result.final_loss
        from ..inference.prediction import computePrediction
        from ..diagnostics.metrics import chi2PerBin

        dataset = context["dataset"]
        mean, cov = computePrediction(result.posterior, dataset, dataset.X)
        var = jnp.diag(cov)
        return chi2PerBin(dataset.y.ravel(), mean, var)


@attrs.define
class BestPPCConfig(SelectionStrategy):
    num_samples: int = 200

    def score(
        self, result: TrainingResult, context: dict[str, Any] | None = None
    ) -> float:
        if context is None or "dataset" not in context:
            return result.final_loss

        dataset = context["dataset"]
        test_data = context.get("test_data")
        transform = context.get("transform")
        rng_key = context.get("rng_key")
        if test_data is None or transform is None:
            return result.final_loss
        if rng_key is None:
            rng_key = jax.random.key(0)
        pred_mean, pred_cov = predictInRealSpace(
            result.posterior,
            dataset,
            test_data,
            transform,
            rng_key=rng_key,
        )
        pred_cov = fixCovarianceMatrix(pred_cov)
        ppc = posteriorPredictiveCheck(
            pred_mean,
            pred_cov,
            test_data,
            num_samples=self.num_samples,
            rng_key=rng_key,
        )
        pvalue = ppc["test_stats"]["chi2"]["all"]["pvalue"]
        logger.info(f"  PPC p-value: {pvalue:.3f}")
        return abs(pvalue - 0.5)


@attrs.define
class RestartConfig:
    num_restarts: int = 3
    strategy: RestartStrategy = attrs.Factory(ResampleConfig)
    criterion: RestartCriterion = attrs.Factory(AlwaysContinueConfig)
    selection: SelectionStrategy = attrs.Factory(BestLossConfig)


@attrs.define
class OptimizationConfig:
    mode: InferenceMode = InferenceMode.OPTIMIZATION
    lr: float = 0.1
    num_iters: int = 200
    optimizer: OptimizerType = OptimizerType.ADAM
    objective: ObjectiveType = ObjectiveType.MLL
    mcmc: MCMCConfig = attrs.Factory(MCMCConfig)
    two_stage: TwoStageConfig = attrs.Factory(TwoStageConfig)
    use_map_priors: bool = False
    map_prior_strength: float = 1.0
    log_interval: int = 50
    weight_decay: float = 1e-4

    lr_schedule_gamma: float | None = None
    lr_schedule_step: int | None = None
    restart: RestartConfig | None = None


def _buildOptimizer(config: OptimizationConfig) -> optax.GradientTransformation:
    base_opt = {
        OptimizerType.ADAM: optax.adam,
        OptimizerType.ADAMW: optax.adamw,
        OptimizerType.SGD: optax.sgd,
    }[config.optimizer](learning_rate=config.lr)  # , weight_decay=config.weight_decay)

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
    model_factory: Callable[[], tuple[Any, Any]] | None = None,
    scoring_fn: Callable[[TrainingResult], float] | None = None,
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
        return runWithRestarts(
            posterior,
            likelihood,
            dataset,
            config,
            rng_key,
            metric_fns=metric_fns,
            model_factory=model_factory,
            scoring_fn=scoring_fn,
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
    # if config.use_l2_regularization and config.l2_regularization_strength > 0:
    #     logger.info(
    #         f"L2 regularization enabled with strength={config.l2_regularization_strength}"
    #     )

    if use_priors:
        log_prior_fn = _build_log_prior_fn(posterior)

        def objective(model: nnx.Module, data: gpjax.Dataset) -> jnp.ndarray:
            nlml = -jnp.sum(base_objective(model, data))

            log_prior = log_prior_fn(model)
            jax.debug.print("Log prior: {}", config.map_prior_strength * log_prior)
            return nlml - config.map_prior_strength * log_prior

    else:

        def objective(p, d):
            base_loss = -jnp.sum(base_objective(p, d))
            return base_loss

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


def _perturbParameters(
    module: nnx.Module, rng_key: jax.Array, scale: float
) -> nnx.Module:
    perturbed = copy.deepcopy(module)
    key_idx = 0
    keys = jax.random.split(rng_key, 10000)
    for path, node in nnx.graph.iter_graph(perturbed):
        if isinstance(node, (gpjax.parameters.Parameter, nnx.Param)):
            noise = jax.random.normal(keys[key_idx], shape=node.value.shape) * scale
            node.value = node.value + noise * jnp.abs(node.value)
            key_idx += 1
    return perturbed


def runWithRestarts(
    posterior: Any,
    likelihood: Any,
    dataset: gpjax.Dataset,
    config: OptimizationConfig,
    rng_key: jax.Array,
    metric_fns: dict[str, Callable] | None = None,
    model_factory: Callable[[], tuple[Any, Any]] | None = None,
    scoring_fn: Callable[[TrainingResult], float] | None = None,
) -> TrainingResult:
    restart_cfg = config.restart
    num_restarts = restart_cfg.num_restarts if restart_cfg else 1

    if num_restarts <= 1 and (
        restart_cfg is None or isinstance(restart_cfg.strategy, ReseedConfig)
    ):
        return runMLE(posterior, likelihood, dataset, config, rng_key, metric_fns)

    if scoring_fn is None:
        selection = restart_cfg.selection
        scoring_fn = lambda r: selection.score(r, {"dataset": dataset})

    all_results: list[TrainingResult] = []

    for i in range(num_restarts):
        run_key, rng_key = jax.random.split(rng_key)

        if i == 0:
            run_posterior, run_likelihood = posterior, likelihood
        else:
            run_posterior, run_likelihood = restart_cfg.strategy.prepareRun(
                posterior, likelihood, i, run_key, model_factory
            )

        logger.info(f"=== Restart {i}/{num_restarts - 1} ===")
        result = runMLE(
            run_posterior, run_likelihood, dataset, config, run_key, metric_fns
        )
        all_results.append(result)

        if i < num_restarts - 1 and not restart_cfg.criterion.shouldContinue(result):
            logger.info(f"  Restart criterion met after run {i}, stopping early")
            break

    best_idx = min(range(len(all_results)), key=lambda i: scoring_fn(all_results[i]))
    best = all_results[best_idx]
    logger.info(
        f"Restarts complete: best run {best_idx}/{len(all_results) - 1}, "
        f"loss={best.final_loss:.4f}"
    )

    all_histories = [r.loss_history for r in all_results]
    best.loss_histories = all_histories
    best.best_restart = best_idx
    return best


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
            if path and path[-1] == "amplitude":
                # Do not freeze amplitude so it is fit simultaneously with the background
                continue
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
        lr=config.two_stage.stage1_lr,
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
            if path and path[-1] == "amplitude":
                continue
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
        lr=config.two_stage.stage2_lr,
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


def fitHyperpriorsFromMC(
    mc_path: str,
    kernel_config: KernelConfig,
    opt_config: OptimizationConfig,
    mc_uncertainty_fraction: float = 0.5,
    domain_mask: jnp.ndarray | None = None,
    rng_key: jax.Array | None = None,
    ndim: int = 2,
) -> dict[str, NormalPriorConfig]:
    from .models import ExactGPConfig
    from .means import ZeroMeanConfig
    from .likelihoods import UniformGaussianNoiseConfig

    if rng_key is None:
        rng_key = jax.random.key(0)

    logger.info(f"Fitting hyperpriors from MC data at {mc_path}")

    loader = FileLoader.forPath(mc_path)
    raw_data = loader.load(mc_path)
    histogram = extractHistogram(raw_data)

    binned_data = histToBinnedData(histogram, variation="central")
    if domain_mask is not None:
        masked_data = binned_data.masked(domain_mask)
    else:
        masked_data = binned_data

    dataset = gpjax.Dataset(X=masked_data.X, y=masked_data.Y.reshape(-1, 1))

    model_config = ExactGPConfig(
        kernel=kernel_config,
        likelihood=UniformGaussianNoiseConfig(),
        mean_function=ZeroMeanConfig(),
    )

    posterior, likelihood, prior = model_config.buildModel(
        dataset, ndim=ndim, rngs=nnx.Rngs(rng_key)
    )

    result = runMLE(posterior, likelihood, dataset, config=opt_config, rng_key=rng_key)
    learned_params = {}
    for path, node in nnx.graph.iter_graph(result.posterior):
        if isinstance(node, gpjax.parameters.Parameter):
            learned_params[path] = float(node.value)

    hyperpriors = {
        path: NormalPriorConfig(
            loc=val,
            scale=jnp.abs(val) * mc_uncertainty_fraction,
        )
        for path, val in learned_params.items()
    }

    logger.info(
        "Extracted MC hyperpriors: "
        + ", ".join(
            f"{k[-1]}={v.loc:.4f}" for k, v in hyperpriors.items() if len(k) > 0
        )
    )

    return hyperpriors
