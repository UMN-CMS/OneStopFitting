"""GP regression inference module (gpjax-based)."""

from ..core.serialization import registerHierarchy
from .kernels import KernelConfig
from .likelihoods import LikelihoodConfig
from .priors import PriorConfig
from .means import MeanFunctionConfig
from .models import GPModelConfig
from .optimization import RestartStrategy, RestartCriterion, SelectionStrategy

# Register all inference hierarchies for polymorphic serialization
# ORDER MATTERS: leaf configs (kernel, likelihood, mean) must register
# before composite configs (model) that reference them.
registerHierarchy(PriorConfig)
registerHierarchy(KernelConfig)
registerHierarchy(LikelihoodConfig)
registerHierarchy(MeanFunctionConfig)
registerHierarchy(GPModelConfig)
registerHierarchy(RestartStrategy)
registerHierarchy(RestartCriterion)
registerHierarchy(SelectionStrategy)
