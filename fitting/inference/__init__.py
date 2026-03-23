"""GP regression inference module (gpjax-based)."""

from ..core.serialization import registerHierarchy
from .kernels import KernelConfig
from .likelihoods import LikelihoodConfig
from .means import MeanFunctionConfig
from .models import GPModelConfig

# Register all inference hierarchies for polymorphic serialization
# ORDER MATTERS: leaf configs (kernel, likelihood, mean) must register
# before composite configs (model) that reference them.
registerHierarchy(KernelConfig)
registerHierarchy(LikelihoodConfig)
registerHierarchy(MeanFunctionConfig)
registerHierarchy(GPModelConfig)
