"""GP regression inference module (gpjax-based)."""

from ..core.serialization import registerHierarchy
from .kernels import KernelConfig
from .likelihoods import LikelihoodConfig
from .means import MeanFunctionConfig
from .models import GPModelConfig

# Register all inference hierarchies for polymorphic serialization
registerHierarchy(KernelConfig)
registerHierarchy(LikelihoodConfig)
registerHierarchy(GPModelConfig)
registerHierarchy(MeanFunctionConfig)
