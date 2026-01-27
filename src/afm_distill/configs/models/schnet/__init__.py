__codegen__ = True

from afm_distill.models.schnet.model import SchNetCutoffFnConfig as SchNetCutoffFnConfig
from afm_distill.models.schnet.model import SchNetNeighborListConfig as SchNetNeighborListConfig
from afm_distill.models.schnet.model import SchNetRBFConfig as SchNetRBFConfig
from afm_distill.models.schnet.model import SchNetStudentModelConfig as SchNetStudentModelConfig
from afm_distill.models.schnet.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.models.schnet.model import SchNetCutoffFnConfig as SchNetCutoffFnConfig
from afm_distill.models.schnet.model import SchNetNeighborListConfig as SchNetNeighborListConfig
from afm_distill.models.schnet.model import SchNetRBFConfig as SchNetRBFConfig
from afm_distill.models.schnet.model import SchNetStudentModelConfig as SchNetStudentModelConfig
from afm_distill.models.schnet.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.models.schnet.model import student_registry as student_registry

from . import model as model

__all__ = [
    "SchNetCutoffFnConfig",
    "SchNetNeighborListConfig",
    "SchNetRBFConfig",
    "SchNetStudentModelConfig",
    "StudentModuleBaseConfig",
    "model",
    "student_registry",
]
