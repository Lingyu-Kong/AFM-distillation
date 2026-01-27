__codegen__ = True

from afm_distill.models import AllegroStudentModelConfig as AllegroStudentModelConfig
from afm_distill.models import CACECutoffFnConfig as CACECutoffFnConfig
from afm_distill.models import CACERBFConfig as CACERBFConfig
from afm_distill.models.cace.model import CACEReadOutHeadConfig as CACEReadOutHeadConfig
from afm_distill.models import CACEStudentModelConfig as CACEStudentModelConfig
from afm_distill import OfflineDistillationConfig as OfflineDistillationConfig
from afm_distill.models import PaiNNCutoffFnConfig as PaiNNCutoffFnConfig
from afm_distill.models.painn.model import PaiNNNeighborListConfig as PaiNNNeighborListConfig
from afm_distill.models import PaiNNRBFConfig as PaiNNRBFConfig
from afm_distill.models import PaiNNStudentModelConfig as PaiNNStudentModelConfig
from afm_distill.models import SchNetCutoffFnConfig as SchNetCutoffFnConfig
from afm_distill.models.schnet.model import SchNetNeighborListConfig as SchNetNeighborListConfig
from afm_distill.models import SchNetRBFConfig as SchNetRBFConfig
from afm_distill.models import SchNetStudentModelConfig as SchNetStudentModelConfig
from afm_distill import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.models import AllegroStudentModelConfig as AllegroStudentModelConfig
from afm_distill.models import CACECutoffFnConfig as CACECutoffFnConfig
from afm_distill.models import CACERBFConfig as CACERBFConfig
from afm_distill.models.cace.model import CACEReadOutHeadConfig as CACEReadOutHeadConfig
from afm_distill.models import CACEStudentModelConfig as CACEStudentModelConfig
from afm_distill import OfflineDistillationConfig as OfflineDistillationConfig
from afm_distill.models import PaiNNCutoffFnConfig as PaiNNCutoffFnConfig
from afm_distill.models.painn.model import PaiNNNeighborListConfig as PaiNNNeighborListConfig
from afm_distill.models import PaiNNRBFConfig as PaiNNRBFConfig
from afm_distill.models import PaiNNStudentModelConfig as PaiNNStudentModelConfig
from afm_distill.models import SchNetCutoffFnConfig as SchNetCutoffFnConfig
from afm_distill.models.schnet.model import SchNetNeighborListConfig as SchNetNeighborListConfig
from afm_distill.models import SchNetRBFConfig as SchNetRBFConfig
from afm_distill.models import SchNetStudentModelConfig as SchNetStudentModelConfig
from afm_distill.main import StudentModelConfig as StudentModelConfig
from afm_distill import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill import data_registry as data_registry
from afm_distill import student_registry as student_registry

from . import main as main
from . import models as models
from . import registry as registry

__all__ = [
    "AllegroStudentModelConfig",
    "CACECutoffFnConfig",
    "CACERBFConfig",
    "CACEReadOutHeadConfig",
    "CACEStudentModelConfig",
    "OfflineDistillationConfig",
    "PaiNNCutoffFnConfig",
    "PaiNNNeighborListConfig",
    "PaiNNRBFConfig",
    "PaiNNStudentModelConfig",
    "SchNetCutoffFnConfig",
    "SchNetNeighborListConfig",
    "SchNetRBFConfig",
    "SchNetStudentModelConfig",
    "StudentModelConfig",
    "StudentModuleBaseConfig",
    "data_registry",
    "main",
    "models",
    "registry",
    "student_registry",
]
