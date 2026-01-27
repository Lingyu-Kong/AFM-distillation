__codegen__ = True

from afm_distill.main import OfflineDistillationConfig as OfflineDistillationConfig
from afm_distill.main import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.main import OfflineDistillationConfig as OfflineDistillationConfig
from afm_distill.main import StudentModelConfig as StudentModelConfig
from afm_distill.main import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.main import data_registry as data_registry
from afm_distill.main import student_registry as student_registry


__all__ = [
    "OfflineDistillationConfig",
    "StudentModelConfig",
    "StudentModuleBaseConfig",
    "data_registry",
    "student_registry",
]
