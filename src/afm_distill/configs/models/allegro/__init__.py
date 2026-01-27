__codegen__ = True

from afm_distill.models.allegro.model import AllegroStudentModelConfig as AllegroStudentModelConfig
from afm_distill.models.allegro.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.models.allegro.model import AllegroStudentModelConfig as AllegroStudentModelConfig
from afm_distill.models.allegro.model import StudentModuleBaseConfig as StudentModuleBaseConfig

from afm_distill.models.allegro.model import student_registry as student_registry

from . import model as model

__all__ = [
    "AllegroStudentModelConfig",
    "StudentModuleBaseConfig",
    "model",
    "student_registry",
]
