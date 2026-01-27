from __future__ import annotations

from .models.base import StudentModuleBase, StudentModuleBaseConfig
from .main import OfflineDistillation, OfflineDistillationConfig
from .registry import student_registry as student_registry
from .registry import data_registry as data_registry

try:
    from . import configs as configs
except ImportError:
    pass
