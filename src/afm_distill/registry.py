from __future__ import annotations

import nshconfig as C

from .models.base import StudentModuleBaseConfig

student_registry = C.Registry(StudentModuleBaseConfig, discriminator="name")
"""Registry for student modules."""

data_registry = C.Registry(C.Config, discriminator="type")
"""Registry for data modules."""
__all__ = [
    "student_registry",
    "data_registry",
]
