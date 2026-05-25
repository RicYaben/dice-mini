"""
Shared interfaces
"""
from .modules import ModuleDescriptor
from .flags import Flag, Flags, flag
from .models import Label, Tag

__all__ = [
    "ModuleDescriptor",
    "Flag",
    "Flags",
    "Label",
    "Tag"
]