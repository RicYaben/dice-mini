"""
Shared interfaces
"""

from .flags import Flag, Flags, flag
from .models import Label, Tag
from .modules import ModuleDescriptor
from .query import query

__all__ = ["ModuleDescriptor", "Flag", "flag", "Flags", "Label", "Tag", "query"]
