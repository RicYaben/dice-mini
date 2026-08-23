"""
Public interfaces
"""

from dice.internal.modules import new_registry
from dice.shared import query
from dice.shared.flags import Flag, Flags, flag

from .tools import Module, Service

__all__ = [
    "Flag",
    "Flags",
    "Module",
    "Service",
    "flag",
    "new_registry",
    "query",
]
