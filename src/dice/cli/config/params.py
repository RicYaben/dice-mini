from typing import Annotated

from cyclopts import Parameter

from .models import ConfigOptions, DatabaseOptions, LogOptions, ModuleOptions

ConfigOptionsArg = Annotated[
    ConfigOptions,
    Parameter(name="*"),
]

ModuleOptionsArg = Annotated[
    ModuleOptions,
    Parameter(name="*"),
]

LogOptionsArg = Annotated[
    LogOptions,
    Parameter(name="*"),
]

DatabaseOptionsArg = Annotated[
    DatabaseOptions,
    Parameter(name="*"),
]
