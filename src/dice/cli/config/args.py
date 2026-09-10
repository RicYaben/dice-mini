from enum import Enum
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter

from dice.shared.modules import module_type_aliases

from .helpers import token_converter


class Mode(int, Enum):
    strict = 1
    normal = 2


ResultsArg = Annotated[
    Path,
    Parameter(name=["--results", "-res"], help="Path to results"),
]

BatchSizeArg = Annotated[
    int, Parameter(name=["--batch-size", "-bs"], help="Query batch size")
]

RegistriesArg = Annotated[
    list[str],
    Parameter(
        name=["--registries", "-Mr"],
        help="Comma-separated list of registries",
        converter=token_converter(","),
    ),
]

CommandsArg = Annotated[
    list[str],
    Parameter(
        name=["--commands", "-C"],
        choices=module_type_aliases(),
        help="Command-separated list of commands to execute",
        converter=token_converter(","),
    ),
]

ModeArg = Annotated[
    Mode,
    Parameter(
        name=["--mode", "-m"],
        help="Mode to run the recipe in",
    ),
]

ModulesArg = Annotated[
    list[str],
    Parameter(
        name=["--modules", "-M"],
        help="Comma-separated list of modules to load",
        converter=token_converter(","),
    ),
]
