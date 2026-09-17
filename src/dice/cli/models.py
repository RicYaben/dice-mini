import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import ujson
from cyclopts import Parameter

from dice.modules import flags
from dice.shared.modules import (
    ModuleType,
    filter_module_types,
    max_module_types,
)

from .args import (
    BatchSizeArg,
    CommandsArg,
    Mode,
    ModeArg,
    RegistriesArg,
    ResultsArg,
)
from .helpers import token_converter


@dataclass
class ConfigOptions:
    configuration: Annotated[
        Path | None, Parameter(name=["--conf", "-c"], help="Path to configuration file")
    ] = None
    batch_size: BatchSizeArg | None = None
    health: Annotated[
        str | None, Parameter(name=["--health", "-h"], help="Health checks to perform")
    ] = None

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "health": (self.health.split(",") if self.health is not None else None),
        }


@dataclass
class ModuleOptions:
    registries: RegistriesArg | None = None
    fl: Annotated[
        Path | None,
        Parameter(name=["--flags", "-Mf"], help="Path to module flags file"),
    ] = None
    params: Annotated[
        str | None,
        Parameter(name=["--params", "-Mp"], help="Module parameters (JSON string)"),
    ] = None

    def to_dict(self) -> dict:
        fgs = flags(self.fl)
        if self.params is not None:
            params = ujson.loads(self.params)
            fgs.update_all(**params)

        return {
            "registries": (self.registries),
            "flags": fgs,
        }


@dataclass
class LogOptions:
    logs: Annotated[Path | None, Parameter(name=["--logs", "-l"])] = None
    level: Annotated[str | None, Parameter(name=["--verbose-level", "-v"])] = None

    def to_dict(self) -> dict:
        return {
            "level": (
                getattr(logging, self.level.upper()) if self.level is not None else None
            ),
            "logs": self.logs,
        }


@dataclass
class DatabaseOptions:
    results: ResultsArg | None = None
    cookbook: Annotated[
        Path | None,
        Parameter(name=["--cookbook", "-cb"], help="Path to the cookbook file"),
    ] = None

    def to_dict(self) -> dict:
        return {
            "results": self.results,
            "cookbook": self.cookbook,
        }


@dataclass
class CommandOptions:
    commands: CommandsArg | None = None
    mode: ModeArg | None = Mode.normal

    def resolve(self) -> list[ModuleType] | None:
        if not self.commands:
            return None

        match self.mode:
            case Mode.strict | None:
                return filter_module_types(self.commands)
            case Mode.normal:
                return max_module_types(self.commands)


@dataclass
class SearchOptions:
    query: Annotated[
        str | None, Parameter(name=["--query", "-q"], help="Search query")
    ] = None
    include: Annotated[
        list[str] | None,
        Parameter(
            name=["--include", "-i"],
            help="Comma-separated list of columns to include",
            converter=token_converter(","),
        ),
    ] = None
    exclude: Annotated[
        list[str] | None,
        Parameter(
            name=["--exclude", "-e"],
            help="Comma-separated list of columns to exclude",
            converter=token_converter(","),
        ),
    ] = None
    limit: Annotated[
        int | None,
        Parameter(
            name=["--limit", "-l"],
            help="Limit the number of results",
        ),
    ] = None
    offset: Annotated[
        int | None,
        Parameter(
            name=["--offset", "-o"],
            help="Offset the results",
        ),
    ] = None
