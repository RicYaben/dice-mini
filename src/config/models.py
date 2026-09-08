import logging
from pathlib import Path
from typing import Annotated

import ujson
from cyclopts import Parameter
from pydantic import BaseModel

from dice.internal.config import ModuleFlags, load_flags


class ConfigOptions(BaseModel):
    configuration: Annotated[
        Path | None, Parameter(name=["--conf", "-c"], help="Path to configuration file")
    ] = None
    query_bsize: Annotated[
        int | None, Parameter(name=["--query-bsize", "-q"], help="Query batch size")
    ] = None
    health: Annotated[
        str | None, Parameter(name=["--health", "-h"], help="Health checks to perform")
    ] = None

    def overrides(self) -> dict:
        return {
            "query_bsize": self.query_bsize,
            "health": (self.health.split(",") if self.health is not None else None),
        }


class ModuleOptions(BaseModel):
    registries: Annotated[
        str | None,
        Parameter(
            name=["--registries", "-r"], help="Comma-separated list of registries"
        ),
    ] = None
    flags: Annotated[
        Path | None,
        Parameter(name=["--flags", "-Mf"], help="Path to module flags file"),
    ] = None
    params: Annotated[
        str | None,
        Parameter(name=["--params", "-Mp"], help="Module parameters (JSON string)"),
    ] = None

    def overrides(self) -> dict:
        flags = None

        if self.flags is not None:
            flags = load_flags(self.flags)

        if self.params is not None:
            params = ujson.loads(self.params)

            if flags is None:
                flags = ModuleFlags({})

            flags.update_all(**params)

        return {
            "registries": (
                self.registries.split(",") if self.registries is not None else None
            ),
            "flags": flags,
        }


class LogOptions(BaseModel):
    # TODO: this could be just a "--logs" fpath
    logs: Annotated[str | None, Parameter(name=["--logs", "-l"])] = None
    verbose: Annotated[str | None, Parameter(name=["--verbose", "-v"])] = None

    def overrides(self) -> dict:
        return {
            "level": (
                getattr(logging, self.verbose.upper())
                if self.verbose is not None
                else None
            ),
            "logs": self.logs,
        }


class DatabaseOptions(BaseModel):
    results: Annotated[
        Path | None,
        Parameter(name=["--results", "-db"], help="Path to results"),
    ] = None
    cookbook: Annotated[
        Path | None,
        Parameter(name=["--cookbook", "-cdb"], help="Path to the cookbook file"),
    ] = None

    def overrides(self) -> dict:
        return {
            "results": self.results,
            "cookbook": self.cookbook,
        }
