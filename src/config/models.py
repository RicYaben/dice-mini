import logging
from pathlib import Path

import ujson
from pydantic import BaseModel

from dice.internal.config import ModuleFlags, load_flags


class ConfigOptions(BaseModel):
    configuration: Path | None = None
    query_bsize: int | None = None
    health: str | None = None

    def overrides(self) -> dict:
        return {
            "query_bsize": self.query_bsize,
            "health": (
                self.health.split(",")
                if self.health is not None
                else None
            ),
        }


class ModuleOptions(BaseModel):
    registries: str | None = None
    module_flags: Path | None = None
    module_params: str | None = None

    def overrides(self) -> dict:
        flags = None

        if self.module_flags is not None:
            flags = load_flags(self.module_flags)

        if self.module_params is not None:
            params = ujson.loads(self.module_params)

            if flags is None:
                flags = ModuleFlags({})

            flags.update_all(**params)

        return {
            "registries": (
                self.registries.split(",")
                if self.registries is not None
                else None
            ),
            "flags": flags,
        }


class LogOptions(BaseModel):
    log_level: str | None = None
    log_file: str | None = None
    log_error: str | None = None

    def overrides(self) -> dict:
        return {
            "level": (
                getattr(logging, self.log_level.upper())
                if self.log_level is not None
                else None
            ),
            "file": self.log_file,
            "errors": self.log_error,
        }


class DatabaseOptions(BaseModel):
    database: str | None = None
    cookbook: str | None = None

    def overrides(self) -> dict:
        return {
            "results": self.database,
            "cookbook": self.cookbook,
        }
