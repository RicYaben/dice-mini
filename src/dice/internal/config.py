import logging
import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, RootModel
from tomlkit import dumps

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class NoSectionError(Exception):
    pass


class Config(BaseModel):
    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, "rb") as f:
            return cls.model_validate(tomllib.load(f))

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(dumps(self.model_dump()))

    def override(self, **kwargs: Any) -> Self:
        for key, value in kwargs.items():
            if value is None:
                continue

            current = getattr(self, key, None)

            if isinstance(current, Config) and isinstance(value, dict):
                current.override(**value)
            else:
                setattr(self, key, value)

        return self


class ModuleFlags(RootModel[dict[str, dict[str, Any]]]):
    def update(self, module: str, **kwargs: Any) -> None:
        self.root.setdefault(module, {}).update(kwargs)

    def update_all(self, **kwargs: dict[str, Any]) -> Self:
        for module, values in kwargs.items():
            self.update(module, **values)
        return self

    def __getitem__(self, module: str) -> dict[str, Any]:
        return self.root[module]

    def __setitem__(self, module: str, values: dict[str, Any]) -> None:
        self.root[module] = values


def load_flags(fpath: str | Path | None) -> ModuleFlags:
    if not fpath:
        return ModuleFlags({})

    with open(fpath, "rb") as f:
        return ModuleFlags.model_validate(tomllib.load(f))


class ModulesConf(Config):
    registries: list[str] = Field(default_factory=list)
    flags: ModuleFlags = Field(default_factory=lambda: ModuleFlags({}))


class DatabasesConf(Config):
    results: str | None = None
    cookbook: str | None = None


class LogsConf(Config):
    level: int = logging.INFO
    logs: Path | None = None


class DiceConfig(Config):
    batch_size: int = 50_000
    health: list[str] = Field(default_factory=list)

    databases: DatabasesConf = Field(default_factory=DatabasesConf)
    modules: ModulesConf = Field(default_factory=ModulesConf)
    logs: LogsConf = Field(default_factory=LogsConf)


def load_configuration(fpath: str | Path) -> DiceConfig:
    return DiceConfig.load(fpath)


def make_configuration(fpath: str | Path | None = None) -> DiceConfig:
    if fpath is not None:
        return DiceConfig.load(fpath)
    return DiceConfig()
