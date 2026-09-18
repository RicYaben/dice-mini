from pathlib import Path

from dice.internal.engine.modules import ModuleRegistry, new_registry
from dice.shared.config import ModuleFlags, load_flags


def flags(fpath: Path | None):
    return load_flags(fpath) if fpath else ModuleFlags({})


def registry(name: str) -> ModuleRegistry:
    return new_registry(name)
