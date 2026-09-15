from pathlib import Path

from dice.internal.config import ModuleFlags, load_flags


def flags(fpath: Path | None):
    return load_flags(fpath) if fpath else ModuleFlags({})
