from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# TODO: I really despise globals, this should go away

DATA_PREFIX: str = "data_"
DEFAULT_MODULES_DIR: str = "modules"
DEFAULT_BSIZE: int = 50_000

@dataclass(frozen=True)
class ModuleType:
    command: str
    name: str
    alias: str

    def alias_name(self) -> str:
        return self.alias or self.name
    
    def __str__(self) -> str:
        return self.name


class ModuleFactory:
    def __init__(self):
        self._lookup: OrderedDict[str, ModuleType] = OrderedDict()

    def register(self, mt: ModuleType):
        # Map each of the three identifiers to the same MType
        for key in (mt.command, mt.name, mt.alias):
            if key:  # allows alias=None
                self._lookup[key] = mt

    def get(self, key: str) -> ModuleType:
        try:
            return self._lookup[key]
        except KeyError:
            raise KeyError(f"No module type found for key: {key!r}")

    def all(self) -> list[ModuleType]:
        # dedupe by .command
        ret = []
        for v in self._lookup.values():
            if v not in ret: 
                ret.append(v)
        return ret

# ---- Define your module types ----

class ModuleEnum(Enum):
    SCANNER = ModuleType(command="scan", name="scanner", alias="s")
    CLASSIFIER = ModuleType(command="classify", name="classifier", alias="c")
    FINGERPRINTER = ModuleType(command="fingerprint", name="fingerprinter", alias="f")
    TAGGER = ModuleType(command="tag", name="tag", alias="t")

# ---- Build factory with registry ----

MFACTORY = ModuleFactory()
for m in ModuleEnum:
    MFACTORY.register(m.value)

def find_module(mod: str) -> ModuleType:
    return MFACTORY.get(mod)