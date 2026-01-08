from dataclasses import dataclass
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_PREFIX: str = "data_"
DEFAULT_MODULES_DIR: str = "modules"
DEFAULT_BSIZE: int = 50_000

@dataclass(frozen=True)
class MType:
    command: str
    name: str
    alias: str

    def alias_name(self) -> str:
        return self.alias or self.name
    
    def __str__(self) -> str:
        return self.name


class MFactory:
    def __init__(self):
        self._lookup: OrderedDict[str, MType] = OrderedDict()

    def register(self, mt: MType):
        # Map each of the three identifiers to the same MType
        for key in (mt.command, mt.name, mt.alias):
            if key:  # allows alias=None
                self._lookup[key] = mt

    def get(self, key: str) -> MType:
        try:
            return self._lookup[key]
        except KeyError:
            raise KeyError(f"No module type found for key: {key!r}")

    def all(self) -> list[MType]:
        # dedupe by .command
        ret = []
        for v in self._lookup.values():
            if v not in ret: 
                ret.append(v)
        return ret

# ---- Define your module types ----

SCANNER = MType(command="scan", name="scanner", alias="s")
CLASSIFIER = MType(command="classify", name="classifier", alias="c")
FINGERPRINTER = MType(command="fingerprint", name="fingerprinter", alias="f")
TAGGER = MType(command="tag", name="tag", alias="t")

# ---- Build factory with registry ----

MFACTORY = MFactory()
for m in [SCANNER, FINGERPRINTER, CLASSIFIER, TAGGER]:
    MFACTORY.register(m)

