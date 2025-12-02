from dataclasses import dataclass
from typing import Dict

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
        self._lookup: Dict[str, MType] = {}

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

    def all(self):
        # dedupe by .command
        seen = set()
        uniq = []
        for mt in self._lookup.values():
            if mt.command not in seen:
                uniq.append(mt)
                seen.add(mt.command)
        return uniq

# ---- Define your module types ----

SCANNER = MType(command="scan", name="scanner", alias="s")
CLASSIFIER = MType(command="classify", name="classifier", alias="c")
FINGERPRINTER = MType(command="fingerprint", name="fingerprinter", alias="f")
TAGGER = MType(command="tag", name="tag", alias="t")

# ---- Build factory with registry ----

MFACTORY = MFactory()
for m in [SCANNER, CLASSIFIER, FINGERPRINTER, TAGGER]:
    MFACTORY.register(m)

