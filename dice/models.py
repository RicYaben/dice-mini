import pandas as pd
from dice.loaders import Loader
from dataclasses import dataclass, asdict

from abc import ABC

@dataclass
class Model(ABC):
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Source(Model):
    # ID of this source. Useful to distinguish between datasets
    # Example: query for all records from source zgrab2 with ID study-1
    id: str
    # name of the source, e.g., zgrab2
    name: str
    # path to the source: a list of directories, or files. Accepts globs.
    # Example: "dir/*/*.jsonl"
    paths: list[str]

    # an explicit loader to use for this source
    _handler: Loader

    def load(self) -> pd.DataFrame:
        return self._handler(self.id, self.name, self.paths)
    
@dataclass
class Host(Model):
    # ID of the host
    id: int
    # address of the host
    addr: str
    
@dataclass
class Fingerprint(Model):
    # ID of the fingerprint
    id: str
    # ID of the host
    host: str
    # ID of the record related to
    record_id: str
    # name of the module that created the fingerprint
    module_name: str
    # data, is a dict
    data: bytes
    
@dataclass
class Label(Model):
    # ID of the label
    id: str
    # name of the label
    name: str
    # descriptor
    description: str
    # mitigation strategy
    mitigation: str
    # name of the module that created this label
    module_name: str

@dataclass
class FingerprintLabel(Model):
    # ID of the fingerprint
    fingerprint_id: str
    # ID of the label
    label_id: str