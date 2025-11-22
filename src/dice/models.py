import pandas as pd
from dice.loaders import Loader
from dataclasses import dataclass, asdict

from abc import ABC
from typing import Generator

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
    # id of the study
    study: str
    # path to the source: a list of directories, or files. Accepts globs.
    # Example: "dir/*/*.jsonl"
    paths: list[str]
    # size of the batches for loading and saving
    batch_size: int

    # an explicit loader to use for this source
    _handler: Loader
    

    def load(self) -> Generator[pd.DataFrame, None, None]:
        return self._handler(self.id, self.name, self.study, self.paths, self.batch_size)
    
@dataclass
class Host(Model):
    # ID of the host
    id: str
    # address of the host
    ip: str
    domain: str
    # prefix
    prefix: str
    asn: str
    
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
    data: str

    # port and protocol from the record
    port: int
    protocol: str
    
@dataclass
class Label(Model):
    # ID of the label
    id: str
    # name of the label
    name: str
    # short descriptor
    short: str
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

@dataclass
class Tag(Model):
    id: str
    name: str
    description: str
    module_name: str

@dataclass
class HostTag(Model):
    # Host ID
    host: str
    # ID of hte Tag
    tag_id: str
    # further details
    details: str
    # Protocol and Port (optional)
    protocol: str
    port: int