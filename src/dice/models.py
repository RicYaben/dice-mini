import hashlib
import uuid
import pandas as pd
from dice.config import DEFAULT_BSIZE
from dice.loaders import Loader
from dataclasses import KW_ONLY, dataclass, asdict, field

from abc import ABC
from typing import Generator

@dataclass
class Model(ABC):
    _: KW_ONLY
    # synthetic primary key ALWAYS present
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        # if the model wants deterministic composite identity,
        # override generate_pk() to compute it from its fields.
        self.id = self.generate_pk()

    def generate_pk(self) -> str:
        if self.primary_key() == ("id",):
            return self.id  # keep UUID
        return self.compute_composite_hash()

    def compute_composite_hash(self) -> str:
        fields = self.primary_key()
        h = hashlib.sha256()
        for f in fields:
            h.update(str(getattr(self, f)).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def primary_key(cls) -> tuple[str, ...]:
        return ("id",)
    
    @classmethod
    def table(cls) -> str:
        raise NotImplementedError
    
    @classmethod
    def from_series(cls, row: pd.Series) -> "Model":
        return cls(**row.to_dict())
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["Model"]:
        return [cls.from_series(row) for _, row in df.iterrows()]

@dataclass
class Source(Model):
    # name of the source, e.g., zgrab2
    name: str
    # id of the study
    study: str
    # path to the source: a list of directories, or files. Accepts globs.
    # Example: "dir/*/*.jsonl"
    paths: list[str]

    # an explicit loader to use for this source
    _handler: Loader

    # size of the batches for loading and saving
    _batch_size: int = DEFAULT_BSIZE

    @classmethod
    def table(cls) -> str:
        return "sources"

    def load(self) -> Generator[pd.DataFrame, None, None]:
        return self._handler(self.id, self.name, self.study, self.paths, self._batch_size)
    
@dataclass
class Host(Model):
    # address of the host
    ip: str
    domain: str
    # prefix
    prefix: str
    asn: str

    @classmethod
    def primary_key(cls):
        return ("ip",)
    
    @classmethod
    def table(cls) -> str:
        return "hosts"
    
@dataclass
class Fingerprint(Model):
    # Host (ip)
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

    @classmethod
    def primary_key(cls):
        return ("record_id", "module_name")
    
    @classmethod
    def table(cls) -> str:
        return "fingerprints"
    
@dataclass
class Label(Model):
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

    @classmethod
    def primary_key(cls):
        return ("name", "module_name")
    
    @classmethod
    def table(cls) -> str:
        return "labels"

@dataclass
class FingerprintLabel(Model):
    # ID of the fingerprint
    fingerprint_id: str
    # ID of the label
    label_id: str

    @classmethod
    def primary_key(cls):
        return ("fingerprint_id", "label_id")
    
    @classmethod
    def table(cls) -> str:
        return "fingerprint_labels"

@dataclass
class Tag(Model):
    name: str
    description: str
    module_name: str

    @classmethod
    def primary_key(cls):
        return ("name", "module_name")
    
    @classmethod
    def table(cls) -> str:
        return "tags"

@dataclass
class HostTag(Model):
    # Host (ip)
    host: str
    # ID of hte Tag
    tag_id: str
    # further details
    details: str
    # Protocol and Port (optional)
    protocol: str
    port: int

    @classmethod
    def primary_key(cls):
        return ("host", "tag_id")
    
    @classmethod
    def table(cls) -> str:
        return "host_tags"
    
M_REQUIRED = [Host, Tag, HostTag, Fingerprint, Label, FingerprintLabel]