import hashlib
import uuid
import re
import pandas as pd

from abc import ABC
from typing import Generator, Optional

from dice.config import DEFAULT_BSIZE
from dice.loaders import Loader, file_loader
from dataclasses import KW_ONLY, dataclass, asdict, field

@dataclass
class Model(ABC):
    _: KW_ONLY
    _pk: tuple[str, ...] = ("id", )
    # synthetic primary key ALWAYS present
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        # if the model wants deterministic composite identity,
        # override generate_pk() to compute it from its fields.
        self.id = self.generate_pk()

    def generate_pk(self) -> str:
        if self._pk == ("id",):
            return self.id  # keep UUID
        return self.compute_composite_hash()

    def compute_composite_hash(self) -> str:
        fields = self._pk
        h = hashlib.sha256()
        for f in fields:
            h.update(str(getattr(self, f)).encode())
        return h.hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def table(cls) -> str:
        # Split CamelCase into words
        parts = re.findall(r'[A-Z][a-z0-9]*', cls.__name__)
        name = "_".join(p.lower() for p in parts)

        # Ensure it ends with 's'
        if not name.endswith("s"):
            name += "s"

        return name
    
    @classmethod
    def from_series(cls, row: pd.Series) -> "Model":
        return cls(**row.to_dict())
    
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["Model"]:
        return [cls.from_series(row) for _, row in df.iterrows()]
    
    @classmethod
    def mock(cls) -> pd.DataFrame:
        return pd.DataFrame([cls().to_dict()])

@dataclass
class Source(Model):
    _pk = ("name",)

    # name of the source, e.g., zgrab2
    name: str = "-"
    # id of the study
    study: str = "-"
    # path to the source: a glob
    # Example: "dir/*/*.jsonl"
    path: str = "-"

    # size of the batches for loading and saving
    batch_size: int = DEFAULT_BSIZE

    # an explicit loader to use for this source
    _handler: Loader = file_loader

    def load(self) -> Generator[pd.DataFrame, None, None]:
        return self._handler(self.id, self.name, self.study, self.path, self.batch_size)
    
@dataclass
class Counter(Model):
    _pk = ("name",)

    name: str = "-"
    value: int = 0

@dataclass
class Cursor(Model):
    _pk = ("source",)
    source: str = "-"
    index: int = 0
    
    def reset(self):
        self.index=0
    
    def update(self, idx: int=1):
        self.index+=idx
    
@dataclass
class Host(Model):
    _pk = ("ip",)
    # address of the host
    ip: str = "-"
    domain: str = "-"
    # prefix
    prefix: str = "-"
    asn: str = "-"
    
@dataclass
class Fingerprint(Model):
    _pk = ("record_id", "module_name")
    # Host (ip)
    host: str = "-"
    # ID of the record related to
    record_id: str = "-"
    # name of the module that created the fingerprint
    module_name: str = "-"
    # data, is a dict
    data: str = "{}"

    # port and protocol from the record
    port: int = 0
    protocol: str = "-"
    
@dataclass
class Label(Model):
    _pk = ("name", "module_name")
    # name of the label
    name: str = "-"
    # short descriptor
    short: str = "-"
    # descriptor
    description: str = "-"
    # mitigation strategy
    mitigation: str = "-"
    # name of the module that created this label
    module_name: str = "-"

@dataclass
class FingerprintLabel(Model):
    _pk = ("fingerprint_id", "label_id")
    # ID of the fingerprint
    fingerprint_id: str = "-"
    # ID of the label
    label_id: str = "-"

@dataclass
class Tag(Model):
    _pk = ("name", "module_name")
    name: str = "-"
    description: str = "-"
    module_name: str = "-"

@dataclass
class HostTag(Model):
    _pk = ("host", "tag_id")
    # Host (ip)
    host: str = "-"
    # ID of hte Tag
    tag_id: str = "-"
    # further details
    details: str = "-"

    # Protocol and Port (optional)
    protocol: Optional[str] = None
    port: Optional[int] = None

M_REQUIRED = [
    Host, 
    Tag, 
    HostTag, 
    Fingerprint, 
    Label, 
    FingerprintLabel, 
    Counter,
    Cursor,
    Source,
]