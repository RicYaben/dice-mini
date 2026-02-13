import uuid
import pandas as pd

from typing import Optional
from sqlmodel import Field, Relationship, SQLModel, UniqueConstraint


class Model(SQLModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_series(cls, row: pd.Series):
        return cls(**row.to_dict())

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        return [cls.from_series(row) for _, row in df.iterrows()]


class Source(Model, table=True):
    "A source represents the content of a set of resources (datasets)"

    # name of the source, e.g., zgrab2
    name: str = Field(unique=True)
    resources: list["Resource"] = Relationship()


class Resource(Model, table=True):
    fpath: str = Field(unique=True)
    source_id: str = Field(default=None, foreign_key="source.id")

    cursor: "Cursor" = Relationship()


class Cursor(Model, table=True):
    resource_id: Optional[str] = Field(default=None, foreign_key="resource.id", unique=True)
    index: int = 0
    done: bool = False


class Host(Model, table=True):
    ip: str = Field(unique=True)

    domain: Optional[str] = None
    prefix: Optional[str] = None
    asn: Optional[str] = None
    

class Fingerprint(Model, table=True):
    # Host (ip)
    host_id: Optional[str] = Field(default=None, foreign_key="host.id")
    # ID of the record related to
    record_id: Optional[str]
    source_id: Optional[str] = Field(default=None, foreign_key="source.id")

    # data, is a dict
    data: str
    # name of the module that created the fingerprint
    module_name: str

    # port and protocol from the record
    port: Optional[int] = None
    protocol: Optional[str] = None

    __table_args__ = (UniqueConstraint("record_id", "host_id", "module_name"),)


class Label(Model, table=True):
    # name of the label
    name: str = Field(unique=True)
    # name of the module that created this label
    module_name: str
    # descriptor
    description: Optional[str] = None
    # short descriptor
    short: Optional[str] = None
    # mitigation strategy
    mitigation: Optional[str] = None
    level: int = 0


class FingerprintLabel(Model, table=True):
    # ID of the fingerprint
    fingerprint_id: Optional[str] = Field(default=None, foreign_key="fingerprint.id")
    # ID of the label
    label_id: Optional[str] = Field(default=None, foreign_key="label.id")

    __table_args__ = (UniqueConstraint("fingerprint_id", "label_id"),)


class Tag(Model, table=True):
    name: str = Field(unique=True)
    module_name: str
    description: str


class HostTag(Model, table=True):
    # Host (ip)
    host_id: Optional[str] = Field(default=None, foreign_key="host.id")
    # ID of hte Tag
    tag_id: Optional[str] = Field(default=None, foreign_key="tag.id")
    # further details
    details: Optional[str] = None

    # Protocol and Port (optional)
    protocol: Optional[str] = None
    port: Optional[int] = None

    __table_args__ = (UniqueConstraint("host_id", "tag_id"),)
