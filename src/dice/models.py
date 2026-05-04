import pandas as pd

from typing import Optional
from sqlalchemy import Connection, MetaData, delete
from sqlmodel import Field, Relationship, SQLModel, Session, Table, UniqueConstraint, select


class Model(SQLModel):
    id: int | None = Field(default=None, primary_key=True)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_series(cls, row: pd.Series):
        return cls(**row.to_dict()) # type: ignore

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
    source_id: int = Field(default=None, foreign_key="source.id")

    cursor: "Cursor" = Relationship(back_populates="resource")

    def flush_records(self, con: Connection):
        with Session(con) as s:
            src = s.exec(select(Source).where(Source.id == self.source_id)).first()
            if not src:
                return
            
            tab = get_records_table(con, src.name, suffix="records")
            stmt = delete(tab).where(tab.c.resource_id == self.id)
            s.exec(stmt)


class Cursor(Model, table=True):
    resource_id: int = Field(
        default=None,
        foreign_key="resource.id",
        unique=True
    )
    idx: int = 0

    resource: Resource = Relationship(back_populates="cursor")

    def update(self, con: Connection, i: int=1):
        self.idx += i
        with Session(con) as s:
            s.add(self)
            s.commit()

    def done(self, con: Connection):
        self.idx=-1
        with Session(con) as s:
            s.add(self)
            s.commit()

class Host(Model, table=True):
    ip: str = Field(unique=True)

    domain: Optional[str] = None
    prefix: Optional[str] = None
    asn: Optional[str] = None
    

class Fingerprint(Model, table=True):
    # Host (ip)
    host: Optional[str] = Field(default=None, foreign_key="host.ip")
    # ID of the record related to
    record_id: Optional[int]
    resource_id: Optional[int] = Field(default=None, foreign_key="resource.id")

    # data, is a dict
    data: str
    # name of the module that created the fingerprint
    module_name: str

    # port and protocol from the record
    port: Optional[int] = None
    protocol: Optional[str] = None

    __table_args__ = (UniqueConstraint("record_id", "host", "module_name"),)


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
    fingerprint_id: Optional[int] = Field(default=None, foreign_key="fingerprint.id")
    # ID of the label
    label_id: Optional[int] = Field(default=None, foreign_key="label.id")

    __table_args__ = (UniqueConstraint("fingerprint_id", "label_id"),)


class Tag(Model, table=True):
    name: str = Field(unique=True)
    module_name: str
    description: str


class HostTag(Model, table=True):
    # Host (ip)
    host: Optional[str] = Field(default=None, foreign_key="host.ip")
    # ID of hte Tag
    tag_id: Optional[int] = Field(default=None, foreign_key="tag.id")
    # further details
    details: Optional[str] = None

    # Protocol and Port (optional)
    protocol: Optional[str] = None
    port: Optional[int] = None

    __table_args__ = (UniqueConstraint("host", "tag_id"),)


def get_records_table(con: Connection, name: str, suffix: Optional[str] = "", sep: Optional[str] ="_") -> Table:
    meta = MetaData()
    if sep and suffix:
        name = sep.join([name, suffix])
    table = Table(name, meta, autoload_with=con)
    return table