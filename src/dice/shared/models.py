import pandas as pd
from sqlalchemy import Connection, MetaData, delete
from sqlmodel import (
    JSON,
    Column,
    Field,
    Relationship,
    Session,
    SQLModel,
    Table,
    UniqueConstraint,
    select,
)


class Model(SQLModel):
    id: int | None = Field(default=None, primary_key=True)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_series(cls, row: pd.Series):
        return cls(**row.to_dict())  # type: ignore

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


class Record(Model, table=True):
    host: str | None = Field(default=None, foreign_key="host.ip")
    source: str | None = Field(default=None, foreign_key="source.name")
    resource_id: int = Field(default=None, foreign_key="resource.id")

    data: dict = Field(sa_column=Column(JSON))
    port: int | None = None
    protocol: str | None = None

    __table_args__ = (UniqueConstraint("resource_id", "host"),)


class Cursor(Model, table=True):
    resource_id: int = Field(default=None, foreign_key="resource.id", unique=True)
    idx: int = 0

    resource: Resource = Relationship(back_populates="cursor")

    def update(self, s: Session, i: int = 1):
        self.idx += i
        s.commit()

    def done(self, s: Session):
        self.idx = -1
        s.commit()


class Host(Model, table=True):
    ip: str = Field(unique=True)
    domain: str | None = None

    prefix: str | None = None
    asn: str | None = None


class Fingerprint(Model, table=True):
    host: str | None = Field(default=None, foreign_key="host.ip")
    record_id: int | None = Field(default=None, foreign_key="record.id")

    data: dict = Field(sa_column=Column(JSON))
    module_name: str | None = None
    protocol: str | None = None

    __table_args__ = (UniqueConstraint("record_id", "host", "module_name"),)


class Label(Model, table=True):
    name: str = Field(unique=True)
    module_name: str | None = None
    description: str | None = None
    short: str | None = None
    mitigation: str | None = None
    level: int = 0


class FingerprintLabel(Model, table=True):
    fingerprint_id: int | None = Field(default=None, foreign_key="fingerprint.id")
    label_id: int | None = Field(default=None, foreign_key="label.id")

    __table_args__ = (UniqueConstraint("fingerprint_id", "label_id"),)


class Tag(Model, table=True):
    name: str = Field(unique=True)
    module_name: str | None = None
    description: str | None = None


class HostTag(Model, table=True):
    host: str | None = Field(default=None, foreign_key="host.ip")
    tag_id: int | None = Field(default=None, foreign_key="tag.id")
    details: str | None = None
    protocol: str | None = None
    port: int | None = None

    __table_args__ = (UniqueConstraint("host", "tag_id"),)


def get_records_table(
    con: Connection, name: str, suffix: str | None = "", sep: str | None = "_"
) -> Table:
    meta = MetaData()
    if sep and suffix:
        name = sep.join([name, suffix])
    table = Table(name, meta, autoload_with=con)
    return table
