from sqlite3 import IntegrityError
from typing import Iterable, Literal, Optional, Type
from sqlalchemy import Connection, Engine
from sqlmodel import SQLModel, Session, UniqueConstraint, exists, select, insert, values, create_engine

def insert_records(session: Session, model: Type, records: Iterable[dict]):
    """
    Quickest way to blindly insert thousands of records into a database.
    Mainly used by the sourcerer to insert actual records.
    https://github.com/fastapi/sqlmodel/discussions/659
    """
    session.exec(insert(model), params=records) # type: ignore
    session.commit()

def _get_unique_columns(model) -> list[str]:
    table = model.__table__

    # Prefer composite unique constraints
    for c in table.constraints:
        if isinstance(c, UniqueConstraint):
            return [col.name for col in c.columns]

    # Fallback to column-level unique=True
    cols = [c.name for c in table.columns if c.unique]
    return cols


def insert_or_ignore(
    session: Session,
    model: Type,
    items: Iterable[dict],
) -> None:
    items = list(items)
    if not items:
        return

    table = model.__table__
    cols = list(items[0].keys())
    unique_cols = _get_unique_columns(model)

    # VALUES table (src_df equivalent)
    src = values(
        *[table.c[c] for c in cols],
        name="src"
    ).data(
        [tuple(item[c] for c in cols) for item in items]
    )

    # WHERE NOT EXISTS (...)
    not_exists = ~exists(
        select(1).where(
            *[
                table.c[c] == src.c[c]
                for c in unique_cols
            ]
        )
    )

    stmt = (
        insert(table)
        .from_select(
            cols,
            select(*[src.c[c] for c in cols]).where(not_exists)
        )
    )

    session.exec(stmt)
    session.commit()

def get_or_create(session: Session, model, **kwargs):
    # Try to get existing
    obj = session.exec(select(model).filter_by(**kwargs)).first() # type: ignore
    if obj:
        return obj, False

    # Try to insert
    obj = model(**kwargs)
    session.add(obj)
    try:
        session.commit()
        session.refresh(obj)
        return obj, True
    except IntegrityError:
        session.rollback()
        # Another process created it first
        obj = session.exec(select(model).filter_by(**kwargs)).one()
        return obj, False


class Connector:
    def __init__(
        self,
        location: Optional[str],
        driver: Literal["sqlite"] = "sqlite",
    ) -> None:
        self.location = location if location else ":memory:"
        self.driver = driver
        self.engine: Optional[Engine] = None

    def load(self):
        e = create_engine(f"{self.driver}:///{self.location}")
        SQLModel.metadata.create_all(e)
        
        self.engine = e
        return e

    def connection(self) -> Connection:
        if not self.engine:
            self.load()
        return self.engine.connect() # type: ignore
    
    def session(self) -> Session:
        return Session(self.connection())


def new_connector(db: Optional[str], name: Literal["sqlite"] = "sqlite") -> Connector:
    return Connector(db, driver=name)